"""Single-process Ollama chat router with online contextual-bandit selection."""
import asyncio
import contextlib
import math
import os
import time
from contextlib import asynccontextmanager

import httpx
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bandit import LinUCB, features
from category import classify
from monitor import collect_all, read_config


class ChatRequest(BaseModel):
    input_tokens: int = Field(ge=0, strict=True)
    category: str | None = Field(default=None, min_length=1)
    messages: list[dict] = Field(min_length=1)


class CategoryRequest(BaseModel):
    text: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app):
    config = read_config(os.environ.get('ROUND_CONFIG', 'config.json'))
    bundle = joblib.load(config['predictor_path'])
    meta = bundle['metadata']
    if meta.get('model_digest') != config['model_digest'] or meta.get('model_name') != config['model']:
        raise RuntimeError('Predictor model metadata does not match configured model/digest')
    classifier = joblib.load(config.get('category_model_path', 'category_model.joblib'))
    if set(classifier['categories']) != set(meta['categories']):
        raise RuntimeError('Classifier and token predictor categories do not match')
    app.state.config, app.state.bundle = config, bundle
    app.state.classifier = classifier
    app.state.bandit = LinUCB(config['state_db'])
    app.state.lock = asyncio.Lock()
    app.state.active = {s['id']: 0 for s in config['servers']}
    app.state.status = {}
    app.state.client = httpx.AsyncClient(timeout=config['request_timeout_seconds'], trust_env=False)
    app.state.monitor_client = httpx.AsyncClient(timeout=3, trust_env=False)

    async def refresh():
        while True:
            rows = await collect_all(app.state.monitor_client, config)
            async with app.state.lock:
                app.state.status = {r['server_id']: r for r in rows}
            await asyncio.sleep(config['poll_seconds'])

    task = asyncio.create_task(refresh())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await app.state.client.aclose()
        await app.state.monitor_client.aclose()
        app.state.bandit.db.close()


app = FastAPI(lifespan=lifespan)


@app.post('/classify')
async def classify_text(request: CategoryRequest):
    try:
        return classify(app.state.classifier, request.text)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/status')
async def status():
    async with app.state.lock:
        return {'servers': app.state.status, 'inflight': app.state.active,
                'observations': {s['id']: app.state.bandit.state(s['id'])[2] for s in app.state.config['servers']}}


@app.post('/chat')
async def chat(request: ChatRequest):
    state = app.state
    meta = state.bundle['metadata']
    classification = None
    category = request.category
    if category is None:
        user_messages = [m.get('content') for m in request.messages if m.get('role') == 'user']
        try:
            classification = classify(state.classifier, user_messages[-1] if user_messages else None)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        category = classification['category']
    if category not in meta['categories']:
        raise HTTPException(422, 'Unknown category. Use a category from the predictor training data.')
    predicted_float = float(state.bundle['estimator'].predict([[request.input_tokens, category]])[0])
    if not math.isfinite(predicted_float):
        raise HTTPException(500, 'Predictor returned a non-finite value')
    predicted = max(0, round(predicted_float))
    async with state.lock:
        eligible = []
        for server in state.config['servers']:
            sid = server['id']
            snapshot = state.status.get(sid, {})
            if not snapshot.get('healthy') or time.time() - snapshot['checked_at'] > state.config['max_metric_age_seconds']:
                continue
            if state.active[sid] >= server.get('max_inflight', 1):
                continue
            x = features(snapshot['metrics'], request.input_tokens, predicted, state.active[sid])
            eligible.append((state.bandit.score(sid, x), server, x))
        if not eligible:
            raise HTTPException(503, 'No healthy server with available capacity')
        # Stable tie breaking; confidence bonuses drive exploration once rewards arrive.
        _, server, x = max(eligible, key=lambda item: (item[0], item[1]['id']))
        sid = server['id']
        state.active[sid] += 1
    payload = {'model': state.config['model'], 'messages': request.messages,
               'stream': False, 'think': False, 'keep_alive': '30m', 'options': meta['settings']}
    started = time.monotonic()
    success, actual = False, None
    try:
        response = await state.client.post(server['ollama_url'].rstrip('/') + '/api/chat', json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get('done') is not True or not isinstance(body.get('eval_count'), int):
            raise ValueError('Invalid Ollama response or missing token count')
        actual = body['eval_count']
        success = True
        return {'server_id': sid, 'category': category, 'classification': classification, 'predicted_output_tokens': predicted,
                'outside_training_range': not meta['input_min'] <= request.input_tokens <= meta['input_max'],
                'ollama_response': body}
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f'Upstream request failed: {type(exc).__name__}') from exc
    finally:
        duration = time.monotonic() - started
        async with state.lock:
            state.active[sid] -= 1
            state.bandit.update(sid, x, duration, success, request.input_tokens, category, predicted, actual)
