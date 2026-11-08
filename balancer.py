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

from bandit import LinUCB
from category import classify
from dispatcher import Dispatcher
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
    app.state.bandit = LinUCB(config['state_db'], alpha=config.get('bandit_alpha', 0.7))
    app.state.lock = asyncio.Lock()
    app.state.active = {s['id']: 0 for s in config['servers']}
    app.state.status = {}
    app.state.dispatcher = Dispatcher(app.state)
    app.state.client = httpx.AsyncClient(timeout=config['request_timeout_seconds'], trust_env=False)
    app.state.monitor_client = httpx.AsyncClient(timeout=3, trust_env=False)

    async def refresh():
        while True:
            rows = await collect_all(app.state.monitor_client, config)
            async with app.state.dispatcher.condition:
                app.state.status = {r['server_id']: r for r in rows}
                app.state.dispatcher.condition.notify_all()
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
        return {'policy': app.state.config.get('policy', 'linucb'),
                'training_enabled': app.state.config.get('training_enabled', True),
                'bandit_alpha': app.state.config.get('bandit_alpha', 0.7),
                'servers': app.state.status, 'inflight': app.state.active,
                'queue': {'waiting': len(app.state.dispatcher.waiters),
                          'max_size': app.state.config.get('queue_max_size', 100),
                          'rejected': app.state.dispatcher.rejected, 'timed_out': app.state.dispatcher.timed_out},
                'recent_performance': {s['id']: app.state.bandit.recent_performance(s['id']) for s in app.state.config['servers']},
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
    server, x, queue_seconds = await state.dispatcher.reserve(request.input_tokens, predicted)
    sid = server['id']
    payload = {'model': state.config['model'], 'messages': request.messages,
               'stream': False, 'think': False, 'keep_alive': '30m', 'options': meta['settings']}
    started = time.monotonic()
    success, actual, prompt_count, prompt_duration, eval_duration = False, None, None, None, None
    try:
        response = await state.client.post(server['ollama_url'].rstrip('/') + '/api/chat', json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get('done') is not True or not isinstance(body.get('eval_count'), int):
            raise ValueError('Invalid Ollama response or missing token count')
        actual = body['eval_count']
        prompt_count, prompt_duration, eval_duration = (body.get('prompt_eval_count'), body.get('prompt_eval_duration'), body.get('eval_duration'))
        def positive_number(value):
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
        prompt_count, prompt_duration, eval_duration = map(positive_number, (prompt_count, prompt_duration, eval_duration))
        success = True
        return {'server_id': sid, 'category': category, 'classification': classification, 'predicted_output_tokens': predicted,
                'queue_seconds': queue_seconds,
                'outside_training_range': not meta['input_min'] <= request.input_tokens <= meta['input_max'],
                'ollama_response': body}
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f'Upstream request failed: {type(exc).__name__}') from exc
    finally:
        duration = time.monotonic() - started
        async with state.dispatcher.condition:
            state.active[sid] -= 1
            try:
                state.bandit.update(sid, x, duration + queue_seconds, success, request.input_tokens, category,
                                    predicted, actual, prompt_count, prompt_duration, eval_duration, queue_seconds,
                                    learn=state.config.get('training_enabled', True))
            finally:
                state.dispatcher.condition.notify_all()
