"""Single-process Ollama chat router with online contextual-bandit selection."""
import asyncio
import contextlib
import math
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bandit import LinUCB
from buckets import predict_bucket
from category import classify
from dispatcher import Dispatcher
from monitor import collect_all, read_config
from workload import snapshot


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
    app.state.bucket_model = joblib.load(config['bucket_model_path']) if config.get('bucket_model_path') else None
    app.state.bandit = LinUCB(config['state_db'], alpha=config.get('bandit_alpha', 0.7))
    app.state.lock = asyncio.Lock()
    app.state.active = {s['id']: 0 for s in config['servers']}
    app.state.active_work = {s['id']: {} for s in config['servers']}
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
                'estimated_work': {sid: [snapshot(work) for work in works.values()] for sid, works in app.state.active_work.items()},
                'queue': {'waiting': len(app.state.dispatcher.waiters),
                          'max_size': app.state.config.get('queue_max_size', 100),
                          'rejected': app.state.dispatcher.rejected, 'timed_out': app.state.dispatcher.timed_out},
                'recent_performance': {s['id']: app.state.bandit.recent_performance(s['id']) for s in app.state.config['servers']},
                'observations': {s['id']: app.state.bandit.state(s['id'])[2] for s in app.state.config['servers']}}


def prepare(request):
    state = app.state
    meta = state.bundle['metadata']
    classification = None
    user_messages = [m.get('content') for m in request.messages if m.get('role') == 'user']
    user_text = user_messages[-1] if user_messages else None
    category = request.category
    if category is None:
        try:
            classification = classify(state.classifier, user_text)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        category = classification['category']
    if category not in meta['categories']:
        raise HTTPException(422, 'Unknown category. Use a category from the predictor training data.')
    predicted_float = float(state.bundle['estimator'].predict([[request.input_tokens, category]])[0])
    if not math.isfinite(predicted_float):
        raise HTTPException(500, 'Predictor returned a non-finite value')
    predicted = max(0, round(predicted_float))
    bucket_prediction = predict_bucket(state.bucket_model, user_text, request.input_tokens, category) if state.bucket_model else None
    return state, meta, category, classification, predicted, bucket_prediction


@app.post('/chat')
async def chat(request: ChatRequest):
    state, meta, category, classification, predicted, bucket_prediction = prepare(request)
    server, x, queue_seconds, work_id = await state.dispatcher.reserve(request.input_tokens, predicted)
    sid = server['id']
    payload = {'model': state.config['model'], 'messages': request.messages,
               'stream': False, 'think': False, 'keep_alive': '30m', 'options': meta['settings']}
    started = time.monotonic()
    success, actual, prompt_count, prompt_duration, eval_duration, prompt_cached = False, None, None, None, None, None
    try:
        response = await state.client.post(server['ollama_url'].rstrip('/') + '/api/chat', json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get('done') is not True or not isinstance(body.get('eval_count'), int):
            raise ValueError('Invalid Ollama response or missing token count')
        actual = body['eval_count']
        prompt_count, prompt_duration, eval_duration, prompt_cached = (body.get('prompt_eval_count'), body.get('prompt_eval_duration'), body.get('eval_duration'), body.get('prompt_eval_cached_count'))
        def positive_number(value):
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
        prompt_count, prompt_duration, eval_duration, prompt_cached = map(positive_number, (prompt_count, prompt_duration, eval_duration, prompt_cached))
        success = True
        return {'server_id': sid, 'category': category, 'classification': classification, 'predicted_output_tokens': predicted,
                'bucket_prediction': bucket_prediction,
                'queue_seconds': queue_seconds,
                'outside_training_range': not meta['input_min'] <= request.input_tokens <= meta['input_max'],
                'ollama_response': body}
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f'Upstream request failed: {type(exc).__name__}') from exc
    finally:
        duration = time.monotonic() - started
        async with state.dispatcher.condition:
            state.active[sid] -= 1
            state.active_work[sid].pop(work_id, None)
            try:
                state.bandit.update(sid, x, duration + queue_seconds, success, request.input_tokens, category,
                                    predicted, actual, prompt_count, prompt_duration, eval_duration, queue_seconds,
                                    learn=state.config.get('training_enabled', True), prompt_cached_count=prompt_cached)
            finally:
                state.dispatcher.condition.notify_all()


@app.post('/chat/stream')
async def chat_stream(request: ChatRequest):
    accepted = time.monotonic()
    state, meta, category, classification, predicted, bucket_prediction = prepare(request)

    def event(kind, **kwargs):
        return json.dumps({'event': kind, **kwargs}, ensure_ascii=False) + '\n'

    async def generate():
        try:
            server, x, queue_seconds, work_id = await state.dispatcher.reserve(request.input_tokens, predicted)
        except HTTPException as exc:
            yield event('error', status=exc.status_code, detail=exc.detail)
            return
        sid = server['id']
        started = time.monotonic()
        payload = {'model': state.config['model'], 'messages': request.messages, 'stream': True,
                   'think': False, 'keep_alive': '30m', 'options': meta['settings']}
        success, cancelled, actual, prompt_count, prompt_duration, eval_duration, prompt_cached = False, False, None, None, None, None, None
        first_content, last_content, intervals = None, None, []
        try:
            yield event('routed', server_id=sid, category=category, classification=classification,
                        predicted_output_tokens=predicted, bucket_prediction=bucket_prediction,
                        queue_seconds=queue_seconds)
            async with state.client.stream('POST', server['ollama_url'].rstrip('/') + '/api/chat', json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    part = json.loads(line)
                    if not isinstance(part, dict) or part.get('error'):
                        raise ValueError('Invalid or error upstream chunk')
                    message = part.get('message') or {}
                    if not isinstance(message, dict):
                        raise ValueError('Invalid message chunk')
                    content = message.get('content')
                    if content:
                        now = time.monotonic()
                        if first_content is None:
                            first_content = now
                        if last_content is not None:
                            intervals.append(now - last_content)
                        last_content = now
                    if part.get('done'):
                        if not isinstance(part.get('eval_count'), int) or isinstance(part.get('eval_count'), bool):
                            raise ValueError('Missing output token count')
                        actual = part['eval_count']
                        def number(v):
                            return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 else None
                        prompt_count, prompt_duration, eval_duration, prompt_cached = map(number, (part.get('prompt_eval_count'), part.get('prompt_eval_duration'), part.get('eval_duration'), part.get('prompt_eval_cached_count')))
                        success = True
                    yield event('chunk', ollama=part)
                    if success:
                        break
                if not success:
                    raise ValueError('Upstream closed before final chunk')
            yield event('complete', server_id=sid, queue_seconds=queue_seconds,
                        ttft_seconds=first_content - accepted if first_content is not None else None,
                        mean_inter_chunk_seconds=sum(intervals) / len(intervals) if intervals else None)
        except (asyncio.CancelledError, GeneratorExit):
            cancelled = True
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            yield event('error', status=502, detail=type(exc).__name__)
        finally:
            duration = time.monotonic() - started
            async with state.dispatcher.condition:
                state.active[sid] -= 1
                state.active_work[sid].pop(work_id, None)
                try:
                    if not cancelled:
                        state.bandit.update(sid, x, duration + queue_seconds, success, request.input_tokens, category,
                                            predicted, actual, prompt_count, prompt_duration, eval_duration, queue_seconds,
                                            learn=state.config.get('training_enabled', True),
                                            ttft_seconds=first_content - accepted if first_content is not None else None,
                                            mean_inter_chunk_seconds=sum(intervals) / len(intervals) if intervals else None,
                                            prompt_cached_count=prompt_cached)
                finally:
                    state.dispatcher.condition.notify_all()

    return StreamingResponse(generate(), media_type='application/x-ndjson')
