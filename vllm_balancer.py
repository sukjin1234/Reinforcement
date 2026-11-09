"""vLLM router; policy and observations are isolated from the Ollama deployment."""
import asyncio
import contextlib
import json
import math
import os
import time
from contextlib import asynccontextmanager

import httpx
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from category import classify
from dispatcher import Dispatcher
from monitor import read_config
from vllm_bandit import LinUCB
from vllm_monitor import collect_all
from workload import snapshot


class ChatRequest(BaseModel):
    messages: list[dict] = Field(min_length=1)
    category: str | None = Field(default=None, min_length=1)


@asynccontextmanager
async def lifespan(app):
    state = app.state
    state.config = read_config(os.environ.get('ROUND_CONFIG', 'config.vllm.json'))
    if state.config.get('backend') != 'vllm':
        raise RuntimeError('vLLM backend must be explicitly configured')
    state.bundle = joblib.load(state.config['predictor_path'])
    state.classifier = joblib.load(state.config['category_model_path'])
    if set(state.classifier['categories']) != set(state.bundle['metadata']['categories']):
        raise RuntimeError('classifier and predictor categories differ')
    state.bandit = LinUCB(state.config['state_db'], alpha=state.config.get('bandit_alpha', .7))
    state.lock = asyncio.Lock()
    state.active = {s['id']: 0 for s in state.config['servers']}
    state.active_work = {s['id']: {} for s in state.config['servers']}
    state.status = {}
    state.dispatcher = Dispatcher(state)
    state.client = httpx.AsyncClient(timeout=state.config['request_timeout_seconds'], trust_env=False)
    state.monitor_client = httpx.AsyncClient(timeout=3, trust_env=False)

    async def refresh():
        while True:
            rows = await collect_all(state.monitor_client, state.config)
            async with state.dispatcher.condition:
                state.status = {r['server_id']: r for r in rows}
                state.dispatcher.condition.notify_all()
            await asyncio.sleep(state.config['poll_seconds'])
    task = asyncio.create_task(refresh())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await state.client.aclose()
        await state.monitor_client.aclose()
        state.bandit.db.close()


app = FastAPI(lifespan=lifespan)


@app.get('/status')
async def status():
    state = app.state
    async with state.lock:
        return {'backend': 'vllm', 'model': state.config['model'],
                'predictor_validation': state.config.get('predictor_validation', 'unvalidated_for_vllm'),
                'policy': state.config.get('policy', 'linucb'),
                'training_enabled': state.config.get('training_enabled', True),
                'servers': state.status, 'inflight': state.active,
                'estimated_work': {sid: [snapshot(w) for w in works.values()] for sid, works in state.active_work.items()},
                'queue': {'waiting': len(state.dispatcher.waiters), 'rejected': state.dispatcher.rejected,
                          'timed_out': state.dispatcher.timed_out},
                'observations': {s['id']: state.bandit.state(s['id'])[2] for s in state.config['servers']}}


async def prepare(request):
    state = app.state
    messages = request.messages
    user_text = next((m.get('content') for m in reversed(messages) if m.get('role') == 'user'), None)
    if not isinstance(user_text, str) or not user_text.strip():
        raise HTTPException(422, 'A non-empty text user message is required')
    classification = None
    category = request.category
    if category is None:
        classification = classify(state.classifier, user_text)
        category = classification['category']
    if category not in state.bundle['metadata']['categories']:
        raise HTTPException(422, 'Unknown category')
    payload = {'model': state.config['model'], 'messages': messages,
               'chat_template_kwargs': {'enable_thinking': False}, 'add_generation_prompt': True}
    count, max_model_len = None, None
    for server in state.config['servers']:
        row = state.status.get(server['id'], {})
        if not row.get('healthy') or time.time() - row['checked_at'] > state.config['max_metric_age_seconds']:
            continue
        try:
            response = await state.client.post(server['vllm_url'].rstrip('/') + '/tokenize', json=payload)
            response.raise_for_status()
            body = response.json()
            count = body.get('count')
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError('invalid token count')
            max_model_len = body.get('max_model_len')
            if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len < 1:
                raise ValueError('missing model context limit')
            break
        except (httpx.HTTPError, ValueError, TypeError):
            count = None
            continue
    if count is None:
        raise HTTPException(503, 'No healthy tokenizer available')
    if count >= max_model_len:
        raise HTTPException(422, 'Input exceeds the model context limit')
    max_output = min(state.config.get('max_output_tokens', 4096), max_model_len - count)
    predicted = float(state.bundle['estimator'].predict([[count, category]])[0])
    if not math.isfinite(predicted):
        raise HTTPException(500, 'Predictor returned a non-finite value')
    return count, category, classification, max(0, round(predicted)), max_output


@app.post('/chat')
async def chat(request: ChatRequest):
    state = app.state
    count, category, classification, predicted, max_output = await prepare(request)
    server, x, queue_seconds, work_id = await state.dispatcher.reserve(count, predicted)
    sid, started = server['id'], time.monotonic()
    actual, prompt, success = None, None, False
    try:
        payload = {'model': state.config['model'], 'messages': request.messages, 'stream': False,
                   'chat_template_kwargs': {'enable_thinking': False}, 'max_tokens': max_output,
                   'temperature': .7, 'top_p': .9, 'seed': 42}
        response = await state.client.post(server['vllm_url'].rstrip('/') + '/v1/chat/completions', json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body['usage']
        actual, prompt = usage['completion_tokens'], usage['prompt_tokens']
        if not isinstance(actual, int) or not isinstance(prompt, int):
            raise ValueError('missing token usage')
        success = True
        return {'server_id': sid, 'input_tokens': count, 'actual_prompt_tokens': prompt,
                'token_count_matches': count == prompt, 'category': category, 'classification': classification,
                'predicted_output_tokens': predicted, 'predictor_validation': state.config.get('predictor_validation', 'unvalidated_for_vllm'),
                'queue_seconds': queue_seconds, 'vllm_response': body}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(502, f'Upstream request failed: {type(exc).__name__}') from exc
    finally:
        async with state.dispatcher.condition:
            state.active[sid] -= 1
            state.active_work[sid].pop(work_id, None)
            try:
                state.bandit.update(sid, x, time.monotonic() - started + queue_seconds, success,
                                    count, category, predicted, actual, prompt_count=prompt,
                                    queue_seconds=queue_seconds, learn=state.config.get('training_enabled', True))
            finally:
                state.dispatcher.condition.notify_all()


@app.post('/chat/stream')
async def chat_stream(request: ChatRequest):
    state, accepted = app.state, time.monotonic()
    count, category, classification, predicted, max_output = await prepare(request)

    def event(kind, **data):
        return json.dumps({'event': kind, **data}, ensure_ascii=False) + '\n'

    async def generate():
        try:
            server, x, queue_seconds, work_id = await state.dispatcher.reserve(count, predicted)
        except HTTPException as exc:
            yield event('error', status=exc.status_code, detail=exc.detail)
            return
        sid, started = server['id'], time.monotonic()
        success, cancelled, actual, prompt, first, last = False, False, None, None, None, None
        intervals = []
        try:
            yield event('routed', server_id=sid, input_tokens=count, category=category,
                        classification=classification, predicted_output_tokens=predicted,
                        predictor_validation=state.config.get('predictor_validation', 'unvalidated_for_vllm'),
                        queue_seconds=queue_seconds)
            payload = {'model': state.config['model'], 'messages': request.messages, 'stream': True,
                       'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False},
                       'max_tokens': max_output,
                       'temperature': .7, 'top_p': .9, 'seed': 42}
            done = False
            async with state.client.stream('POST', server['vllm_url'].rstrip('/') + '/v1/chat/completions', json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        done = True
                        break
                    part = json.loads(data)
                    if part.get('error'):
                        raise ValueError('upstream stream error')
                    usage = part.get('usage')
                    if usage:
                        actual, prompt = usage.get('completion_tokens'), usage.get('prompt_tokens')
                    for choice in part.get('choices', []):
                        if (choice.get('delta') or {}).get('content'):
                            now = time.monotonic()
                            if first is None:
                                first = now
                            if last is not None:
                                intervals.append(now - last)
                            last = now
                    yield event('chunk', vllm=part)
            if not done or not isinstance(actual, int) or not isinstance(prompt, int):
                raise ValueError('stream ended without token usage')
            success = True
            yield event('complete', server_id=sid, actual_prompt_tokens=prompt,
                        token_count_matches=count == prompt, actual_output_tokens=actual,
                        queue_seconds=queue_seconds, ttft_seconds=first - accepted if first else None,
                        mean_inter_chunk_seconds=sum(intervals) / len(intervals) if intervals else None)
        except (asyncio.CancelledError, GeneratorExit):
            cancelled = True
            raise
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            yield event('error', status=502, detail=type(exc).__name__)
        finally:
            async with state.dispatcher.condition:
                state.active[sid] -= 1
                state.active_work[sid].pop(work_id, None)
                try:
                    if not cancelled:
                        state.bandit.update(sid, x, time.monotonic() - started + queue_seconds, success,
                                            count, category, predicted, actual, prompt_count=prompt,
                                            queue_seconds=queue_seconds, learn=state.config.get('training_enabled', True),
                                            ttft_seconds=first - accepted if first else None,
                                            mean_inter_chunk_seconds=sum(intervals) / len(intervals) if intervals else None)
                finally:
                    state.dispatcher.condition.notify_all()

    return StreamingResponse(generate(), media_type='application/x-ndjson')
