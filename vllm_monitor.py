"""Read per-instance vLLM metrics without confusing estimates with measurements."""
import asyncio
import math
import re
import time

import httpx

NAMES = ('vllm:num_requests_running', 'vllm:num_requests_waiting',
         'vllm:kv_cache_usage_perc', 'vllm:prompt_tokens_total',
         'vllm:generation_tokens_total')


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.match(r'^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)', line)
        if not match or match.group(1) not in NAMES:
            continue
        value = float(match.group(2))
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid vLLM metric')
        values[match.group(1)] = values.get(match.group(1), 0) + value
    if any(name not in values for name in NAMES[:3]):
        raise ValueError('missing required vLLM metrics')
    return {name.removeprefix('vllm:'): value for name, value in values.items()}


async def collect_one(client, server, config):
    result = {'server_id': server['id'], 'healthy': False, 'checked_at': time.time()}
    try:
        import os
        token = os.environ.get(config.get('metrics_token_env', ''), '')
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        host, models, metrics = await asyncio.gather(
            client.get(server['metrics_url'], headers=headers),
            client.get(server['vllm_url'].rstrip('/') + '/v1/models'),
            client.get(server['vllm_url'].rstrip('/') + '/metrics'))
        for response in (host, models, metrics):
            response.raise_for_status()
        if config['model'] not in [m.get('id') for m in models.json().get('data', [])]:
            raise ValueError('expected model not served')
        machine = host.json()
        if not machine.get('gpus'):
            raise ValueError('GPU metrics unavailable')
        engine = parse_metrics(metrics.text)
        if engine['kv_cache_usage_perc'] > 1:
            raise ValueError('invalid KV cache fraction')
        result.update(healthy=True, metrics=machine, vllm=engine)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['checked_at'] = time.time()
    return result


async def collect_all(client, config):
    return await asyncio.gather(*(collect_one(client, s, config) for s in config['servers']))
