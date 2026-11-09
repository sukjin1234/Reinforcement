"""Read per-instance vLLM metrics without confusing estimates with measurements."""
import asyncio
import math
import re
import time

import httpx

NAMES = ('vllm:num_requests_running', 'vllm:num_requests_waiting',
         'vllm:kv_cache_usage_perc', 'vllm:prompt_tokens_total',
         'vllm:generation_tokens_total', 'vllm:request_prefill_kv_computed_tokens_sum',
         'vllm:request_prefill_time_seconds_sum', 'vllm:request_generation_tokens_sum',
         'vllm:request_decode_time_seconds_sum', 'vllm:request_queue_time_seconds_sum',
         'vllm:request_queue_time_seconds_count')


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
        def valid_percent(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 100
        if not all(valid_percent(machine.get(key)) for key in ('cpu_pct', 'memory_pct')):
            raise ValueError('invalid CPU or memory metrics')
        for gpu in machine['gpus']:
            used, total = gpu.get('memory_used_mib'), gpu.get('memory_total_mib')
            if not valid_percent(gpu.get('utilization_pct')) or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in (used, total)) or not 0 <= used <= total or total <= 0:
                raise ValueError('invalid GPU metrics')
        engine = parse_metrics(metrics.text)
        if engine['kv_cache_usage_perc'] > 1:
            raise ValueError('invalid KV cache fraction')
        def ratio(numerator, denominator, minimum=0):
            n, d = engine.get(numerator), engine.get(denominator)
            return n / d if n is not None and n >= minimum and d is not None and d > 0 else None
        perf = {
            'uncached_input_tokens_per_second': ratio('request_prefill_kv_computed_tokens_sum', 'request_prefill_time_seconds_sum', 32),
            'input_tokens_per_second': ratio('request_prefill_kv_computed_tokens_sum', 'request_prefill_time_seconds_sum', 32),
            'output_tokens_per_second': ratio('request_generation_tokens_sum', 'request_decode_time_seconds_sum'),
            'mean_engine_queue_seconds': ratio('request_queue_time_seconds_sum', 'request_queue_time_seconds_count'),
            'source': 'cumulative completed-request histogram; not instantaneous',
        }
        result.update(healthy=True, metrics=machine, vllm=engine, vllm_performance=perf)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['checked_at'] = time.time()
    return result


async def collect_all(client, config):
    return await asyncio.gather(*(collect_one(client, s, config) for s in config['servers']))
