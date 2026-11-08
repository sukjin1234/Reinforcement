"""Collect host metrics and independently verify the expected Ollama model."""
import asyncio
import json
import math
import os
import time
from pathlib import Path
import httpx


def read_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    servers = config['servers']
    if not servers or len({s['id'] for s in servers}) != len(servers):
        raise ValueError('servers must have unique IDs')
    for server in servers:
        if server.get('max_inflight', 1) < 1:
            raise ValueError('max_inflight must be positive')
    for name in ('poll_seconds', 'max_metric_age_seconds', 'request_timeout_seconds'):
        if config[name] <= 0:
            raise ValueError(f'{name} must be positive')
    if config.get('queue_max_size', 100) < 1 or config.get('queue_wait_seconds', 60) <= 0:
        raise ValueError('queue capacity and wait time must be positive')
    return config


async def collect_one(client, server, config):
    result = {'server_id': server['id'], 'checked_at': time.time(), 'healthy': False}
    try:
        token = os.environ.get(config.get('metrics_token_env', ''), '')
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        metrics_response, tags_response = await asyncio.gather(
            client.get(server['metrics_url'], headers=headers),
            client.get(server['ollama_url'].rstrip('/') + '/api/tags'))
        metrics_response.raise_for_status()
        tags_response.raise_for_status()
        metrics = metrics_response.json()
        models = tags_response.json().get('models', [])
        matches = [m for m in models if m.get('name') == config['model'] and m.get('digest') == config['model_digest']]
        if not matches:
            raise ValueError('expected model or digest not present')
        gpus = metrics['gpus']
        if not gpus:
            raise ValueError('GPU metrics unavailable')
        for v in [metrics['cpu_pct'], metrics['memory_pct'], *[g['utilization_pct'] for g in gpus]]:
            if not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 100:
                raise ValueError('invalid metrics')
        for g in gpus:
            if not all(isinstance(g[k], (float, int)) and math.isfinite(g[k]) for k in ('memory_used_mib', 'memory_total_mib')) or not 0 <= g['memory_used_mib'] <= g['memory_total_mib'] or g['memory_total_mib'] <= 0:
                raise ValueError('invalid GPU memory metrics')
        result.update({'healthy': True, 'metrics': metrics})
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['checked_at'] = time.time()
    return result


async def collect_all(client, config):
    return await asyncio.gather(*(collect_one(client, server, config) for server in config['servers']))
