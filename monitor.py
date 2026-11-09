"""Validate vLLM router configuration and collect server metrics."""
import json
from pathlib import Path
from vllm_monitor import collect_all, collect_one


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
    if config.get('policy', 'linucb') not in ('linucb', 'round_robin', 'shortest_predicted'):
        raise ValueError('policy must be linucb, round_robin or shortest_predicted')
    if not isinstance(config.get('training_enabled', True), bool) or config.get('bandit_alpha', 0.7) < 0:
        raise ValueError('invalid training_enabled or bandit_alpha')
    return config


