"""Compare frozen policies on identical prompts using isolated router databases."""
import argparse
import asyncio
import json
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=8011)
    parser.add_argument('--concurrency', type=int, default=6)
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    base = json.loads(Path(args.config).read_text(encoding='utf-8'))
    prompts = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['questions']
    order = ['linucb', 'round_robin', 'shortest_predicted']
    random.Random(20261109).shuffle(order)
    report = {'order': order, 'policies': {}, 'linucb_alpha': 0.0, 'training_enabled': False,
              'limitations': 'One run per policy, English SQuAD prompts with synthetic style instructions; shared caches and time ordering can affect results. Frozen LinUCB uses alpha=0; the live router may use another alpha. Not statistical proof of superiority.'}
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        for policy in order:
            db = directory / (policy + '.sqlite3')
            with sqlite3.connect(base['state_db']) as source, sqlite3.connect(db) as destination:
                source.backup(destination)
            config = dict(base, state_db=str(db), policy=policy, training_enabled=False, bandit_alpha=0.0)
            cfg = directory / (policy + '.json')
            cfg.write_text(json.dumps(config), encoding='utf-8')
            import os
            env = dict(os.environ, ROUND_CONFIG=str(cfg), OMP_NUM_THREADS='2')
            with (directory / (policy + '.log')).open('w') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'vllm_balancer:app', '--host', '127.0.0.1', '--port', str(args.port)], env=env, stdout=log, stderr=log)
                try:
                    base_url = f'http://127.0.0.1:{args.port}'
                    for _ in range(120):
                        if proc.poll() is not None:
                            raise RuntimeError(f'router for {policy} exited; see log')
                        try:
                            status = (await client.get(base_url + '/status', timeout=2)).json()
                            if len(status['servers']) == len(base['servers']) and all(v['healthy'] for v in status['servers'].values()):
                                break
                        except (httpx.HTTPError, ValueError, KeyError):
                            pass
                        await asyncio.sleep(1)
                    else:
                        raise RuntimeError('router did not become ready')
                    sem = asyncio.Semaphore(args.concurrency)
                    async def send(item):
                        async with sem:
                            started = time.monotonic()
                            try:
                                r = await client.post(base_url + '/chat', json={'messages': [{'role': 'user', 'content': item['prompt']}]})
                                r.raise_for_status()
                                body = r.json()
                                return {'id': item['id'], 'ok': True, 'server': body['server_id'], 'latency': time.monotonic()-started, 'queue': body['queue_seconds'], 'tokens': body['vllm_response']['usage']['completion_tokens']}
                            except (httpx.HTTPError, ValueError, KeyError) as exc:
                                return {'id': item['id'], 'ok': False, 'error': type(exc).__name__}
                    started = time.monotonic()
                    tasks = [asyncio.create_task(send(item)) for item in prompts]
                    rows = []
                    with (directory / (policy + '.jsonl')).open('x', encoding='utf-8') as output:
                        for task in asyncio.as_completed(tasks):
                            row = await task
                            rows.append(row)
                            output.write(json.dumps(row) + '\n')
                            output.flush()
                            print(policy, len(rows), '/', len(prompts), flush=True)
                    duration = time.monotonic()-started
                    good = [r for r in rows if r['ok']]
                    report['policies'][policy] = {'success': len(good), 'errors': len(rows)-len(good), 'total_seconds': duration,
                        'mean_latency_seconds': statistics.mean(r['latency'] for r in good) if good else None,
                        'mean_queue_seconds': statistics.mean(r['queue'] for r in good) if good else None,
                        'requests_per_second': len(good)/duration, 'generated_tokens_per_second': sum(r['tokens'] for r in good)/duration}
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
    (directory / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
