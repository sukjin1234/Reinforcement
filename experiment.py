"""Train LinUCB and compare frozen policy with round robin on the same requests."""
import argparse
import asyncio
from contextlib import closing
import json
import os
import random
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path

import httpx

TEST_DOMAINS = {'경제·경영', '학교·교육'}


def select_questions(question_path, result_path, seed, train_count, test_count):
    eligible = {}
    with open(result_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'ok' and r.get('done_reason') == 'stop' and 100 <= r.get('output_tokens', 0) <= 700:
                eligible[r['id']] = r
    groups = {'train': [], 'test': []}
    with open(question_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            result = eligible.get(row['id'])
            if result is None:
                continue
            item = {'id': row['id'], 'domain': row['domain'], 'category': row['category'],
                    'input_tokens': result['input_tokens'], 'prompt': row['prompt']}
            groups['test' if row['domain'] in TEST_DOMAINS else 'train'].append(item)
    rng = random.Random(seed)
    if len(groups['train']) < train_count or len(groups['test']) < test_count:
        raise ValueError('Not enough eligible questions')
    return rng.sample(groups['train'], train_count), rng.sample(groups['test'], test_count)


def summary(rows, elapsed):
    success = [r for r in rows if r['success']]
    values = sorted(r['latency_seconds'] for r in success)
    return {'requests': len(rows), 'successes': len(success), 'failures': len(rows) - len(success),
            'elapsed_seconds': elapsed, 'throughput_requests_per_second': len(success) / elapsed,
            'mean_latency_seconds': statistics.mean(values) if values else None,
            'median_latency_seconds': statistics.median(values) if values else None,
            'p95_latency_seconds': values[max(0, (95 * len(values) + 99) // 100 - 1)] if values else None,
            'mean_queue_seconds': statistics.mean(r['queue_seconds'] for r in success) if success else None,
            'servers': {server: sum(r.get('server') == server for r in success) for server in ('ai', 'ai1', 'ai2')}}


def copy_database(src, dst):
    with closing(sqlite3.connect(src)) as source, closing(sqlite3.connect(dst)) as target:
        source.backup(target)


async def run_phase(base, run_dir, config, name, tasks, batch_size, port):
    phase_config = dict(config)
    phase_config.update(state_db=str(run_dir / (name + '.sqlite3')),
                        policy='round_robin' if name == 'round_robin' else 'linucb',
                        training_enabled=name == 'train', bandit_alpha=0.7 if name == 'train' else 0.0)
    if name != 'train':
        copy_database(run_dir / 'train.sqlite3', run_dir / (name + '.sqlite3'))
    config_path = run_dir / (name + '.json')
    config_path.write_text(json.dumps(phase_config, ensure_ascii=False, indent=2), encoding='utf-8')
    env = dict(os.environ, ROUND_CONFIG=str(config_path), OMP_NUM_THREADS='2')
    proc = subprocess.Popen([os.sys.executable, '-m', 'uvicorn', 'balancer:app', '--host', '127.0.0.1', '--port', str(port)],
                            cwd=base, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    address = f'http://127.0.0.1:{port}'
    rows = []
    try:
        async with httpx.AsyncClient(timeout=330, trust_env=False) as client:
            for _ in range(80):
                if proc.poll() is not None:
                    raise RuntimeError(f'{name}: experiment router failed to start')
                try:
                    response = await client.get(address + '/status')
                    response.raise_for_status()
                    state = response.json()
                    if len(state['servers']) == 3 and all(s['healthy'] for s in state['servers'].values()):
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.5)
            else:
                raise RuntimeError(f'{name}: not all servers became healthy')

            async def send(item):
                start = time.monotonic()
                record = {'id': item['id'], 'domain': item['domain'], 'category': item['category'], 'success': False}
                try:
                    r = await client.post(address + '/chat', json={'input_tokens': item['input_tokens'], 'category': item['category'],
                                                                  'messages': [{'role': 'user', 'content': item['prompt']}]})
                    r.raise_for_status()
                    data = r.json()
                    record.update(success=True, server=data['server_id'], queue_seconds=data['queue_seconds'],
                                  predicted_output_tokens=data['predicted_output_tokens'],
                                  actual_output_tokens=data['ollama_response']['eval_count'])
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    record['error'] = type(exc).__name__
                record['latency_seconds'] = time.monotonic() - start
                return record

            start = time.monotonic()
            for i in range(0, len(tasks), batch_size):
                rows.extend(await asyncio.gather(*(send(item) for item in tasks[i:i + batch_size])))
                with (run_dir / (name + '.jsonl')).open('a', encoding='utf-8') as f:
                    for item in rows[i:]:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f'{name}: {len(rows)}/{len(tasks)} complete; errors={sum(not r["success"] for r in rows)}', flush=True)
            return summary(rows, time.monotonic() - start)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', required=True, help='New directory; must not already exist')
    p.add_argument('--train-count', type=int, default=90)
    p.add_argument('--test-count', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--port', type=int, default=8010)
    args = p.parse_args()
    if min(args.train_count, args.test_count, args.batch_size) < 1:
        p.error('counts and batch size must be positive')
    base = Path.cwd().resolve()
    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    train, test = select_questions(args.questions, args.results, args.seed, args.train_count, args.test_count)
    manifest = {'seed': args.seed, 'test_domains': sorted(TEST_DOMAINS), 'train_ids': [i['id'] for i in train],
                'test_ids': [i['id'] for i in test], 'filter': 'status=ok, done_reason=stop, observed output 100..700 tokens',
                'batch_size': args.batch_size, 'order': ['train', 'linucb', 'round_robin']}
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'method': manifest, 'warning': 'Synthetic template questions; output-length filter uses observed results; held-out domains may share templates; predictor previously trained on same corpus. No randomization of evaluation order.'}
    for name, tasks in [('train', train), ('linucb', test), ('round_robin', test)]:
        report[name] = await run_phase(base, run_dir, config, name, tasks, args.batch_size, args.port)
        (run_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
