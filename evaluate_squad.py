"""Paired repeated evaluation on independently authored SQuAD prompts."""
import argparse
import asyncio
import json
import os
import random
import sqlite3
import statistics
import subprocess
import time
from contextlib import closing
from pathlib import Path

import httpx
import numpy as np
from bandit import DIM, LinUCB
from experiment import copy_database, summary


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--config', default='config.json')
    p.add_argument('--bucket-model', default='bucket_model.joblib')
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--port', type=int, default=8010)
    args = p.parse_args()
    if args.repeats < 1 or args.batch_size < 1:
        p.error('repeats and batch size must be positive')
    base = Path.cwd().resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    calibration = {r['id']: r for r in (json.loads(x) for x in Path(args.calibration).read_text(encoding='utf-8').splitlines())}
    if len(calibration) != len(manifest['questions']) or any(calibration.get(q['id'], {}).get('status') != 'ok' for q in manifest['questions']):
        raise ValueError('Complete calibration required')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    seed_db = out / 'seed.sqlite3'
    copy_database(config['state_db'], seed_db)
    seed_model = LinUCB(str(seed_db), alpha=0)
    for row in calibration.values():
        seed_model.update(row['server_id'], np.zeros(DIM), row['elapsed_seconds'], True, row['input_tokens'],
                          row['category'], row['predicted_tokens'], row['output_tokens'], row.get('input_tokens'),
                          row.get('prompt_eval_duration'), row.get('eval_duration'), 0, learn=False,
                          prompt_cached_count=row.get('prompt_eval_cached_count'))
    seed_model.db.close()
    policies = ['linucb', 'round_robin', 'shortest_predicted']
    rng = random.Random(args.seed)
    schedule = []
    for repeat in range(args.repeats):
        order = list(policies)
        rng.shuffle(order)
        schedule.extend((repeat, name) for name in order)
    results = []
    for repeat, policy in schedule:
        phase = f'{repeat + 1}_{policy}'
        phase_db = out / (phase + '.sqlite3')
        copy_database(seed_db, phase_db)
        cfg = dict(config, state_db=str(phase_db), policy=policy, training_enabled=False,
                   bandit_alpha=0.0, bucket_model_path=str(Path(args.bucket_model).resolve()))
        cfg_path = out / (phase + '.json')
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        proc = subprocess.Popen([os.sys.executable, '-m', 'uvicorn', 'balancer:app', '--host', '127.0.0.1', '--port', str(args.port)],
                                cwd=base, env=dict(os.environ, ROUND_CONFIG=str(cfg_path), OMP_NUM_THREADS='2'),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        address = f'http://127.0.0.1:{args.port}'
        rows = []
        try:
            async with httpx.AsyncClient(timeout=330, trust_env=False) as client:
                for _ in range(80):
                    if proc.poll() is not None:
                        raise RuntimeError(f'{phase}: router failed to start')
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
                    raise RuntimeError(f'{phase}: servers not healthy')

                async def send(item):
                    started = time.monotonic()
                    row = {'id': item['id'], 'success': False}
                    request = {'input_tokens': calibration[item['id']]['input_tokens'],
                               'messages': [{'role': 'user', 'content': item['prompt']}]}
                    try:
                        async with client.stream('POST', address + '/chat/stream', json=request) as r:
                            r.raise_for_status()
                            async for line in r.aiter_lines():
                                if not line.strip():
                                    continue
                                event = json.loads(line)
                                if event['event'] == 'routed':
                                    row.update(server=event['server_id'], queue_seconds=event['queue_seconds'], category=event['category'])
                                elif event['event'] == 'chunk' and event['ollama'].get('done'):
                                    row.update(actual_output_tokens=event['ollama']['eval_count'],
                                               actual_input_tokens=event['ollama'].get('prompt_eval_count'))
                                elif event['event'] == 'complete':
                                    row.update(success=True, ttft_seconds=event['ttft_seconds'],
                                               mean_inter_chunk_seconds=event['mean_inter_chunk_seconds'])
                                elif event['event'] == 'error':
                                    row['error'] = event.get('detail', 'unknown')
                    except (httpx.HTTPError, ValueError, KeyError) as exc:
                        row['error'] = type(exc).__name__
                    row['latency_seconds'] = time.monotonic() - started
                    return row

                items = list(manifest['questions'])
                random.Random(args.seed + repeat).shuffle(items)
                started = time.monotonic()
                for i in range(0, len(items), args.batch_size):
                    rows.extend(await asyncio.gather(*(send(q) for q in items[i:i + args.batch_size])))
                    print(f'{phase}: {len(rows)}/{len(items)}; failures={sum(not x["success"] for x in rows)}', flush=True)
                result = summary(rows, time.monotonic() - started)
                ttft = [r['ttft_seconds'] for r in rows if r['success'] and r.get('ttft_seconds') is not None]
                result.update(policy=policy, repeat=repeat + 1,
                              mean_ttft_seconds=statistics.mean(ttft) if ttft else None,
                              input_count_mismatches=sum(r.get('actual_input_tokens') != calibration[r['id']]['input_tokens'] for r in rows if r['success']))
                (out / (phase + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
                results.append(result)
                (out / 'partial_report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    report = {'source': manifest['source'], 'source_url': manifest['url'], 'source_license': manifest['license'],
              'source_sha256': manifest['data_sha256'], 'sample_count': len(manifest['questions']),
              'sampling': manifest['sampling'], 'seed': args.seed, 'batch_size': args.batch_size,
              'policy_order': schedule, 'runs': results,
              'limitations': 'English reading-comprehension dataset, artificial response-length instructions, only 24 examples and 2 repeats. Original output-token count measured beforehand and replays may differ. All policies are frozen; streaming and automatic category classification used. TTFT is first nonempty content chunk, not necessarily one token.'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
