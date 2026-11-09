"""Probe Ollama parallelism in a separate local process on one GPU host."""
import argparse
import asyncio
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx

MODEL = 'qwen3:8b-q4_K_M'
EXPECTED_DIGEST = '500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'


async def phase(parallel, output_dir, port, repeats):
    address = f'http://127.0.0.1:{port}'
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError('Probe port is already in use')
    env = dict(os.environ, OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_NUM_PARALLEL=str(parallel),
               OLLAMA_MODELS='/usr/share/ollama/.ollama/models', OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_NO_CLOUD='1')
    with (output_dir / f'ollama_{parallel}.log').open('w') as log:
        proc = subprocess.Popen(['ollama', 'serve'], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    try:
        async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError('Probe server failed; see log')
                try:
                    r = await client.get(address + '/api/tags')
                    r.raise_for_status()
                    models = r.json()['models']
                    if any(m.get('name') == MODEL and m.get('digest') == EXPECTED_DIGEST for m in models):
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.5)
            else:
                raise RuntimeError('Probe server not ready')
            options = {'num_ctx': 16384, 'num_predict': 256, 'temperature': .7, 'top_p': .9, 'seed': 42}
            warm = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Say hello.'}],
                    'think': False, 'stream': False, 'keep_alive': '5m', 'options': {**options, 'num_predict': 8}}
            # Warm the concurrent path as well as model loading, before timing.
            warm_responses = await asyncio.gather(*(client.post(address + '/api/chat', json={**warm,
                'messages': [{'role': 'user', 'content': f'Say hello. Warmup {i}.'}]}) for i in range(2)))
            for r in warm_responses:
                r.raise_for_status()

            async def send(index, repeat):
                payload = {'model': MODEL, 'messages': [{'role': 'user', 'content':
                           f'Explain how computer networks work in about 100 words. Round {repeat}, request {index}.'}],
                           'think': False, 'stream': True, 'keep_alive': '5m', 'options': options}
                started = time.monotonic()
                first, done = None, None
                async with client.stream('POST', address + '/api/chat', json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        row = json.loads(line)
                        if row.get('error'):
                            raise RuntimeError('Ollama returned an error')
                        if first is None and (row.get('message') or {}).get('content'):
                            first = time.monotonic() - started
                        if row.get('done'):
                            done = row
                            break
                if not done or first is None:
                    raise RuntimeError('Incomplete streamed response')
                return {'ttft_seconds': first, 'latency_seconds': time.monotonic() - started,
                        'actual_output_tokens': done.get('eval_count')}

            rounds = []
            fields = []
            for repeat in range(repeats):
                start = time.monotonic()
                tasks = [asyncio.create_task(send(i, repeat)) for i in range(2)]
                # The API reports loaded models and memory, not per-request progress.
                await asyncio.sleep(.1)
                ps = (await client.get(address + '/api/ps')).json()
                fields = sorted(ps.get('models', [{}])[0].keys()) if ps.get('models') else []
                rows = await asyncio.gather(*tasks)
                rounds.append({'total_elapsed_seconds': time.monotonic() - start, 'requests': rows})
            return {'parallel': parallel, 'rounds': rounds, 'api_ps_fields': fields}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, default=11435)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=False)
    if args.repeats < 1:
        p.error('repeats must be positive')
    result = {'warning': 'One host, two synthetic prompts, fixed 1-then-2 order, warm concurrent path. Separate local process; system service settings unchanged.', 'phases': []}
    for n in (1, 2):
        result['phases'].append(await phase(n, path, args.port, args.repeats))
        (path / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('parallel', n, 'complete', flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
