"""Measure a running vLLM instance with controlled concurrent requests."""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


async def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite output')
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        async def request(index):
            payload = {'model': args.model, 'messages': [{'role': 'user', 'content':
                       f'Explain GPU scheduling briefly. Test request {index}.'}],
                       'max_tokens': args.max_tokens, 'temperature': 0,
                       'chat_template_kwargs': {'enable_thinking': False}}
            start = time.monotonic()
            response = await client.post(args.url.rstrip('/') + '/v1/chat/completions', json=payload)
            response.raise_for_status()
            usage = response.json()['usage']
            return {'elapsed_seconds': time.monotonic() - start, 'input_tokens': usage['prompt_tokens'],
                    'output_tokens': usage['completion_tokens']}
        await request('warmup')
        phases = []
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                start = time.monotonic()
                rows = await asyncio.gather(*(request(f'{concurrency}-{repeat}-{i}') for i in range(concurrency)))
                phases.append({'concurrency': concurrency, 'repeat': repeat + 1,
                               'batch_seconds': time.monotonic() - start, 'requests': rows})
        report = {'model': args.model, 'phases': phases,
                  'mean_batch_seconds': {str(n): statistics.mean(p['batch_seconds'] for p in phases if p['concurrency'] == n)
                                         for n in args.concurrency},
                  'limitations': 'Short prompts and fixed ordering; concurrent external workloads and cache can affect results.'}
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8100')
    parser.add_argument('--model', default='qwen3-8b')
    parser.add_argument('--output', required=True)
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1 or any(x < 1 for x in args.concurrency):
        parser.error('counts must be positive')
    asyncio.run(run(args))
