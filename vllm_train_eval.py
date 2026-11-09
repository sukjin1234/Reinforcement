"""Collect vLLM training observations and evaluate on a separate question set."""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-count', type=int, default=60)
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--url', default='http://127.0.0.1:8008')
    args = parser.parse_args()
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261109)
    # Reservoir sampling avoids loading the entire 100k question file into memory.
    sample = []
    with open(args.questions, encoding='utf-8') as source:
        for i, line in enumerate(source):
            item = json.loads(line)
            if i < args.train_count:
                sample.append(item)
            else:
                j = rng.randrange(i + 1)
                if j < args.train_count:
                    sample[j] = item
    evaluation = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['questions']
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        async def send(item, labelled):
            async with semaphore:
                body = {'messages': [{'role': 'user', 'content': item['prompt']}]}
                if labelled:
                    body['category'] = item['category']
                started = time.monotonic()
                try:
                    response = await client.post(args.url + '/chat', json=body)
                    response.raise_for_status()
                    result = response.json()
                    usage = result['vllm_response']['usage']
                    return {'id': item['id'], 'server_id': result['server_id'], 'success': True,
                            'category': result['category'], 'input_tokens': result['input_tokens'],
                            'predicted_output_tokens': result['predicted_output_tokens'],
                            'actual_output_tokens': usage['completion_tokens'],
                            'finish_reason': result['vllm_response']['choices'][0]['finish_reason'],
                            'token_count_matches': result['token_count_matches'],
                            'queue_seconds': result['queue_seconds'], 'elapsed_seconds': time.monotonic() - started}
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    return {'id': item['id'], 'success': False, 'error': type(exc).__name__}

        async def phase(name, items, labelled):
            with (path / (name + '.jsonl')).open('x', encoding='utf-8') as f:
                tasks = [asyncio.create_task(send(item, labelled)) for item in items]
                rows = []
                for task in asyncio.as_completed(tasks):
                    row = await task
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    f.flush()
                    print(name, len(rows), '/', len(items), row.get('server_id', row.get('error')), flush=True)
            valid = [r for r in rows if r['success'] and r['finish_reason'] == 'stop']
            return {'count': len(rows), 'success': sum(r['success'] for r in rows),
                    'completed_without_limit': len(valid),
                    'token_mae': statistics.mean(abs(r['predicted_output_tokens'] - r['actual_output_tokens']) for r in valid) if valid else None,
                    'mean_latency_seconds': statistics.mean(r['elapsed_seconds'] for r in rows if r['success']) if any(r['success'] for r in rows) else None,
                    'token_count_mismatches': sum(not r['token_count_matches'] for r in rows if r['success'])}

        summary = {'training': await phase('training', sample, True)}
        # Evaluation requests also produce online policy updates; therefore the
        # measurements are a prediction check, not a frozen-policy comparison.
        summary['prediction_check'] = await phase('prediction_check', evaluation, False)
        summary['limitations'] = ('SQuAD questions are English, not representative Korean traffic. '
                                  'Online learning is enabled during evaluation; this is not a policy comparison.')
        (path / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
