"""Calibrate actual Qwen token counts on held-out human-written questions."""
import argparse
import asyncio
import json
import time
import collections
from pathlib import Path

import httpx
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from buckets import predict_bucket
from category import classify


async def run(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    predictor = joblib.load(config['predictor_path'])
    classifier = joblib.load(config['category_model_path'])
    bucket_model = joblib.load(args.bucket_model)
    endpoints = [s['ollama_url'].rstrip('/') for s in config['servers']]
    locks = [asyncio.Lock() for _ in endpoints]
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        async def send(index, item):
            endpoint_index = index % len(endpoints)
            async with locks[endpoint_index]:
                payload = {'model': config['model'], 'messages': [{'role': 'user', 'content': item['prompt']}],
                           'stream': False, 'think': False, 'keep_alive': '30m', 'options': predictor['metadata']['settings']}
                started = time.monotonic()
                try:
                    response = await client.post(endpoints[endpoint_index] + '/api/chat', json=payload)
                    response.raise_for_status()
                    result = response.json()
                    actual_in, actual_out = result['prompt_eval_count'], result['eval_count']
                    classification = classify(classifier, item['prompt'])
                    cat = classification['category']
                    predicted = max(0, round(float(predictor['estimator'].predict([[actual_in, cat]])[0])))
                    estimated_bucket = predict_bucket(bucket_model, item['prompt'], actual_in, cat)
                    row = {'id': item['id'], 'status': 'ok', 'server_id': config['servers'][endpoint_index]['id'],
                           'input_tokens': actual_in, 'output_tokens': actual_out, 'done_reason': result.get('done_reason'),
                           'prompt_eval_duration': result.get('prompt_eval_duration'), 'prompt_eval_cached_count': result.get('prompt_eval_cached_count'),
                           'eval_duration': result.get('eval_duration'),
                           'category': cat, 'category_score': classification['score'], 'predicted_tokens': predicted,
                           'bucket_prediction': estimated_bucket, 'elapsed_seconds': time.monotonic() - started}
                except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
                    row = {'id': item['id'], 'status': 'error', 'error': type(exc).__name__}
                print('calibrated', index + 1, '/', len(manifest['questions']), row['status'], flush=True)
                return row
        rows = await asyncio.gather(*(send(i, item) for i, item in enumerate(manifest['questions'])))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite calibration results')
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    valid = [r for r in rows if r['status'] == 'ok' and r['done_reason'] == 'stop']
    def true_bucket(n): return int(np.searchsorted(bucket_model['bounds'], n, side='right'))
    bucket_counts = collections.Counter(bucket_model['labels'][true_bucket(r['output_tokens'])] for r in valid)
    report = {'source': manifest['source'], 'count': len(rows), 'valid': len(valid),
              'censored': sum(r.get('status') == 'ok' and r.get('done_reason') != 'stop' for r in rows),
              'errors': sum(r['status'] != 'ok' for r in rows),
              'token_mae': float(mean_absolute_error([r['output_tokens'] for r in valid], [r['predicted_tokens'] for r in valid])) if valid else None,
              'bucket_accuracy': float(np.mean([bucket_model['labels'][true_bucket(r['output_tokens'])] == r['bucket_prediction']['bucket'] for r in valid])) if valid else None,
              'actual_bucket_counts': dict(bucket_counts),
              'majority_bucket_accuracy': max(bucket_counts.values()) / len(valid) if valid else None,
              'under_bucket_rate': float(np.mean([bucket_model['labels'].index(r['bucket_prediction']['bucket']) < true_bucket(r['output_tokens']) for r in valid])) if valid else None,
              'warning': '24 human-authored English SQuAD questions with synthetic style wrappers, not Korean production traffic; automatic category is not ground-truth verified.'}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', default='config.json')
    p.add_argument('--bucket-model', default='bucket_model.joblib')
    p.add_argument('--output', required=True)
    p.add_argument('--report', required=True)
    asyncio.run(run(p.parse_args()))
