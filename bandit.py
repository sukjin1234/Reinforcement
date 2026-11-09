"""Per-server contextual LinUCB, updated from observed request outcomes."""
import json
import math
import sqlite3
import time
import numpy as np

DIM = 11


def features(metrics, input_tokens, output_tokens, inflight, performance=None):
    gpus = metrics['gpus']
    gpu_util = max(g['utilization_pct'] for g in gpus) / 100
    gpu_mem = max(g['memory_used_mib'] / g['memory_total_mib'] for g in gpus)
    performance = performance or {}
    input_rate = performance.get('input_tokens_per_second')
    output_rate = performance.get('output_tokens_per_second')
    return np.array([1.0, metrics['cpu_pct'] / 100, metrics['memory_pct'] / 100,
                     gpu_util, gpu_mem, min(input_tokens / 4096, 2),
                     min(output_tokens / 4096, 2) + min(inflight, 10),
                     min(input_rate / 1000, 2) if input_rate is not None else 0,
                     min(output_rate / 100, 2) if output_rate is not None else 0,
                     float(input_rate is not None), float(output_rate is not None)], dtype=float)


class LinUCB:
    def __init__(self, path, alpha=0.7):
        self.db = sqlite3.connect(path)
        self.alpha = alpha
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS arms (server_id TEXT PRIMARY KEY, a TEXT NOT NULL, b TEXT NOT NULL, n INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS outcomes (id INTEGER PRIMARY KEY, timestamp REAL NOT NULL,
              server_id TEXT NOT NULL, input_tokens INTEGER NOT NULL, category TEXT NOT NULL,
              predicted_tokens INTEGER NOT NULL, actual_tokens INTEGER, duration_seconds REAL NOT NULL,
              success INTEGER NOT NULL, reward REAL NOT NULL, context TEXT NOT NULL);
        ''')
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(outcomes)')}
        for name in ('prompt_count', 'prompt_duration_ns', 'eval_duration_ns', 'queue_seconds', 'ttft_seconds', 'mean_inter_chunk_seconds', 'prompt_cached_count'):
            if name not in columns:
                self.db.execute(f'ALTER TABLE outcomes ADD COLUMN {name} REAL')
        self.db.commit()

    def state(self, server_id):
        row = self.db.execute('SELECT a,b,n FROM arms WHERE server_id=?', (server_id,)).fetchone()
        if row is None:
            return np.eye(DIM), np.zeros(DIM), 0
        a, b = np.array(json.loads(row[0])), np.array(json.loads(row[1]))
        if a.shape == (7, 7) and b.shape == (7,):
            migrated_a, migrated_b = np.eye(DIM), np.zeros(DIM)
            migrated_a[:7, :7], migrated_b[:7] = a, b
            return migrated_a, migrated_b, row[2]
        if a.shape != (DIM, DIM) or b.shape != (DIM,):
            raise ValueError('Incompatible saved policy dimensions')
        return a, b, row[2]

    def score(self, server_id, x):
        a, b, _ = self.state(server_id)
        inv_x = np.linalg.solve(a, x)
        return float(b @ inv_x + self.alpha * np.sqrt(max(0, x @ inv_x)))

    def update(self, server_id, x, duration, success, input_tokens, category, predicted, actual,
               prompt_count=None, prompt_duration_ns=None, eval_duration_ns=None, queue_seconds=None, learn=True,
               ttft_seconds=None, mean_inter_chunk_seconds=None, prompt_cached_count=None):
        # Bounded reward prevents one timeout dominating the policy.
        reward = -min(math.log1p(max(duration, 0) / 10), 5.0) if success else -5.0
        a, b, n = self.state(server_id)
        a += np.outer(x, x)
        b += reward * x
        with self.db:
            if learn:
                self.db.execute('INSERT OR REPLACE INTO arms VALUES (?,?,?,?)',
                                (server_id, json.dumps(a.tolist()), json.dumps(b.tolist()), n + 1))
            self.db.execute('INSERT INTO outcomes(timestamp,server_id,input_tokens,category,predicted_tokens,actual_tokens,duration_seconds,success,reward,context,prompt_count,prompt_duration_ns,eval_duration_ns,queue_seconds,ttft_seconds,mean_inter_chunk_seconds,prompt_cached_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (time.time(), server_id, input_tokens, category, predicted, actual,
                             duration, int(success), reward, json.dumps(x.tolist()), prompt_count,
                             prompt_duration_ns, eval_duration_ns, queue_seconds, ttft_seconds, mean_inter_chunk_seconds, prompt_cached_count))
        return reward

    def recent_performance(self, server_id, limit=20):
        rows = self.db.execute('''SELECT duration_seconds, prompt_count, prompt_duration_ns,
                   actual_tokens, eval_duration_ns, prompt_cached_count, ttft_seconds, mean_inter_chunk_seconds FROM outcomes
                   WHERE server_id=? AND success=1 ORDER BY id DESC LIMIT ?''', (server_id, limit)).fetchall()
        def rate(count_idx, duration_idx):
            valid = [(r[count_idx], r[duration_idx]) for r in rows
                     if r[count_idx] is not None and r[duration_idx] is not None
                     and r[count_idx] >= 0 and r[duration_idx] > 0]
            return (sum(x for x, _ in valid) * 1e9 / sum(d for _, d in valid)) if valid else None
        # A one-token cache miss is dominated by fixed overhead and does not
        # estimate throughput for a new prompt. Require a meaningful sample.
        cached_valid = [(r[1] - r[5], r[2]) for r in rows if r[1] is not None and r[5] is not None
                        and r[2] is not None and 0 <= r[5] <= r[1] and r[1] - r[5] >= 32 and r[2] > 0]
        def mean_optional(index):
            values = [r[index] for r in rows if r[index] is not None]
            return sum(values) / len(values) if values else None
        return {'samples': len(rows), 'mean_latency_seconds': sum(r[0] for r in rows) / len(rows) if rows else None,
                'input_tokens_per_second': rate(1, 2),
                'uncached_input_tokens_per_second': sum(n for n, _ in cached_valid) * 1e9 / sum(d for _, d in cached_valid) if cached_valid else None,
                'uncached_rate_samples': len(cached_valid),
                'output_tokens_per_second': rate(3, 4), 'mean_ttft_seconds': mean_optional(6),
                'mean_inter_chunk_seconds': mean_optional(7)}
