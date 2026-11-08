"""Per-server contextual LinUCB, updated from observed request outcomes."""
import json
import math
import sqlite3
import time
import numpy as np

DIM = 7


def features(metrics, input_tokens, output_tokens, inflight):
    gpus = metrics['gpus']
    gpu_util = max(g['utilization_pct'] for g in gpus) / 100
    gpu_mem = max(g['memory_used_mib'] / g['memory_total_mib'] for g in gpus)
    return np.array([1.0, metrics['cpu_pct'] / 100, metrics['memory_pct'] / 100,
                     gpu_util, gpu_mem, min(input_tokens / 4096, 2),
                     min(output_tokens / 4096, 2) + min(inflight, 10)], dtype=float)


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

    def state(self, server_id):
        row = self.db.execute('SELECT a,b,n FROM arms WHERE server_id=?', (server_id,)).fetchone()
        if row is None:
            return np.eye(DIM), np.zeros(DIM), 0
        return np.array(json.loads(row[0])), np.array(json.loads(row[1])), row[2]

    def score(self, server_id, x):
        a, b, _ = self.state(server_id)
        inv_x = np.linalg.solve(a, x)
        return float(b @ inv_x + self.alpha * np.sqrt(max(0, x @ inv_x)))

    def update(self, server_id, x, duration, success, input_tokens, category, predicted, actual):
        # Bounded reward prevents one timeout dominating the policy.
        reward = -min(math.log1p(max(duration, 0) / 10), 5.0) if success else -5.0
        a, b, n = self.state(server_id)
        a += np.outer(x, x)
        b += reward * x
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO arms VALUES (?,?,?,?)',
                            (server_id, json.dumps(a.tolist()), json.dumps(b.tolist()), n + 1))
            self.db.execute('INSERT INTO outcomes(timestamp,server_id,input_tokens,category,predicted_tokens,actual_tokens,duration_seconds,success,reward,context) VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (time.time(), server_id, input_tokens, category, predicted, actual,
                             duration, int(success), reward, json.dumps(x.tolist())))
        return reward
