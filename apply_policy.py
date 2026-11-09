"""Import an evaluated experiment's learned policy into a stopped router DB."""
import argparse
from contextlib import closing
import json
import sqlite3
from pathlib import Path
from bandit import DIM


def apply(source_path, target_path, backup_path, run_id):
    source_path, target_path, backup_path = map(Path, (source_path, target_path, backup_path))
    if not source_path.is_file() or not target_path.is_file() or backup_path.exists():
        raise ValueError('Source and target must exist and backup must be new')
    if len({p.resolve() for p in (source_path, target_path, backup_path)}) != 3:
        raise ValueError('Source, target and backup must be different files')
    with closing(sqlite3.connect(source_path)) as source, closing(sqlite3.connect(target_path)) as target:
        arms = source.execute('select server_id,a,b,n from arms').fetchall()
        if {r[0] for r in arms} != {'ai', 'ai1', 'ai2'}:
            raise ValueError('Expected trained policy for each of ai, ai1, ai2')
        for name, a, b, n in arms:
            a, b = json.loads(a), json.loads(b)
            if len(a) != DIM or any(len(row) != DIM for row in a) or len(b) != DIM or n < 1:
                raise ValueError('Invalid policy dimensions or missing observations for ' + name)
        columns = ['timestamp', 'server_id', 'input_tokens', 'category', 'predicted_tokens', 'actual_tokens',
                   'duration_seconds', 'success', 'reward', 'context', 'prompt_count', 'prompt_duration_ns',
                   'eval_duration_ns', 'queue_seconds']
        source_columns = {r[1] for r in source.execute('PRAGMA table_info(outcomes)')}
        target_columns = {r[1] for r in target.execute('PRAGMA table_info(outcomes)')}
        for optional in ('ttft_seconds', 'mean_inter_chunk_seconds', 'prompt_cached_count'):
            if optional in source_columns and optional in target_columns:
                columns.append(optional)
        records = source.execute('SELECT ' + ','.join(columns) + ' FROM outcomes ORDER BY id').fetchall()
        if not records or any(not r[7] for r in records):
            raise ValueError('Training records are absent or include failures')
        with closing(sqlite3.connect(backup_path)) as backup:
            target.backup(backup)
        with target:
            target.execute('CREATE TABLE IF NOT EXISTS policy_deployments (run_id TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)')
            target.execute('INSERT INTO policy_deployments(run_id) VALUES (?)', (run_id,))
            target.executemany('INSERT OR REPLACE INTO arms(server_id,a,b,n) VALUES (?,?,?,?)', arms)
            target.executemany('INSERT INTO outcomes(' + ','.join(columns) + ') VALUES (' + ','.join('?' for _ in columns) + ')', records)
    return {'servers': {row[0]: row[3] for row in arms}, 'training_records': len(records), 'backup': str(backup_path)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--backup', required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    print(json.dumps(apply(args.source, args.target, args.backup, args.run_id), ensure_ascii=False, indent=2))
