"""Periodically write a JSONL record per server."""
import argparse
import asyncio
import json
import time
import httpx
from monitor import collect_all, read_config


async def run(config, output, once):
    async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
        while True:
            started = time.monotonic()
            rows = await collect_all(client, config)
            with open(output, 'a', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps(rows, ensure_ascii=False), flush=True)
            if once:
                return
            await asyncio.sleep(max(0, config['poll_seconds'] - (time.monotonic() - started)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='metrics.jsonl')
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    asyncio.run(run(read_config(args.config), args.output, args.once))
