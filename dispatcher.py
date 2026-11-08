"""Bounded FIFO admission and atomic server reservation."""
import asyncio
import time
from collections import deque
from fastapi import HTTPException
from bandit import features


class Dispatcher:
    def __init__(self, state):
        self.state = state
        self.condition = asyncio.Condition(state.lock)
        self.waiters = deque()
        self.rejected = 0
        self.timed_out = 0

    async def reserve(self, input_tokens, predicted):
        state = self.state
        marker = object()
        started = time.monotonic()
        deadline = started + state.config.get('queue_wait_seconds', 60)
        async with self.condition:
            if len(self.waiters) >= state.config.get('queue_max_size', 100):
                self.rejected += 1
                raise HTTPException(503, 'Request queue is full')
            self.waiters.append(marker)
            try:
                while True:
                    eligible = []
                    if self.waiters[0] is marker:
                        for server in state.config['servers']:
                            sid = server['id']
                            snapshot = state.status.get(sid, {})
                            if not snapshot.get('healthy') or time.time() - snapshot['checked_at'] > state.config['max_metric_age_seconds']:
                                continue
                            if state.active[sid] >= server.get('max_inflight', 1):
                                continue
                            x = features(snapshot['metrics'], input_tokens, predicted, state.active[sid])
                            eligible.append((state.bandit.score(sid, x), server, x))
                    if eligible:
                        _, server, x = max(eligible, key=lambda item: (item[0], item[1]['id']))
                        state.active[server['id']] += 1
                        self.waiters.popleft()
                        self.condition.notify_all()
                        return server, x, time.monotonic() - started
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.timed_out += 1
                        raise HTTPException(503, 'Timed out waiting for a healthy server')
                    try:
                        await asyncio.wait_for(self.condition.wait(), remaining)
                    except asyncio.TimeoutError:
                        self.timed_out += 1
                        raise HTTPException(503, 'Timed out waiting for a healthy server')
            finally:
                if marker in self.waiters:
                    self.waiters.remove(marker)
                    self.condition.notify_all()
