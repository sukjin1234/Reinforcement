"""Bounded FIFO admission and atomic server reservation."""
import asyncio
import time
import uuid
from collections import deque
from fastapi import HTTPException
from bandit import features
from workload import estimate_seconds, snapshot as work_snapshot


class Dispatcher:
    def __init__(self, state):
        self.state = state
        self.condition = asyncio.Condition(state.lock)
        self.waiters = deque()
        self.rejected = 0
        self.timed_out = 0
        self.next_server = 0

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
                            perf = state.bandit.recent_performance(sid)
                            if state.config.get('backend') == 'vllm':
                                perf = {**perf, **snapshot.get('vllm_performance', {})}
                                from vllm_bandit import features as vllm_features
                                x = vllm_features(snapshot['metrics'], input_tokens, predicted, state.active[sid], perf, snapshot['vllm'])
                            else:
                                x = features(snapshot['metrics'], input_tokens, predicted, state.active[sid], perf)
                            own = estimate_seconds(input_tokens, predicted, perf)
                            existing = [work_snapshot(w)['estimated_remaining_seconds'] for w in state.active_work[sid].values()]
                            estimate = own + sum(existing) if own is not None and all(v is not None for v in existing) else None
                            eligible.append((state.bandit.score(sid, x), server, x, estimate, own))
                    if eligible:
                        policy = state.config.get('policy', 'linucb')
                        if policy == 'round_robin' or (policy == 'shortest_predicted' and any(item[3] is None for item in eligible)):
                            ordered = state.config['servers']
                            eligible_ids = {item[1]['id']: item for item in eligible}
                            for offset in range(len(ordered)):
                                index = (self.next_server + offset) % len(ordered)
                                if ordered[index]['id'] in eligible_ids:
                                    _, server, x, _, own = eligible_ids[ordered[index]['id']]
                                    self.next_server = (index + 1) % len(ordered)
                                    break
                        elif policy == 'shortest_predicted':
                            _, server, x, _, own = min(eligible, key=lambda item: (item[3], item[1]['id']))
                        else:
                            _, server, x, _, own = max(eligible, key=lambda item: (item[0], item[1]['id']))
                        state.active[server['id']] += 1
                        work_id = uuid.uuid4().hex
                        state.active_work[server['id']][work_id] = {'start': time.monotonic(), 'input_tokens': input_tokens,
                            'predicted_output_tokens': predicted, 'estimated_service_seconds': own}
                        self.waiters.popleft()
                        self.condition.notify_all()
                        return server, x, time.monotonic() - started, work_id
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
