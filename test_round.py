import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient
import balancer
from dispatcher import Dispatcher
from bandit import LinUCB, features
from monitor import collect_one
from experiment import select_questions
from apply_policy import apply
from workload import estimate_seconds, snapshot

METRICS = {'cpu_pct': 10, 'memory_pct': 30, 'gpus': [{'utilization_pct': 20, 'memory_used_mib': 100, 'memory_total_mib': 1000}]}
CONFIG = {'model': 'model', 'model_digest': 'digest'}
SERVER = {'id': 'a', 'metrics_url': 'http://a/metrics', 'ollama_url': 'http://a'}


class Tests(unittest.TestCase):
    def test_bandit_reward_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'policy.sqlite3')
            model = LinUCB(path)
            x = features(METRICS, 100, 500, 0)
            prior = model.score('a', x)
            model.update('a', x, 20, True, 100, 'code', 500, 450,
                         prompt_count=100, prompt_duration_ns=2e9, eval_duration_ns=9e9, queue_seconds=1, prompt_cached_count=20)
            self.assertLess(model.score('a', x), prior)
            self.assertEqual(model.state('a')[2], 1)
            model.db.close()
            reopened = LinUCB(path)
            self.assertEqual(reopened.state('a')[2], 1)
            self.assertEqual(reopened.db.execute('select actual_tokens from outcomes').fetchone()[0], 450)
            performance = reopened.recent_performance('a')
            self.assertEqual(performance['input_tokens_per_second'], 50)
            self.assertEqual(performance['uncached_input_tokens_per_second'], 40)
            self.assertEqual(performance['output_tokens_per_second'], 50)
            reopened.update('a', x, 10, True, 100, 'code', 500, 450, learn=False)
            self.assertEqual(reopened.state('a')[2], 1)
            self.assertEqual(reopened.db.execute('select count(*) from outcomes').fetchone()[0], 2)
            reopened.db.close()

    def test_feature_includes_prediction(self):
        self.assertFalse(np.array_equal(features(METRICS, 100, 500, 0), features(METRICS, 100, 1000, 0)))
        self.assertFalse(np.array_equal(features(METRICS, 100, 500, 0),
                                        features(METRICS, 100, 500, 0, {'input_tokens_per_second': 1000, 'output_tokens_per_second': 50})))

    def test_old_policy_dimensions_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            model = LinUCB(str(Path(directory) / 'old.sqlite3'))
            old_a = np.eye(7) * 2
            old_b = np.arange(7)
            model.db.execute('insert into arms values (?,?,?,?)', ('old', json.dumps(old_a.tolist()), json.dumps(old_b.tolist()), 3))
            model.db.commit()
            a, b, n = model.state('old')
            np.testing.assert_array_equal(a[:7, :7], old_a)
            np.testing.assert_array_equal(b[:7], old_b)
            self.assertEqual(a.shape, (11, 11))
            self.assertEqual(n, 3)
            model.db.close()

    def test_collection_rejects_wrong_model_and_failed_agent(self):
        async def check():
            def handler(request):
                if request.url.path == '/metrics':
                    return httpx.Response(200, json=METRICS)
                return httpx.Response(200, json={'models': [{'name': 'model', 'digest': 'wrong'}]})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await collect_one(client, SERVER, CONFIG)
                self.assertFalse(result['healthy'])
                self.assertIn('digest', result['error'])
            def healthy(request):
                return httpx.Response(200, json=METRICS if request.url.path == '/metrics' else {'models': [{'name': 'model', 'digest': 'digest'}]})
            async with httpx.AsyncClient(transport=httpx.MockTransport(healthy)) as client:
                self.assertTrue((await collect_one(client, SERVER, CONFIG))['healthy'])
        asyncio.run(check())

    def test_router_selection_and_unavailable_server(self):
        class Estimator:
            def predict(self, values):
                return [500]
        class Classifier:
            classes_ = np.array(['code'])
            def predict_proba(self, values):
                return np.array([[1.0]])
        with tempfile.TemporaryDirectory() as directory:
            config = {'model': 'model', 'model_digest': 'digest', 'predictor_path': 'unused',
                      'state_db': str(Path(directory) / 'state.sqlite3'), 'servers': [SERVER | {'max_inflight': 1}],
                      'poll_seconds': 60, 'max_metric_age_seconds': 10, 'request_timeout_seconds': 10,
                      'queue_wait_seconds': 0.05}
            bundle = {'estimator': Estimator(), 'metadata': {'model_name': 'model', 'model_digest': 'digest',
                      'settings': {}, 'input_min': 10, 'input_max': 2000, 'categories': ['code']}}
            classifier = {'estimator': Classifier(), 'categories': ['code'], 'max_chars': 1000}
            async def collected(*args):
                return [{'server_id': 'a', 'healthy': False, 'checked_at': __import__('time').time()}]
            def load_model(path):
                return bundle if path == 'unused' else classifier
            with patch.object(balancer, 'read_config', return_value=config), patch.object(balancer.joblib, 'load', side_effect=load_model), patch.object(balancer, 'collect_all', collected):
                with TestClient(balancer.app) as client:
                    self.assertEqual(client.post('/classify', json={'text': 'write code'}).json()['category'], 'code')
                    self.assertEqual(client.post('/classify', json={'text': '  '}).status_code, 422)
                    request = {'input_tokens': 100, 'category': 'code', 'messages': [{'role': 'user', 'content': 'hello'}]}
                    self.assertEqual(client.post('/chat', json=request).status_code, 503)
                    balancer.app.state.status = {'a': {'healthy': True, 'checked_at': __import__('time').time(), 'metrics': METRICS}}
                    async def post(*args, **kwargs):
                        return httpx.Response(200, json={'done': True, 'eval_count': 12}, request=httpx.Request('POST', 'http://a/api/chat'))
                    with patch.object(balancer.app.state.client, 'post', side_effect=post):
                        response = client.post('/chat', json=request)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['server_id'], 'a')
                    self.assertEqual(response.json()['predicted_output_tokens'], 500)
                    self.assertEqual(client.get('/status').json()['observations']['a'], 1)
                    self.assertEqual(client.get('/status').json()['policy'], 'linucb')
                    self.assertTrue(client.get('/status').json()['training_enabled'])
                    self.assertEqual(balancer.app.state.active['a'], 0)
                    self.assertEqual(client.post('/chat', json={**request, 'category': 'unknown'}).status_code, 422)
                    with patch.object(balancer.app.state.client, 'post', side_effect=post):
                        response = client.post('/chat', json={k: v for k, v in request.items() if k != 'category'})
                    self.assertEqual(response.json()['category'], 'code')
                    self.assertEqual(response.json()['classification']['score'], 1.0)
                    class StreamResponse:
                        def raise_for_status(self):
                            pass
                        async def aiter_lines(self):
                            yield json.dumps({'message': {'content': 'hello'}, 'done': False})
                            yield json.dumps({'message': {'content': ' world'}, 'done': False})
                            yield json.dumps({'message': {'content': ''}, 'done': True, 'eval_count': 2,
                                              'prompt_eval_count': 10, 'prompt_eval_duration': 100,
                                              'eval_duration': 200})
                    @asynccontextmanager
                    async def stream(*args, **kwargs):
                        yield StreamResponse()
                    with patch.object(balancer.app.state.client, 'stream', side_effect=stream):
                        response = client.post('/chat/stream', json=request)
                    events = [json.loads(line) for line in response.text.splitlines()]
                    self.assertEqual([e['event'] for e in events], ['routed', 'chunk', 'chunk', 'chunk', 'complete'])
                    self.assertIsNotNone(events[-1]['ttft_seconds'])
                    self.assertIsNotNone(events[-1]['mean_inter_chunk_seconds'])
                    self.assertEqual(client.get('/status').json()['inflight']['a'], 0)
                    with closing(sqlite3.connect(config['state_db'])) as db:
                        self.assertIsNotNone(db.execute('SELECT ttft_seconds FROM outcomes ORDER BY id DESC').fetchone()[0])
                    class FailedStream(StreamResponse):
                        async def aiter_lines(self):
                            yield json.dumps({'error': 'upstream failed'})
                    @asynccontextmanager
                    async def failed_stream(*args, **kwargs):
                        yield FailedStream()
                    with patch.object(balancer.app.state.client, 'stream', side_effect=failed_stream):
                        response = client.post('/chat/stream', json=request)
                    events = [json.loads(line) for line in response.text.splitlines()]
                    self.assertEqual(events[-1]['event'], 'error')
                    self.assertEqual(client.get('/status').json()['inflight']['a'], 0)

    def test_queue_fifo_full_and_cancellation(self):
        async def check():
            from fastapi import HTTPException
            import time
            with tempfile.TemporaryDirectory() as directory:
                bandit = LinUCB(str(Path(directory) / 'queue.sqlite3'))
                state = SimpleNamespace(lock=asyncio.Lock(), bandit=bandit,
                    config={'servers': [SERVER | {'max_inflight': 1}], 'max_metric_age_seconds': 10,
                            'queue_max_size': 1, 'queue_wait_seconds': 2},
                    active={'a': 1}, active_work={'a': {}}, status={'a': {'healthy': True, 'checked_at': time.time(), 'metrics': METRICS}})
                q = Dispatcher(state)
                first = asyncio.create_task(q.reserve(100, 500))
                await asyncio.sleep(0.01)
                self.assertEqual(len(q.waiters), 1)
                with self.assertRaises(HTTPException) as error:
                    await q.reserve(100, 500)
                self.assertEqual(error.exception.status_code, 503)
                async with q.condition:
                    state.active['a'] = 0
                    q.condition.notify_all()
                server, _, delay, _ = await first
                self.assertEqual(server['id'], 'a')
                self.assertGreaterEqual(delay, 0)
                self.assertEqual(state.active['a'], 1)
                pending = asyncio.create_task(q.reserve(100, 500))
                await asyncio.sleep(0.01)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertEqual(len(q.waiters), 0)
                bandit.db.close()
        asyncio.run(check())

    def test_queue_preserves_fifo_order(self):
        async def check():
            import time
            with tempfile.TemporaryDirectory() as directory:
                bandit = LinUCB(str(Path(directory) / 'fifo.sqlite3'))
                state = SimpleNamespace(lock=asyncio.Lock(), bandit=bandit,
                    config={'servers': [SERVER | {'max_inflight': 1}], 'max_metric_age_seconds': 10,
                            'queue_max_size': 3, 'queue_wait_seconds': 2},
                    active={'a': 1}, active_work={'a': {}}, status={'a': {'healthy': True, 'checked_at': time.time(), 'metrics': METRICS}})
                q = Dispatcher(state)
                first = asyncio.create_task(q.reserve(100, 500))
                await asyncio.sleep(0.02)
                second = asyncio.create_task(q.reserve(200, 500))
                await asyncio.sleep(0.02)
                self.assertEqual(len(q.waiters), 2)
                async with q.condition:
                    state.active['a'] = 0
                    q.condition.notify_all()
                await first
                self.assertFalse(second.done())
                async with q.condition:
                    state.active['a'] = 0
                    q.condition.notify_all()
                await second
                self.assertEqual(len(q.waiters), 0)
                bandit.db.close()
        asyncio.run(check())

    def test_round_robin_uses_each_available_server(self):
        async def check():
            import time
            with tempfile.TemporaryDirectory() as directory:
                bandit = LinUCB(str(Path(directory) / 'rr.sqlite3'))
                servers = [{'id': x, 'max_inflight': 1} for x in ('a', 'b', 'c')]
                state = SimpleNamespace(lock=asyncio.Lock(), bandit=bandit,
                    config={'servers': servers, 'policy': 'round_robin', 'max_metric_age_seconds': 10},
                    active=dict.fromkeys(('a', 'b', 'c'), 0), active_work={x: {} for x in ('a', 'b', 'c')},
                    status={x: {'healthy': True, 'checked_at': time.time(), 'metrics': METRICS} for x in ('a', 'b', 'c')})
                dispatcher = Dispatcher(state)
                chosen = []
                for _ in range(3):
                    server, _, _, _ = await dispatcher.reserve(100, 200)
                    chosen.append(server['id'])
                self.assertEqual(chosen, ['a', 'b', 'c'])
                bandit.db.close()
        asyncio.run(check())

    def test_remaining_estimate_marks_overdue_and_missing_rates(self):
        self.assertIsNone(estimate_seconds(100, 200, {}))
        self.assertEqual(estimate_seconds(100, 200, {'uncached_input_tokens_per_second': 100, 'output_tokens_per_second': 50}), 5)
        result = snapshot({'start': 10, 'input_tokens': 100, 'predicted_output_tokens': 200, 'estimated_service_seconds': 5}, now=16)
        self.assertTrue(result['overdue'])
        self.assertEqual(result['estimated_remaining_seconds'], 0)

    def test_shortest_predicted_policy_and_unknown_speed_fallback(self):
        async def check():
            import time
            with tempfile.TemporaryDirectory() as directory:
                bandit = LinUCB(str(Path(directory) / 'speed.sqlite3'))
                x = features(METRICS, 100, 500, 0)
                for sid, duration in [('a', 10e9), ('b', 2e9)]:
                    bandit.update(sid, x, 10, True, 100, 'code', 500, 500,
                                  prompt_count=100, prompt_duration_ns=1e9,
                                  eval_duration_ns=duration, prompt_cached_count=0, learn=False)
                servers = [{'id': sid, 'max_inflight': 1} for sid in ('a', 'b')]
                state = SimpleNamespace(lock=asyncio.Lock(), bandit=bandit,
                    config={'servers': servers, 'policy': 'shortest_predicted', 'max_metric_age_seconds': 10},
                    active={'a': 0, 'b': 0}, active_work={'a': {}, 'b': {}},
                    status={sid: {'healthy': True, 'checked_at': time.time(), 'metrics': METRICS} for sid in ('a', 'b')})
                server, _, _, _ = await Dispatcher(state).reserve(100, 500)
                self.assertEqual(server['id'], 'b')
                bandit.db.close()
        asyncio.run(check())

    def test_experiment_holds_out_domains_and_filters_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = [
                {'id': 'train', 'domain': '소프트웨어', 'category': 'code', 'prompt': 'test'},
                {'id': 'test', 'domain': '학교·교육', 'category': 'code', 'prompt': 'test2'},
                {'id': 'bad', 'domain': '학교·교육', 'category': 'code', 'prompt': 'bad'},
            ]
            results = [{'id': x['id'], 'status': 'ok', 'done_reason': 'length' if x['id'] == 'bad' else 'stop',
                        'input_tokens': 100, 'output_tokens': 300} for x in questions]
            for name, rows in [('questions', questions), ('results', results)]:
                (root / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
            train, test = select_questions(root / 'questions', root / 'results', 42, 1, 1)
            self.assertEqual([r['id'] for r in train], ['train'])
            self.assertEqual([r['id'] for r in test], ['test'])

    def test_apply_policy_preserves_backup_and_rejects_duplicates(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = LinUCB(str(root / 'src.sqlite3'))
            dst = LinUCB(str(root / 'dst.sqlite3'))
            x = features(METRICS, 100, 500, 0)
            dst.update('ai', x, 4, True, 100, 'code', 500, 100)
            for server in ('ai', 'ai1', 'ai2'):
                src.update(server, x, 3, True, 100, 'code', 500, 100)
            src.db.close()
            dst.db.close()
            result = apply(root / 'src.sqlite3', root / 'dst.sqlite3', root / 'backup.sqlite3', 'run1')
            self.assertEqual(result['training_records'], 3)
            with closing(sqlite3.connect(root / 'backup.sqlite3')) as backup:
                self.assertEqual(backup.execute('select count(*) from outcomes').fetchone()[0], 1)
            with closing(sqlite3.connect(root / 'dst.sqlite3')) as live:
                self.assertEqual(live.execute('select count(*) from outcomes').fetchone()[0], 4)
            with self.assertRaises(sqlite3.IntegrityError):
                apply(root / 'src.sqlite3', root / 'dst.sqlite3', root / 'backup2.sqlite3', 'run1')
            with closing(sqlite3.connect(root / 'dst.sqlite3')) as live:
                self.assertEqual(live.execute('select count(*) from outcomes').fetchone()[0], 4)


if __name__ == '__main__':
    unittest.main()
