import asyncio
import time
import unittest
import tempfile
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

import vllm_balancer
from vllm_monitor import parse_metrics, collect_one
from vllm_balancer import ChatRequest, app, prepare
from vllm_bandit import DIM, features


class MetricsTests(unittest.TestCase):
    def test_running_waiting_and_cache(self):
        result = parse_metrics('''# HELP vllm:num_requests_running current requests
vllm:num_requests_running{model_name="qwen"} 2
vllm:num_requests_waiting{model_name="qwen"} 3
vllm:kv_cache_usage_perc{model_name="qwen"} 0.35
vllm:prompt_tokens_total{model_name="qwen"} 42
''')
        self.assertEqual(result['num_requests_running'], 2)
        self.assertEqual(result['num_requests_waiting'], 3)
        self.assertEqual(result['kv_cache_usage_perc'], .35)

    def test_missing_and_invalid_metrics_fail_closed(self):
        for text in ('vllm:num_requests_running 1',
                     'vllm:num_requests_running NaN\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0'):
            with self.assertRaises(ValueError):
                parse_metrics(text)

    def test_policy_receives_engine_load(self):
        host = {'cpu_pct': 0, 'memory_pct': 0, 'gpus': [{'utilization_pct': 0, 'memory_used_mib': 0, 'memory_total_mib': 1}]}
        x = features(host, 30, 40, 0, engine={'num_requests_running': 2, 'num_requests_waiting': 3, 'kv_cache_usage_perc': .4})
        self.assertEqual(len(x), DIM)
        self.assertEqual(list(x[-3:]), [.2, .3, .4])

    def test_health_checks_served_model_and_host_metrics(self):
        async def check():
            host = {'cpu_pct': 10, 'memory_pct': 20, 'gpus': [{'utilization_pct': 5, 'memory_used_mib': 100, 'memory_total_mib': 1000}]}
            metrics = 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.2\n'
            def handler(request):
                if request.url.host == 'agent':
                    return httpx.Response(200, json=host)
                if request.url.path == '/v1/models':
                    return httpx.Response(200, json={'data': [{'id': 'qwen3-8b'}]})
                return httpx.Response(200, text=metrics)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                server = {'id': 'ai', 'metrics_url': 'http://agent/metrics', 'vllm_url': 'http://engine'}
                self.assertTrue((await collect_one(client, server, {'model': 'qwen3-8b'}))['healthy'])
                self.assertFalse((await collect_one(client, server, {'model': 'other'}))['healthy'])
                host['gpus'][0]['memory_used_mib'] = 2000
                self.assertFalse((await collect_one(client, server, {'model': 'qwen3-8b'}))['healthy'])
        asyncio.run(check())

    def test_chat_uses_vllm_token_count_and_records_result(self):
        class Estimator:
            def predict(self, rows): return [40]
        class Classifier:
            classes_ = np.array(['code'])
            def predict_proba(self, rows): return np.array([[1.]])
        host = {'cpu_pct': 10, 'memory_pct': 20, 'gpus': [{'utilization_pct': 5, 'memory_used_mib': 100, 'memory_total_mib': 1000}]}
        with tempfile.TemporaryDirectory() as directory:
            config = {'backend': 'vllm', 'model': 'qwen3-8b', 'predictor_path': 'predictor', 'category_model_path': 'classifier',
                      'state_db': str(Path(directory) / 'test.sqlite3'), 'servers': [{'id': 'ai', 'vllm_url': 'http://engine', 'max_inflight': 1}],
                      'poll_seconds': 60, 'max_metric_age_seconds': 10, 'request_timeout_seconds': 10}
            bundle = {'estimator': Estimator(), 'metadata': {'categories': ['code']}}
            classifier = {'estimator': Classifier(), 'categories': ['code'], 'max_chars': 1000}
            async def collect(*args):
                return [{'server_id': 'ai', 'healthy': True, 'checked_at': time.time(), 'metrics': host,
                         'vllm': {'num_requests_running': 0, 'num_requests_waiting': 0, 'kv_cache_usage_perc': 0}}]
            async def post(url, **kwargs):
                data = {'count': 17, 'max_model_len': 128} if url.endswith('/tokenize') else {'usage': {'prompt_tokens': 17, 'completion_tokens': 8}}
                return httpx.Response(200, json=data, request=httpx.Request('POST', url))
            with patch.object(vllm_balancer, 'read_config', return_value=config), patch.object(vllm_balancer.joblib, 'load', side_effect=[bundle, classifier]), patch.object(vllm_balancer, 'collect_all', collect):
                with TestClient(vllm_balancer.app) as client:
                    with patch.object(vllm_balancer.app.state.client, 'post', side_effect=post):
                        response = client.post('/chat', json={'messages': [{'role': 'user', 'content': 'write code'}]})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['input_tokens'], 17)
                    self.assertTrue(response.json()['token_count_matches'])
                    self.assertEqual(response.json()['predicted_output_tokens'], 40)
                    self.assertEqual(client.get('/status').json()['observations']['ai'], 1)
                    self.assertEqual(client.post('/classify', json={'text': 'write code'}).json()['category'], 'code')
                    class Stream:
                        def raise_for_status(self): pass
                        async def aiter_lines(self):
                            yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'hello'}}]})
                            yield 'data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': 17, 'completion_tokens': 8}})
                            yield 'data: [DONE]'
                    @asynccontextmanager
                    async def stream(*args, **kwargs):
                        yield Stream()
                    with patch.object(vllm_balancer.app.state.client, 'post', side_effect=post), patch.object(vllm_balancer.app.state.client, 'stream', side_effect=stream):
                        response = client.post('/chat/stream', json={'messages': [{'role': 'user', 'content': 'write code'}]})
                    events = [json.loads(line) for line in response.text.splitlines()]
                    self.assertEqual([event['event'] for event in events], ['routed', 'chunk', 'chunk', 'complete'])
                    self.assertEqual(events[-1]['actual_output_tokens'], 8)
                    self.assertTrue(events[-1]['token_count_matches'])
                    self.assertEqual(client.get('/status').json()['inflight']['ai'], 0)


class TokenizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_template_is_applied_before_prediction(self):
        captured = []
        def handler(request):
            import json
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={'count': 17, 'tokens': [0] * 17, 'max_model_len': 20})
        class Estimator:
            def predict(self, rows):
                self.rows = rows
                return [23]
        estimator = Estimator()
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.state.config = {'model': 'qwen3-8b', 'max_metric_age_seconds': 10,
                            'servers': [{'id': 'ai', 'vllm_url': 'http://ai'}]}
        app.state.status = {'ai': {'healthy': True, 'checked_at': time.time()}}
        app.state.client = client
        app.state.bundle = {'metadata': {'categories': ['qa']}, 'estimator': estimator}
        request = ChatRequest(messages=[{'role': 'user', 'content': '안녕하세요'}], category='qa')
        try:
            count, _, _, predicted, max_output = await prepare(request)
        finally:
            await client.aclose()
        self.assertEqual((count, predicted), (17, 23))
        self.assertEqual(max_output, 3)
        self.assertEqual(estimator.rows, [[17, 'qa']])
        self.assertEqual(captured[0]['chat_template_kwargs'], {'enable_thinking': False})


if __name__ == '__main__':
    unittest.main()
