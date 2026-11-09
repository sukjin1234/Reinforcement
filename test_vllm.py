import asyncio
import time
import unittest
from types import SimpleNamespace

import httpx

from vllm_monitor import parse_metrics
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
