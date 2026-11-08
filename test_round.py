import asyncio
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient
import balancer
from bandit import LinUCB, features
from monitor import collect_one

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
            model.update('a', x, 20, True, 100, 'code', 500, 450)
            self.assertLess(model.score('a', x), prior)
            self.assertEqual(model.state('a')[2], 1)
            model.db.close()
            reopened = LinUCB(path)
            self.assertEqual(reopened.state('a')[2], 1)
            self.assertEqual(reopened.db.execute('select actual_tokens from outcomes').fetchone()[0], 450)
            reopened.db.close()

    def test_feature_includes_prediction(self):
        self.assertFalse(np.array_equal(features(METRICS, 100, 500, 0), features(METRICS, 100, 1000, 0)))

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
                      'poll_seconds': 60, 'max_metric_age_seconds': 10, 'request_timeout_seconds': 10}
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
                    self.assertEqual(balancer.app.state.active['a'], 0)
                    self.assertEqual(client.post('/chat', json={**request, 'category': 'unknown'}).status_code, 422)
                    with patch.object(balancer.app.state.client, 'post', side_effect=post):
                        response = client.post('/chat', json={k: v for k, v in request.items() if k != 'category'})
                    self.assertEqual(response.json()['category'], 'code')
                    self.assertEqual(response.json()['classification']['score'], 1.0)


if __name__ == '__main__':
    unittest.main()
