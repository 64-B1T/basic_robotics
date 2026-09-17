import os
import time
import unittest

from basic_robotics.plotting import vis_3js_server


class test_plotting_vis_3js_server(unittest.TestCase):

    def setUp(self):
        vis_3js_server.ALL_DATA = {}
        vis_3js_server.app.testing = True
        self.client = vis_3js_server.app.test_client()

    # -- POST /api/json --

    def test_server_post_no_json_body(self):
        resp = self.client.post('/api/json', data='null', content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()['message'], 'No JSON Provided')

    def test_server_post_no_key_specified(self):
        resp = self.client.post('/api/json', json={"Value": 1})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()['message'], 'No Key(s) Specified')

    def test_server_post_single_key_create_then_update(self):
        resp = self.client.post('/api/json', json={"Key": "foo", "Value": 1})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(vis_3js_server.ALL_DATA['foo'], [{"Key": "foo", "Value": 1}])

        resp = self.client.post('/api/json', json={"Key": "foo", "Value": 2})
        self.assertEqual(resp.status_code, 200)
        # Updates in place rather than appending a new entry.
        self.assertEqual(vis_3js_server.ALL_DATA['foo'], [{"Key": "foo", "Value": 2}])

    def test_server_post_keys_create_then_update(self):
        resp = self.client.post('/api/json', json={"Keys": {"a": {"X": 1}, "b": {"X": 2}}})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(vis_3js_server.ALL_DATA['a'], [{"X": 1}])
        self.assertEqual(vis_3js_server.ALL_DATA['b'], [{"X": 2}])

        resp = self.client.post('/api/json', json={"Keys": {"a": {"X": 99}}})
        self.assertEqual(resp.status_code, 200)
        # Existing key 'a' gets updated in place, not duplicated.
        self.assertEqual(vis_3js_server.ALL_DATA['a'], [{"X": 99}])
        self.assertEqual(vis_3js_server.ALL_DATA['b'], [{"X": 2}])

    # -- GET /api/json --

    def test_server_get_latest(self):
        vis_3js_server.ALL_DATA['foo'] = [{"Key": "foo", "Value": 1}, {"Key": "foo", "Value": 2}]
        resp = self.client.get('/api/json?Latest=1')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"foo": {"Key": "foo", "Value": 2}})

    def test_server_get_complete(self):
        vis_3js_server.ALL_DATA['foo'] = [{"Key": "foo", "Value": 1}]
        resp = self.client.get('/api/json?Complete=1')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"foo": [{"Key": "foo", "Value": 1}]})

    def test_server_get_category_filter(self):
        vis_3js_server.ALL_DATA['a'] = [{"Category": "Model"}]
        vis_3js_server.ALL_DATA['b'] = [{"Category": "Frame"}]
        vis_3js_server.ALL_DATA['c'] = [{"NoCategory": True}]
        resp = self.client.get('/api/json?Category=Model&Complete=1')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('a', data)
        self.assertIn('c', data)
        self.assertNotIn('b', data)

    def test_server_get_key_without_index(self):
        vis_3js_server.ALL_DATA['foo'] = [{"Key": "foo", "Value": 1}, {"Key": "foo", "Value": 2}]
        resp = self.client.get('/api/json?Key=foo')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"foo": [{"Key": "foo", "Value": 1}, {"Key": "foo", "Value": 2}]})

    def test_server_get_key_with_index(self):
        vis_3js_server.ALL_DATA['foo'] = [{"Key": "foo", "Value": 1}, {"Key": "foo", "Value": 2}]
        resp = self.client.get('/api/json?Key=foo&Index=0')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"foo": {"Key": "foo", "Value": 1}})

    def test_server_get_key_multiple(self):
        vis_3js_server.ALL_DATA['a'] = [{"Key": "a", "Value": 1}]
        vis_3js_server.ALL_DATA['b'] = [{"Key": "b", "Value": 2}]
        resp = self.client.get('/api/json?Key=a,b')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('a', data)
        self.assertIn('b', data)

    # -- PUT /api/json --

    def test_server_put_single_key_overrides(self):
        vis_3js_server.ALL_DATA['foo'] = [{"Key": "foo", "Value": 1}, {"Key": "foo", "Value": 2}]
        resp = self.client.put('/api/json', json={"Key": "foo", "Value": 99})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['message'], 'Data Overridden')
        # A single-key PUT replaces the whole history, not just index 0.
        self.assertEqual(vis_3js_server.ALL_DATA['foo'], [{"Key": "foo", "Value": 99}])

    def test_server_put_keys_updates(self):
        resp = self.client.put('/api/json', json={"Keys": {"a": {"X": 1}, "b": {"X": 2}}})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['message'], 'Keys Updated')
        self.assertEqual(vis_3js_server.ALL_DATA['a'], [{"X": 1}])
        self.assertEqual(vis_3js_server.ALL_DATA['b'], [{"X": 2}])

    def test_server_put_delete_all(self):
        vis_3js_server.ALL_DATA['a'] = [{"X": 1}]
        resp = self.client.put('/api/json', json={"DeleteAll": 1})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['message'], 'All Data Deleted')
        self.assertEqual(vis_3js_server.ALL_DATA, {})

    def test_server_put_delete_key(self):
        vis_3js_server.ALL_DATA['a'] = [{"X": 1}]
        vis_3js_server.ALL_DATA['b'] = [{"X": 2}]
        resp = self.client.put('/api/json', json={"DeleteKey": "a"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['message'], 'Data Deleted')
        self.assertNotIn('a', vis_3js_server.ALL_DATA)
        self.assertIn('b', vis_3js_server.ALL_DATA)

    def test_server_put_delete_keys(self):
        vis_3js_server.ALL_DATA['a'] = [{"X": 1}]
        vis_3js_server.ALL_DATA['b'] = [{"X": 2}]
        resp = self.client.put('/api/json', json={"DeleteKeys": ["a"]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['message'], 'Data Deleted')
        self.assertNotIn('a', vis_3js_server.ALL_DATA)
        self.assertIn('b', vis_3js_server.ALL_DATA)

    def test_server_put_delete_category(self):
        vis_3js_server.ALL_DATA['a'] = [{"Category": "Model"}]
        vis_3js_server.ALL_DATA['b'] = [{"Category": "Frame"}]
        resp = self.client.put('/api/json', json={"DeleteCategory": "Model"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['message'], 'Data Deleted')
        self.assertNotIn('a', vis_3js_server.ALL_DATA)
        self.assertIn('b', vis_3js_server.ALL_DATA)

    def test_server_put_no_key(self):
        resp = self.client.put('/api/json', json={"Nonsense": 1})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()['message'], 'No Key')

    # -- PruneExpired via /api/json --

    def test_server_prune_expired_removes_stale_entry(self):
        vis_3js_server.ALL_DATA['foo'] = [
            {"UnixTime": time.time() - 10, "TimeToLive": 1},
        ]
        # Any request triggers PruneExpired() as a side effect.
        resp = self.client.get('/api/json?Latest=1')
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn('foo', vis_3js_server.ALL_DATA)

    def test_server_prune_expired_keeps_fresh_entry(self):
        vis_3js_server.ALL_DATA['foo'] = [
            {"UnixTime": time.time(), "TimeToLive": 1000},
        ]
        resp = self.client.get('/api/json?Latest=1')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('foo', vis_3js_server.ALL_DATA)

    def test_server_prune_expired_skips_empty_history(self):
        vis_3js_server.ALL_DATA['foo'] = []
        resp = self.client.get('/api/json?Complete=1')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(vis_3js_server.ALL_DATA['foo'], [])

    # -- index / static file routes --

    def test_server_index_renders_frame(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'<title>Visualizer</title>', resp.data)

    def test_server_index_not_found(self):
        original = vis_3js_server.package_directory
        vis_3js_server.package_directory = '/no/such/directory'
        try:
            resp = self.client.get('/')
        finally:
            vis_3js_server.package_directory = original
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()['message'], 'File Not Found')

    def test_server_get_internal_file_found(self):
        resp = self.client.get('/internal/web_assets/favicon.ico')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['Content-Type'], 'image/x-icon')

    def test_server_get_internal_file_unknown_extension_falls_back(self):
        resp = self.client.get('/internal/web_assets/CtrlIntro.json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['Content-Type'], 'application/octet-stream')

    def test_server_get_internal_file_not_found(self):
        resp = self.client.get('/internal/web_assets/does_not_exist.ico')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()['message'], 'File Not Found')

    def test_server_get_external_file_found(self):
        package_directory = os.path.dirname(os.path.abspath(vis_3js_server.__file__))
        abs_path = os.path.join(package_directory, 'web_assets', 'favicon.ico')
        resp = self.client.get(abs_path)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['Content-Type'], 'image/x-icon')

    def test_server_get_external_file_unknown_extension_falls_back(self):
        package_directory = os.path.dirname(os.path.abspath(vis_3js_server.__file__))
        abs_path = os.path.join(package_directory, 'web_assets', 'CtrlIntro.json')
        resp = self.client.get(abs_path)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['Content-Type'], 'application/octet-stream')

    def test_server_get_external_file_not_found(self):
        resp = self.client.get('/definitely/not/a/real/path.ico')
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()['message'], 'File Not Found')


if __name__ == '__main__':
    unittest.main()
