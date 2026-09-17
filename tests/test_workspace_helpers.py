import json
import os
import tempfile
import unittest
from io import StringIO
from contextlib import redirect_stdout

import numpy as np

from basic_robotics.general import tm
from basic_robotics.workspace.helpers import (
    WaitText, complete_trajectory, convert_from_json, convert_to_json,
    filter_manipulability_at_threshold, load_point_cloud_from_file, post_flag,
    score_point, sort_cloud, wait_for,
)


class test_workspace_helpers(unittest.TestCase):

    def test_score_point_thresholds(self):
        self.assertEqual(score_point(0.95), 'limegreen')
        self.assertEqual(score_point(0.85), 'green')
        self.assertEqual(score_point(0.75), 'teal')
        self.assertEqual(score_point(0.65), 'dodgerblue')
        self.assertEqual(score_point(0.55), 'blue')
        self.assertEqual(score_point(0.45), 'yellow')
        self.assertEqual(score_point(0.35), 'orange')
        self.assertEqual(score_point(0.25), 'peru')
        self.assertEqual(score_point(0.15), 'red')
        self.assertEqual(score_point(0.05), 'darkred')

    def test_post_flag_returns_argument_after_flag(self):
        cmds = ['analyze', '-f', 'input.txt', '-plot']
        self.assertEqual(post_flag('-f', cmds), 'input.txt')

    def test_post_flag_raises_when_flag_missing(self):
        with self.assertRaises(ValueError):
            post_flag('-missing', ['a', 'b'])

    def test_sort_cloud_deduplicates_close_points(self):
        cloud = [
            tm([1.0, 2.0, 3.0, 0, 0, 0]),
            tm([1.04, 2.04, 3.04, 0, 0, 0]),  # rounds to the same point at 1 decimal
            tm([5.0, 5.0, 5.0, 0, 0, 0]),
        ]
        result = sort_cloud(cloud)
        self.assertEqual(result.shape, (2, 3))

    def test_filter_manipulability_at_threshold(self):
        results = [
            ['p1', 0.9, [], []],
            ['p2', 0.3, [], []],
            ['p3', 0.55, [], []],
        ]
        filtered = filter_manipulability_at_threshold(results, 0.5)
        self.assertEqual([r[0] for r in filtered], ['p1', 'p3'])

    def test_complete_trajectory_noop_when_interpolation_disabled(self):
        points = [tm([0, 0, 0, 0, 0, 0]), tm([10, 0, 0, 0, 0, 0])]
        result = complete_trajectory(points, -1, 1)
        self.assertEqual(result, points)

    def test_complete_trajectory_linear_interpolation_fills_gaps(self):
        points = [tm([0, 0, 0, 0, 0, 0]), tm([4, 0, 0, 0, 0, 0])]
        result = complete_trajectory(points, 1.0, point_interpolation_mode=1)
        # Interpolated points plus the final waypoint.
        self.assertGreater(len(result), 2)
        self.assertAlmostEqual(result[-1].TAA.flatten()[0], 4.0, places=3)

    def test_complete_trajectory_arc_interpolation_fills_gaps(self):
        points = [tm([0, 0, 0, 0, 0, 0]), tm([4, 0, 0, 0, 0, 0])]
        result = complete_trajectory(points, 1.0, point_interpolation_mode=2)
        self.assertGreater(len(result), 2)

    def test_load_point_cloud_from_file(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write('[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]\n')
            f.write('[0.0, 0.0, 0.0, np.pi/2, 0.0, 0.0]\n')
            fname = f.name
        try:
            points = load_point_cloud_from_file(fname)
            self.assertEqual(len(points), 2)
            np.testing.assert_allclose(points[0].TAA.flatten()[0:3], [1.0, 2.0, 3.0])
            self.assertAlmostEqual(points[1].TAA.flatten()[3], np.pi / 2, places=5)
        finally:
            os.remove(fname)

    def test_convert_to_json_and_back_round_trips(self):
        results = [
            [tm([1, 2, 3, 0, 0, 0]), 0.75,
             [tm([0.1, 0.2, 0.3, 0, 0, 0])], [np.array([0.1, 0.2, 0.3])]],
        ]
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            fname = f.name
        try:
            convert_to_json(results, fname)
            with open(fname) as fh:
                raw = json.load(fh)
            # Regression test: the original code never initialized the
            # 'thetas' sub-dict before writing to it, which raised a KeyError.
            key = str(results[0][0])
            self.assertIn('thetas', raw[key])
            self.assertIn('successes', raw[key])

            round_tripped = convert_from_json(fname)
            self.assertEqual(len(round_tripped), 1)
            _, score, successes, thetas = round_tripped[0]
            self.assertAlmostEqual(score, 0.75)
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(thetas), 1)
            np.testing.assert_allclose(thetas[0], [0.1, 0.2, 0.3])
        finally:
            os.remove(fname)

    def test_wait_text_prints_dots_and_done(self):
        out = StringIO()
        with redirect_stdout(out):
            waiter = WaitText('Working', iter_limit=2)
            waiter.print()
            waiter.print()
            waiter.print()  # wraps around iter_limit
            waiter.done()
        printed = out.getvalue()
        self.assertIn('Working', printed)
        self.assertIn('Done', printed)

    def test_wait_for_returns_once_result_is_ready(self):
        class ImmediateResult:
            def ready(self):
                return True

        out = StringIO()
        with redirect_stdout(out):
            wait_for(ImmediateResult(), 'Waiting')  # should not block


if __name__ == '__main__':
    unittest.main()
