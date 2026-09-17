import unittest

import numpy as np

from basic_robotics.workspace.alpha_shape import AlphaShape, alpha_shape_3d


def jittered_point_cloud(num_points=80, extent=2.0, seed=42):
    """A pseudo-random point cloud filling a cube, avoiding the degenerate
    (co-planar/co-spherical) geometry a perfectly regular grid produces,
    which both alpha shape backends handle poorly."""
    rng = np.random.default_rng(seed)
    return rng.random((num_points, 3)) * extent


class test_workspace_alpha_shape(unittest.TestCase):

    def setUp(self):
        self.points = jittered_point_cloud()

    def test_alpha_shape_3d_raw_function(self):
        vertices, edges, triangles = alpha_shape_3d(self.points, alpha=1.0)
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(edges), 0)
        self.assertGreater(len(triangles), 0)
        self.assertEqual(triangles.shape[1], 3)
        self.assertEqual(edges.shape[1], 2)

    def test_mode_1_alpha_shape_populates_triangles(self):
        # Regression test: the original code overwrote its own `triangles`
        # variable with an empty list before ever using it, so triangle_inds
        # and triangles were always empty for mode=1. They must be populated.
        shape = AlphaShape(self.points, alpha=1.5, mode=1)
        self.assertGreater(len(shape.triangle_inds), 0)
        self.assertGreater(len(shape.triangles), 0)
        self.assertEqual(len(shape.triangles), len(shape.triangle_inds))
        # Each triangle should be a 3x3 array of the three vertex coordinates.
        self.assertEqual(shape.triangles[0].shape, (3, 3))

    def test_mode_0_alpha_shape_uses_alphashape_library(self):
        shape = AlphaShape(self.points, alpha=1.5, mode=0)
        self.assertGreater(len(shape.verts), 0)
        self.assertGreater(len(shape.triangle_inds), 0)
        self.assertEqual(len(shape.triangles), len(shape.triangle_inds))

    def test_calculate_bounds_is_finite_and_within_point_cloud(self):
        # Regression test: the original code used the numpy 1.x alias
        # `np.Inf`, which was removed in numpy 2.0 and would crash outright.
        shape = AlphaShape(self.points, alpha=1.5, mode=1)
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        for axis, (lo, hi) in enumerate(shape.bounds):
            self.assertTrue(np.isfinite(lo))
            self.assertTrue(np.isfinite(hi))
            self.assertLessEqual(lo, hi)
            # The hull is a subset of the input cloud, so its bounds can't
            # extend beyond the cloud's own bounding box.
            self.assertGreaterEqual(lo, mins[axis] - 1e-9)
            self.assertLessEqual(hi, maxs[axis] + 1e-9)

    def test_draw_renders_without_error(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        shape = AlphaShape(self.points, alpha=1.5, mode=1)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        shape.draw(ax)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
