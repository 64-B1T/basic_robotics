import itertools
import os
import pickle
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from basic_robotics.general import tm
from basic_robotics.workspace.viewer import WorkspaceViewer


def make_grid_workspace():
    """A small, evenly-spaced point cloud + score pairs, in the
    ``[[point, score], ...]`` shape WorkspaceViewer.process_slices expects."""
    coords = [0.0, 0.5, 1.0]
    work_space = []
    for x, y, z in itertools.product(coords, coords, coords):
        score = 1.0 if (x, y, z) == (0.5, 0.5, 0.5) else 0.2
        work_space.append([tm([x, y, z, 0, 0, 0]), score])
    return work_space


class test_workspace_viewer(unittest.TestCase):

    def setUp(self):
        self.work_space = make_grid_workspace()
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f:
            pickle.dump(self.work_space, f)
            self.fname = f.name

    def tearDown(self):
        os.remove(self.fname)
        plt.close('all')

    def _make_axes(self):
        fig = plt.figure()
        axis_1 = fig.add_subplot(1, 3, 1)
        axis_2 = fig.add_subplot(1, 3, 2)
        axis_3 = fig.add_subplot(1, 3, 3)
        return fig, np.array([[axis_2, axis_3], [axis_1, 0]])

    def test_loads_and_bins_a_saved_workspace(self):
        _, axes = self._make_axes()
        viewer = WorkspaceViewer(self.fname, axes, plot_3d_slice=False)
        self.assertEqual(len(viewer.cloud_data.shape), 3)
        for dim in viewer.cloud_data.shape:
            self.assertGreaterEqual(dim, 1)
        # The center point was scored highest; it must show up somewhere
        # in the binned volume.
        self.assertAlmostEqual(viewer.cloud_data.max(), 1.0, places=3)

    def test_update_and_scroll_do_not_error(self):
        _, axes = self._make_axes()
        viewer = WorkspaceViewer(self.fname, axes, plot_3d_slice=False)
        viewer.update()

        class FakeScrollEvent:
            button = 'up'

        viewer.on_scroll(FakeScrollEvent())
        FakeScrollEvent.button = 'down'
        viewer.on_scroll(FakeScrollEvent())


if __name__ == '__main__':
    unittest.main()
