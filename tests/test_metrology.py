import unittest

import numpy as np

from basic_robotics.general import tm
from basic_robotics.metrology.virtual_vision import Scene, Observed, SceneObj, Camera


def make_camera(camT=None, sigma=0.001, id=0):
    if camT is None:
        camT = tm()
    return Camera(aptx=500, apty=500, pixX=512, pixY=512,
            maxX=1024, maxY=1024, sigma=sigma, camT=camT, id=id)


class test_metrology(unittest.TestCase):

    # -- Camera --

    def test_metrology_Camera_construction(self):
        cam = make_camera(sigma=2.0)
        # sigma is squared internally.
        self.assertAlmostEqual(cam.sigma, 4.0)
        self.assertEqual(cam.focx, 500)
        self.assertEqual(cam.pixX, 512)

    def test_metrology_Camera_getFrameSize(self):
        cam = make_camera()
        fs = cam.getFrameSize()
        self.assertAlmostEqual(fs[0], cam.maxX / cam.focx / 2)
        self.assertAlmostEqual(fs[1], cam.maxY / cam.focy / 2)

    def test_metrology_Camera_getLocalPos_accepts_flat_and_column(self):
        cam = make_camera()
        flat = cam.getLocalPos([512, 512])
        column = cam.getLocalPos(np.array([[512.0], [512.0]]))
        self.assertAlmostEqual(flat[0], column[0])
        self.assertAlmostEqual(flat[1], column[1])

    def test_metrology_Camera_getPhoto_in_and_out_of_view(self):
        cam = make_camera()
        in_view_point = tm([0, 0, 5, 0, 0, 0])
        img, Q, success = cam.getPhoto(in_view_point)
        self.assertTrue(success)
        self.assertAlmostEqual(float(np.asarray(img).flatten()[0]), cam.pixX, places=3)
        self.assertAlmostEqual(float(np.asarray(img).flatten()[1]), cam.pixY, places=3)

        # Far off to the side, well outside the sensor bounds.
        out_of_view_point = tm([500, 500, 5, 0, 0, 0])
        img2, Q2, success2 = cam.getPhoto(out_of_view_point)
        self.assertFalse(success2)
        # Out-of-view observations get an inflated (less trustworthy) covariance.
        self.assertGreater(Q2[0, 0], Q[0, 0])

    def test_metrology_Camera_updateFocal_and_updateResolution(self):
        cam = make_camera()
        cam.updateFocal(600, 700)
        self.assertEqual(cam.focx, 600)
        self.assertEqual(cam.focy, 700)

        cam.updateResolution(800, 900)
        self.assertEqual(cam.pixX, 800)
        self.assertEqual(cam.pixY, 900)

    def test_metrology_Camera_moveCamera(self):
        cam = make_camera()
        new_pose = tm([1, 2, 3, 0, 0, 0])
        cam.moveCamera(new_pose)
        np.testing.assert_allclose(cam.CamT.gTAA().flatten(), new_pose.gTAA().flatten())

    def test_metrology_Camera_equality(self):
        cam_a = make_camera(id=1)
        cam_b = make_camera(id=1)
        cam_c = make_camera(id=2)
        self.assertTrue(cam_a == cam_b)
        # __eq__ has no explicit False branch, so mismatched ids fall through
        # to an implicit None (falsy, but not literally False).
        self.assertFalse(cam_a == cam_c)

    def test_metrology_Camera_getProbability_zero_outside_sensor(self):
        cam = make_camera()
        out_of_view_point = tm([500, 500, 5, 0, 0, 0])
        prob = cam.getProbability(np.array([0, 0]), out_of_view_point)
        self.assertEqual(prob, 0)

    def test_metrology_Camera_getProbability_nonnegative_in_view(self):
        cam = make_camera()
        in_view_point = tm([0, 0, 5, 0, 0, 0])
        img, _, _ = cam.getPhoto(in_view_point)
        prob = cam.getProbability(np.asarray(img).flatten(), in_view_point)
        self.assertGreaterEqual(prob, 0)

    def test_metrology_Camera_dhdx_jacobian_shape(self):
        cam = make_camera()
        point = tm([0, 0, 5, 0, 0, 0])
        jac = cam.dhdx(point)
        # 2 pixel outputs (x, y) with respect to 3 position inputs (x, y, z).
        self.assertEqual(jac.shape, (2, 3))

    # -- SceneObj --

    def test_metrology_SceneObj_from_point_list(self):
        points = [tm([0, 0, 0, 0, 0, 0]), tm([1, 0, 0, 0, 0, 0]), tm([0, 1, 0, 0, 0, 0])]
        obj = SceneObj(points, tol=0.1, name='triangle')
        self.assertEqual(obj.sz, 3)
        self.assertIs(obj.lead, points[0])
        self.assertEqual(len(obj.rels), 2)
        self.assertEqual(len(obj.dists), 2)
        self.assertAlmostEqual(obj.dists[0], 1.0, places=6)

    def test_metrology_SceneObj_from_single_point_generates_extras(self):
        lead = tm([1, 1, 1, 0, 0, 0])
        obj = SceneObj(lead, tol=0.1, name='generated')
        self.assertEqual(obj.sz, 4)
        self.assertEqual(len(obj.objs), 4)
        self.assertIs(obj.lead, lead)

    def test_metrology_SceneObj_adjRot_zero_at_exact_match(self):
        points = [tm([0, 0, 0, 0, 0, 0]), tm([2, 0, 0, 0, 0, 0])]
        obj = SceneObj(points, tol=0.1, name='pair')
        # Scoring the lead pose's own (unrotated) orientation against itself
        # should find (near) zero error, since the points already match.
        score = obj.adjRot([0, 0, 0], tm([0, 0, 0, 0, 0, 0]))
        self.assertAlmostEqual(score, 0.0, places=3)

    # -- Scene / Observed (multi-camera triangulation pipeline) --

    def test_metrology_Scene_newSceneObj_and_addCam(self):
        scene = Scene()
        cam = make_camera()
        scene.addCam(cam)
        scene.newSceneObj(tm([0, 0, 5, 0, 0, 0]), tol=0.5, name='pt')
        self.assertEqual(len(scene.camList), 1)
        self.assertEqual(len(scene.objList), 1)
        self.assertEqual(scene.objList[0].name, 'pt')

    def test_metrology_Scene_addSceneObj(self):
        scene = Scene()
        obj = SceneObj(tm([0, 0, 5, 0, 0, 0]))
        scene.addSceneObj(obj)
        self.assertIs(scene.objList[0], obj)

    def test_metrology_Scene_triangulates_point_from_two_cameras(self):
        # Two cameras on either side of the origin, both looking at a point
        # roughly in front of them - enough to triangulate its 3D position.
        scene = Scene()
        scene.addCam(make_camera(tm([-2, 0, 0, 0, 0, 0]), sigma=0.001, id=1))
        scene.addCam(make_camera(tm([2, 0, 0, 0, 0, 0]), sigma=0.001, id=2))
        scene.newSceneObj(tm([0, 0, 5, 0, 0, 0]), tol=0.5, name='target')

        result = scene.GetObjPositionsFromPoints()

        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        for observed_point in result:
            self.assertGreaterEqual(observed_point.inView, 2)
            self.assertIsNotNone(observed_point.cPos)

        # CalculateGrid should now produce a symmetric distance matrix sized
        # to the reconstructed points.
        grid = scene.CalculateGrid()
        self.assertEqual(grid.shape, (len(result), len(result)))
        np.testing.assert_allclose(grid, grid.T, atol=1e-9)

    def test_metrology_Scene_no_cameras_returns_none(self):
        scene = Scene()
        scene.newSceneObj(tm([0, 0, 5, 0, 0, 0]))
        result = scene.GetObjPositionsFromPoints()
        self.assertIsNone(result)

    # -- Observed (unit-level, with a hand-built camera/pixel observation) --

    def test_metrology_Observed_collateVector_is_unit_length(self):
        cam = make_camera(tm())
        point = tm([0, 0, 5, 0, 0, 0])
        img, Q, success = cam.getPhoto(point)
        self.assertTrue(success)

        observed = Observed(img, Q, cam)

        gl = observed.gl.gTAA().flatten()[0:3]
        self.assertAlmostEqual(float(np.linalg.norm(gl)), 1.0, places=6)
        self.assertEqual(observed.inView, 1)
        self.assertEqual(observed.camerasViewing, [cam])

    def test_metrology_Observed_eq_and_sync_same_point(self):
        cam1 = make_camera(tm([-2, 0, 0, 0, 0, 0]), id=1)
        cam2 = make_camera(tm([2, 0, 0, 0, 0, 0]), id=2)
        point = tm([0, 0, 5, 0, 0, 0])

        img1, Q1, _ = cam1.getPhoto(point)
        img2, Q2, _ = cam2.getPhoto(point)
        obs1 = Observed(img1, Q1, cam1, tol=0.01)
        obs2 = Observed(img2, Q2, cam2, tol=0.01)

        self.assertTrue(obs1.eq(obs2))
        self.assertGreater(len(obs1.draftPoses), 0)

        obs1.sync(obs2)
        self.assertEqual(obs1.inView, 2)
        self.assertIn(cam2, obs1.camerasViewing)

        avg = obs1.CalcAvgGuess()
        self.assertIsInstance(avg, tm)
        np.testing.assert_allclose(
                avg.gTAA().flatten()[0:3], point.gTAA().flatten()[0:3], atol=0.5)

    def test_metrology_Observed_eq_false_for_same_camera(self):
        cam = make_camera()
        point = tm([0, 0, 5, 0, 0, 0])
        img, Q, _ = cam.getPhoto(point)
        obs1 = Observed(img, Q, cam)
        obs2 = Observed(img, Q, cam)
        # Two observations from the *same* camera can't be triangulated
        # against each other.
        self.assertFalse(obs1.eq(obs2))


if __name__ == '__main__':
    unittest.main()
