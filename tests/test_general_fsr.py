import unittest
import numpy as np
from basic_robotics.general import tm, fsr, fmr

class TestFSR(unittest.TestCase):
    def test_TAAtoTM(self):
        pass #TODO

    def test_TMtoTAA(self):
        pass #TODO

    def test_localToGlobal(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        tmb = tm([3, 2, 1, 0, 0, 0])
        tmc = fsr.localToGlobal(tma, tmb)
        self.assertEqual(tmc[0], 4)
        self.assertEqual(tmc[1], 4)
        self.assertEqual(tmc[2], 4)

    def test_globalToLocal(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        tmb = tm([3, 2, 1, 0, 0, 0])
        tmc = fsr.globalToLocal(tma, tmb)
        self.assertEqual(tmc[0], 2)
        self.assertEqual(tmc[1], 0)
        self.assertEqual(tmc[2], -2)

    def test_planeFromThreePoints(self):
        tma = tm([1, -2, -0, 0, 0, 0])
        tmb = tm([3, 1, 4, 0, 0, 0])
        tmc = tm([0, -1, 2, 0, 0, 0])
        a, b, c, d = fsr.planeFromThreePoints(tma, tmb, tmc)
        self.assertEqual(a, -2)
        self.assertEqual(b, 8)
        self.assertEqual(c, -5)
        self.assertEqual(d, -18)
        tma = tm([1, 1, 1, 0, 0, 0])
        tmb = tm([1, 2, 0, 0, 0, 0])
        tmc = tm([-1, 2, 1, 0, 0, 0])
        a, b, c, d = fsr.planeFromThreePoints(tma, tmb, tmc)
        self.assertEqual(a, -1)
        self.assertEqual(b, -2)
        self.assertEqual(c, -2)
        self.assertEqual(d, -5)

    def test_planePointsFromTransform(self):
        tmb = tm()
        tm1, tm2, tm3 = fsr.planePointsFromTransform(tmb)
        self.assertEqual(tm2[0], 1)
        self.assertEqual(tm3[1], 1)
        tm1, tm2, tm3 = fsr.planePointsFromTransform(tm([0, 0, 0, np.pi/2, 0, 0]))
        self.assertEqual(tm2[0], 1)
        self.assertEqual(tm3[2], 1)

    def test_mirror(self):
        mirror_plane = tm()
        mirror_point = tm([1, 2, 3, 0, 0, 0])
        mirrored_point = fsr.mirror(mirror_plane, mirror_point)
        #print(mirrored_point)
        self.assertEqual(mirrored_point[0], 1)
        self.assertEqual(mirrored_point[1], 2)
        self.assertEqual(mirrored_point[2], -3)

    def test_adjustRotationToMidpoint(self):
        active = tm()
        ref1 = tm([0, 0, -1, 0, 0, 0])
        ref2 = tm([0, 0, 1, 0, 0, 0])
        mod0 = fsr.adjustRotationToMidpoint(active, ref1, ref2)
        mod1 = fsr.adjustRotationToMidpoint(active, ref1, ref2, 1)
        for i in range(6):
            self.assertEqual(mod0[i], 0)
            self.assertEqual(mod1[i], 0)
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        mod0 = fsr.adjustRotationToMidpoint(active, ref1, ref2)
        mod1 = fsr.adjustRotationToMidpoint(active, ref1, ref2, 1)
        for i in range(6):
            self.assertEqual(mod0[i], 0)

    def test_tmAvgMidpoint(self):
        tma = tm([2, 4, 6, 0, np.pi, 0])
        tmb = tm()
        tmc = fsr.tmAvgMidpoint(tma, tmb)
        self.assertEqual(tmc[0], 1)
        self.assertEqual(tmc[1], 2)
        self.assertEqual(tmc[2], 3)
        self.assertAlmostEqual(tmc[4], np.pi/2, 3)

    def test_tmInterpMidpoint(self):
        tma = tm([2, 4, 6, 0, np.pi/2, 0])
        tmb = tm()
        tmc = fsr.tmInterpMidpoint(tma, tmb)
        self.assertEqual(tmc[0], 1)
        self.assertEqual(tmc[1], 2)
        self.assertEqual(tmc[2], 3)
        #TODO

    def test_getSurfaceNormal(self):
        tma = tm()
        tmb = tm([0, 1, 0, 0, np.pi/2, 0])
        tmc = tm([1, 0, 0, 0, 0, 0])
        tmd, unit = fsr.getSurfaceNormal([tmc, tmb, tma])
        self.assertAlmostEqual(tmd[0], .33333, 2)
        self.assertAlmostEqual(tmd[1], .33333, 2)
        self.assertAlmostEqual(tmd[2], 0, 2)
        self.assertAlmostEqual(unit[2], 1, 5)
        object_center = tm([0, 0, 1, 0, 0, 0])
        tmd, unit = fsr.getSurfaceNormal([tmc, tmb, tma], object_center)
        self.assertAlmostEqual(tmd[0], .33333, 2)
        self.assertAlmostEqual(tmd[1], .33333, 2)
        self.assertAlmostEqual(tmd[2], 0, 2)
        self.assertAlmostEqual(unit[2], -1, 5)

    def test_rotationFromVector(self):
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        ref1p = fsr.rotationFromVector(ref1, ref2)
        self.assertAlmostEqual(fsr.distance(ref1p @ tm([0, 0, fsr.distance(ref1, ref2), 0, 0, 0]), ref2), 0, 4)
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 5, 7, 0, 0, 0])
        ref1p = fsr.rotationFromVector(ref1, ref2)
        self.assertAlmostEqual(fsr.distance(ref1p @ tm([0, 0, fsr.distance(ref1, ref2), 0, 0, 0]), ref2), 0, 3)
        #pass #TODO

    def test_lookAt(self):
        pass #TODO

    def test_poseError(self):
        pass #TODO

    def test_geometricError(self):
        pass #TODO

    def test_distance(self):
        pass #TODO

    def test_arcDistance(self):
        pass #TODO

    def test_closeLinearGap(self):
        pass #TODO

    def test_closeArcGap(self):
        pass #TODO

    def test_IKPath(self):
        pass #TODO

    def test_rad2Deg(self):
        pass #TODO

    def test_angleMod(self):
        pass #TODO

    def test_angleBetween(self):
        pass #TODO

    def test_makeWrench(self):
        pass #TODO

    def test_transformWrenchFrame(self):
        pass #TODO

    def test_twistToScrew(self):
        pass #TODO

    def test_normalizeTwist(self):
        pass #TODO

    def test_twistFromTransform(self):
        pass #TODO

    def test_transformFromTwist(self):
        pass #TODO

    def test_transformByVector(self):
        pass #TODO

    def test_fiboSphere(self):
        pass #TODO

    def test_unitSphere(self):
        pass #TODO

    def test_getUnitVec(self):
        pass #TODO

    def test_chainJacobian(self):
        pass #TODO

    def test_numericalJacobian(self):
        pass #TODO

    def test_boxSpatialInertia(self):
        pass #TODO

    def test_delMini(self):
        pass #TODO

    def test_setElements(self):
        pass #TODO



if __name__ == '__main__':
    unittest.main()
