import unittest
import numpy as np
from basic_robotics.general import tm, fsr, fmr

class test_general_fsr(unittest.TestCase):
    def test_TAAtoTM(self):
        test_tm = tm([1, 2, 3, 0.1, 0.2, 0.3])
        trnsfm = test_tm.gTM()
        ntm = fsr.TAAtoTM(test_tm.gTAA())
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(trnsfm[i,j], ntm[i,j])

    def test_general_fsr_TMtoTAA(self):
        test_tm = tm([21, 19, 1, -0.1, -.05, -.03])
        rnsfm = test_tm.gTM()
        ntaa = test_tm.gTAA().flatten()
        taa = fsr.TMtoTAA(rnsfm).flatten()
        for i in range(6):
            self.assertAlmostEqual(taa[i], ntaa[i])

    def test_general_fsr_localToGlobal(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        tmb = tm([3, 2, 1, 0, 0, 0])
        tmc = fsr.localToGlobal(tma, tmb)
        self.assertEqual(tmc[0], 4)
        self.assertEqual(tmc[1], 4)
        self.assertEqual(tmc[2], 4)

    def test_general_fsr_globalToLocal(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        tmb = tm([3, 2, 1, 0, 0, 0])
        tmc = fsr.globalToLocal(tma, tmb)
        self.assertEqual(tmc[0], 2)
        self.assertEqual(tmc[1], 0)
        self.assertEqual(tmc[2], -2)

    def test_general_fsr_planeFromThreePoints(self):
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

    def test_general_fsr_planePointsFromTransform(self):
        tmb = tm()
        tm1, tm2, tm3 = fsr.planePointsFromTransform(tmb)
        self.assertEqual(tm2[0], 1)
        self.assertEqual(tm3[1], 1)
        tm1, tm2, tm3 = fsr.planePointsFromTransform(tm([0, 0, 0, np.pi/2, 0, 0]))
        self.assertEqual(tm2[0], 1)
        self.assertEqual(tm3[2], 1)

    def test_general_fsr_mirror(self):
        mirror_plane = tm()
        mirror_point = tm([1, 2, 3, 0, 0, 0])
        mirrored_point = fsr.mirror(mirror_plane, mirror_point)
        #print(mirrored_point)
        self.assertEqual(mirrored_point[0], 1)
        self.assertEqual(mirrored_point[1], 2)
        self.assertEqual(mirrored_point[2], -3)

    def test_general_fsr_adjustRotationToMidpoint(self):
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

    def test_general_fsr_tmAvgMidpoint(self):
        tma = tm([2, 4, 6, 0, np.pi, 0])
        tmb = tm()
        tmc = fsr.tmAvgMidpoint(tma, tmb)
        self.assertEqual(tmc[0], 1)
        self.assertEqual(tmc[1], 2)
        self.assertEqual(tmc[2], 3)
        self.assertAlmostEqual(tmc[4], np.pi/2, 3)

    def test_general_fsr_tmInterpMidpoint(self):
        tma = tm([2, 4, 6, 0, np.pi/2, 0])
        tmb = tm()
        tmc = fsr.tmInterpMidpoint(tma, tmb)
        self.assertEqual(tmc[0], 1)
        self.assertEqual(tmc[1], 2)
        self.assertEqual(tmc[2], 3)
        #TODO

    def test_general_fsr_getSurfaceNormal(self):
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

    def test_general_fsr_rotationFromVector(self):
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        ref1p = fsr.rotationFromVector(ref1, ref2)
        self.assertAlmostEqual(fsr.distance(ref1p @ tm([0, 0, fsr.distance(ref1, ref2), 0, 0, 0]), ref2), 0, 4)
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 5, 7, 0, 0, 0])
        ref1p = fsr.rotationFromVector(ref1, ref2)
        self.assertAlmostEqual(fsr.distance(ref1p @ tm([0, 0, fsr.distance(ref1, ref2), 0, 0, 0]), ref2), 0, 3)

    def test_general_fsr_lookAt(self):
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        ref1p = fsr.lookAt(ref1, ref2)
        self.assertAlmostEqual(fsr.distance(ref1p @ tm([0, 0, fsr.distance(ref1, ref2), 0, 0, 0]), ref2), 0, 4)
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 5, 7, 0, 0, 0])
        ref1p = fsr.lookAt(ref1, ref2)
        self.assertAlmostEqual(fsr.distance(ref1p @ tm([0, 0, fsr.distance(ref1, ref2), 0, 0, 0]), ref2), 0, 3)

        # Test Case Where The object is directly above
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([-1, 0, 7, 0, 0, 0])
        ref1p = fsr.lookAt(ref1, ref2)
        self.assertAlmostEqual(fsr.distance(ref1p @ tm([0, 0, fsr.distance(ref1, ref2), 0, 0, 0]), ref2), 0, 3)

    def test_general_fsr_poseError(self):
        tm1 = tm([1, 2, 3, 4, 5, 6])
        tm2 = tm([-1, 2, 0, 4, 5, 0])

        tm3 = fsr.poseError(tm1, tm2)
        self.assertEqual(tm3[0], 2)
        self.assertEqual(tm3[1], 0)
        self.assertEqual(tm3[2], 3)
        self.assertEqual(tm3[5], 6)

    def test_general_fsr_geometricError(self):
        tm1 = tm([1, 2, 3, 0, 0, 0])
        tm2 = tm([-1, 2, 0, 4, 5, 0])

        tm3 = fsr.geometricError(tm1, tm2)
        self.assertEqual(tm3[0], 2)
        self.assertEqual(tm3[1], 0)
        self.assertEqual(tm3[2], 3)

    def test_general_fsr_distance(self):
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        dist = fsr.distance(ref1, ref2)
        self.assertAlmostEqual(dist, 2, 2)

        a = np.array([0, 1])
        b = np.array([0, 4])

        dist = fsr.distance(a, b)

        self.assertAlmostEqual(dist, 3, 2)

    def test_general_fsr_arcDistance(self):
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        geo_err = fsr.arcDistance(ref1, ref2)
        self.assertAlmostEqual(geo_err, 2, 2)
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, np.pi/6, 0])
        geo_err = fsr.arcDistance(ref1, ref2)
        self.assertAlmostEqual(geo_err, np.sqrt(4 + (np.pi/6)**2), 2)

    def test_general_fsr_closeLinearGap(self):
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        ntm = fsr.closeLinearGap(ref1, ref2, 0.1)
        self.assertAlmostEqual(ntm[0], -0.9, 3)

        ref1 = tm([0, 0, 0, np.pi/4, 0, 0])
        ref2 = tm([0, 0, 0, np.pi/2, 0, 0])
        ntm = fsr.closeLinearGap(ref1, ref2, np.pi/8)
        self.assertAlmostEqual(ntm[3], np.pi/4 + np.pi/8, 3)

        ref1 = tm([0, 0, 0, np.pi/4, 0, 0])
        ref2 = tm([0, 0, 0, np.pi/4, 0, 0])
        ntm = fsr.closeLinearGap(ref1, ref2, np.pi/8)
        self.assertAlmostEqual(ntm[3], np.pi/4, 3)

    def test_general_fsr_closeArcGap(self):
        ref1 = tm([-1, 0, 0, 0, 0, 0])
        ref2 = tm([1, 0, 0, 0, 0, 0])
        ntm = fsr.closeArcGap(ref1, ref2, 0.1)
        self.assertAlmostEqual(ntm[0], -0.9, 3)

        ref1 = tm([0, 0, 0, np.pi/4, 0, 0])
        ref2 = tm([0, 0, 0, np.pi/2, 0, 0])
        ntm = fsr.closeArcGap(ref1, ref2, np.pi/8)
        self.assertAlmostEqual(ntm[3], np.pi/4 + np.pi/8, 3)

        ref1 = tm([1, 2, 3, np.pi/4, 0, np.pi/8])
        ref2 = tm([-.1, 0, 0, np.pi/2, 0, 0])
        ref3 = tm([ 0.980213, 2.010569, 2.905226, 0.805147, 0.007977, 0.383584 ])
        ntm = fsr.closeArcGap(ref1, ref2, 0.1)
        for i in range(6):
            self.assertAlmostEqual(ref3[i], ntm[i], 2)

        ntm = fsr.closeArcGap(ref1, ref1, 0.1)
        for i in range(6):
            self.assertAlmostEqual(ref1[i], ntm[i], 2)

    def test_general_fsr_IKPath(self):
        ref1 = tm([0, 0, 0, 0, 0, 0])
        ref2 = tm([1, 2, 3, 0, np.pi/4, 0])

        path = fsr.IKPath(ref1, ref2, 100)

        for i in range(6):
            self.assertAlmostEqual(ref1[i], path[0][i], 2)
        for i in range(6):
            self.assertAlmostEqual(ref2[i], path[-1][i], 2)
        self.assertEqual(len(path), 100)
        for i in range(98):
            self.assertAlmostEqual(
                fsr.distance(path[i], path[i+1]),
                fsr.distance(path[i+1],path[i+2]), 4)

    def test_general_fsr_deg2Rad(self):
        d1 = 90
        d2 = 180

        self.assertEqual(fsr.deg2Rad(d1), np.pi/2)
        self.assertEqual(fsr.deg2Rad(d2), np.pi)

    def test_general_fsr_rad2Deg(self):
        d1 = np.pi/2
        d2 = np.pi

        self.assertEqual(fsr.rad2Deg(d1), 90)
        self.assertEqual(fsr.rad2Deg(d2), 180)

    def test_general_fsr_angleMod(self):
        sin_non_mod = np.pi/2
        sin_mod = 5 * np.pi/2

        self.assertAlmostEqual(fsr.angleMod(sin_mod), sin_non_mod, 2)

        arr_non_mod = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
        arr_mod = np.array([5 * np.pi/2, 5 * np.pi/2, 5 * np.pi/2, 5 * np.pi/2])

        self.assertAlmostEqual(fsr.angleMod(arr_mod)[0], arr_non_mod[0], 2)

        arr_non_mod = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0, 0])
        arr_mod = np.array([5 * np.pi/2, 5 * np.pi/2, 5 * np.pi/2, 5 * np.pi/2, 0, 0])

        self.assertAlmostEqual(fsr.angleMod(arr_mod)[3], arr_non_mod[3], 2)
        self.assertNotEqual(fsr.angleMod(arr_mod)[0], arr_non_mod[0], 2)

        tmtest = tm(arr_mod)
        tmtest2 = fsr.angleMod(tmtest)
        self.assertAlmostEqual(tmtest2[3], arr_non_mod[3], 2)


    def test_general_fsr_angleBetween(self):
        p1 = tm()
        p2 = tm([1, 2, 3, 0, 0, 0])
        p3 = tm([4, 1, 5, 0, 0, 0])

        ang = fsr.angleBetween(p2, p1, p3)
        self.assertAlmostEqual(ang, 30*np.pi/180, 4)

        p1 = tm()
        p2 = tm([6, 22, 3, 0, 0, 0])
        p3 = tm([6, 7, 6, 0, 0, 0])

        ang = fsr.angleBetween(p2, p1, p3)
        self.assertAlmostEqual(ang, 34.7*np.pi/180, 4)

    def test_general_fsr_makeWrench(self):
        tmp = tm([1, 2, 1, 0, 0, 0])
        v = [0, 0, -9.81]
        f = 20
        wrench = fsr.makeWrench(tmp, f, v)

        v = np.array(v)
        wrench2 = fsr.makeWrench(tmp, f, v)

        for i in range(6):
            self.assertEqual(wrench[i][0], wrench2[i][0])

        self.assertAlmostEqual(wrench[0,0], -392.4)
        self.assertAlmostEqual(wrench[5,0], -196.2)

    def test_general_fsr_transformWrenchFrame(self):
        frame_a = tm()
        frame_b = tm([1, 2, 3, np.pi/6, 0, np.pi/9])
        frame_c = tm([2, 2, 0, 0, 0, 0])

        v = [0, 0, -9.81]
        f = 20 # N

        wrench_c_a = fsr.makeWrench(frame_c, f, v)
        wrench_c_c = fsr.makeWrench(tm(), f, v)
        wrench_c_a_p = fsr.transformWrenchFrame(wrench_c_c, frame_c, frame_a)

        for i in range(6):
            self.assertEqual(wrench_c_a[i][0], wrench_c_a_p[i][0])
            #self.assertNotEqual(wrench_c_a[i][0], wrench_c_c[i][0])

        wrench_b_a = fsr.makeWrench(frame_b, f, v)
        frame_b_c = fsr.globalToLocal(frame_c, frame_b)
        wrench_b_c = fsr.makeWrench(frame_b_c, f, v)
        wrench_b_a_p = fsr.transformWrenchFrame(wrench_b_c, frame_c, frame_a)
        for i in range(6):
            self.assertEqual(wrench_b_a[i][0], wrench_b_a_p[i][0])

    def test_general_fsr_twistToScrew(self):
        #Test derived from L7.m 2018 Robotics Control Class
        input_1 = np.array([0, 0, 1, 1, -1, 1]).reshape((6,1))
        w, th, q, h = fsr.twistToScrew(input_1)
        w_exp = np.array([0.0, 0.0, 1.0]).reshape((3,1))
        th_exp = 1.0
        q_exp = np.array([1.0, 1.0, 0.0]).reshape((3,1))
        h_exp = 1.0
        for i in range(3):
            self.assertAlmostEqual(w[i,0], w_exp[i,0])
            self.assertAlmostEqual(q[i,0], q_exp[i,0])
        self.assertAlmostEqual(th, th_exp)
        self.assertAlmostEqual(h, h_exp)

        #Round 2
        input_1 = np.array([4, 3, 1, 1, 1, -4]).reshape((6,1))
        w, th, q, h = fsr.twistToScrew(input_1)
        w_exp = np.array([0.7845, 0.5883, 0.1961]).reshape((3,1))
        th_exp = 5.0990
        q_exp = np.array([-.5000, .6538, 0.0385]).reshape((3,1))
        h_exp = 0.1154
        for i in range(3):
            self.assertAlmostEqual(w[i,0], w_exp[i,0], 3)
            self.assertAlmostEqual(q[i,0], q_exp[i,0], 3)
        self.assertAlmostEqual(th, th_exp, 3)
        self.assertAlmostEqual(h, h_exp, 3)

        #Round 3
        input_1 = np.array([0, 0, 0, -1, 2, 1]).reshape((6,1))
        w, th, q, h = fsr.twistToScrew(input_1)
        w_exp = np.array([-.4082, .8165, .4082]).reshape((3,1))
        th_exp = 2.4495
        q_exp = np.array([0.0, 0.0, 0.0]).reshape((3,1))
        h_exp = np.inf
        for i in range(3):
            self.assertAlmostEqual(w[i,0], w_exp[i,0], 3)
            self.assertAlmostEqual(q[i,0], q_exp[i,0], 3)
        self.assertAlmostEqual(th, th_exp, 3)
        self.assertAlmostEqual(h, h_exp, 3)

    def test_general_fsr_normalizeTwist(self):
        input_twist = np.array([[1],
            [2],
            [3],
            [25],
            [11],
            [-19]])
        output_twist = np.array([[0.26726],
            [0.53452],
            [0.80178],
            [6.6815],
            [2.9399],
            [-5.078]])
        result = fsr.normalizeTwist(input_twist)
        for i in range(6):
            self.assertAlmostEqual(result[i,0],output_twist[i,0], 3)

        input_twist = np.array([0, 0, 0, 3, 4, 5])
        output_twist = np.array([0, 0, 0, .4243, .5657, .7071])
        result = fsr.normalizeTwist(input_twist)
        for i in range(6):
            self.assertAlmostEqual(result[0],output_twist[0], 3)

    def test_general_fsr_twistFromTransform(self):
        #Test derived from L6.m 2018 Robotics Control Class
        def test_combo(input_tm, output):
            test_out = fsr.twistFromTransform(tm(input_tm)).flatten()
            for i in range(4):
                self.assertAlmostEqual(test_out[i], output[i], 3)

        expected = np.array([0, 0, 0.2094, 0, 0.0524, 0.30])
        input_tm = np.array([
                [0.9781,   -0.2079,  0, -0.0055],
                [0.2079,    0.9781, 0,    0.0520],
                [0, 0,    1.0000,    0.3000],
                [0,    0,   0,   1.0000]
            ])
        test_combo(input_tm, expected)

        expected = np.array([0, 0, .0698, 0, 0.0175, 0.1])
        input_tm = np.array([[0.99756,-0.069756,0,-0.00060899],
            [0.069756,0.99756,0,0.017439],
            [0,0,1,0.1],
            [0,0,0,1]])
        test_combo(input_tm, expected)

    def test_general_fsr_transformFromTwist(self):
        #Test derived from L6.m 2018 Robotics Control Class
        def test_combo(input_twist, output):
            test_out = fsr.transformFromTwist(input_twist).gTM()
            for i in range(4):
                for j in range(4):
                    self.assertAlmostEqual(test_out[i,j], output[i,j], 3)

        input_twist = np.array([0, 0, 0.2094, 0, 0.0524, 0.30])
        expected = np.array([
                [0.9781,   -0.2079,  0, -0.0055],
                [0.2079,    0.9781, 0,    0.0520],
                [0, 0,    1.0000,    0.3000],
                [0,    0,   0,   1.0000]
            ])
        test_combo(input_twist, expected)

        input_twist = np.array([0, 0, .0698, 0, 0.0175, 0.1])
        expected = np.array([[0.99756,-0.069756,0,-0.00060899],
            [0.069756,0.99756,0,0.017439],
            [0,0,1,0.1],
            [0,0,0,1]])
        test_combo(input_twist, expected)

    def test_general_fsr_transformByVector(self):
        A = tm([1, 2, 3, 4, 5, 6])
        B = np.array([0, 2, 1])

        res_exp = np.array([2.0103, 1.9980, 4.9948])

        C = fsr.transformByVector(A, B)

        self.assertAlmostEqual(res_exp[0], C[0], 2)
        self.assertAlmostEqual(res_exp[1], C[1], 2)
        self.assertAlmostEqual(res_exp[2], C[2], 2)

    def test_general_fsr_fiboSphere(self):
        fsphere = fsr.fiboSphere(100)

        self.assertEqual(len(fsphere), 100)
        for i in range(99):
            self.assertAlmostEqual(
                    fsr.distance(fsphere[i,:], np.array([0, 0, 0])),
                    1, 1)

    def test_general_fsr_unitSphere(self):
        usphere = fsr.unitSphere(100)

        self.assertEqual(len(usphere), 121) # Not exactly 100
        for i in range(99):
            self.assertAlmostEqual(
                    fsr.distance(usphere[i,:], np.array([0, 0, 0])),
                    1, 1)

        usphere, azel = fsr.unitSphere(100, True)

        self.assertEqual(len(azel), 121)

    def test_general_fsr_getUnitVec(self):
        test_tm = tm([1, 2, 3, 0, 0, 0])
        origin_tm = tm()
        unit = fsr.getUnitVec(origin_tm, test_tm)
        res_expected =[0.26726124, 0.53452248, 0.80178373]
        for i in range(3):
            self.assertAlmostEqual(res_expected[i], unit[i])
        res_expected_2 = np.array([0.26726124, 0.53452248, 0.80178373])*3
        unit = fsr.getUnitVec(origin_tm, test_tm, 3)
        for i in range(3):
            self.assertAlmostEqual(res_expected_2[i], unit[i])

        unit, dist = fsr.getUnitVec(origin_tm, test_tm, 3, True)
        for i in range(3):
            self.assertAlmostEqual(res_expected_2[i], unit[i])
        self.assertAlmostEqual(dist, 3.741657, 3)


    def test_general_fsr_chainJacobian(self):
        exscrew = np.array([[0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, -2, -2, 0, -2, 0],
            [0, 0, 0, 2, 0, 2],
            [0, 0, 1.5, 0, 3.1, 0]])
        ref = np.array([[0, 0, 0, -0.98999, 0.019915, 0.54137],
            [0, 1, 1, 0, -0.98999, -0.1068],
            [1, 0, 0, -0.14112, -0.13971, 0.83397],
            [0, -2, -0.73779, 0, 0.50688, 0.054682],
            [0, 0, 0, -0.61604, -0.097872, 0.92229],
            [0, 0, 0.81045, 0, 0.76579, 0.082613]])

        ans = fsr.chainJacobian(exscrew, np.array([0, 1, 2, 3, 4, 5]))

        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(ref[i,j], ans[i,j], 2)

    def test_general_fsr_numericalJacobian(self):
        # Test derived from
        # https://www.mathworks.com/matlabcentral/answers/28066-numerical-jacobian-in-matlab
        fx = lambda x : np.array([x[0]**2 + x[1]**2, x[0]**3 * x[1]**3])

        pa = np.array([1.0, 1.0])
        res = fsr.numericalJacobian(fx, pa, 0.00001)

        ans = np.array([[2, 2], [3, 3]])

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(res[i,j], ans[i,j], 2)

    def test_general_fsr_boxSpatialInertia(self):
        #Test derived from L10.m 2018 Robotics Control Class
        l_in = 1
        w_in = 2
        h_in = 3
        m_in = 1
        expected = np.array([
            [1.0833, 0, 0, 0, 0, 0],
            [0, 0.83333, 0, 0, 0, 0],
            [0, 0, 0.41667, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]]
        )
        result = fsr.boxSpatialInertia(m_in, l_in, w_in, h_in)
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(result[i,j], expected[i,j], 3)

        l_in = 5.0
        w_in = 2.0
        h_in = 7.0
        m_in = 13.0
        expected = np.array([[57.4167, 0, 0, 0, 0, 0],
            [0, 80.1667, 0, 0, 0, 0],
            [0, 0, 31.4167, 0, 0, 0],
            [0, 0, 0, 13, 0, 0],
            [0, 0, 0, 0, 13, 0],
            [0, 0, 0, 0, 0, 13]])
        result = fsr.boxSpatialInertia(m_in, l_in, w_in, h_in)
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(result[i,j], expected[i,j], 3)

    def test_general_fsr_setElements(self):
        test_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        test_inds = np.array([1, 2, 5])
        test_new = np.array([9, 10, 11])

        r_arr = fsr.setElements(test_arr, test_inds, test_new)

        self.assertEqual(r_arr[1], test_new[0])
        self.assertEqual(r_arr[2], test_new[1])
        self.assertEqual(r_arr[5], test_new[2])

if __name__ == '__main__':
    unittest.main()
