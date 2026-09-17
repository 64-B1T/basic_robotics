import unittest
import numpy as np
from basic_robotics.general import tm, fsr, fmr

class test_general_tm(unittest.TestCase):
    def test_general_tm_translation_initialization(self):
        tmDOF6List = tm([1, 2, 3, 0, 0, 0], True)
        tmDOF7List = tm([1, 2, 3, 0, 0, 0, 1])
        tmDOF6Array = tm(np.array([1, 2, 3, 0, 0, 0]))
        tmDOF7Array = tm(np.array([1, 2, 3, 0, 0, 0, 1]))
        tmDOF6ListRPY = tm([1, 2, 3, 0, 0, 0], True)
        tmDOF7ListRPY = tm([1, 2, 3, 0, 0, 0, 1], True)
        tmDOF6ArrayRPY = tm(np.array([1, 2, 3, 0, 0, 0]), True)
        tmDOF7ArrayRPY = tm(np.array([1, 2, 3, 0, 0, 0, 1]), True)
        tmDOF4x4Array = tm(np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]))
        tmDOF4x4ArrayRPY = tm(np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]), True)
        tmDOFArraytm = tm(np.array([tm([1, 2, 3, 0, 0, 0])]))

        tm_list = [tmDOF6List, tmDOF7List,
            tmDOF6Array, tmDOF7Array,
            tmDOF6ListRPY, tmDOF7ListRPY,
            tmDOF6ArrayRPY, tmDOF7ArrayRPY,
            tmDOF4x4Array, tmDOF4x4ArrayRPY,
            tmDOFArraytm]

        for i in range(len(tm_list) - 1):
            self.assertEqual(tm_list[i][0], tm_list[i+1][0])

    def test_general_tm_rotation_initialization(self):
        #Test cases generated with https://www.andre-gaschler.com/rotationconverter/
        #Axis Angle
        tmDOF3List = tm([1, 2, 3])
        tmDOF6List = tm([1, 2, 3, 1, 2, 3])
        tmDOF3Array = tm(np.array([1, 2, 3]))
        tmDOF6Array = tm(np.array([1, 2, 3, 1, 2, 3]))

        #Quaternion
        tmDOF7List = tm([0, 0, 0, 0.25532186,  0.51064372,  0.76596558, -0.29555113])
        tmDOF7Array = tm(np.array([1, 2, 3, 0.25532186,  0.51064372,  0.76596558, -0.29555113]))

        #Euler Angles
        tmDOF3ListRPY = tm([-1.2137551, 0.0894119, -2.3429888], True)
        tmDOF6ListRPY = tm([0, 0, 0, -1.2137551, 0.0894119, -2.3429888], True)
        tmDOF3ArrayRPY = tm(np.array([-1.2137551, 0.0894119, -2.3429888]), True)
        tmDOF6ArrayRPY = tm(np.array([0, 0, 0, -1.2137551, 0.0894119, -2.3429888]), True)

        #Rotation matrix
        tmDOF4x4Array = tm(np.array([
            [ -0.6949205,  0.7135210,  0.0892929, 0],
            [-0.1920070, -0.3037851,  0.9331924, 0],
            [0.6929781,  0.6313497,  0.3481075, 0 ],
            [0, 0, 0, 1]]))

        tmDOF4x4ArrayRPY = tm(np.array([
            [ -0.6949205,  0.7135210,  0.0892929, 0],
            [-0.1920070, -0.3037851,  0.9331924, 0],
            [0.6929781,  0.6313497,  0.3481075, 0 ],
            [0, 0, 0, 1]]), True)

        tm_list = [
            tmDOF3List, tmDOF6List,
            tmDOF3ListRPY, tmDOF6ListRPY,
            tmDOF3Array, tmDOF6Array,
            tmDOF3ArrayRPY, tmDOF6ArrayRPY,
            tmDOF7List, tmDOF7Array,
            tmDOF4x4Array, tmDOF4x4ArrayRPY
        ]

        for i in range(len(tm_list) - 1):
            for ii in range(3):
                for jj in range(3):
                    self.assertAlmostEqual(tm_list[i].TM[ii,jj], tm_list[i+1].TM[ii,jj], 3)

    def test_general_tm_copy_independence(self):
        tma = tm([1, 2, 3, 4, 5, 6])
        tmb = tma.spawnNew([2, 3, 4, 5, 6, 7])

        self.assertEqual(tma[1], tmb[0])
        tmc = tm(tmb)
        self.assertEqual(tmb[0], tmc[0])

        tmd = tma
        tmd[1] = 5
        self.assertEqual(tma[1], 5)
        tme = tma.copy()
        tme[1] = 6
        self.assertNotEqual(tma[1], tme[1])

    def test_general_tm_getSetQuat(self):
        tma = tm([1,2,3])
        quat = tma.getQuat()
        self.assertAlmostEqual(quat[0], 0.25532186)
        self.assertAlmostEqual(quat[1], 0.51064372)
        self.assertAlmostEqual(quat[2], 0.76596558)
        self.assertAlmostEqual(quat[3], -0.29555113)

        tmxx = tm()
        tmxx.setQuat([0.25532186,  0.51064372,  0.76596558, -0.29555113])
        for ii in range(3):
            for jj in range(3):
                self.assertAlmostEqual(tmxx.TM[ii,jj], tma.TM[ii,jj], 3)

    def test_general_tm_setTMIndex(self):
        tma = tm([0, 1, 2, 0, 0, 0])
        tma[3] = np.pi/2
        self.assertAlmostEqual(tma[3], np.pi/2, 5)

    def test_general_tm_sTM(self):
        tma = tm([0, 0, 1, 0, 0, 0])
        gtm = np.array([
            [ -0.6949205,  0.7135210,  0.0892929, 0],
            [-0.1920070, -0.3037851,  0.9331924, 0],
            [0.6929781,  0.6313497,  0.3481075, 0 ],
            [0, 0, 0, 1]])
        tma.sTM(gtm)
        for ii in range(4):
            for jj in range(4):
                self.assertAlmostEqual(tma.TM[ii,jj], gtm[ii,jj], 3)

    def test_general_tm_sTAA(self):
        tma = tm([0, 0, 1, 0, 0, 0])
        gtm = np.array([
            [ -0.6949205,  0.7135210,  0.0892929, 0],
            [-0.1920070, -0.3037851,  0.9331924, 0],
            [0.6929781,  0.6313497,  0.3481075, 0 ],
            [0, 0, 0, 1]])
        tma.sTAA(np.array([0, 0, 0, 1, 2, 3]))
        for ii in range(4):
            for jj in range(4):
                self.assertAlmostEqual(tma.TM[ii,jj], gtm[ii,jj], 3)

    def test_general_tm_transpose(self):
        tma = tm([1, 0, 1, 0, 1, 0])
        transpose_ref = tma.gTM().T
        transpose_tm = tma.T()
        transpose_conj_tm = tma.cT()
        for ii in range(4):
            for jj in range(4):
                self.assertAlmostEqual(transpose_ref[ii,jj], transpose_tm.TM[ii,jj], 3)
                self.assertAlmostEqual(transpose_ref[ii,jj], transpose_conj_tm.TM[ii,jj], 3)

    def test_general_tm_inverse(self):
        tma = tm([1, 2, 3, 4, 5, 6])
        tminv1 = np.linalg.inv(tma.gTM())
        tminv2 = np.linalg.pinv(tma.gTM())
        tminv3 = tma.inv()
        tminv4 = tma.pinv()
        for ii in range(4):
            for jj in range(4):
                self.assertAlmostEqual(tminv1[ii,jj], tminv2[ii,jj], 3)
                self.assertAlmostEqual(tminv1[ii,jj], tminv3.TM[ii,jj], 3)
                self.assertAlmostEqual(tminv1[ii,jj], tminv4.TM[ii,jj], 3)

    def test_general_tm_approx(self):
        tma = tm([1.1, 1.12, 1.123, 1.1234, 1.12345, 1.123456])
        tmb = tm(tma.approx(3))
        self.assertEqual(tmb[3], 1.123)
        tmc = tm(tma.approx(2))
        self.assertEqual(tmc[5], 1.12)

    def test_general_tm_set(self):
        tma = tm([1, 2, 3, 4, 5, 6])
        tma.set(2, 1)
        self.assertEqual(tma[2], 1)
        tma.set(0, 1)
        self.assertEqual(tma[0], 1)

    def test_general_tm_floor_div(self):
        tma = tm([1, 2, 3, 0, 1, 0])
        tmb = tm([-1, 0, 1, 4, -1, 0])
        tmr = tma.gTM()
        tmc = tmb // tma
        tme = tmb // tmr

        tmd = tmc @ tma
        tmf = tme @ tmr
        for ii in range(4):
            for jj in range(4):
                self.assertAlmostEqual(tmd.TM[ii,jj], tmb.TM[ii,jj], 3)
                self.assertAlmostEqual(tmf.TM[ii,jj], tmb.TM[ii,jj], 3)

        tma = tm([10, 20, 31, 0, 0, 0])
        tmb = tma // 2
        self.assertEqual(tmb[0], 5.0)
        self.assertEqual(tmb[2], 15.0)

    def test_general_tm_addition(self):
        tma = tm([0, 1, 2, 0, 0, 1])
        tmb = tm([2, 0, 1, 0, 1, 0])

        tmc = tma + tmb
        self.assertEqual(tmc[0], 2)
        self.assertEqual(tmc[2], 3)

        tmc = tma + 2
        self.assertEqual(tmc[0], 2)
        self.assertEqual(tmc[2], 4)

        tmc = tma + np.array([2, 0, 1, 0, 1, 0])
        self.assertEqual(tmc[0], 2)
        self.assertEqual(tmc[2], 3)

        tmc = tma + np.array([2])
        self.assertEqual(tmc[0], 2)
        self.assertEqual(tmc[2], 4)

    def test_general_tm_subtraction(self):
        tma = tm([0, 1, 2, 0, 0, 1])
        tmb = tm([2, 0, 1, 0, 1, 0])

        tmc = tma - tmb
        self.assertEqual(tmc[0], -2)
        self.assertEqual(tmc[2], 1)

        tmc = tma - 2
        self.assertEqual(tmc[0], -2)
        self.assertEqual(tmc[2], 0)

        tmc = tma - np.array([2, 0, 1, 0, 1, 0])
        self.assertEqual(tmc[0], -2)
        self.assertEqual(tmc[2], 1)

        tmc = tma - np.array([2])
        self.assertEqual(tmc[0], -2)
        self.assertEqual(tmc[2], 0)

    def test_general_tm_rmatmul(self):
        tma = tm([0, 1, 2, 0, 0, 1])
        tmb = tm([2, 0, 1, 0, 1, 0])

        tmc = tma * tmb
        #TODO Add more

    def test_general_tm_eq(self):
        tma = tm()
        tmb = tm()

        self.assertTrue(tma == tmb)

        self.assertFalse(tma == 1)

        tma = tm([1, 2, 3, 4, 5, 6])
        self.assertTrue(tma != tmb)

        tmb = tm([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertTrue(tma == tmb)

        tmb = tmb.copy()
        self.assertTrue(tma == tmb)

    def test_general_tm_exp6(self):
        tma = tm([1, 2, 3, 0, 0, np.pi / 4])
        exp6_result = tma.exp6()
        self.assertEqual(exp6_result.shape, (4, 4))

    def test_general_tm_gRot(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        rot = tma.gRot()
        np.testing.assert_allclose(rot, np.eye(3))

    def test_general_tm_dunder_sum(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        np.testing.assert_allclose(np.asarray(tma.__sum__()).flatten(), [6.0])

    def test_general_tm_rmatmul_dunder(self):
        tma = tm([1, 0, 0, 0, 0, 0])
        tmb = tm([0, 1, 0, 0, 0, 0])

        tm_case = tma.__rmatmul__(tmb)
        self.assertIsInstance(tm_case, tm)
        np.testing.assert_allclose(tm_case.TM, tmb.TM @ tma.TM)

        array_case = tma.__rmatmul__(np.eye(4))
        np.testing.assert_allclose(array_case.TM, tma.TM)

        scalar_case = tma.__rmatmul__(2)
        np.testing.assert_allclose(scalar_case.TAA.flatten(), (2 * tma.TAA).flatten())

    def test_general_tm_mul_ndarray(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        result = tma * np.eye(4)
        np.testing.assert_allclose(result.TM, tma.TM * np.eye(4))

    def test_general_tm_rmul(self):
        tma = tm([1, 0, 0, 0, 0, 0])
        tmb = tm([0, 1, 0, 0, 0, 0])

        tm_case = tma.__rmul__(tmb)
        np.testing.assert_allclose(tm_case.TM, tmb.TM @ tma.TM)

        array_case = tma.__rmul__(np.eye(4))
        np.testing.assert_allclose(array_case.TM, np.eye(4) * tma.TM)

    def test_general_tm_gt(self):
        big = tm([2, 2, 2, 1, 1, 1])
        small = tm([1, 1, 1, 0, 0, 0])

        self.assertTrue(big > small)
        self.assertFalse(small > big)
        self.assertTrue(big > 0)
        self.assertFalse(small > 10)

    def test_general_tm_lt(self):
        big = tm([2, 2, 2, 1, 1, 1])
        small = tm([1, 1, 1, 0, 0, 0])

        self.assertTrue(small < big)
        self.assertFalse(big < small)
        self.assertTrue(small < 10)
        self.assertFalse(big < 0)

    def test_general_tm_le(self):
        tma = tm([1, 1, 1, 0, 0, 0])
        tmb = tm([1, 1, 1, 0, 0, 0])
        tmc = tm([2, 2, 2, 1, 1, 1])

        self.assertTrue(tma <= tmb)
        self.assertTrue(tma <= tmc)
        self.assertFalse(tmc <= tma)

    def test_general_tm_ge(self):
        tma = tm([1, 1, 1, 0, 0, 0])
        tmb = tm([1, 1, 1, 0, 0, 0])
        tmc = tm([2, 2, 2, 1, 1, 1])

        self.assertTrue(tma >= tmb)
        self.assertTrue(tmc >= tma)
        self.assertFalse(tma >= tmc)

    def test_general_tm_str(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        expected = "[ 1.000000, 2.000000, 3.000000, 0.000000, 0.000000, 0.000000 ]"
        self.assertEqual(str(tma), expected)



if __name__ == '__main__':
    unittest.main()
