import unittest
import numpy as np
from basic_robotics.modern_robotics_numba import modern_high_performance as mr

class test_modern_robotics_numba(unittest.TestCase):

    def matrix_equality_assertion(self, mat_a, mat_b, num_dec = 3):
        shape_a = mat_a.shape
        shape_b = mat_b.shape
        self.assertEqual(len(shape_a), len(shape_b))
        for i in range(len(shape_a)):
            self.assertEqual(shape_a[i], shape_b[i])
        mat_a_flat = mat_a.flatten()
        mat_b_flat = mat_b.flatten()
        for i in range(len(mat_a_flat)):
            self.assertAlmostEqual(mat_a_flat[i], mat_b_flat[i], num_dec)

    def test_modern_robotics_numba_NearZero(self):
        z = -1e-7
        self.assertTrue(mr.NearZero(z))

    def test_modern_robotics_numba_Normalize(self):
        V = np.array([1, 2, 3])
        output = mr.Normalize(V)
        expected = np.array([0.26726124, 0.53452248, 0.80178373])
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_AngleMod(self):
        #TODO
        pass


    def test_modern_robotics_numba_Norm(self):
        #TODO
        pass


    def test_modern_robotics_numba_RotInv(self):
        R = np.array([[0, 0.0, 1],
                      [1, 0.0, 0],
                      [0, 1, 0]])
        expected = np.array([[0, 1, 0],
                  [0, 0.0, 1],
                  [1, 0.0, 0]])
        output = mr.RotInv(R)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_VecToso3(self):
        omg = np.array([1, 2, 3])
        expected = np.array([[ 0.0, -3,  2],
                  [ 3,  0.0, -1],
                  [-2,  1,  0]])
        output = mr.VecToso3(omg)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_so3ToVec(self):
        so3mat = np.array([[ 0.0, -3,  2],
                           [ 3,  0.0, -1],
                           [-2,  1,  0]])
        expected = np.array([1, 2, 3])
        output = mr.so3ToVec(so3mat)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_AxisAng3(self):
        #TODO
        pass
        expected_1 = np.array([0.26726124, 0.53452248, 0.80178373])
        expected_2 = 3.7416573867739413
        output = mr.AxisAng3(np.array([1, 2, 3]))
        self.matrix_equality_assertion(output[0], expected_1)
        self.assertAlmostEqual(expected_2, output[1])


    def test_modern_robotics_numba_MatrixExp3(self):
        expected = np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
        output = mr.MatrixExp3(np.array([[ 0.0, -3,  2],
                           [ 3,  0.0, -1],
                           [-2,  1,  0]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_SafeTrace(self):
        #TODO
        pass

    def test_modern_robotics_numba_SafeClip(self):
        #TODO
        pass

    def test_modern_robotics_numba_MatrixLog3(self):
        expected = np.array([[          0.0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0.0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
        output = mr.MatrixLog3(np.array([[0, 0.0, 1],
                      [1, 0.0, 0],
                      [0, 1, 0]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_RpToTrans(self):
        R = np.array([[1, 0.0,  0],
                      [0, 0.0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
        expected = np.array([[1, 0.0,  0.0, 1],
                  [0, 0.0, -1, 2],
                  [0, 1,  0.0, 5],
                  [0, 0.0,  0.0, 1]])
        output = mr.RpToTrans(R, p)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_TransToRp(self):
        #MUST USE FLOAT
        expected = (np.array([[1, 0.0,  0],
                   [0, 0.0, -1],
                   [0, 1,  0]]),
         np.array([0, 0.0, 3]))
        output = mr.TransToRp(np.array([[1, 0.0,  0.0, 0.0],
                      [0, 0.0, -1, 0],
                      [0, 1.0,  0.0, 3],
                      [0, 0.0,  0.0, 1]]))
        self.matrix_equality_assertion(output[0], expected[0])
        self.matrix_equality_assertion(output[1], expected[1])

    def test_modern_robotics_numba_TransInv(self):
        expected =np.array([[1,  0.0, 0.0,  0],
                  [0,  0.0, 1, -3],
                  [0, -1, 0.,  0],
                  [0,  0.0, 0.0,  1]])
        output = mr.TransInv(np.array([[1.0, 0.0,  0.0, 0.0],
                      [0, 0.0, -1, 0],
                      [0, 1,  0.0, 3],
                      [0, 0.0,  0.0, 1]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_VecTose3(self):
        expected = np.array([[ 0.0, -3,  2, 4],
                  [ 3,  0.0, -1, 5],
                  [-2,  1,  0.0, 6],
                  [ 0.0,  0.0,  0.0, 0]])
        output = mr.VecTose3(np.array([1, 2, 3, 4, 5, 6]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_se3ToVec(self):
        expected = np.array([1, 2, 3, 4, 5, 6])
        output = mr.se3ToVec(np.array([[ 0.0, -3,  2, 4],
                           [ 3,  0.0, -1, 5],
                           [-2,  1,  0.0, 6],
                           [ 0.0,  0.0,  0.0, 0]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_Adjoint(self):
        expected =np.array([[1, 0.0,  0.0, 0.0, 0.0,  0],
                  [0, 0.0, -1, 0.0, 0.0,  0],
                  [0, 1,  0.0, 0.0, 0.0,  0],
                  [0, 0.0,  3, 1, 0.0,  0],
                  [3, 0.0,  0.0, 0.0, 0.0, -1],
                  [0, 0.0,  0.0, 0.0, 1,  0]])
        output = mr.Adjoint(np.array([[1, 0.0,  0.0, 0],
                      [0, 0.0, -1, 0],
                      [0, 1,  0.0, 3],
                      [0, 0.0,  0.0, 1]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_ScrewToAxis(self):
        expected = np.array([0, 0.0, 1, 0.0, -3, 2])
        output = mr.ScrewToAxis(np.array([3, 0.0, 0]), np.array([0, 0.0, 1]), 2)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_AxisAng6(self):
        expected = (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
        output = mr.AxisAng6(np.array([1, 0.0, 0.0, 1, 2, 3]))
        self.matrix_equality_assertion(output[0], expected[0])
        self.assertAlmostEqual(output[1], expected[1])

    def test_modern_robotics_numba_MatrixExp6(self):
        expected = np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0.0,   0.0,    0.0,   1]])
        output = mr.MatrixExp6(np.array([[0,     0.0,       0.0,      0],
                           [0,   0.0, -1.57079632, 2.35619449],
                           [0, 1.57079632,       0.0, 2.35619449],
                           [0,   0.0,       0.0,      0]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_MatMul(self):
        #TODO
        pass

    def test_modern_robotics_numba_LocalToGlobal(self):
        #TODO
        pass

    def test_modern_robotics_numba_GlobalToLocal(self):
        #TODO
        pass

    def test_modern_robotics_numba_MatrixLog6(self):
        expected = np.array([[0,          0.0,           0.0,           0],
                  [0,          0.0, -1.57079633,  2.35619449],
                  [0, 1.57079633,           0.0,  2.35619449],
                  [0,          0.0,           0.0,           0]])
        output = mr.MatrixLog6(np.array([[1, 0.0,  0.0, 0],
                      [0, 0.0, -1, 0],
                      [0, 1,  0.0, 3],
                      [0, 0.0,  0.0, 1]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_SafeDot(self):
        #TODO
        pass

    def test_modern_robotics_numba_ProjectToSO3(self):
        expected = np.array([[ 0.67901136,  0.14894516,  0.71885945],
                  [ 0.37320708,  0.77319584, -0.51272279],
                  [-0.63218672,  0.61642804,  0.46942137]])
        output = mr.ProjectToSO3(np.array([[ 0.675,  0.150,  0.720],
                        [ 0.370,  0.771, -0.511],
                        [-0.630,  0.619,  0.472]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_ProjectToSE3(self):
        expected = np.array([[ 0.67901136,  0.14894516,  0.71885945,  1.2 ],
                  [ 0.37320708,  0.77319584, -0.51272279,  5.4 ],
                  [-0.63218672,  0.61642804,  0.46942137,  3.6 ],
                  [ 0.        ,  0.        ,  0.        ,  1.  ]])
        output = mr.ProjectToSE3(np.array([[ 0.675,  0.150,  0.720,  1.2],
                        [ 0.370,  0.771, -0.511,  5.4],
                        [-0.630,  0.619,  0.472,  3.6],
                        [ 0.003,  0.002,  0.010,  0.9]]))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_DistanceToSO3(self):
        expected = 0.08835
        output = mr.DistanceToSO3(np.array([[ 1.0,  0.0,   0.0 ],
                        [ 0.0,  0.1,  -0.95],
                        [ 0.0,  1.0,   0.1 ]]))
        self.assertAlmostEqual(output, expected, 4)

    def test_modern_robotics_numba_DistanceToSE3(self):
        expected = 0.134931
        output = mr.DistanceToSE3(np.array([[ 1.0,  0.0,   0.0,   1.2 ],
                        [ 0.0,  0.1,  -0.95,  1.5 ],
                        [ 0.0,  1.0,   0.1,  -0.9 ],
                        [ 0.0,  0.0,   0.1,   0.98 ]]))

        self.assertAlmostEqual(output, expected, 4)

    def test_modern_robotics_numba_TestIfSO3(self):
        output = mr.TestIfSO3(mat = np.array([[1.0, 0.0,  0.0 ],
                        [0.0, 0.1, -0.95],
                        [0.0, 1.0,  0.1 ]]))
        self.assertFalse(output)

    def test_modern_robotics_numba_TestIfSE3(self):
        output = mr.TestIfSE3(np.array([[1.0, 0.0,   0.0,  1.2],
                        [0.0, 0.1, -0.95,  1.5],
                        [0.0, 1.0,   0.1, -0.9],
                        [0.0, 0.0,   0.1, 0.98]]))
        self.assertFalse(output)

    def test_modern_robotics_numba_FKinBody(self):
        M = np.array([[-1, 0.0,  0.0, 0],
                      [ 0.0, 1,  0.0, 6],
                      [ 0.0, 0.0, -1, 2],
                      [ 0.0, 0.0,  0.0, 1]])
        Blist = np.array([[0, 0.0, -1, 2, 0.0,   0],
                          [0, 0.0,  0.0, 0.0, 1,   0],
                          [0, 0.0,  1, 0.0, 0.0, 0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
        expected = np.array([[0, 1,  0.0,         -5],
                  [1, 0.0,  0.0,          4],
                  [0, 0.0, -1, 1.68584073],
                  [0, 0.0,  0.0,          1]])
        output = mr.FKinBody(M, Blist, thetalist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_FKinSpace(self):
        M = np.array([[-1, 0.0,  0.0, 0],
                      [ 0.0, 1,  0.0, 6],
                      [ 0.0, 0.0, -1, 2],
                      [ 0.0, 0.0,  0.0, 1]])
        Slist = np.array([[0, 0.0,  1,  4, 0.0,    0],
                          [0, 0.0,  0.0,  0.0, 1,    0],
                          [0, 0.0, -1, -6, 0.0, -0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
        expected = np.array([[0, 1,  0.0,         -5],
                  [1, 0.0,  0.0,          4],
                  [0, 0.0, -1, 1.68584073],
                  [0, 0.0,  0.0,          1]])
        output = mr.FKinSpace(M, Slist, thetalist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_SafeCopy(self):
        #TODO
        pass

    def test_modern_robotics_numba_JacobianBody(self):
        Blist = np.array([[0, 0.0, 1,   0.0, 0.2, 0.2],
                          [1, 0.0, 0.0,   2,   0.0,   3],
                          [0, 1, 0.0,   0.0,   2,   1],
                          [1, 0.0, 0.0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
        expected = np.array([[-0.04528405, 0.99500417,           0.0,   1],
                  [ 0.74359313, 0.09304865,  0.36235775,   0],
                  [-0.66709716, 0.03617541, -0.93203909,   0],
                  [ 2.32586047,    1.66809,  0.56410831, 0.2],
                  [-1.44321167, 2.94561275,  1.43306521, 0.3],
                  [-2.06639565, 1.82881722, -1.58868628, 0.4]])
        output = mr.JacobianBody(Blist, thetalist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_JacobianSpace(self):
        Slist = np.array([[0, 0.0, 1,   0.0, 0.2, 0.2],
                          [1, 0.0, 0.0,   2,   0.0,   3],
                          [0, 1, 0.0,   0.0,   2,   1],
                          [1, 0.0, 0.0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
        expected = np.array([[  0.0, 0.98006658, -0.09011564,  0.95749426],
                  [  0.0, 0.19866933,   0.4445544,  0.28487557],
                  [  1,          0.0,  0.89120736, -0.04528405],
                  [  0.0, 1.95218638, -2.21635216, -0.51161537],
                  [0.2, 0.43654132, -2.43712573,  2.77535713],
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])
        output = mr.JacobianSpace(Slist, thetalist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_IKinBody(self):
        Blist = np.array([[0, 0.0, -1, 2, 0.0,   0],
                          [0, 0.0,  0.0, 0.0, 1,   0],
                          [0, 0.0,  1, 0.0, 0.0, 0.1]]).T
        M = np.array([[-1, 0.0,  0.0, 0],
                      [ 0.0, 1,  0.0, 6],
                      [ 0.0, 0.0, -1, 2],
                      [ 0.0, 0.0,  0.0, 1]])
        T = np.array([[0, 1,  0.0,     -5],
                      [1, 0.0,  0.0,      4],
                      [0, 0.0, -1, 1.6858],
                      [0, 0.0,  0.0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
        expected = (np.array([1.57073819, 2.999667, 3.14153913]), True)
        output = mr.IKinBody(Blist, M, T, thetalist0, eomg, ev)
        self.matrix_equality_assertion(output[0], expected[0])
        self.assertTrue(output[1])

    def test_modern_robotics_numba_IKinSpace(self):
        Slist = np.array([[0, 0.0,  1,  4, 0.0,    0],
                          [0, 0.0,  0.0,  0.0, 1,    0],
                          [0, 0.0, -1, -6, 0.0, -0.1]]).T
        M = np.array([[-1, 0.0,  0.0, 0],
                      [ 0.0, 1,  0.0, 6],
                      [ 0.0, 0.0, -1, 2],
                      [ 0.0, 0.0,  0.0, 1]])
        T = np.array([[0, 1,  0.0,     -5],
                      [1, 0.0,  0.0,      4],
                      [0, 0.0, -1, 1.6858],
                      [0, 0.0,  0.0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
        expected = (np.array([ 1.57073783,  2.99966384,  3.1415342 ]), True)
        output = mr.IKinSpace(Slist, M, T, thetalist0, eomg, ev)
        self.matrix_equality_assertion(output[0], expected[0])
        self.assertTrue(output[1])

    def test_modern_robotics_numba_ad(self):
        expected = np.array([[ 0.0, -3,  2,  0.0,  0.0,  0],
                  [ 3,  0.0, -1,  0.0,  0.0,  0],
                  [-2,  1,  0.0,  0.0,  0.0,  0],
                  [ 0.0, -6,  5,  0.0, -3,  2],
                  [ 6,  0.0, -4,  3,  0.0, -1],
                  [-5,  4,  0.0, -2,  1,  0]])
        input_arr = np.array([1.0, 2, 3, 4, 5, 6])
        output = mr.ad(input_arr)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_InverseDynamics(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        g = np.array([0, 0.0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0.0, 0.0,        0],
                        [0, 1, 0.0,        0],
                        [0, 0.0, 1, 0.089159],
                        [0, 0.0, 0.0,        1]])
        M12 = np.array([[ 0.0, 0.0, 1,    0.28],
                        [ 0.0, 1, 0.0, 0.13585],
                        [-1, 0.0, 0.0,       0],
                        [ 0.0, 0.0, 0.0,       1]])
        M23 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0, -0.1197],
                        [0, 0.0, 1,   0.395],
                        [0, 0.0, 0.0,       1]])
        M34 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0,       0],
                        [0, 0.0, 1, 0.14225],
                        [0, 0.0, 0.0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0.0, 1,      0.0, 1,     0],
                          [0, 1, 0.0, -0.089, 0.0,     0],
                          [0, 1, 0.0, -0.089, 0.0, 0.425]]).T
        expected = np.array([74.69616155, -33.06766016, -3.23057314])
        output = mr.InverseDynamics(thetalist, dthetalist,
                ddthetalist, g, Ftip, Mlist, Glist, Slist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_MassMatrix(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        M01 = np.array([[1, 0.0, 0.0,        0],
                        [0, 1, 0.0,        0],
                        [0, 0.0, 1, 0.089159],
                        [0, 0.0, 0.0,        1]])
        M12 = np.array([[ 0.0, 0.0, 1,    0.28],
                        [ 0.0, 1, 0.0, 0.13585],
                        [-1, 0.0, 0.0,       0],
                        [ 0.0, 0.0, 0.0,       1]])
        M23 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0, -0.1197],
                        [0, 0.0, 1,   0.395],
                        [0, 0.0, 0.0,       1]])
        M34 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0,       0],
                        [0, 0.0, 1, 0.14225],
                        [0, 0.0, 0.0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0.0, 1,      0.0, 1,     0],
                          [0, 1, 0.0, -0.089, 0.0,     0],
                          [0, 1, 0.0, -0.089, 0.0, 0.425]]).T
        expected = np.array([[ 2.25433380e+01, -3.07146754e-01, -7.18426391e-03],
                  [-3.07146754e-01,  1.96850717e+00,  4.32157368e-01],
                  [-7.18426391e-03,  4.32157368e-01,  1.91630858e-01]])
        output = mr.MassMatrix(thetalist, Mlist, Glist, Slist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_VelQuadraticForces(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        M01 = np.array([[1, 0.0, 0.0,        0],
                        [0, 1, 0.0,        0],
                        [0, 0.0, 1, 0.089159],
                        [0, 0.0, 0.0,        1]])
        M12 = np.array([[ 0.0, 0.0, 1,    0.28],
                        [ 0.0, 1, 0.0, 0.13585],
                        [-1, 0.0, 0.0,       0],
                        [ 0.0, 0.0, 0.0,       1]])
        M23 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0, -0.1197],
                        [0, 0.0, 1,   0.395],
                        [0, 0.0, 0.0,       1]])
        M34 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0,       0],
                        [0, 0.0, 1, 0.14225],
                        [0, 0.0, 0.0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0.0, 1,      0.0, 1,     0],
                          [0, 1, 0.0, -0.089, 0.0,     0],
                          [0, 1, 0.0, -0.089, 0.0, 0.425]]).T
        expected = np.array([0.26453118, -0.05505157, -0.00689132])
        output = mr.VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_GravityForces(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        g = np.array([0, 0.0, -9.8])
        M01 = np.array([[1, 0.0, 0.0,        0],
                        [0, 1, 0.0,        0],
                        [0, 0.0, 1, 0.089159],
                        [0, 0.0, 0.0,        1]])
        M12 = np.array([[ 0.0, 0.0, 1,    0.28],
                        [ 0.0, 1, 0.0, 0.13585],
                        [-1, 0.0, 0.0,       0],
                        [ 0.0, 0.0, 0.0,       1]])
        M23 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0, -0.1197],
                        [0, 0.0, 1,   0.395],
                        [0, 0.0, 0.0,       1]])
        M34 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0,       0],
                        [0, 0.0, 1, 0.14225],
                        [0, 0.0, 0.0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0.0, 1,      0.0, 1,     0],
                          [0, 1, 0.0, -0.089, 0.0,     0],
                          [0, 1, 0.0, -0.089, 0.0, 0.425]]).T
        expected = np.array([28.40331262, -37.64094817, -5.4415892])
        output = mr.GravityForces(thetalist, g, Mlist, Glist, Slist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_EndEffectorForces(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0.0, 0.0,        0],
                        [0, 1, 0.0,        0],
                        [0, 0.0, 1, 0.089159],
                        [0, 0.0, 0.0,        1]])
        M12 = np.array([[ 0.0, 0.0, 1,    0.28],
                        [ 0.0, 1, 0.0, 0.13585],
                        [-1, 0.0, 0.0,       0],
                        [ 0.0, 0.0, 0.0,       1]])
        M23 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0, -0.1197],
                        [0, 0.0, 1,   0.395],
                        [0, 0.0, 0.0,       1]])
        M34 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0,       0],
                        [0, 0.0, 1, 0.14225],
                        [0, 0.0, 0.0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0.0, 1,      0.0, 1,     0],
                          [0, 1, 0.0, -0.089, 0.0,     0],
                          [0, 1, 0.0, -0.089, 0.0, 0.425]]).T
        expected = np.array([1.40954608, 1.85771497, 1.392409])
        output = mr.EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_ForwardDynamics(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taulist = np.array([0.5, 0.6, 0.7])
        g = np.array([0, 0.0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0.0, 0.0,        0],
                        [0, 1, 0.0,        0],
                        [0, 0.0, 1, 0.089159],
                        [0, 0.0, 0.0,        1]])
        M12 = np.array([[ 0.0, 0.0, 1,    0.28],
                        [ 0.0, 1, 0.0, 0.13585],
                        [-1, 0.0, 0.0,       0],
                        [ 0.0, 0.0, 0.0,       1]])
        M23 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0, -0.1197],
                        [0, 0.0, 1,   0.395],
                        [0, 0.0, 0.0,       1]])
        M34 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0,       0],
                        [0, 0.0, 1, 0.14225],
                        [0, 0.0, 0.0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0.0, 1,      0.0, 1,     0],
                          [0, 1, 0.0, -0.089, 0.0,     0],
                          [0, 1, 0.0, -0.089, 0.0, 0.425]]).T
        expected = np.array([-0.97392907, 25.58466784, -32.91499212])
        output = mr. ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_EulerStep(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        dt = 0.1
        expected = (np.array([ 0.11,  0.12,  0.13]), np.array([ 0.3 ,  0.35,  0.4 ]))
        output = mr.EulerStep(thetalist, dthetalist, ddthetalist, dt)
        self.matrix_equality_assertion(output[0], expected[0])
        self.matrix_equality_assertion(output[1], expected[1])

    def test_modern_robotics_numba_InverseDynamicsTrajectory(self):
        #TODO
        pass

    def test_modern_robotics_numba_ForwardDynamicsTrajectory(self):
        #TODO
        pass

    def test_modern_robotics_numba_CubicTimeScaling(self):
        expected = 0.216
        Tf = 2
        t = 0.6
        output = mr.CubicTimeScaling(Tf, t)
        self.assertAlmostEqual(output, expected, 4)

    def test_modern_robotics_numba_JointTrajectory(self):
        thetastart = np.array([1, 0.0, 0.0, 1, 1, 0.2, 0.0, 1])
        thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1])
        Tf = 4
        N = 6
        method = 3
        expected = np.array([[     1,     0.0,      0.0,      1,     1,    0.2,      0.0, 1],
                  [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1],
                  [1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1],
                  [1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1],
                  [1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1],
                  [   1.2,   0.5,    0.6,    1.1,     2,      2,    0.9, 1]])

        output = mr.JointTrajectory(thetastart, thetaend, Tf, N, method)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_ScrewTrajectory(self):
        Xstart = np.array([[1, 0.0, 0.0, 1],
                           [0, 1, 0.0, 0],
                           [0, 0.0, 1, 1],
                           [0, 0.0, 0.0, 1]])
        Xend = np.array([[0, 0.0, 1, 0.1],
                         [1, 0.0, 0.0,   0],
                         [0, 1, 0.0, 4.1],
                         [0, 0.0, 0.0,   1]])
        tF = 5
        N = 4
        method = 3
        expected = np.array([np.array([[1, 0.0, 0.0, 1],
                   [0, 1, 0.0, 0],
                   [0, 0.0, 1, 1],
                   [0, 0.0, 0.0, 1]]),
         np.array([[0.904, -0.25, 0.346, 0.441],
                   [0.346, 0.904, -0.25, 0.529],
                   [-0.25, 0.346, 0.904, 1.601],
                   [    0.0,     0.0,     0.0,     1]]),
         np.array([[0.346, -0.25, 0.904, -0.117],
                   [0.904, 0.346, -0.25,  0.473],
                   [-0.25, 0.904, 0.346,  3.274],
                   [    0.0,     0.0,     0.0,      1]]),
         np.array([[0, 0.0, 1, 0.1],
                   [1, 0.0, 0.0,   0],
                   [0, 1, 0.0, 4.1],
                   [0, 0.0, 0.0,   1]])])
        output = np.array(mr.ScrewTrajectory(Xstart, Xend, tF, N, method))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_ComputedTorque(self):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        eint = np.array([0.2, 0.2, 0.2])
        g = np.array([0, 0.0, -9.8])
        M01 = np.array([[1, 0.0, 0.0,        0],
                        [0, 1, 0.0,        0],
                        [0, 0.0, 1, 0.089159],
                        [0, 0.0, 0.0,        1]])
        M12 = np.array([[ 0.0, 0.0, 1,    0.28],
                        [ 0.0, 1, 0.0, 0.13585],
                        [-1, 0.0, 0.0,       0],
                        [ 0.0, 0.0, 0.0,       1]])
        M23 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0, -0.1197],
                        [0, 0.0, 1,   0.395],
                        [0, 0.0, 0.0,       1]])
        M34 = np.array([[1, 0.0, 0.0,       0],
                        [0, 1, 0.0,       0],
                        [0, 0.0, 1, 0.14225],
                        [0, 0.0, 0.0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0.0, 1,      0.0, 1,     0],
                          [0, 1, 0.0, -0.089, 0.0,     0],
                          [0, 1, 0.0, -0.089, 0.0, 0.425]]).T
        thetalistd = np.array([1.0, 1.0, 1.0])
        dthetalistd = np.array([2, 1.2, 2])
        ddthetalistd = np.array([0.1, 0.1, 0.1])
        Kp = 1.3
        Ki = 1.2
        Kd = 1.1
        expected = np.array([133.00525246, -29.94223324, -3.03276856])
        output = mr.ComputedTorque(thetalist, dthetalist, eint, g, Mlist, Glist, Slist, \
                           thetalistd, dthetalistd, ddthetalistd, Kp, Ki, Kd)
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_CartesianTrajectory(self):
        Xstart = np.array([[1, 0.0, 0.0, 1],
                           [0, 1, 0.0, 0],
                           [0, 0.0, 1, 1],
                           [0, 0.0, 0.0, 1]])
        Xend = np.array([[0, 0.0, 1, 0.1],
                         [1, 0.0, 0.0,   0],
                         [0, 1, 0.0, 4.1],
                         [0, 0.0, 0.0,   1]])
        tF= 5
        N = 4
        method = 5
        expected = np.array([np.array([[1, 0.0, 0.0, 1],
                   [0, 1, 0.0, 0],
                   [0, 0.0, 1, 1],
                   [0, 0.0, 0.0, 1]]),
         np.array([[ 0.937, -0.214,  0.277, 0.811],
                   [ 0.277,  0.937, -0.214,     0],
                   [-0.214,  0.277,  0.937, 1.651],
                   [     0.0,      0.0,      0.0,     1]]),
         np.array([[ 0.277, -0.214,  0.937, 0.289],
                   [ 0.937,  0.277, -0.214,     0],
                   [-0.214,  0.937,  0.277, 3.449],
                   [     0.0,      0.0,      0.0,     1]]),
         np.array([[0, 0.0, 1, 0.1],
                   [1, 0.0, 0.0,   0],
                   [0, 1, 0.0, 4.1],
                   [0, 0.0, 0.0,   1]])])
        output = np.array(mr.CartesianTrajectory(Xstart, Xend, tF, N, method))
        self.matrix_equality_assertion(output, expected)

    def test_modern_robotics_numba_SimulateControl(self):
        #TODO
        pass
