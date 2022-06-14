import unittest
import numpy as np
from basic_robotics.general import tm
from basic_robotics.general import Screw
from basic_robotics.utilities.disp import disp

class test_general_screw(unittest.TestCase):
    
    def matrix_equality_assertion(self, mat_a, mat_b, num_dec = 3, eq = True):
        def matrix_equality_message(a, b):
            return ('\nMatrix Equality Error\n' + disp(a, 'Matrix A', noprint=True) +
                    '\nIs Not Equal To\n' + disp(b, 'Matrix B', noprint=True))
        def matrix_shape_message(a, b):
            return ('\nMatrix Shape Error\n' + disp(a.shape, 'Matrix A', noprint=True) +
                    '\nIs Not Equal To\n' + disp(b.shape, 'Matrix B', noprint=True))
        shape_a = mat_a.shape
        shape_b = mat_b.shape
        self.assertEqual(len(shape_a), len(shape_b), matrix_shape_message(mat_a, mat_b))
        for i in range(len(shape_a)):
            self.assertEqual(shape_a[i], shape_b[i], matrix_shape_message(mat_a, mat_b))
        mat_a_flat = mat_a.flatten()
        mat_b_flat = mat_b.flatten()
        for i in range(len(mat_a_flat)):
            if eq:
                self.assertAlmostEqual(
                    mat_a_flat[i], mat_b_flat[i], num_dec, matrix_equality_message(mat_a, mat_b))
            else:
                self.assertNotAlmostEqual(
                    mat_a_flat[i], mat_b_flat[i], num_dec, matrix_equality_message(mat_a, mat_b))

    def test_general_screw_Copy(self):
        screw1 = Screw(np.array([1.0,2,3,4,5,6]))
        screw2 = screw1.copy()
        
        ref_frame = tm([1, 2, 3, np.pi/7, 0, np.pi/7])
        screw2.changeFrame(ref_frame)
        screw2[2] = 7

        self.assertEqual(screw2[2], 7)
        self.assertEqual(screw1[2], 3) 

        self.matrix_equality_assertion(screw1.data, screw2.data, eq=False)

        ref_frame[2] = -1

        self.assertNotEqual(ref_frame[2], screw2.frame_applied[2])
        
    def test_general_screw_changeFrame(self):
        screw1 = Screw(np.array([1.0,2,3,4,5,6]))
        screw2 = screw1.copy()
        ref_frame = tm([1, 2, 3, np.pi/7, 0, np.pi/7])
        screw2.changeFrame(ref_frame)
        screw2.changeFrame(ref_frame)
        self.matrix_equality_assertion(screw2.frame_applied.gTM(), ref_frame.gTM())

        screw2.changeFrame(tm())
        self.matrix_equality_assertion(screw1.data, screw2.data)

    def test_general_screw_getPitch(self):
        screw1 = Screw(np.array([1.0,2,3,4,5,6]))
        self.assertAlmostEqual(screw1.getPitch(), 0.42640143)

    def test_general_screw_cross(self):
        test_screw_1 = Screw(np.array([0, 1, 0, 1, 2, 0]))
        test_screw_2 = Screw(np.array([1, 0, 1, 0, 1, 0]))

        test_screw_3 = test_screw_1 * test_screw_2
        screw_cross_ref = np.array([1.000000, 0.000000, -1.000000, 2.000000, -1.000000, -2.000000]).reshape((6,1))
        self.matrix_equality_assertion(test_screw_3, screw_cross_ref)

        test_screw_4 = Screw(np.array([1, 0, 1, 0, 1, 0])) 
        test_screw_4.changeFrame(tm([1, 2, 3, 1, 2, 3]))

        test_screw_5 = test_screw_1 * test_screw_4
        self.matrix_equality_assertion(test_screw_5, screw_cross_ref)

    def test_general_screw_dot(self):
        test_screw_1 = Screw(np.array([0, 1, 0, 1, 2, 0]))
        test_screw_2 = Screw(np.array([1, 0, 1, 0, 1, 0]))

        test_ds = test_screw_1 @ test_screw_2
        ref_ds = np.array([ 0.000,    2.000])
        self.matrix_equality_assertion(test_ds, ref_ds)

        test_screw_2.changeFrame(tm([1, 2, 3, 1, 2, 3]))
        test_ds = test_screw_1 @ test_screw_2
        self.matrix_equality_assertion(test_ds, ref_ds)

    def test_general_screw_dualScalarMultiply(self):
        test_screw_1 = Screw(np.array([0, 1, 0, 1, 2, 0]))

        ref_val = np.array([ 0.000000, 2.000000, 0.000000, 2.000000, 8.000000, 0.000000 ]).reshape((6,1))
        self.matrix_equality_assertion(test_screw_1 * np.array([2, 4]), ref_val)

    def test_general_screw_sum(self):
        test_screw_1 = Screw(np.array([0, 1, 0, 1, 2, 0]))
        self.assertEqual(sum(test_screw_1), 4.0)

    def test_general_screw_array(self):
        test_screw_0 = Screw(np.array([1, 2, 0, 0, 0, 0]))
        test_screw_1 = Screw(np.array([0, 1, 2, 0, 0, 0]))
        test_screw_2 = Screw(np.array([0, 0, 1, 2, 0, 0]))
        test_screw_3 = Screw(np.array([0, 0, 0, 1, 2, 0]))
        test_screw_4 = Screw(np.array([0, 0, 0, 0, 1, 2]))
        test_list = [test_screw_0, test_screw_1, test_screw_2, test_screw_3, test_screw_4] 

        test_array = np.array(test_list).T[0]

        ref_array = np.array([[1.000,    0.000,    0.000,    0.000,    0.000],
                [     2.000,    1.000,    0.000,    0.000,    0.000 ],
                [     0.000,    2.000,    1.000,    0.000,    0.000 ],
                [     0.000,    0.000,    2.000,    1.000,    0.000 ],
                [     0.000,    0.000,    0.000,    2.000,    1.000 ],
                [     0.000,    0.000,    0.000,    0.000,    2.000]])
        self.matrix_equality_assertion(test_array, ref_array)



