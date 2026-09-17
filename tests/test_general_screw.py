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

    def test_general_screw_reshape(self):
        screw1 = Screw(np.array([1.0, 2, 3, 4, 5, 6]))
        reshaped = screw1.reshape((3, 2))
        self.assertEqual(reshaped.shape, (3, 2))
        np.testing.assert_allclose(reshaped.flatten(), [1, 2, 3, 4, 5, 6])

    def test_general_screw_dunder_sum(self):
        test_screw_1 = Screw(np.array([0, 1, 0, 1, 2, 0]))
        self.assertEqual(test_screw_1.__sum__(), 4.0)

    def test_general_screw_setitem_array(self):
        screw1 = Screw(np.array([1.0, 2, 3, 4, 5, 6]))
        screw1[0:3] = np.array([9.0, 8, 7]).reshape((3, 1))
        np.testing.assert_allclose(screw1.data[0:3, 0], [9, 8, 7])

    def test_general_screw_add(self):
        screw1 = Screw(np.array([0, 1, 0, 1, 2, 0]))

        screw_plus_array = screw1 + np.array([1, 1, 1, 1, 1, 1])
        self.assertIsInstance(screw_plus_array, Screw)
        np.testing.assert_allclose(screw_plus_array.data.flatten(), [1, 2, 1, 2, 3, 1])

        scalar_result = screw1 + 5
        np.testing.assert_allclose(np.asarray(scalar_result).flatten(), [5, 6, 5, 6, 7, 5])

    def test_general_screw_radd(self):
        screw1 = Screw(np.array([0, 1, 0, 1, 2, 0]))
        result = 5 + screw1
        np.testing.assert_allclose(np.asarray(result).flatten(), [5, 6, 5, 6, 7, 5])

    def test_general_screw_sub(self):
        screw1 = Screw(np.array([5, 5, 5, 5, 5, 5]))
        screw2 = Screw(np.array([1, 2, 3, 4, 5, 6]))

        diff = screw1 - screw2
        self.assertIsInstance(diff, Screw)
        np.testing.assert_allclose(diff.data.flatten(), [4, 3, 2, 1, 0, -1])

        diff_arr = screw1 - np.array([1, 1, 1, 1, 1, 1])
        self.assertIsInstance(diff_arr, Screw)
        np.testing.assert_allclose(diff_arr.data.flatten(), [4, 4, 4, 4, 4, 4])

        diff_scalar = screw1 - 2
        np.testing.assert_allclose(np.asarray(diff_scalar).flatten(), [3, 3, 3, 3, 3, 3])

    def test_general_screw_rsub(self):
        screw1 = Screw(np.array([1, 2, 3, 4, 5, 6]))
        screw2 = Screw(np.array([5, 5, 5, 5, 5, 5]))

        # Both operands being Screws always dispatches through the left
        # operand's __sub__, so __rsub__'s Screw branch needs a direct call.
        direct = screw1.__rsub__(screw2)
        np.testing.assert_allclose(direct.data.flatten(), [4, 3, 2, 1, 0, -1])

        array_result = np.array([5.0, 5, 5, 5, 5, 5]) - screw1
        self.assertIsInstance(array_result, Screw)
        np.testing.assert_allclose(array_result.data.flatten(), [4, 3, 2, 1, 0, -1])

        scalar_result = 10 - screw1
        np.testing.assert_allclose(np.asarray(scalar_result).flatten(), [9, 8, 7, 6, 5, 4])

    def test_general_screw_matmul_fallback(self):
        screw1 = Screw(np.array([1.0, 2, 3, 4, 5, 6]))
        result = screw1 @ np.array([[2.0]])
        np.testing.assert_allclose(result.flatten(), screw1.data.flatten() * 2)

    def test_general_screw_mul_fallback(self):
        screw1 = Screw(np.array([1.0, 2, 3, 4, 5, 6]))
        result = screw1 * np.array([1, 1, 1, 1, 1, 1])
        self.assertEqual(result.shape, (6, 6))

    def test_general_screw_truediv_fallback(self):
        screw1 = Screw(np.array([2.0, 4, 6, 8, 10, 12]))
        result = screw1 / np.array([[2.0]])
        np.testing.assert_allclose(result.flatten(), [1, 2, 3, 4, 5, 6])

    def test_general_screw_rtruediv_fallback(self):
        screw1 = Screw(np.array([2.0, 2, 2, 2, 2, 2]))
        result = np.array([4.0, 4, 4, 4, 4, 4]).reshape((6, 1)) / screw1
        np.testing.assert_allclose(np.asarray(result).flatten(), [2, 2, 2, 2, 2, 2])

    def test_general_screw_floordiv(self):
        screw1 = Screw(np.array([5.0, 7, 9, 11, 13, 15]))

        int_result = screw1 // 2
        self.assertIsInstance(int_result, Screw)
        np.testing.assert_allclose(int_result.data.flatten(), [2, 3, 4, 5, 6, 7])

        array_result = screw1 // np.array([2.0, 2, 2, 2, 2, 2]).reshape((6, 1))
        np.testing.assert_allclose(
                np.asarray(array_result).flatten(), screw1.data.flatten() // 2)

    def test_general_screw_rfloordiv(self):
        screw1 = Screw(np.array([2.0, 2, 2, 2, 2, 2]))

        int_result = screw1.__rfloordiv__(9)
        self.assertIsInstance(int_result, Screw)
        np.testing.assert_allclose(int_result.data.flatten(), [4, 4, 4, 4, 4, 4])

        array_result = np.array([9.0, 9, 9, 9, 9, 9]).reshape((6, 1)) // screw1
        np.testing.assert_allclose(np.asarray(array_result).flatten(), [4, 4, 4, 4, 4, 4])

    def test_general_screw_eq_non_screw(self):
        screw1 = Screw(np.array([1.0, 2, 3, 4, 5, 6]))
        self.assertFalse(screw1 == "not a screw")

    def test_general_screw_comparison_operators(self):
        small = Screw(np.array([1.0, 1, 1, 1, 1, 1]))
        big = Screw(np.array([2.0, 2, 2, 2, 2, 2]))

        np.testing.assert_array_equal((small > big).flatten(), [False] * 6)
        np.testing.assert_array_equal((small > 0).flatten(), [True] * 6)

        np.testing.assert_array_equal((small < big).flatten(), [True] * 6)
        np.testing.assert_array_equal((small < 0).flatten(), [False] * 6)

        np.testing.assert_array_equal((small <= big).flatten(), [True] * 6)
        np.testing.assert_array_equal((small <= 1).flatten(), [True] * 6)

        np.testing.assert_array_equal((big >= small).flatten(), [True] * 6)
        np.testing.assert_array_equal((big >= 2).flatten(), [True] * 6)

    def test_general_screw_str(self):
        screw1 = Screw(np.array([1.0, 2, 3, 4, 5, 6]))
        expected = "[ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000 ]"
        self.assertEqual(str(screw1), expected)

