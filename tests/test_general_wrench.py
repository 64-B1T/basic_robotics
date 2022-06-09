import unittest
import numpy as np
from basic_robotics.general import tm
from basic_robotics.general import Wrench
from basic_robotics.utilities.disp import disp

class test_general_wrench(unittest.TestCase):

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
    
    def test_general_wrench_getMoment(self):
        test_wrench = Wrench(np.array([0, 0, -9.81]) * 5)
        moment = test_wrench.getMoment()
        self.assertEqual(moment[0], 0)

    def test_general_wrench_getForce(self):
        test_wrench = Wrench(np.array([0, 0, -9.81]) * 5)
        force = test_wrench.getForce()
        self.assertEqual(force[2], -9.81*5)

    def test_general_wrench_flatten(self):
        #TODO
        pass

    def test_general_wrench_changeFrame(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)
        wrench_b = Wrench(np.array([0, 0, -9.81]) * 5, tm([1, 0, -5, 0, 0, 0]), tm([-1, 0, 5, 0, 0, 0]))
        self.assertNotEqual(wrench_a[1], wrench_b[1])
        wrench_b.changeFrame(tm())
        self.matrix_equality_assertion(wrench_a.wrench_arr, wrench_b.wrench_arr)

    def test_general_wrench_copy(self):
        #TODO
        pass
    
    def test_general_wrench_getitem(self):
        #TODO
        pass

    def test_general_wrench_setitem(self):
        #TODO
        pass

    def test_general_wrench_floordiv(self):
        #TODO
        pass

    def test_general_wrench_abs(self):
        #TODO
        pass

    def test_general_wrench_sum(self):
        #TODO
        pass

    def test_general_wrench_add(self):
        #TODO
        pass

    def test_general_wrench_sub(self):
        #TODO
        pass

    def test_general_wrench_matmul(self):
        #TODO
        pass

    def test_general_wrench_rmatmul(self):
        #TODO
        pass

    def test_general_wrench_mul(self):
        #TODO
        pass

    def test_general_wrench_rmul(self):
        #TODO
        pass

    def test_general_wrench_truediv(self):
        #TODO
        pass

    def test_general_wrench_eq(self):
        #TODO
        pass

    def test_general_wrench_gt(self):
        #TODO
        pass

    def test_general_wrench_lt(self):
        #TODO
        pass

    def test_general_wrench_le(self):
        #TODO
        pass

    def test_general_wrench_ge(self):
        #TODO
        pass

    def test_general_wrench_str(self):
        #TODO
        pass
