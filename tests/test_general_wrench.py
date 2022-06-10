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
        test_wrench = Wrench(np.array([0, 0, -9.81]) * 5)
        flat_wrench = test_wrench.flatten()
        self.assertFalse(isinstance(flat_wrench, Wrench))
        self.assertEqual(len(flat_wrench.shape), 1)

    def test_general_wrench_changeFrame(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)
        wrench_b = Wrench(np.array([0, 0, -9.81]) * 5, tm([1, 0, -5, 0, 0, 0]), tm([-1, 0, 5, 0, 0, 0]))
        self.assertNotEqual(wrench_a[1], wrench_b[1])
        wrench_b.changeFrame(tm())
        self.matrix_equality_assertion(wrench_a.getData(), wrench_b.getData())


        wrench_c = wrench_a.copy()
        wrench_c.changeFrame(tm())
        self.matrix_equality_assertion(wrench_a.getData(), wrench_c.getData())

    def test_general_wrench_copy(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)
        wrench_b = wrench_a.copy()
        wrench_c = wrench_a

        wrench_c[2] = 5 
        wrench_b[1] = 10

        self.assertEqual(wrench_c[2], 5)
        self.assertEqual(wrench_a[2], 5)
        self.assertEqual(wrench_b[2], 0)
        self.assertEqual(wrench_b[1], 10)
        
        wrench_a.frame_applied[2] = 3
        self.assertEqual(wrench_c.frame_applied[2], 3)
        self.assertEqual(wrench_b.frame_applied[2], 0)
    

    def test_general_wrench_abs(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)
        wrench_abs = abs(wrench_a)
        self.assertAlmostEqual(wrench_abs[5], 49.05)

    def test_general_wrench_add(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)
        wrench_b = Wrench(np.array([0, 0, -9.81]) * 5, tm([1, 0, -5, 0, 0, 0]), tm([-1, 0, 5, 0, 0, 0]))

        wrench_c_test = wrench_a + wrench_b
        self.assertTrue(isinstance(wrench_c_test, Wrench))
        self.assertNotEqual(wrench_a[1], wrench_b[1])

        wrench_b.changeFrame(tm())
        wrench_c_ref = wrench_a + wrench_b
        self.matrix_equality_assertion(wrench_c_test.getData(), wrench_c_ref.getData())

    def test_general_wrench_sub(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)
        wrench_b = Wrench(np.array([0, 0, -9.81]) * 5, tm([1, 0, -5, 0, 0, 0]), tm([-1, 0, 5, 0, 0, 0]))

        wrench_c_test = wrench_a - wrench_b
        self.assertTrue(isinstance(wrench_c_test, Wrench))
        self.assertNotEqual(wrench_a[1], wrench_b[1])

        wrench_b.changeFrame(tm())
        wrench_c_ref = wrench_a - wrench_b
        self.matrix_equality_assertion(wrench_c_ref.getData(), np.zeros((6,1)))

    def test_general_wrench_mul(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)

        wrench_b = Wrench(np.array([0, 0, -9.81]) * 10)

        wrench_c = wrench_a * 2
        wrench_d = 2 * wrench_a

        self.matrix_equality_assertion(wrench_c.getData(), wrench_d.getData())
        self.matrix_equality_assertion(wrench_b.getData(), wrench_d.getData())


        test_m = np.ones((6,6))
        self.assertFalse(isinstance(test_m * wrench_c, Wrench))

    def test_general_wrench_truediv(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)

        wrench_b = Wrench(np.array([0, 0, -9.81]) * 10)
        wrench_b_ref = Wrench(np.array([0, 0, -9.81]) * 10)

        self.matrix_equality_assertion(wrench_b.getData(), wrench_b_ref.getData())
        wrench_c = wrench_b / 2
        self.matrix_equality_assertion(wrench_b.getData(), wrench_b_ref.getData())
        wrench_d = 2 / wrench_b 

        self.matrix_equality_assertion(wrench_c.getData(), wrench_a.getData())
        self.assertAlmostEqual(wrench_d[5], 2/(-98.1))
        pass

    def test_general_wrench_eq(self):
        wrench_a = Wrench(np.array([0, 0, -9.81]) * 5)
        wrench_b = Wrench(np.array([0, 0, 0, 0, 0, 0]))
        self.assertTrue(wrench_a != wrench_b)

        wrench_c = Wrench(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -49.05]))
        self.assertTrue(wrench_a == wrench_c)

        wrench_d = Wrench(np.array([0, 0, 0, 0, 0, -49.05]), frame_applied = tm([1, 2, 3, 4, 5, 6]))
        self.assertTrue(wrench_a != wrench_d)

        self.assertFalse(wrench_a == tm())
