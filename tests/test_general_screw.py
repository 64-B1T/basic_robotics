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

