import io
import time
import unittest
from contextlib import redirect_stdout

import numpy as np
from basic_robotics.general import tm
from basic_robotics.general import Wrench
from basic_robotics.utilities.disp import disp, dispa, disptex, printTFlist, progressBar

class test_utilities_disp(unittest.TestCase):

    def test_utilities_disp_lists(self):
        tma = tm([1, 2, 3, 4, 5, 6])
        tmb = tm([2, 3, 4, 5, 6, 7])
        tmc = tm([3, 4, 5, 6, 7, 8])
        tmd = tm([4, 5, 6, 7, 8, 9])

        tml = [tma, tmb, tmc, tmd]

        disp(tml, noprint=True)

        disp(tml, 'example matrix title', noprint=True)

        wr1 = Wrench(np.array([0, 1, 2]), tma)
        wr2 = Wrench(np.array([0, 1, 2]), tmb)
        wr3 = Wrench(np.array([0, 1, 2]), tmc)
        wr4 = Wrench(np.array([0, 1, 2]), tmd)

        disp([wr1, wr2, wr3, wr4], noprint=True)

    def test_utilities_disp_latex_mode(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = disp(mat, 'Latex Title', mode=1, noprint=True)
        self.assertIn('\\begin{table}', result)
        self.assertIn('Latex Title', result)

    def test_utilities_disptex_no_shape_fallback(self):
        # Plain lists have no .shape, so disptex falls back to dispa.
        result = disptex([1, 2, 3], 'List Title')
        self.assertIsInstance(result, str)

    def test_utilities_dispa_single_tm(self):
        tma = tm([1, 2, 3, 0, 0, 0])
        result = dispa(tma, 'Single TM')
        self.assertIsInstance(result, str)

    def test_utilities_dispa_mixed_list(self):
        # A list that is neither all-tm nor all-Wrench exercises the
        # general per-item recursion branch of dispa.
        tma = tm([1, 2, 3, 0, 0, 0])
        mixed = [tma, np.zeros((2, 2))]
        result = dispa(mixed, 'Mixed List')
        self.assertIn('Mixed List BEGIN', result)
        # A tm item (not a list/tuple) is printed with its compact str(),
        # rather than recursed into the boxed matrix format.
        self.assertIn(str(tma), result)
        self.assertIn('Dim 1:', result)

    def test_utilities_dispa_1d_large_and_inf_values(self):
        result = dispa(np.array([float('inf'), 12345678.0, 1.0]), 'Vector')
        self.assertIsInstance(result, str)
        self.assertIn('inf', result.lower())

    def test_utilities_dispa_higher_dimensions(self):
        arr3 = np.arange(8, dtype=float).reshape((2, 2, 2))
        result3 = dispa(arr3, 'Cube')
        self.assertIn('Cube', result3)

        arr4 = np.arange(16, dtype=float).reshape((2, 2, 2, 2))
        result4 = dispa(arr4, 'Tesseract')
        self.assertIn('Tesseract', result4)

        arr5 = np.arange(32, dtype=float).reshape((2, 2, 2, 2, 2))
        result5 = dispa(arr5, 'Penteract')
        self.assertIn('Penteract', result5)

    def test_utilities_printTFlist_no_names_and_large_values(self):
        tma = tm([12345678, 0, 0, 0, 0, 0])
        tmb = tm([1, 2, 3, 0, 0, 0])
        result = printTFlist([tma, tmb], 'No Names', 3, print_names=False)
        self.assertIsInstance(result, str)
        self.assertNotIn('Xm', result)

    def test_utilities_progressBar_eta(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            progressBar(1, 10, ETA=time.time() - 1)
        self.assertIn('ETA:', buf.getvalue())
