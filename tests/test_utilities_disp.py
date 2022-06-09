import unittest
import numpy as np
from basic_robotics.general import tm
from basic_robotics.general import Wrench
from basic_robotics.utilities.disp import disp

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
