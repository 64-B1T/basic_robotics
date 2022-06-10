import unittest
import sys
sys.path.append('tests')
from test_general_transform import test_general_tm
from test_general_fsr import test_general_fsr
from test_general_wrench import test_general_wrench
from test_general_screw import test_general_screw
from test_kinematics_sp import test_kinematics_sp
from test_kinematics_arm import test_kinematics_arm
from test_modern_robotics_numba import test_modern_robotics_numba
from test_utilities_disp import test_utilities_disp

if __name__ == '__main__':
    unittest.main()
