import unittest
import sys
sys.path.append('tests')
from test_general_transform import test_general_tm
from test_general_fsr import test_general_fsr
from test_kinematics_sp import test_kinematics_sp
from test_kinematics_arm import test_kinematics_arm
from test_modern_robotics_numba import test_modern_robotics_numba

if __name__ == '__main__':
    unittest.main()
