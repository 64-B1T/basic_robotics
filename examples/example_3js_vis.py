import time
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.plotting.vis_3js_client import *
import os
import numpy as np

def run_example():

    c = DrawClient()
    c.delete()
    c.makeFloor()
    arm = loadArmFromURDF(
        os.path.abspath(
            './tests/test_helpers/ur_description/ur10.urdf'))
    arm_vis = ArmPlot("UR5", arm, c)
    for i in np.linspace(-np.pi, np.pi, 6000):
        time.sleep(0.001)#30FPS
        arm.FK(np.array([i, i, i, i, 0, 0]))
        arm_vis.update(True)

if __name__ == '__main__':
    run_example()
