import requests
import json
import time
import sys

from basic_robotics.general import fsr, tm
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.plotting.vis_3js_client import *
import os
import numpy as np
import copy

def run_example():

    c = DrawClient()
    c.delete()
    c.makeFloor()
    arm = loadArmFromURDF(os.path.abspath('./tests/test_helpers/ur_description/ur10.urdf'))
    base = TubePlot(tm([0, 0, .5, 0, 0, 0]), .5, .03, c)
    arm_vis = ArmPlot("UR5", arm, c)
    for j in range(100):
        for i in np.linspace(-np.pi/2, np.pi/2, 1000):
            time.sleep(0.005)
            arm.FK(np.array([i, i, i, i, 0, 0]))
            arm_vis.update(True)

