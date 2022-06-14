import json
import os
import matplotlib.pyplot as plt

import basic_robotics #Import the General Library
from basic_robotics.general import tm #Import transformation library
from basic_robotics.utilities.disp import disp
# SP Tests
disp("Beginning SP Test")
from basic_robotics.kinematics import loadSP
from basic_robotics.plotting.Draw import DrawSP

def run_example():
    basic_sp = {
    "Name":"Basic SP","Type":"SP","BottomPlate":{"Thickness":0.1,"JointRadius":0.9,"JointSpacing":9,"Mass": 6},
    "TopPlate":{"Thickness":0.16,"JointRadius":0.3,"JointSpacing":25,"Mass": 1},
    "Actuators":{"MinExtension":0.75,"MaxExtension":1.5,"MotorMass":0.5,"ShaftMass":0.9,"ForceLimit": 800,"MotorCOGD":0.2,"ShaftCOGD":0.2},
    "Drawing":{"TopRadius":1,"BottomRadius":1,"ShaftRadius": 0.1,"MotorRadius": 0.2},
    "Settings":{"MaxAngleDev":55,"GenerateActuators":0,"IgnoreRestHeight":1,"UseSpin":0,"AssignMasses":1,"InferActuatorCOG":1},
    "Params":{"RestHeight":1.2,"Spin":30}}

    basic_sp_string = json.dumps(basic_sp)
    with open ('sp_test_data.json', 'w') as outfile:
        outfile.write(basic_sp_string)

    sp_model = loadSP('sp_test_data.json', '')
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(0,2)

    #Delete file
    os.remove('sp_test_data.json')
    DrawSP(sp_model, ax)

    plt.show()

if __name__ == '__main__':
    run_example()