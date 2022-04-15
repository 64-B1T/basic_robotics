import unittest
import numpy as np
from basic_robotics.general import tm, fsr, fmr
import json
import os
from basic_robotics.kinematics import loadSP
from basic_robotics.plotting.Draw import DrawSP, DrawAxes
from basic_robotics.utilities.disp import disp
import matplotlib.pyplot as plt
basic_sp = {
   "Name":"Basic SP",
   "Type":"SP",
   "BottomPlate":{
      "Thickness":0.1,
      "JointRadius":0.9,
      "JointSpacing":9,
      "Mass": 6
   },
   "TopPlate":{
      "Thickness":0.16,
      "JointRadius":0.3,
      "JointSpacing":25,
      "Mass": 1
   },
   "Actuators":{
      "MinExtension":0.75,
      "MaxExtension":1.5,
      "MotorMass":0.5,
      "ShaftMass":0.9,
      "ForceLimit": 800,
      "MotorCOGD":0.2,
      "ShaftCOGD":0.2
   },
   "Drawing":{
     "TopRadius":1,
     "BottomRadius":1,
     "ShaftRadius": 0.1,
     "MotorRadius": 0.2
   },
   "Settings":{
      "MaxAngleDev":55,
      "GenerateActuators":0,
      "IgnoreRestHeight":1,
      "UseSpin":0,
      "AssignMasses":1,
      "InferActuatorCOG":1
   },
   "Params":{
      "RestHeight":1.2,
      "Spin":30
   }
}

basic_sp_string = json.dumps(basic_sp)
with open ('sp_test_data.json', 'w') as outfile:
    outfile.write(basic_sp_string)

sp = loadSP('sp_test_data.json', '')
sp.IK(top_plate_pos = tm([0, 0, 1.2, 0, 0, 0]))
#Delete file
os.remove('sp_test_data.json')

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,2)

goal = tm([0.25, 0.25, 1.1, np.pi/8, 0, 0])
sp.IK(top_plate_pos = goal)
DrawSP(sp, ax, 'r')
lens = sp.getLens().flatten()
result_top = sp.getTopT()
disp(result_top)
DrawAxes(sp.gtopT(), .2, ax)
sp.IK(top_plate_pos = tm([.1, 0, 1.2, np.pi/16, 0, 0]))

sp.FKSolve(lens)
#DrawSP(sp, ax, 'g')
print(lens)
print(sp.getLens().flatten())
DrawSP(sp, ax, 'r')
DrawAxes(sp.getTopT(), .3, ax)

sp.IK(top_plate_pos = tm([ 0.368977, 0.237965, -0.579408, 0.000000, 0.000000, 0.000000 ]))
#DrawSP(sp, ax)
print(result_top)
result_lens = sp.getLens().flatten()

plt.show()


result_top = sp.getTopT()
