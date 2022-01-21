from basic_robotics.faser_math import tm, fsr, fmr
from basic_robotics.faser_robot_kinematics import *
from basic_robotics.faser_plotting.Draw.Draw import *

meld_platform = loadSP('../../z_MELD/meld_sp_definition.json', '')
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,2)

DrawSP(meld_platform, ax)

plt.show()
