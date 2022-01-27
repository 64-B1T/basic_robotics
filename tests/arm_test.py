from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Arm
from basic_robotics.plotting.Draw import *

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlim3d(-3,3)
ax.set_ylim3d(-3,3)
ax.set_zlim3d(0,6)

Base_T = tm() # Set a transformation for the base

# Define some link lengths
L1 = 2
L2 = 2
L3 = 2
W = 0.1
Ln = [.5 ,L1, L2, L3, W, W, W]

# Define the transformations of each joint
Tspace = [tm(np.array([[0],[0],[L1/2],[0],[0],[0]])),
    tm(np.array([[L2/2],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+(L3/2)],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+L3+(W/2)],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+L3+W+(W/2)],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+L3+W+W+(W/2)],[0],[L1],[0],[0],[0]]))]
basic_arm_end_effector_home = fsr.TAAtoTM(np.array([[L2+L3+W+W+W],[0],[L1],[0],[0],[0]]))

basic_arm_joint_axes = np.array([[0, 0, 1],[0, 1, 0],[0, 1, 0],[1, 0, 0],[0, 1, 0],[1, 0, 0]]).conj().T
basic_arm_joint_homes = np.array([[0, 0, 0],[0, 0, L1],[L2, 0, L1],[L2+L3, 0, L1],[L2+L3+W, 0, L1],[L2+L3+2*W, 0, L1]]).conj().T
basic_arm_screw_list = np.zeros((6,6))

#Create the screw list
for i in range(0,6):
    basic_arm_screw_list[0:6,i] = np.hstack((basic_arm_joint_axes[0:3,i],np.cross(basic_arm_joint_homes[0:3,i],basic_arm_joint_axes[0:3,i])))

#Input some basic dimensions
basic_arm_link_box_dims = np.array([[W, W, W],[W, W, L1],[W, W, L2],[W, W, L3],[W, W, W],[W, W, W],[W, W, W]]).conj().T
basic_arm_link_mass_transforms = [None] * (len(basic_arm_screw_list) + 1)
basic_arm_link_mass_transforms[0] = Tspace[0]

#Set mass transforms
for i in range(1,6):
    basic_arm_link_mass_transforms[i] = (Tspace[i-1].inv() @ Tspace[i])
basic_arm_link_mass_transforms[6] = (Tspace[5].inv() @ basic_arm_end_effector_home)
masses = np.array([5, 5, 5, 1, 1, 1])
basic_arm_inertia_list = np.zeros((6,6,6))

#Create spatial inertia matrices for links
for i in range(6):
    basic_arm_inertia_list[i,:,:] = fsr.boxSpatialInertia(masses[i],basic_arm_link_box_dims[0,i],basic_arm_link_box_dims[1,i],basic_arm_link_box_dims[2,i])

# Create the arm from the above paramters
arm = Arm(Base_T,basic_arm_screw_list,basic_arm_end_effector_home,basic_arm_joint_homes,basic_arm_joint_axes)

#ALTERNATIVELY, JUST LOAD A URDF USING THE 'loadArmFromURDF' function in basic_robotics.kinematics
arm.setDynamicsProperties(basic_arm_link_mass_transforms,Tspace,basic_arm_inertia_list,basic_arm_link_box_dims)
arm.jointMins = np.array([np.pi, np.pi/2, np.pi/2, np.pi, np.pi, np.pi])
arm.jointMaxs = np.array([np.pi, np.pi/2, np.pi/2, np.pi, np.pi, np.pi]) * -1

#Draw the arm at a couple positions
DrawArm(arm, ax, jdia = .3)
goal = arm.FK(np.array([np.pi/2, np.pi/4, -np.pi/4+.1, 0, 0, 0]))
DrawArm(arm, ax, jdia = .3)
plt.show()
