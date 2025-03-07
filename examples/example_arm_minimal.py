import numpy as np
from basic_robotics.general import tm, fsr, Wrench
from basic_robotics.kinematics import Arm
from basic_robotics.plotting.vis_matplotlib import plt, DrawArm
from basic_robotics.utilities.disp import disp


def run_example():
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlim3d(-3,3)
    ax.set_ylim3d(-3,3)
    ax.set_zlim3d(0,6)
    Base_T = tm() # Set a transformation for the base

            # Define some link lengths
    L1 = 3
    L2 = 3
    L3 = 3
    W = 0.1
    # Define the transformations of each link
    basic_arm_end_effector_home = fsr.TAAtoTM(np.array([[L2+L3+W+W+W],[0],[L1],[0],[0],[0]]))

    basic_arm_joint_axes = np.array([[0, 0, 1],[0, 1, 0],[0, 1, 0],[1, 0, 0],[0, 1, 0],[1, 0, 0]]).conj().T
    basic_arm_joint_homes = np.array([[0, 0, 0],[0, 0, L1],[L2, 0, L1],[L2+L3, 0, L1],[L2+L3+W, 0, L1],[L2+L3+2*W, 0, L1]]).conj().T
    basic_arm_screw_list = np.zeros((6,6))

    #Create the screw list
    for i in range(0,6):
        basic_arm_screw_list[0:6,i] = np.hstack((basic_arm_joint_axes[0:3,i],np.cross(basic_arm_joint_homes[0:3,i],basic_arm_joint_axes[0:3,i])))

    #Input some basic dimensions
    basic_arm_link_box_dims = np.array([[W, W, L1],[L2, W, W],[L3, W, W],[W, W, W],[W, W, W],[W, W, W]]).conj().T

    # Create the arm from the above paramters
    arm = Arm(Base_T,basic_arm_screw_list,basic_arm_end_effector_home,basic_arm_joint_homes,basic_arm_joint_axes)

    #ALTERNATIVELY, JUST LOAD A URDF USING THE 'loadArmFromURDF' function in basic_robotics.kinematics
    arm.setJointProperties(
            np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])* -2,
            np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) * 2)
    arm.setVisColProperties(link_dimensions = basic_arm_link_box_dims)

    #Draw the arm at a couple positions
    #DrawArm(arm, ax, jdia = .3)
    goal = arm.FK(np.array([np.pi/2, np.pi/4, -np.pi/4+.1, -np.pi/6, np.pi/6, np.pi/8]))
    wrench = Wrench(np.array([3, 5, -40]), arm.getEEPos(), tm())
    torques = arm.staticForces(wrench)
    wrench_list = arm.staticForces(wrench)

    disp(torques)
    disp(wrench_list)

    DrawArm(arm, ax, jdia = .3)
    plt.show()

if __name__ == '__main__':
    run_example()