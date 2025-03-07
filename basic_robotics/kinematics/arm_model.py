"""
Model a Serial Arm.

Primarily uses functions from Modern Robotics (Lynch).
"""
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import scipy as sci
import scipy.integrate as integrate
import scipy.linalg as ling

from .robot_model import Robot
from ..general import fmr, fsr, tm, Wrench, Twist
from ..metrology.virtual_vision import Camera
from ..plotting.vis_matplotlib import DrawArm  # , DrawRectangle
from ..utilities.disp import disp
from .visual_info import vis_info


class Arm(Robot):
    """Models a Serial Arm."""

    def __init__(self, base_pos_global : tm, screw_list : 'np.ndarray[float]', end_effector_home : tm,
            joint_poses_home : 'list[tm]', joint_axes : 'np.ndarray[float]' = None) -> 'Arm':
        """
        Create a new Arm Instance.

        Args:
            base_pos_global (tm): Base link position of the arm.
            screw_list (np.ndarray[float]): Screw list in the global space
            end_effector_home (tm): End effector pose in global space
            joint_poses_home (list[tm]): Joint poses in global space
            joint_axes (np.ndarray[float], optional): Joint Axes list. Defaults to None.

        Returns:
            Arm: New Serial arm.
        """        
    # Variable Declarations (Pre Initialization)
        # Number of Degrees of Freedom (All Important)
        self.num_dof = np.shape(screw_list)[1]

        # Names
        super().__init__("Arm")
        self.link_names = []
        self.joint_names = []
        self.setNames()

        #Tolerances and Bounds
        self.pos_tolerance = 0.0001
        self.rot_tolerance = 0.00001
        self.joint_mins = np.ones(self.num_dof) * np.pi * -1
        self.joint_maxs = np.ones(self.num_dof) * np.pi
        self.max_vels = np.ones(self.num_dof) * np.Inf
        self.max_effort = np.ones(self.num_dof) * np.inf

        #Visual and Collision
        self._vis_props = None
        self._col_props = None
        self._link_dimensions = None

        #Origins
        self._link_homes_global = None # Home Positions of Links in Global Space
        self._joint_homes_global = None # Home Positions of Joints in Global Space
        self._prev_joints_to_next_joints = None # Origins of Joints Relative to Previous Joint
        self._eef_to_last_joint = None # Transform from End Effector to Previous Joint
        self._end_effector_home = None # Home position of end effector (Global)
        self._end_effector_home_local = None # Home position of the end effector (Local)
        self._original_end_effector_home = None #Default end effector position 
        self._fixed_base_offset = None

        #Mass Information
        self._link_masses = None
        self._link_mass_grav_centers = [] #Local Link Mass Centers Relative To Previous Joint

        #State Information
        self._theta = np.zeros(self.num_dof) # Current Theta
        self._base_pos_global = base_pos_global # Current Base Position
        self._end_effector_pos_global = end_effector_home.copy() # Current EE Position
        self._last_tau = np.zeros(self.num_dof)
        self.fail_count = 0
        self._reversable = False

        #Statics and Dynamics
        self._box_spatial_links = 0
        self.grav = np.array([0, 0, -9.81])

        #Other
        self.cameras = []

        if joint_axes is not None:
            self._reversable = True
            self.reversed = False
            self.joint_axes = joint_axes
            self.original_joint_axes = joint_axes
    # Initialization
        self.screw_list_body = np.zeros((6, self.num_dof))
        self.initialize(base_pos_global, screw_list, end_effector_home, joint_poses_home)

        for i in range(0, self.num_dof):
            self.screw_list_body[:, i] = (
                fmr.Adjoint(self._end_effector_home.inv().gTM()) @
                self.screw_list[:, i])
    # Backups
        self.original_screw_list = screw_list.copy()
        self.FK(np.zeros((self.num_dof)))

    def initialize(self, base_pos_global : 'tm', screw_list : 'np.ndarray[float]', 
            end_effector_home : 'tm', joint_poses_home : 'list[tm]') -> None:
        """
        Initialize internal elements of the serial arm.

        Used when creating a new arm, and when moving it from one location to another.

        Args:
            base_pos_global (tm): Base location of the arm.
            screw_list (np.ndarray[float]): Screw list in the global space.
            end_effector_home (tm): End effector home in global space.
            joint_poses_home (list[tm]): Joint poses in the global space. 
        """        
        self.screw_list = screw_list
        self.original_screw_list_body = np.copy(screw_list)
        if self._joint_homes_global is not None:
            for i in range((self.num_dof)):
                base_to_link = fsr.globalToLocal(self._base_pos_global, self._joint_homes_global[i])
                new_global = fsr.localToGlobal(base_pos_global, base_to_link)
                self._joint_homes_global[i] = new_global
        else:
            self._joint_homes_global = [tm()]
            for i in range(joint_poses_home.shape[1]):
                self._joint_homes_global.append(
                        tm([joint_poses_home[0][i],
                            joint_poses_home[1][i],
                            joint_poses_home[2][i], 0, 0, 0]))
            self._joint_homes_global = self._joint_homes_global[1:]
        self.original_joint_poses_home = joint_poses_home
        self.joint_poses_home = np.zeros((3, self.num_dof))
        if joint_poses_home.size > 1:
            for i in range(0, self.num_dof):
                self.joint_poses_home[0:3, i] = fsr.transformByVector(
                        base_pos_global, joint_poses_home[0:3, i])
                #Convert transformByVector
        for i in range(0, self.num_dof):
            self.screw_list[:, i] = fmr.Adjoint(base_pos_global.gTM()) @ screw_list[:, i]
            if joint_poses_home.size <= 1:
                _, _, joint_pose_temp, _ = fsr.twistToScrew(self.screw_list[:, i])
                #Convert TwistToScrew
                self.joint_poses_home[0:3, i] = joint_pose_temp # For plotting purposes
        self._end_effector_home_local = end_effector_home
        self._end_effector_home = base_pos_global @ end_effector_home
        self._helper_determine_eef_to_last_joint()
        self._end_effector_pos_global = self._end_effector_home.copy()
        self._original_end_effector_home = self._end_effector_home.copy()
        self._base_pos_global = base_pos_global.copy()

    """
    Kinematics
    """

    def thetaProtector(self, theta : 'np.ndarray[float]') -> 'np.ndarray[float]':
        """
        Ensure thetas are within joint mins and maxes.

        Args:
            theta (np.ndarray[float]): Input thetas.

        Returns:
            np.ndarray[float]: Output protected thetas.
        """        
        theta_len = len(theta)
        if (np.any(theta[0:theta_len] < self.joint_mins[0:theta_len]) or
                np.any(theta[0:theta_len] > self.joint_maxs[0:theta_len])):
            theta[np.where(theta<self.joint_mins[0:theta_len])] = (
                    self.joint_mins[np.where(theta<self.joint_mins[0:theta_len])])
            theta[np.where(theta>self.joint_maxs[0:theta_len])] = (
                    self.joint_maxs[np.where(theta>self.joint_maxs[0:theta_len])])
        return theta

    #Converted to python -Liam
    def FK(self, theta : 'np.ndarray[float]', protect : bool = False) -> tm:
        """
        Perform Forward Kinematics, yielding an end effector transform from joint configuration.

        Updates internal arm state.
        Args:
            theta (np.ndarray[float]): joint configuration
            protect (bool, optional): disable error checking.. Defaults to False.

        Returns:
            tm: end effector configuration.
        """        
        if theta is None:
            return self.getEEPos()
        if not protect:
            theta = self.thetaProtector(theta)
        self._theta = fsr.angleMod(theta.reshape(len(theta)))
        end_effector_transform = tm(fmr.FKinSpace(
            self._end_effector_home.gTM(), self.screw_list, theta))
        self._end_effector_pos_global = end_effector_transform
        return end_effector_transform

    #Converted to python - Liam
    def FKLink(self, theta : 'np.ndarray[float]', i : int, protect : bool = False) -> tm:
        """
        Get a given link origin pose in global space for a given theta configuration.

        Args:
            theta (np.ndarray[float]): joint configuration
            i (int): link index
            protect (bool, optional): Disable error checking. Defaults to False.

        Returns:
            tm: Link transform.
        """        
        # Returns the TM of link i
        # Lynch 4.1
        if not protect:
            theta = self.thetaProtector(theta)
        end_effector_pos =  tm(fmr.FKinSpace(self._link_homes_global[i].TM,
            self.screw_list[0:6, 0:i], theta[0:i+1]))
        return end_effector_pos

    def FKJoint(self, theta : 'np.ndarray[float]', i : int, protect : bool = False) -> tm:
        """
        Get a given joint origin pose in global space for a given theta configuration.

        Args:
            theta (np.ndarray[float]): joint configuration
            i (int): joint index
            protect (bool, optional): Disable error checking. Defaults to False.

        Returns:
            tm: joint transform.
        """        
        # Returns the TM of link i
        # Lynch 4.1
        if not protect:
            theta = self.thetaProtector(theta)
        if i == self.num_dof - 1:
            jh = self._end_effector_home.TM
        else:
            jh = self._joint_homes_global[i].TM
        end_effector_pos = tm(fmr.FKinSpace(jh,
            self.screw_list[0:6, 0:i+1], theta[0:i+1]))
        return end_effector_pos

    #Converted to python - Liam
    def IK(self, goal_position : tm, theta_init : 'np.ndarray[float]' = None, 
            check : bool = True, level : int = 6, max_iters : int = 30, 
            protect : bool = False): 
        """
        Calculate joint configuration of an arm to reach a desired end effector configuration.

        There may be more than one appropriate configuration depending on the arm.
        Updates internal arm state.
        Args:
            goal_position (tm): Goal Position to reach.
            theta_init (np.ndarray[float], optional): Initial Guess Thetas. Defaults to None/use current thetas.
            check (bool, optional): If failure, try again with a different guess. Defaults to True.
            level (int, optional): Maximum number of guesses. Defaults to 6.
            max_iters (int, optional): Maximum number of iterations per attempt. Defaults to 30.
            protect: (bool, optional): Disable error checking. Defaults to False.
        Returns:
            theta_list (np.ndarray[float]): solution thetas
            success (bool): success of the operation.
        """        
        if theta_init is None:
            theta_init = fsr.angleMod(self._theta.reshape(len(self._theta)))
        if not protect:
            return self.constrainedIK(goal_position, theta_init, check, level, max_iters)
        theta, success = fmr.IKinSpace(
                self.screw_list, self._end_effector_home.gTM(),
                goal_position.gTM(), theta_init,
                self.pos_tolerance, self.rot_tolerance, max_iters=max_iters)
        theta = fsr.angleMod(theta)
        self._theta = theta
        if success:
            self._end_effector_pos_global = goal_position
        else:
            if check:
                i = 0
                while i < level and success == 0:
                    theta_temp = np.zeros((len(self._theta)))
                    for j in range(len(theta_temp)):
                        theta_temp[j] = random.uniform(-np.pi, np.pi)
                    theta, success = fmr.IKinSpace(
                            self.screw_list, self._end_effector_home.gTM(),
                            goal_position.gTM(), theta_init,
                            self.pos_tolerance, self.rot_tolerance, max_iters=max_iters)
                    i = i + 1
                if success:
                    self._end_effector_pos_global = goal_position
        return theta, success

    def constrainedIK(self, goal_position : tm, theta_init : 'np.ndarray[float]' = None,
            check : bool = True, level : int = 6,
            max_iters : int= 30): 
        """
        Calculate joint positions of a serial arm, provided rotational constraints on the Joints.

        All parameters are optional except the desired end effector position
        Joint constraints are set through the joint_maxs and joint_mins properties, and should be
        arrays the same size as the number of DOFS

        Args:
            goal_position (tm): Goal Position to reach.
            theta_init (np.ndarray[float], optional): Initial Guess Thetas. Defaults to None/use current thetas.
            check (bool, optional): If failure, try again with a different guess. Defaults to True.
            level (int, optional): Maximum number of guesses. Defaults to 6.
            max_iters (int, optional): Maximum number of iterations per attempt. Defaults to 30.

        Returns:
            theta_list (np.ndarray[float]): solution thetas
            success (bool): success of the operation.
        """        
        if not isinstance(goal_position, tm):
            print(goal_position)
            print('Attempted pass ^')
            return self._theta
        screw_list = self.screw_list.copy()
        end_effector_home_c = self._end_effector_home.copy()
        if theta_init is None:
            theta_init = self._theta.copy()
        i = 0

        theta_list, success = fmr.IKinSpaceConstrained(
                screw_list, end_effector_home_c.gTM(), goal_position.gTM(),
                theta_init, self.pos_tolerance, self.rot_tolerance,
                self.joint_mins, self.joint_maxs, max_iters)

        if success:
            self._end_effector_pos_global = goal_position
        else:
            if check:
                i = 0
                self.fail_count = 0
                while i < level and success == 0:
                    theta_temp = np.zeros((len(self._theta)))
                    for j in range(len(theta_temp)):
                        theta_temp[j] = random.uniform(self.joint_mins[j], self.joint_maxs[j])
                    try:
                        theta_list, success = fmr.IKinSpaceConstrained(
                                screw_list, end_effector_home_c.gTM(),
                                goal_position.gTM(),
                                theta_temp, self.pos_tolerance,
                                self.rot_tolerance, self.joint_mins,
                                self.joint_maxs, max_iters)
                    except Exception as e:
                        theta_list, success = self.constrainedIK(
                                goal_position, theta_temp, check=False)
                        disp('FMR Failure: ' + str(e))
                    i = i + 1
                if success:
                    self._end_effector_pos_global = goal_position
        if not success:
            if not check:
                self.fail_count += 1
            else:
                # print('Total Cycle Failure')
                self.FK(np.zeros(len(self._theta)))
        else:
            if self.fail_count != 0:
                print('Success + ' + str(self.fail_count) + ' failures')
            self.FK(theta_list)

        return theta_list, success

    #def IKForceOptimal(self, T, theta_init, forcev, random_sample = 1000, mode = 'MAX'):
    #    """
    #    Early attempt at creating a force optimization package for a serial arm.
    #    Absolutely NOT the optimial way to do this. Only works for overactuated arms.
    #    Args:
    #        T: Desired end effector position to calculate for
    #        theta_init: Intial theta guess for desired end effector position.
    #            Set to 0s if not provided.
    #        forcev: Force applied to the end effector of the arm. Wrench.
    #        random_sample: number of samples to test to look for the most optimal solution
    #        mode: Set Mode to reduce. Max: Max Force. Sum: Sum of forces. Mean: Mean Force
    #    Returns:
    #        List of thetas
    #    """
    #    thetas = []
    #    for i in range(random_sample):
    #        theta_temp = np.zeros((len(self._theta)))
    #        for j in range(len(theta_init)):
    #            theta_temp[j] = random.uniform(-np.pi, np.pi)
    #        thetas.append(theta_temp)
    #    force_thetas = []
    #    temp_moment = np.cross(T[0:3].reshape((3)), forcev)
    #    wrench = np.array([temp_moment[0], temp_moment[1],
    #        temp_moment[2], forcev[0], forcev[1], forcev[2]]).reshape((6, 1))
    #    for i in range(len(thetas)):
    #        candidate_theta, success =self.IK(T, thetas[i])
    #        if success and sum(abs(fsr.poseError(self.FK(candidate_theta), T))) < .0001:
    #            force_thetas.append(candidate_theta)
    #    max_force = []
    #    for i in range(len(force_thetas)):
    #        if mode == 'MAX':
    #            max_force.append(max(abs(self.staticForces(force_thetas[i], wrench))))
    #        elif mode == 'SUM':
    #            max_force.append(sum(abs(self.staticForces(force_thetas[i], wrench))))
    #        elif mode == 'MEAN':
    #            max_force.append(sum(abs(
    #                self.staticForces(force_thetas[i], wrench))) / len(force_thetas))
    #    index = max_force.index(min(max_force))
    #    self._theta = force_thetas[index]
    #    return force_thetas[index]

    #def IKMotion(self, T, theta_init):
    #    """
    #    This calculates IK by numerically moving the end effector
    #    from the pose defined by theta_init in the direction of the desired
    #    pose T.  If the pose cannot be reached, it gets as close as
    #    it can.  This can sometimes return a better result than standard IK.
    #    An approximate explanation an be found in Lynch 9.2
    #    Args:
    #        T: Desired end effector position to calculate for
    #        theta_init: Intial theta guess for desired end effector position.
    #    Returns:
    #        theta: list of theta lists
    #        success: boolean for success
    #        t: goal
    #        thall: integration results
    #    """
    #    #This function does not work at all, and is based on matlabs
    #    dt = 1e-5
    #    theta_init = theta_init.flatten()
    #    start_transform = self.FK(theta_init)
    #    start_direction = T @ start_transform.inv()
    #    twist_direction = fsr.twistFromTransform(start_direction)
    #    jac = lambda t, x : self.jacobian(x)
    #    res = lambda t, x : np.linalg.pinv(self.jacobian(x)) @ twist_direction
    #    disp(res(0, theta_init), 'res?')
    #    t = integrate.solve_ivp(res, [0.0, 1.0], theta_init, first_step=0.1, method='Radau')
    #    print(t.sol)
    #    return t.sol

    def IKFree(self, goal_position : tm, theta_init : 'np.ndarray[float]', 
            inds):
        """
        Perform IK but only allow 'free_inds' to be varied.

        Method not covered in Lynch.
        SetElements inserts the variable vector x into the positions
        indicated by freeinds in theta_init.  The remaining elements are
        unchanged.
        This function is sensitive to initial conditions.
        Args:
            goal_position: Desired end effector position to calculate for
            theta_init: Intial theta guess for desired end effector position.
            inds: Free indexes to move
        Returns:
            theta: list of theta lists
            success: boolean for success
        """
        #free_thetas = fsolve(@(x)(obj.FK(SetElements(theta_init,
            #freeinds, x))-T), theta_init(freeinds))
        res = lambda x : (self.FK(fsr.setElements(theta_init, inds, x)) - goal_position).gTAA().flatten()
        #Use newton_krylov instead of fsolve
        solver_result = sci.optimize.root(res, theta_init[inds], method='lm')
        # free_thetas = fsolve(@(x)(self.FK(SetElements(theta_init,
            #freeinds, x))-T), theta_init(freeinds));
        free_thetas = solver_result.x
        theta = np.squeeze(theta_init)
        theta[inds] = np.squeeze(free_thetas)
        if fmr.Norm6((goal_position - self.FK(theta))[0:6]) < 0.001:
            return (theta, True)
        return (theta, False)


    """
    Kinematics Helpers
    """

    def randomPos(self) -> tm:
        """
        Generate a random position.

        Returns:
            tm: Randomly assigned configuration of the end effector.
        """       
        theta_temp = np.zeros((len(self._theta)))
        for j in range(len(theta_temp)):
            theta_temp[j] = random.uniform(self.joint_mins[j], self.joint_maxs[j])
        pos = self.FK(theta_temp)
        return pos


    #def reverse(self):
    #    """
    #    Flip around the serial arm so that the end effector is not the base and vice versa.
    #    Keep the same end pose
    #    """
    #    	old_thetas = np.copy(self._theta)
    #    new_theta = np.zeros((len(self._theta)))
    #    for i in range(self.num_dof):
    #        new_theta[i] = old_thetas[len(old_thetas) - 1 - i]
    #    new_screw_list = np.copy(self.original_screw_list)
    #    new_end_effector_home = self._end_effector_home.copy()
    #    new_thetas = self.FK(self._theta)
    #    new_joint_axes = np.copy(self.joint_axes)
    #    new_joint_poses_home = np.copy(self.original_joint_poses_home)
    #    for i in range(new_joint_axes.shape[1]):
    #        new_joint_axes[0:3, i] = self.joint_axes[0:3, new_joint_axes.shape[1] - 1 - i]
    #    differences = np.zeros((3, new_joint_poses_home.shape[1]-1))
    #    for i in range(new_joint_poses_home.shape[1]-1):
    #        differences[0:3, i] = (self.original_joint_poses_home[0:3,(
    #            self.original_joint_poses_home.shape[1] - 1 - i)] -
    #            self.original_joint_poses_home[0:3,(
    #            self.original_joint_poses_home.shape[1] - 2 - i)])
    #    #print(differences, 'differences')
    #    for i in range(new_joint_poses_home.shape[1]):
    #        if i == 0:
    #            new_joint_poses_home[0:3, i] = (self.original_joint_poses_home[0:3, (
    #                self.original_joint_poses_home.shape[1] - 1)] - np.sum(differences, axis = 1))
    #        else:
    #            new_joint_poses_home[0:3, i] = (new_joint_poses_home[0:3, i -1] +
    #                differences[0:3, i - 1])
    #    for i in range(self.num_dof):
    #        new_screw_list[0:6, i] = np.hstack((new_joint_axes[0:3, i],
    #            np.cross(new_joint_poses_home[0:3, i], new_joint_axes[0:3, i])))
    #    new_thetas = (new_thetas @
    #        tm([0, 0, 0, 0, np.pi, 0]) @ tm([0, 0, 0, 0, 0, np.pi]))
    #    if np.size(self._link_dimensions) != 1:
    #        new_link_dimensions = np.zeros((self._link_dimensions.shape))
    #        for i in range(self._link_dimensions.shape[1]):
    #            new_link_dimensions[0:3, i] = (
    #                self._link_dimensions[0:3,(self._link_dimensions.shape[1] - i -1)])
    #        self._link_dimensions = new_link_dimensions
    #    if len(self._joint_homes_global) != 1:
    #        new_joint_homes_global = [None] * len(self._joint_homes_global)
    #        for i in range(len(new_joint_homes_global)):
    #            new_joint_homes_global[i] = (
    #                self._joint_homes_global[len(new_joint_homes_global) - i -1])
    #        self._joint_homes_global = new_joint_homes_global
    #    self.screw_list = new_screw_list
    #    self.original_screw_list = np.copy(new_screw_list)
    #    #print(self._base_pos_global, '')
    #    new_end_effector_home = new_thetas @ self._end_effector_home_local
    #    self._base_pos_global = new_thetas
    #    self.original_joint_poses_home = new_joint_poses_home
    #    self.joint_poses_home = np.zeros((3, self.num_dof))
    #    self.screw_list_body = np.zeros((6, self.num_dof))
    #    if new_joint_poses_home.size > 1:
    #        for i in range(0, self.num_dof):
    #            self.joint_poses_home[0:3, i] = fsr.transformByVector(new_thetas,
    #                new_joint_poses_home[0:3, i])
    #            #Convert transformByVector
    #    for i in range(0, self.num_dof):
    #        self.screw_list[:, i] = fmr.Adjoint(new_thetas.gTM()) @ new_screw_list[:, i]
    #        if new_joint_poses_home.size <= 1:
    #            [w, th, joint_pose_temp, h] = fsr.twistToScrew(self.screw_list[:, i])
    #            #Convert TwistToScrew
    #            self.joint_poses_home[0:3, i] = joint_pose_temp; # For plotting purposes
    #    self._end_effector_home = new_end_effector_home
    #    self.original_end_effector_home = self._end_effector_home.copy()
    #    if len(self._joint_homes_global) != 1:
    #        new_link_mass_grav_centers = [None] * len(self._joint_homes_global) # Merged into link_mass_grav_centers
    #        new_link_mass_grav_centers[0] = self._joint_homes_global[0] # Merged into link_mass_grav_centers
    #        for i in range(1, 6):
    #            new_link_mass_grav_centers[i] = ( # Merged into link_mass_grav_centers
    #                self._joint_homes_global[i-1].inv() @ self._joint_homes_global[i])
    #        new_link_mass_grav_centers[len(self._joint_homes_global) -1] = ( # Merged into link_mass_grav_centers
    #            self._joint_homes_global[5].inv() @ self._end_effector_home)
    #        self._link_mass_grav_centers = new_link_mass_grav_centers # Merged into link_mass_grav_centers
    #    self._box_spatial_links = 0
    #    for i in range(0, self.num_dof):
    #        self.screw_list_body[:, i] = (
    #            fmr.Adjoint(self._end_effector_home.inv().gTM()) @ self.screw_list[:, i])
    #    #print(new_theta)
    #    self.FK(new_theta)

    """
    Motion Planning
    """

    def lineTrajectory(self, target : tm, initial : tm = None, 
            execute  : bool = True, delt : float = .01):
        """
        Create a trajectory moving the end effector in a straight line torwards the target.

        Args:
            target (tm): Target configuration
            initial (tm, optional): Initial Pose. Defaults to None/Current Pose.
            execute (bool, optional): Execute motion, or remain at current config. Defaults to True.
            delt (float, optional): Distance of end effector betweene ach pose in meters. Defaults to .01.

        Returns:
            theta_list (list[np.ndarray[float]]): List of theta configurations.
        """
        if initial is None:
            initial = self._end_effector_pos_global.copy()
        satisfied = False
        init_theta = np.copy(self._theta)
        theta_list = []
        count = 0
        while not satisfied and count < 2500:
            count+=1
            error = fsr.poseError(target, initial).gTAA().flatten()
            satisfied = True
            if (np.any(error[0:3] > self.pos_tolerance) or
                    np.any(error[3:6] > self.rot_tolerance)):
                satisfied = False
            initial = fsr.closeLinearGap(initial, target, delt)
            theta_list.append(np.copy(self._theta))
            self.IK(initial, self._theta)
        self.IK(target, self._theta)
        theta_list.append(self._theta)
        if (execute == False):
            self.FK(init_theta)
        return theta_list


    def visualServoToTarget(self, target : tm, pixel_tol : int = 2, desired_dist : float = 1.0,
            pose_delta : float = 0.1, pose_tol : float = 0.2, max_iter: int = 1000,
            cam_ind : int = 0):
        """
        Perform visual servoing to a target using virtual cameras.

        Args:
            target (tm): Object of interest to move towards.
            pixel_tol (int, optional): Pixel Tolerance. Defaults to 2.
            desired_dist (float, optional): Desired distance to object in meters. Defaults to 1.0.
            pose_delta (float, optional): Distance in meters to record between poses at end effector. Defaults to 0.1.
            pose_tol (float, optional): Pose tolarance in meters. Defaults to 0.2.
            max_iter (int, optional): Maximum iterations before giving up. Defaults to 1000.
            cam_ind (int, optional): Virtual camera to use. Defaults to 0.

        Returns:
            theta_list (np.ndarray[Float]): final thetas
            thetas_list (list[np.ndarray[float]]): list of all theta configurations along trajectory.
        """               
        if (len(self.cameras) == 0):
            print('NO CAMERA CONNECTED')
            return self._theta, []
        at_target = False
        done = False
        start_pos = self.FK(self._theta)
        theta = 0
        j = 0
        theta_list = []
        while not (at_target and done ):
            pose_adjust = tm()
            at_target = True
            done = True
            img, _, suc = self.cameras[cam_ind][0].getPhoto(target)
            if not suc:
                print('Failed to locate Target')
                return self._theta, []
            if img[0] < self.cameras[cam_ind][2][0] - pixel_tol:
                pose_adjust[0] = -pose_delta
                at_target = False
            if img[0] > self.cameras[cam_ind][2][0] + pixel_tol:
                pose_adjust[0] = pose_delta
                at_target = False
            if img[1] < self.cameras[cam_ind][2][1] - pixel_tol:
                pose_adjust[1] = -pose_delta
                at_target = False
            if img[1] > self.cameras[cam_ind][2][1] + pixel_tol:
                pose_adjust[1] = pose_delta
                at_target = False
            if at_target:
                d = fsr.distance(self._end_effector_pos_global, target)
                #print(d)
                if d < desired_dist - pose_tol:
                    done = False
                    pose_adjust[2] = -.01
                if d > desired_dist + pose_tol:
                    done = False
                    pose_adjust[2] = .01
            start_pos =start_pos @ pose_adjust
            theta = self.IK(start_pos, self._theta)
            theta_list.append(theta)
            self.updateCams()
            j = j + 1
            if j > max_iter:
                print('Failed to find solution, max iterations')
                return self._theta, []
        return theta, theta_list

    def PDControlToGoalEE(self, goal_position : tm, theta : 'np.ndarray[float]' = None,
            prev_theta : 'np.ndarray[float]' = None, p_gain : float = 100.0,
            d_gain : float = 100.0, max_theta_dot : float = 0.1) -> 'np.ndarray[float]':
        """
        Use PD Control to move to the goal position.

        Args:
            goal_position (tm): Goal Position
            theta (np.ndarray[float], optional): Initial Theta. Defaults to None/Current Position.
            prev_theta (np.ndarray[float], optional): previous theta. Defaults to None/Dead stop.
            p_gain (float, optional): Proportional Gain. Defaults to 100.0.
            d_gain (float, optional): Derivative Gain. Defaults to 100.0.
            max_theta_dot (float, optional): Max joint velocity. Defaults to 0.1.

        Returns:
            np.ndarray[float]: Scaled joint velocities.
        """        
        if prev_theta is None:
            prev_theta = np.zeros(self.num_dof)
        backup_theta = self._theta.copy()
        current_end_effector_pos = self.FK(theta)
        previous_end_effector_pos = self.FK(prev_theta)
        error_ee_to_goal = fmr.Norm(current_end_effector_pos[0:3]-goal_position[0:3])
        delt_distance_to_goal = (error_ee_to_goal-
            fmr.Norm(previous_end_effector_pos[0:3]-goal_position[0:3]))
        scale = p_gain * error_ee_to_goal + d_gain * min(0, delt_distance_to_goal)
        #twist = fsr.twistToGoal(current_end_effector_pos, goal_position)
        twist = fsr.twistToGoal(current_end_effector_pos, goal_position)
        normalized_twist = fsr.normalizeTwist(twist)
        theta_dot = np.linalg.pinv(self.jacobian(theta)) @ normalized_twist
        scaled_theta_dot = max_theta_dot/max(abs(theta_dot)) * theta_dot * scale
        self.FK(backup_theta)
        return scaled_theta_dot

    """
    Getters and Setters
    """

    def setNames(self, arm_name : str = None,
            link_names = None,
            joint_names = None) -> None:
        """
        Set names for Arm elements.

        Will use generics if not specified.
        Args:
            arm_name (str, optional): Arm name. Defaults to None.
            link_names (list[str], optional): Names for each link. Defaults to None.
            joint_names (list[str], optional): Names for each joint. Defaults to None.
        """        
        if arm_name is not None and self.name != "Arm":
            self.name = arm_name
        if link_names is not None:
            self.link_names = link_names
        elif self.link_names == []:
            for i in range(self.num_dof):
                self.link_names.append('link' + str(i))
            self.link_names.append('end_effector')
        if joint_names is not None:
            self.joint_names = joint_names
        elif self.joint_names == []:
            for i in range(self.num_dof):
                self.joint_names.append('joint' + str(i))

    def setJointProperties(self, joint_mins : 'np.ndarray[float]' = None,
            joint_maxs : 'np.ndarray[float]' = None,
            max_vels : 'np.ndarray[float]'= None,
            max_effort : 'np.ndarray[float]'= None) -> None:
        """
        Set joint properties for the arm.

        Args:
            joint_mins (np.ndarray[float], optional): Joint minimum rotations. Defaults to None.
            joint_maxs (np.ndarray[float], optional): Joint maximum rotations. Defaults to None.
            max_vels (np.ndarray[float], optional): Joint maximum velocities. Defaults to None.
            max_effort (np.ndarray[float], optional): Joint maximum efforts/torques. Defaults to None.
        """ 
        if joint_mins is not None:
            self.joint_mins = joint_mins
        if joint_maxs is not None:
            self.joint_maxs = joint_maxs
        if max_vels is not None:
            self.max_vels = max_vels
        if max_effort is not None:
            self.max_effort = max_effort

    def setVisColProperties(self, vis_props : list = None,
            col_props : list = None,
            link_dimensions : 'np.ndarray[float]' = None) -> None:
        """
        Set visual and collision properties for the arm.

        All arguments are optional.
        Args:
            vis_props (list, optional): Visual Properties. Defaults to None.
            col_props (list, optional): Collision Properties. Defaults to None.
            link_dimensions (np.ndarray[float], optional): Link Box Dimensions. Defaults to None.
        """        
        if vis_props is not None:
            self._vis_props = vis_props
        if col_props is not None:
            self._col_props = col_props
        if link_dimensions is not None:
            self._link_dimensions = link_dimensions

    def setOrigins(self, prev_joints_to_next_joints = None,
            joint_homes_global = None,
            link_homes_global = None,
            eef_to_last_joint : tm = None,
            base_to_fixed_joint : tm = None
            ) -> None:
        """
        Set origins of joints and links for the arm.

        Args:
            prev_joints_to_next_joints (list[tm], optional): Local transforms from one joint to the next. Defaults to None.
            joint_homes_global (list[tm], optional): Joint positions in global space. Defaults to None.
            link_homes_global (list[tm], optional): Link centers in global space. Defaults to None.
            eef_to_last_joint (tm, optional): End effector transform to the last joint. Defaults to None.
            base_to_fixed_joint (tm, optional): offset from arm base to actual base. Defaults to None.
        """        
        if prev_joints_to_next_joints is not None:
            self._prev_joints_to_next_joints = prev_joints_to_next_joints
        if joint_homes_global is not None:
            self._joint_homes_global = joint_homes_global
        if link_homes_global is not None:
            self._link_homes_global = link_homes_global
        if eef_to_last_joint is not None:
            self._eef_to_last_joint = eef_to_last_joint
        if base_to_fixed_joint is not None: 
            self._fixed_base_offset = base_to_fixed_joint

    def setMassProperties(self, link_masses : 'np.ndarray[float]' = None,
            mass_grav_centers = None,
            box_spatial_links : 'np.ndarray[float]' = None) -> None:
        """
        Set mass properties for the arm.

        Args:
            link_masses (np.ndarray[float], optional): Link masses in Kg. Defaults to None.
            mass_grav_centers (list[tm], optional): Link centers of mass in local frame. Defaults to None.
            box_spatial_links (np.ndarray[float], optional): Box spatial inertia. Defaults to None.
        """        
        if link_masses is not None:
            self._link_masses = link_masses
        if mass_grav_centers is not None:
            self._link_mass_grav_centers = mass_grav_centers
        if self._link_mass_grav_centers is None and self._link_masses is not None:
            self._link_mass_grav_centers = [tm() for x in self._link_masses]
        if box_spatial_links is not None:
            self._box_spatial_links = box_spatial_links

    def testArmValues(self) -> None:   # pragma: no cover
        """Print a detailed description of the arm."""
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        print('S')
        print(self.screw_list, title = 'screw_list')
        print('screw_list_body')
        print(self.screw_list_body, title = 'screw_list_body')
        print('Q')
        print(self.joint_poses_home, title = 'joint_poses_home list')
        print('end_effector_home')
        print(self._end_effector_home, title = 'end_effector_home')
        print('_original_end_effector_home')
        print(self._original_end_effector_home, title = 'original_end_effector_home')
        print('_Mlinks')
        print(self._link_mass_grav_centers, title = 'Link Masses')
        print('_Mhome')
        print(self._link_homes_global, title = '_Mhome')
        print('_Glinks')
        print(self._box_spatial_links, title = '_Glinks')
        print('_dimensions')
        print(self._link_dimensions, title = 'Dimensions')


    def getJointTransforms(self, return_base : bool = True):
        """
        Return joint information for each link in the arm, including the base.

        If the end effector is not coincident with the last joint of the arm,
        returns end effector also, even though it is not a true joint
        Base return behavior can be disabled by setting 'return_base' to false

        Args:
            return_base (bool, optional): Return the base or not. Defaults to True.

        Returns:
            list[tm]: list of each joint location.
        """        
        if return_base:
            if self._fixed_base_offset is not None:
                poses = [self._base_pos_global.copy() @ self._fixed_base_offset]
            else:
                poses = [self._base_pos_global.copy()]
        else:
            poses = []
        for i in range((self.num_dof)):
            poses.append(self.FKJoint(self._theta, i))
        if self._eef_to_last_joint is not None:
            poses.insert(-1, poses[-1] @ self._eef_to_last_joint)
        return poses

    def setArbitraryHome(self, new_home_global : tm, theta : 'np.ndarray[float]' = None) -> None:
        """
        Set a new end effector position relative to last joint.

        Args:
            new_home_global (tm): New relative home position
            theta (np.ndarray[float], optional): configuration for joints. Defaults to None.
        """        
        end_effector_temp = self.FK(theta)
        old_to_new = fsr.globalToLocal(end_effector_temp, new_home_global)
        new_home = fsr.localToGlobal(self._end_effector_home, old_to_new)
        self._end_effector_home = new_home
        self._helper_determine_eef_to_last_joint()

    #Converted to Python - Joshua
    def restoreOriginalEE(self) -> None:
        """Restore the original End effector configuration of the arm."""
        self._end_effector_home = self._original_end_effector_home
        self._helper_determine_eef_to_last_joint()

    def getScrewList(self) -> 'np.ndarray[float]':
        """
        Get the space frame screw list for the arm.

        Returns:
            np.ndarray[float]: space frame screw list.
        """        
        return self.screw_list.copy()

    def getLinkDimensions(self) -> 'np.ndarray[float]':
        """
        Get box link dimensions for the arm.

        Returns:
            np.ndarray[float]: Box link dimensions.
        """        
        return self._link_dimensions.copy()

    """
    Forces and Dynamics
    """

    def staticForcesWithCrossMoments(self, end_effector_wrench : Wrench = Wrench(), 
            theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Calculate joint wrenches given an end effector wrench.

        Args:
            end_effector_wrench (Wrench, optional): End effector wrench. Defaults to None.
            theta (np.ndarray[float], optional): Joint configuration. Defaults to None.

        Returns:
            list[wrench] wrenches at each joint
        """  

        # Given Jacobian matrix (6x6)
        J = np.random.rand(6, 6)

        # Given end effector wrench (6x1)
        F = np.random.rand(6, 1)

        # Initialize arrays to store forces and moments at each joint
        joint_forces = np.zeros((6, 3))
        joint_moments = np.zeros((6, 3))

        # Initialize total wrench at each joint to zero
        total_wrench = np.zeros((6, 1))

        # Iterate through joints starting from the base
        for i in range(6):
            # Calculate the total wrench at this joint
            total_wrench[i] = np.dot(J[:, i].T, F).flatten()
            
            # Calculate forces and moments at this joint
            joint_forces[i] = total_wrench[i, :3]
            joint_moments[i] = total_wrench[i, 3:]

            # Update end effector wrench by subtracting the contribution from this joint
            F -= np.dot(J[:, i].reshape(6, 1), total_wrench[i].reshape(1, 1))

        # Print forces and moments for each joint
        for i in range(6):
            print(f"Joint {i+1} - Forces (X, Y, Z): {joint_forces[i]} - Moments (X, Y, Z): {joint_moments[i]}")




    def staticForcesWithLinkMasses(self, end_effector_wrench : Wrench = Wrench(), 
            theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Calculate joint torques given end effector wrench, and all joint masses.

        Args:
            end_effector_wrench (Wrench, optional): End effector wrench. Defaults to None.
            theta (np.ndarray[float], optional): Joint configuration. Defaults to None.

        Returns:
            np.ndarray[float]: Joint torques of the arm.
        """        
        if theta is not None:
            self.FK(theta)
        jacobian = self.jacobian(theta)
        tau_init = jacobian.T @ end_effector_wrench
        carry_wrench = end_effector_wrench
        joint_poses = self.getJointTransforms()
        for i in range(self.num_dof, 0, -1):
            link_mass_cg = self._link_mass_grav_centers[i]
            link_mass = self._link_masses[i]
            applied_pos_global = joint_poses[i] @ link_mass_cg
            carry_wrench = carry_wrench + fsr.makeWrench(applied_pos_global, link_mass, self.grav)
            tau = jacobian[0:6, 0:i].T @ carry_wrench
            tau_init[i-1] = tau[-1]
        return tau_init

    def inverseDynamicsEMR(self, theta : 'np.ndarray[float]', theta_dot : 'np.ndarray[float]',
            theta_dot_dot : 'np.ndarray[float]', grav : 'np.ndarray[float]' = None, 
            end_effector_wrench : Wrench = Wrench()) -> 'np.ndarray[float]': 
        """
        Inverse Dynamics using Modern Robotics Formulation.

        Args:
            theta (np.ndarray[float]): Joint thetas.
            theta_dot (np.ndarray[float]): Joint velocities.
            theta_dot_dot (np.ndarray[float]): Joint accelerations
            grav (np.ndarray[float], optional): Gravity vector. Defaults to None.
            end_effector_wrench (Wrench, optional): End effector wrench. Defaults to None.

        Returns:
            np.ndarray[float]: Joint torque list (tau)
        """
        if grav is None:
            grav = self.grav
        # Merged into link_mass_grav_centers
        link_mass_array = np.array([x.gTM() for x in self._link_mass_grav_centers])
        return fmr.InverseDynamics(theta, theta_dot, theta_dot_dot, grav, end_effector_wrench,
            link_mass_array, self._box_spatial_links, self.screw_list)

    def inverseDynamics(self, theta : 'np.ndarray[float]', theta_dot : 'np.ndarray[float]',
            theta_dot_dot : 'np.ndarray[float]', grav : 'np.ndarray[float]' = None, 
            end_effector_wrench : Wrench = Wrench()):
        """
        Inverse Dynamics using Modern Robotics Formulation.

        Args:
            theta (np.ndarray[float]): Joint thetas.
            theta_dot (np.ndarray[float]): Joint velocities.
            theta_dot_dot (np.ndarray[float]): Joint accelerations
            grav (np.ndarray[float], optional): Gravity vector. Defaults to None.
            end_effector_wrench (Wrench, optional): End effector wrench. Defaults to None.

        Returns:
            np.ndarray[float]: Joint torque list (tau)
        """
        #Multiple Bugs Fixed - Liam Aug 4 2019
        if grav is None:
            grav = self.grav

        A = np.zeros((self.screw_list.shape))
        V = np.zeros((self.screw_list.shape))
        vel_dot = np.zeros((self.screw_list.shape))
        for i in range(self.num_dof):
            A[0:6, i] = (self._link_homes_global[i].inv().adjoint() @
                self.screw_list[0:6, i]).reshape((6))

            Ti_im1 = (fmr.MatrixExp6(-fmr.VecTose3(A[0:6, i]) * theta[i]) @
                self._link_mass_grav_centers[i].inv().TM) # Merged into link_mass_grav_centers
            if i > 0:
                V[0:6, i] = (A[0:6, i].reshape((6, 1)) * theta_dot[i] +
                     fmr.Adjoint(Ti_im1) @ V[0:6, i-1].reshape((6, 1))).reshape((6))

                start_term = (A[0:6, i] * theta_dot_dot[i]).reshape((6, 1))
                add_term_1 = (fmr.Adjoint(Ti_im1) @ vel_dot[0:6, i-1]).reshape((6, 1))
                add_term_2 = (fmr.ad(V[0:6, i]) @ A[0:6, i] * theta_dot[i]).reshape((6, 1))
                vel_dot[0:6, i] = ((start_term + add_term_1 + add_term_2).reshape((6)))
            else:
                V[0:6, i] = (A[0:6, i] * theta_dot[i] + fmr.Adjoint(Ti_im1) @ np.zeros((6)))
                vel_dot[0:6, i] = ((A[0:6, i] * theta_dot_dot[i]) +
                    (fmr.Adjoint(Ti_im1) @ np.hstack((np.array([0,0,0]) , -1*grav))) +
                    (fmr.ad(V[0:6, i]) @ A[0:6, i] * theta_dot[i]))
        F = np.zeros((self.screw_list.shape))
        tau = np.zeros((theta.size, 1))
        for i in range(self.num_dof-1, -1, -1):
            if i == self.num_dof-1:
                #continue
                Tip1_i = self._link_mass_grav_centers[i+1].inv().TM
                start_term = (fmr.Adjoint(Tip1_i).conj().T @ end_effector_wrench).flatten()
                add_term = self._box_spatial_links[i,:,:] @ vel_dot[0:6, i]
                sub_term = fmr.ad(V[0:6, i]).conj().T @ self._box_spatial_links[i,:,:] @ V[0:6, i]
                F[0:6, i] = start_term + add_term - sub_term
            else:
                Tip1_i = (fmr.MatrixExp6(-fmr.VecTose3(A[0:6, i+1]) * theta[i + 1]) @
                    self._link_mass_grav_centers[i+1].inv().TM) # Merged into link_mass_grav_centers
                start_term = fmr.Adjoint(Tip1_i).T @ F[0:6, i+1]
                add_term = self._box_spatial_links[i,:,:] @ vel_dot[0:6, i]
                sub_term = fmr.ad(V[0:6, i]).conj().T @ self._box_spatial_links[i,:,:] @ V[0:6, i]
                F[0:6, i] = start_term + add_term - sub_term

            tau[i] = F[0:6, i].conj().T @ A[0:6, i]
        return tau, A, V, vel_dot, F

    def inverseDynamicsC(self, theta : 'np.ndarray[float]', theta_dot : 'np.ndarray[float]',
            theta_dot_dot : 'np.ndarray[float]', grav : 'np.ndarray[float]' = None,
            end_effector_wrench : Wrench = Wrench()):
        """
        Calculate inverse dynamics according to the algorithm in Lynch 8.4.

        Args:
            theta (np.ndarray[float]): Joint thetas.
            theta_dot (np.ndarray[float]): Joint velocities.
            theta_dot_dot (np.ndarray[float]): Joint accelerations
            grav (np.ndarray[float], optional): Gravity vector. Defaults to None.
            end_effector_wrench (Wrench, optional): End effector wrench. Defaults to empty wrench.

        Returns:
            tau (np.ndarray[float]): joint torques
        """        
        if grav is None:
            grav = self.grav
        n = self.num_dof
        A = np.zeros((6*n, n))
        G = np.zeros((6*n, 6*n))

        for i in range(n):
            a_index = (i)*6
            b_index = (i)*6+6
            A[a_index:b_index, i] = (
                self._link_homes_global[i].inv().adjoint() @ self.screw_list[0:6, i])
            G[a_index:b_index, a_index:b_index] = self._box_spatial_links[i,:,:]

        n = self.num_dof
        T10 = self.FKLink(theta, 0).inv()

        term_1 = T10.adjoint() @ np.hstack((np.zeros(3), -grav))
        term_2 = np.zeros((5*n))
        vel_dot_base = np.hstack((term_1, term_2))

        Ttipend = self.FK(theta).inv() @ self.FKLink(theta, n - 1)

        Ftip = np.vstack((np.zeros((5*n, 1)), Ttipend.adjoint().conj().T @ end_effector_wrench))

        joint_axes = np.zeros((6*n, 6*n))
        Vbase = np.zeros((6*n, 1))
        for i in range (1, n):
            Ti_im1 = self.FKLink(theta, i).inv() @ self.FKLink(theta, i-1)
            index_1 = (i)*6
            index_2 = (i)*6+6
            index_3 = (i-1)*6
            index_4 = (i-1)*6+6
            joint_axes[index_1:index_2,index_3:index_4] = Ti_im1.adjoint()
        L = ling.inv(np.identity((6*n))-joint_axes)
        V = L @ (A @ theta_dot.reshape((6,1)) + Vbase)
        #Checkpoint 2
        adV = np.zeros((6*n, 6*n))
        adAthd = np.zeros((6*n, 6*n))
        for i in range(n):
            index_1 = (i) * 6
            index_2 = (i) * 6 + 6
            adV[index_1:index_2,index_1:index_2] = fmr.ad(V[index_1:index_2, 0])
            adAthd[index_1:index_2,index_1:index_2] = fmr.ad(theta_dot[i] * A[index_1:index_2, i])

        #disp(adV)
        #disp(adAthd)
        #         L * (A * thetadotdot                  - adAthd * W          * V - adAthd * Vbase           + Vdotbase)
        #vel_dot = L @ (A @ theta_dot_dot.reshape((6,1)) - adAthd @ joint_axes @ V - adAthd @ Vbase.flatten() + vel_dot_base)
        t1 = A @ theta_dot_dot.reshape((6,1))
        t2 = adAthd @ joint_axes @ V
        t3 = adAthd @ Vbase
        vel_dot = L @ (t1 - t2 - t3 + vel_dot_base.reshape((len(vel_dot_base),1)))

        t1 = G @ vel_dot
        t2 = adV.conj().T @ G @ V
        F = L.conj().T @ (t1 - t2 + Ftip)
        tau = A.conj().T @ F
        M = A.conj().T @ L.conj().T @ G @ L @ A


        return tau, M, G

    def forwardDynamicsE(self, theta : 'np.ndarray[float]', theta_dot : 'np.ndarray[float]',
            tau : 'np.ndarray[float]', grav : 'np.ndarray[float]' = None, 
            end_effector_wrench : Wrench = Wrench()):
        """
        Calculate forward dynamics sans Modern Robotics.

        Args:
            theta (np.ndarray[float]): Joint configuration
            theta_dot (np.ndarray[float]): Joint velocities
            tau (np.ndarray[float]): Joint torques
            grav (np.ndarray[float], optional): gravity vector. Defaults to None.
            end_effector_wrench (Wrench, optional): end effector wrench. Defaults to None.

        Returns:
            theta_dot_dot (np.ndarray[Float]): Joint accelerations
        """        
        if grav is None:
            grav = self.grav
        M = self.massMatrix(theta)
        h = self.coriolisGravity(theta, theta_dot, grav)
        ee = self.inverseDynamics(theta, np.zeros(len(theta)), 
                np.zeros(len(theta)), np.zeros((3)), end_effector_wrench)[0]
        mult_term = (tau-h.flatten()-ee.flatten())
        theta_dot_dot = ling.pinv(M) @ mult_term

        return theta_dot_dot, M, h, ee

    def forwardDynamics(self, theta : 'np.ndarray[float]', theta_dot : 'np.ndarray[float]', 
            tau : 'np.ndarray[float]', grav : 'np.ndarray[float]' = None,
            end_effector_wrench : Wrench = Wrench()) -> 'np.ndarray[float]':
        """
        Calculate forward dynamics sans Modern Robotics.

        Args:
            theta (np.ndarray[float]): Joint configuration
            theta_dot (np.ndarray[float]): Joint velocities
            tau (np.ndarray[float]): Joint torques
            grav (np.ndarray[float], optional): gravity vector. Defaults to None.
            end_effector_wrench (Wrench, optional): end effector wrench. Defaults to None.

        Returns:
            theta_dot_dot (np.ndarray[Float]): Joint accelerations
        """      
        if grav is None:
            grav = self.grav
        link_mass_array = np.array([x.gTM() for x in self._link_mass_grav_centers])
        theta_dot_dot = fmr.ForwardDynamics(
            theta,
            theta_dot,
            tau,
            grav,
            end_effector_wrench,
            link_mass_array,
            self._box_spatial_links,
            self.screw_list)
        return theta_dot_dot
        
    def integrateForwardDynamics(self, theta0 : 'np.ndarray[float]', 
            thetadot0 : 'np.ndarray[float]', tau : 'np.ndarray[float]', dt : float = 1.0, 
            grav : 'np.ndarray[float]' = None, 
            end_effector_wrench : Wrench = Wrench(), 
            t_series : 'np.ndarray[float]' = None):
        """
        Integrate forward kinematics.

        Args:
            theta0 (np.ndarray[float]): Joint configuration init
            thetadot0 (np.ndarray[float]): Joint velocity init
            tau (np.ndarray[float]): Joint torques init.
            dt (float, optional): delta time. Defaults to 1.0.
            grav (np.ndarray[float], optional): gravity vector. Defaults to None.
            end_effector_wrench (Wrench, optional): end effector wrench. Defaults to Wrench().
            t_series (np.ndarray[float], optional): use specified time series. Defaults to None.

        Returns:
            np.ndarray[float] : joint accelerations
        """        
        if grav is None:
            grav = self.grav

        def anon(t, x):
            sol = np.zeros((2*self.num_dof))
            sol[0:self.num_dof] = x[self.num_dof:]
            sol[self.num_dof:] = self.forwardDynamicsE(
                x[0:self.num_dof],x[self.num_dof:], tau, grav, end_effector_wrench)[0].flatten()
            return sol

        init_thetas = np.zeros((2*self.num_dof))
        init_thetas[0:self.num_dof] = theta0
        init_thetas[self.num_dof:] = thetadot0
        sol = integrate.solve_ivp(anon, np.array([0, dt]), init_thetas, t_eval = t_series)

       # merged = np.hstack([i.reshape(-1,1) for i in sol.y])
        return sol.t, sol.y.T

    def massMatrix(self, theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Generate arm mass matrix list.

        Args:
            theta (np.ndarray[float], optional): joint configuration. Defaults to None.

        Returns:
            np.ndarray[float]: mass matrix for arm
        """        
        #Debugged - Liam 8/4/19
        theta = self._helper_ensure_theta_not_none(theta)
        M = np.zeros((len(theta),len(theta)))
        for i in range(len(theta)):
            Ji = self.jacobianLink(i, theta)
            jt = Ji.T @ self._box_spatial_links[i,:,:] @ Ji
            M = M + jt
        #print(M, 'M1')
        #print(fmr.massMatrix(theta, self._link_mass_grav_centers, # Merged into link_mass_grav_centers
        #    self._box_spatial_links, self.screw_list), 'Masses')
        return M

    def coriolisGravity(self, theta : 'np.ndarray[float]',
            theta_dot : 'np.ndarray[float]',
            grav : 'np.ndarray[float]') -> 'np.ndarray[float]':
        """
        Calculate coriolis Gravity.

        Args:
            theta (np.ndarray[float]): Joint configuration
            theta_dot (np.ndarray[float]): Joint acceleration
            grav (np.ndarray[float]): Gravity array

        Returns:
            np.ndarray[float]: Mass matrix for the arm.
        """        
        h = self.inverseDynamics(theta, theta_dot, 0*theta, grav, np.zeros((6, 1)))[0]
        return h

    """
    Jacobian Calculations
    """

    #Converted to Python - Joshua
    def jacobian(self, theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Calculate Space Jacobian for given configuration.

        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        theta = self._helper_ensure_theta_not_none(theta)
        return fmr.JacobianSpace(self.screw_list, theta)

    #Converted to Python - Joshua
    def jacobianBody(self, theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Calculate Body Jacobian for given configuration.

        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        theta = self._helper_ensure_theta_not_none(theta)
        return fmr.JacobianBody(self.screw_list_body, theta)

    def jacobianLink(self, i : int,  theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Calculate Space Jacobian for given configuration link.

        Args:
            i: joint index
            theta: joint configuration
        Returns:
            jacobian
        """
        theta = self._helper_ensure_theta_not_none(theta)
        t_ad = self.FKLink(theta, i).inv().adjoint()
        t_js = fmr.JacobianSpace(self.screw_list[0:6, 0:i+1], theta[0:i+1])
        t_z = np.zeros((6, len(theta) - (i+1)))
        t_mt = t_ad @ t_js
        return np.hstack((t_mt, t_z))

    def jacobianEETrans(self, theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Calculate Jacobian of the end effector (body), but the frame is rotated to be aligned with the space (global) frame.
        
        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        theta = self._helper_ensure_theta_not_none(theta)
        end_effector_temp = self.FK(theta)
        end_effector_temp[3:6] = np.zeros(3)
        jacobian = self.jacobian(theta)
        return end_effector_temp.inv().adjoint() @ jacobian

    def numericalJacobian(self, theta : 'np.ndarray[float]' = None) -> 'np.ndarray[float]':
        """
        Calculate numerical Jacobian for given configuration.

        Args:
            theta: joint configuration
        Returns:
            jacobian
        """
        theta = self._helper_ensure_theta_not_none(theta)
        jacobian = np.zeros((6, self.num_dof))
        temp = lambda x : self.FK(x).gTM().T.flatten()
        numerical_jacobian = fsr.numericalJacobian(temp, theta, 0.0005)
        for i in range(0, self.num_dof):
            inv_ee_t = ling.inv(self.FK(theta).gTM().T)
            jac_re = np.reshape(numerical_jacobian[:, i], ((4, 4)))
            jacobian[0:6, i] = fmr.se3ToVec((inv_ee_t @ jac_re).T)
        return jacobian

    def getManipulability(self, theta : 'np.ndarray[float]' = None):
        """
        Calculate Manipulability at a given configuration.

        Args:
            theta: configuration
        Returns:
            Manipulability parameters
        """
        if theta == None:
            theta = self._theta.copy()
        Jb = self.jacobianBody(theta)
        Jw = Jb[0:3,:] #Angular
        Jv = Jb[3:6,:] #Linear

        Aw = Jw @ Jw.T
        Av = Jv @ Jv.T

        AwEig, AwEigVec = np.linalg.eig(Aw)
        AvEig, AvEigVec = np.linalg.eig(Av)

        uAw = 1/(np.sqrt(max(AwEig))/np.sqrt(min(AwEig)))
        uAv = 1/(np.sqrt(max(AvEig))/np.sqrt(min(AvEig)))

        return AwEig, AwEigVec, uAw, AvEig, AvEigVec, uAv

    """
    Camera
    """


    def addCamera(self, cam : Camera, end_effector_to_cam : tm) -> None:
        """
        Add a camera to the arm.

        Args:
            cam: camera object
            end_effector_to_cam: end effector to camera transform
        """
        cam.moveCamera(self._end_effector_pos_global @ end_effector_to_cam)
        img, _, _ = cam.getPhoto(self._end_effector_pos_global @
            tm([0, 0, 1, 0, 0, 0]))
        camL = [cam, end_effector_to_cam, img]
        self.cameras.append(camL)

    def updateCams(self) -> None:
        """Update camera locations."""
        for i in range(len(self.cameras)):
            self.cameras[i][0].moveCamera(self._end_effector_pos_global @ self.cameras[i][1])

    """
    Class Methods
    """

    def move(self, new_base_pos_global : tm, stationary : bool = False) -> None:
        """
        Move the arm to another location.

        Args:
            T: new base location
            stationary: boolean for keeping the end effector in origianal location while
                moving the base separately
        """
        curpos = self._end_effector_pos_global.copy()
        curth = self._theta.copy()
        self.initialize(new_base_pos_global, self.original_screw_list.copy(),
            self._end_effector_home_local, self.original_joint_poses_home)
        if stationary == False:
            self.FK(self._theta)
        else:
            self.IK(curpos, curth)

    def draw(self, ax):
        """
        Draw the arm using the vis_matplotlib library.

        Args:
            ax: matplotlib axes to plot to.
        """
        DrawArm(self, ax)

    """
    Helpers to avoid code duplication
    """
    def _helper_determine_eef_to_last_joint(self):
        """
        Determine if an eef to last joint transform is required.
        
        If it is required, it would need to be set.
        """
        if not np.allclose(
                self._end_effector_home[0:3],
                self._joint_homes_global[-1][0:3],
                atol = 1e-9, rtol = 0):
            self._eef_to_last_joint = fsr.globalToLocal(
                    self._end_effector_home, self._joint_homes_global[-1])

    def _helper_ensure_theta_not_none(self, theta : 'np.ndarray[float]') -> 'np.ndarray[float]':
        """
        Ensure theta value is present.

        Args:
            theta (np.ndarray[float]): Input theta, may be none.

        Returns:
            np.ndarray[float]: Output theta, either input or current state.
        """        
        if theta is None:
            return self._theta
        return theta

class URDFLoader:
    """Load a URDF into an Arm object."""

    def __init__(self):
        """Init a URDFLoader."""
        self.type = 'LINK'
        self.sub_type = None
        self.axis = None
        self.xyz_origin = None #Use as CG for Joints
        self.mass = None
        self.inertia = None
        self.id = None
        self.name = ''
        self.parent = None
        self.children = []
        self.num_children = 0

        #Link Visuals
        self.vis_properties = vis_info()
        self.col_properties = vis_info()

        #Joint Specific
        self.joint_limits = np.array([-2*np.pi, 2*np.pi])
        self.max_effort = np.inf
        self.max_velocity = np.inf

    def __str__(self):   # pragma: no cover
        """Display properties of calculated object."""
        if self.type == 'link':
            print('link: ' + self.name + ' (' + str(self.id) + ')')
        else:
            print(self.sub_type + ' joint: ' + self.name + ' (' + str(self.id) + ')')
        if self.parent is not None:
            print('\tparent: ' + self.parent.name)
        else:
            print('\tHas no parent')
        print('\tchildren:' + str(self.num_children))
        for child in self.children:
            print('\t\t' + child.name)
        print('\tOrigin: ' + str(self.xyz_origin))
        if self.type == 'link':
            print('\tMass: ' + str(self.mass))
            print('\tVisProperties: ' + str(self.vis_properties))
            print('\tColProperties: ' + str(self.col_properties))
        else:
            print('\tJoint Limits: ' + str(self.joint_limits))
            print('\tMax Effort: ' + str(self.max_effort))
            print('\tMax Velocity: ' + str(self.max_velocity))
        return("")

def load_urdf_spec_file(urdf_fname, package_fname):
    """
    Return a file path from a urdf specified file.

    Args:
        urdf_fname: urdf file name
        package_fname: package_fname
    Returns:
        string of the absolute path
    """
    if 'package://' in package_fname:
        return find_package_dir(urdf_fname, package_fname)
    elif package_fname[0:3] == '../':
        return os.path.abspath(package_fname)
    else:
        return package_fname

def find_package_dir(urdf_fname, package_rel_dir):
    """
    Attempt to find a directory specified by a ros package macro without ROS.

    Args:
        urdf_fname: urdf file name/path *must be absolute
        package_rel_dir: relative package directory
    Returns:
        string of the absolute file path
    """
    search_char = '/'
    urdf_fname = urdf_fname.replace('\\', search_char)
    package_rel_dir = package_rel_dir.replace('\\', search_char)


    real_path = os.path.abspath(urdf_fname).replace('\\', '/')
    real_split_path = real_path.split(search_char)

    #print("Package Rel " + package_rel_dir)
    package_name = search_char.join(package_rel_dir[9:].split(search_char)[1:])
    #print("Package Name! " + package_name)
    
    found_path = False
    i = len(real_split_path) - 1
    #print(real_split_path)
    while not found_path and i > 0:
        test_path_prefix = search_char.join(real_split_path[0:i])
        test_path = test_path_prefix + search_char + package_name
        #print(test_path)
        if os.path.isfile(test_path):
            #print(test_path)
            return test_path
        i -= 1
    #print("Not found:" + package_name)

def loadArmFromURDF(file_name):
    """
    Load an arm from a URDF File.

    Args:
        file_name: file name of urdf object
    Returns:
        Arm object
    """
    #Joints connect parent and child links
    #Each joint has anoriginframethat defines the position
    #    and orientation of thechildlink frame relativeto the
    #    parentlink frame when the joint variable is zero.
    #    Theoriginis on he joint’s axis.
    # Each joint has anaxis3-vector, a unit vector
    #   expressed inthechildlink’s frame,
    #   in the direction of positive rotation
    #   for a revolutejoint or positive translation
    #    for a prismatic joint.
    try:
        tree = ET.parse(file_name)
    except:
        if os.path.exists(file_name):
            print('Malformed URDF or unrecognizeable format')
        else:
            print('File not Found')
        return

    root = tree.getroot()
    elements = []

    def extractOrigin(x_obj):
        """
        Shortcut for pulling from xml.

        Args:
            x: xml object root
        Returns:
            origin of root
        """
        x_org_urdf = x_obj.get('xyz')
        r_org_urdf = x_obj.get('rpy')
        if x_org_urdf is not None:
            t_origin = x_org_urdf.split()
        else:
            t_origin = [0, 0, 0]
        if r_org_urdf is not None:
            r_origin = r_org_urdf.split()
        else:
            r_origin = [0, 0, 0]
        return t_origin, r_origin

    def completeInertiaExtraction(child):
        """
        Extract inertial properties from child.

        Args:
            child: child object
        """
        ixx = child.find('inertia').get('ixx')
        ixy = child.find('inertia').get('ixy')
        ixz = child.find('inertia').get('ixz')
        iyy = child.find('inertia').get('iyy')
        iyz = child.find('inertia').get('iyz')
        izz = child.find('inertia').get('izz')
        inertia_matrix = np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz]], dtype=float)
        return inertia_matrix

    def completeGeometryParse(child):
        """
        Complete Geometry parsing for children.

        Args:
            child: child xml object to be parsed
        """
        geometry_type = 'box'
        origin = tm()
        properties = vis_info()
        for grand_child in child:
            #print(grand_child.tag)
            if grand_child.tag == 'origin':
                cg_xyz_raw, cg_rpy_raw = extractOrigin(grand_child)
                cg_origin_xyz = np.array(cg_xyz_raw, dtype=float)
                cg_origin_rpy = np.array(cg_rpy_raw, dtype=float)

                cg_origin_tm = tm([cg_origin_xyz[0], cg_origin_xyz[1], cg_origin_xyz[2],
                        cg_origin_rpy[0], cg_origin_rpy[1], cg_origin_rpy[2]])

                new_element.xyz_origin = cg_origin_tm
                properties.setOrigin(cg_origin_tm)
            elif grand_child.tag == 'geometry':
                geometry_parent = child.find('geometry')
                for geometry in geometry_parent:
                    if geometry.tag == 'box':
                        properties.geo_type = 'box'
                        properties.box_size = geometry.get('size').split()
                    elif geometry.tag == 'cylinder':
                        properties.geo_type = 'cyl'
                        properties.radius = geometry.get('radius')
                        properties.length = geometry.get('length')
                    elif geometry.tag == 'sphere':
                        properties.geo_type = 'spr'
                        properties.radius = geometry.get('radius')
                    elif geometry.tag == 'mesh':
                        properties.geo_type = 'mesh'
                        properties.file_name = (
                            load_urdf_spec_file(file_name, geometry.get('filename')))
                        properties.setScale(geometry.get('scale'))
        return properties


    def completeLinkParse(new_element, parent):
        #print(new_element.name)
        for child in parent:
            if child.tag == 'inertial':
                cg_xyz_raw, cg_rpy_raw = extractOrigin(child.find('origin'))
                cg_origin_xyz = np.array(cg_xyz_raw, dtype=float)
                cg_origin_rpy = np.array(cg_rpy_raw, dtype=float)

                cg_origin_tm = tm([cg_origin_xyz[0], cg_origin_xyz[1], cg_origin_xyz[2],
                        cg_origin_rpy[0], cg_origin_rpy[1], cg_origin_rpy[2]])

                new_element.xyz_origin = cg_origin_tm
                new_element.inertia = completeInertiaExtraction(child)
                new_element.mass = float(child.find('mass').get('value'))
            elif child.tag == 'visual':
                new_element.vis_properties = \
                        completeGeometryParse(child)
                if (new_element.vis_properties.geo_type == 'mesh' and 
                        new_element.vis_properties.file_name is None):
                    new_element.vis_properties = None
            elif child.tag == 'collision':
                new_element.col_properties = \
                        completeGeometryParse(child)
                if (new_element.col_properties.geo_type == 'mesh' and 
                        new_element.col_properties.file_name is None):
                    new_element.col_properties = None

    def completeJointParse(new_element, parent):
        #print(new_element.name)
        for child in parent:
            if child.tag == 'axis':
                axis = np.array(child.get('xyz').split(), dtype=float)
                new_element.axis = axis
            if child.tag == 'origin':
                cg_xyz_raw, cg_rpy_raw = extractOrigin(child)
                cg_origin_xyz = np.array(cg_xyz_raw, dtype=float)
                cg_origin_rpy = np.array(cg_rpy_raw, dtype=float)

                cg_origin_tm = tm([cg_origin_xyz[0], cg_origin_xyz[1], cg_origin_xyz[2],
                        0, 0, 0])
                cg_origin_tm = cg_origin_tm @ tm([0, 0, 0, 0, 0, cg_origin_rpy[2]])
                cg_origin_tm = cg_origin_tm @ tm([0, 0, 0, 0, cg_origin_rpy[1], 0])
                #cg_origin_rpy[0], cg_origin_rpy[1], cg_origin_rpy[2]
                cg_origin_tm = cg_origin_tm @ tm([0, 0, 0, cg_origin_rpy[0], 0, 0])



                new_element.xyz_origin = cg_origin_tm
            if child.tag == 'limit':
                new_element.joint_limits[0] = child.get('lower')
                new_element.joint_limits[1] = child.get('upper')
                new_element.max_effort = child.get('effort')
                new_element.max_velocity = child.get('velocity')

    def findNamedElement(named_element, element_type='all'):
        for element in elements:
            if element.name == named_element:
                if element_type == 'all':
                    return element
                if element_type == 'link' and element.type == 'link':
                    return element
                if element_type == 'joint' and element.type == 'joint':
                    return element

    def totalChildren(element):
        if element.num_children == 0:
            return 1
        else:
            sum_children = 0
            for child in element.children:
                sum_children += totalChildren(child)
            return sum_children


    def mostChildren(element):
        most_children = totalChildren(element.children[0])
        max_ind = 0
        for i in range(element.num_children):
            child_qty = totalChildren(element.children[i])
            if child_qty > most_children:
                max_ind = i
                most_children = child_qty
        return element.children[max_ind]

    def determineAxis(joint_location, axis):
        joint_rotation = tm([joint_location[3], joint_location[4], joint_location[5]])
        axis_unit = tm([axis[0], axis[1], axis[2], 0, 0, 0])
        axis_new = (joint_rotation @ axis_unit)[0:3]
        #if sum(abs(axis)) > 0:
        #    axis_new = abs(axis_new)
        #else:
        #    axis_new = abs(axis_new) * -1
        return axis_new.flatten()


    #Perform First Pass Parse
    for child in root:
        new_element = URDFLoader()
        new_element.type = child.tag
        new_element.name = child.get('name')
        if new_element.type == 'link':
            completeLinkParse(new_element, child)
        elif new_element.type == 'joint':
            new_element.sub_type = child.get('type')
            completeJointParse(new_element, child)
        elements.append(new_element)


    world_link = URDFLoader()
    world_link.type = 'link'
    world_link.sub_type = 'fixed'
    elements.append(world_link)
    #Assign Parents and Children to complete chain
    for child in root:
        if child.tag == 'joint':
            this_element = findNamedElement(child.get('name'), 'joint')
            parent_name = 'world'
            child_name = ''
            for sub_child in child:
                if sub_child.tag == 'parent':
                    parent_name = sub_child.get('link')
                elif sub_child.tag == 'child':
                    child_name = sub_child.get('link')
            parent_element = findNamedElement(parent_name)
            child_element = findNamedElement(child_name)
            this_element.parent = parent_element
            parent_element.children.append(this_element)
            parent_element.num_children += 1
            child_element.parent = this_element
            this_element.children.append(child_element)
            this_element.num_children += 1

    #Account for cases that don't use world
    
    if world_link.num_children == 0:
        elements.remove(world_link)
        for element in elements:
            if element.type == 'link' and element.parent is None and element.num_children > 0:
                world_link = element
                break
            if (element.type == 'joint' and element.sub_type == 'fixed' and
                    element.parent is None and element.num_children > 0):
                world_link = element
                break
    home = tm()


    num_dof = 0
    #Count the number of degrees of freedom along longest kinematic chain
    temp_element = world_link
    while temp_element.num_children > 0:
        if temp_element.type == 'joint' and temp_element.sub_type != 'fixed':
            num_dof += 1
        temp_element = mostChildren(temp_element)

   
    joint_poses = [home]
    link_poses = []

    joint_axes = np.zeros((3, num_dof))
    joint_homes = np.zeros((3, num_dof))
    arrind = 0

    #Figure out the link home poses
    temp_element = world_link
    link_masses = []
    link_mass_grav_centers = []
    link_names = []
    joint_names = []
    vis_props = []
    col_props = []
    joint_mins = []
    joint_maxs = []
    joint_vel_limits = []
    joint_effort_limits = []
    prev_joints_to_next_joints = []
    inertia_props = []
    eef_to_last_joint = None
    while temp_element.num_children > 0:
        #temp_element.display()
        if temp_element.type == 'link' or temp_element.sub_type == 'fixed':
            #If a Link or a Fixed Joint
            if temp_element.type == 'link':
                if temp_element.mass is not None:
                    link_masses.append(temp_element.mass)
                    link_mass_grav_centers.append(temp_element.xyz_origin)
                    link_poses.append(joint_poses[-1] @ temp_element.xyz_origin)
                    inertia_props.append(temp_element.inertia)
                link_names.append(temp_element.name)
                vis_props.append(temp_element.vis_properties)
                col_props.append(temp_element.col_properties)
            if temp_element.sub_type == 'fixed': #If it's a fixed joint, it's a pseudo link
                #eef_transform = eef_transform @ temp_element.xyz_origin
                prev_joints_to_next_joints.append(temp_element.xyz_origin) # Fixed Joints are still origins
                joint_poses[-1] = joint_poses[-1] @ temp_element.xyz_origin
                eef_to_last_joint = temp_element.xyz_origin.inv()
            temp_element = mostChildren(temp_element)
            continue
        prev_joints_to_next_joints.append(temp_element.xyz_origin)
        joint_poses.append(joint_poses[-1] @ temp_element.xyz_origin)
        joint_names.append(temp_element.name)
        joint_mins.append(temp_element.joint_limits[0])
        joint_maxs.append(temp_element.joint_limits[1])
        joint_vel_limits.append(temp_element.max_velocity)
        joint_effort_limits.append(temp_element.max_effort)
        joint_axes[0:3, arrind] = determineAxis(joint_poses[-1], temp_element.axis)
        joint_homes[0:3, arrind] = joint_poses[-1][0:3].flatten()
        temp_element = mostChildren(temp_element)
        arrind+=1

    #Build the screw list
    make_none = True
    for x in vis_props:
        if x is not None:
            make_none = False
    if make_none:
        vis_props = None
    make_none = True
    for x in col_props:
        if x is not None:
            make_none = False
    if make_none:
        col_props = None   
    screw_list = np.zeros((6, num_dof))
    for i in range(num_dof):
        screw_list[0:6, i] = np.hstack((
            joint_axes[0:3, i],
            np.cross(joint_homes[0:3, i], joint_axes[0:3, i])))

    base_offset = joint_poses[0]
    joint_poses = joint_poses[1:]
    if inertia_props == [] or inertia_props[0] is None:
        inertia_props = None

    
    arm = Arm(tm(), screw_list, joint_poses[-1], joint_homes, joint_axes)
    arm.setNames('Arm', link_names, joint_names)
    arm.setJointProperties(np.array(joint_mins), np.array(joint_maxs),
            np.array(joint_vel_limits), np.array(joint_effort_limits))
    arm.setOrigins(prev_joints_to_next_joints, joint_poses, link_poses, eef_to_last_joint, base_offset)
    arm.setMassProperties(np.array(link_masses), link_mass_grav_centers, inertia_props)
    #Set Names
    #arm.link_names = link_names
    #arm.joint_names = joint_names

    #Max/Min Joint Properties
    #arm.joint_mins = np.array(joint_mins)
    #arm.joint_maxs = np.array(joint_maxs)
    #arm.max_vels = np.array(joint_vel_limits)
    #arm.max_effort = np.array(joint_effort_limits)

    #Vis/Col Properties
    #arm.vis_props = vis_props
    #arm.col_props = col_props

    # Object Origins
    #arm.prev_joints_to_next_joints = prev_joints_to_next_joints # Joint Origins Local
    #arm.joint_homes_global = joint_poses # Joint Origins Global
    #arm.link_homes_global = link_poses # Link Origins Global
    #arm.eef_to_last_joint = eef_to_last_joint # EEF to Last Joint

    #Masses
    #arm.link_mass_grav_centers = link_mass_grav_centers
    #arm.link_masses = np.array(link_masses)

    #Placeholder Dimensions
    dims = np.zeros((3, num_dof + 1))
    for i in range(num_dof + 1):
        dims[0:3,
             i] = np.array([.1, .1, .1])

    arm.setVisColProperties(vis_props, col_props, dims)
    return arm

"""
def loadArmFromJSON(file_name):
    #Load Arm From a JSON File
    #Args:
    #    file_name: filename of the json to be loaded
    #Returns:
    #    Arm object
    #
    with open(file_name, 'r') as arm_file:
        arm_data = json.load(arm_file)

    num_dof = arm_data["NumDof"]
    end_effector_home = tm(arm_data["EndEffectorLocation"])
    base_location = tm(arm_data["BaseLocation"])

    link_mass_centers_raw = []
    joint_centers_raw = []
    link_masses_raw = arm_data["LinkMasses"]
    joint_homes_global_raw = []
    box_dimensions_raw = []
    for i in range(num_dof+1):
        ii = str(i)
        box_dimensions_raw.append(arm_data["LinkBoxDimensions"][ii])
        joint_homes_global_raw.append(arm_data["JointHomePositions"][ii])
        if i == num_dof:
            continue
        link_mass_centers_raw.append(tm(arm_data["LinkCentersOfMass"][ii]))
        joint_centers_raw.append(arm_data["JointAxes"][ii])


    joint_axes = np.array(joint_centers_raw).T

    joint_homes_global = np.array(joint_homes_global_raw).T

    screw_list = np.zeros((6, num_dof))
    for i in range(0, num_dof):
        screw_list[0:6, i] = np.hstack((joint_axes[0:3, i],
            np.cross(joint_homes_global[0:3, i], joint_axes[0:3, i])))

    dimensions = np.array(box_dimensions_raw).T

    Mi = [None] * (num_dof + 1)
    Mi[0] = link_mass_centers_raw[0]
    for i in range(1, num_dof):
        Mi[i] = link_mass_centers_raw[i].inv() @ link_mass_centers_raw[i]
    Mi[num_dof] = link_mass_centers_raw[num_dof - 1] @ end_effector_home

    link_masses = np.array(link_masses_raw)

    box_spatial = np.zeros((num_dof, 6, 6))
    for i in range(num_dof):
        box_spatial[i,:,:] = fsr.boxSpatialInertia(
            link_masses[i], dimensions[0, i], dimensions[1, i], dimensions[2, i])
    arm = Arm(base_location, screw_list,
        end_effector_home, joint_homes_global, joint_axes)

    disp(end_effector_home, 'EEPOS')
    home_poses = []
    for pose in joint_homes_global_raw:
        print(pose)
        home_poses.append(tm([pose[0], pose[1], pose[2], 0, 0, 0]))
    #arm.setDynamicsProperties(Mi, link_mass_centers_raw, box_spatial, dimensions)
    arm.setDynamicsProperties(Mi, home_poses, box_spatial, dimensions)

    return arm
"""
