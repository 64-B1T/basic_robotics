import numpy as np
import numba
from numba import jit
#from modern_high_performance import *
from ..modern_robotics_numba.modern_high_performance import *

#Parallel Disabled For Now. Something is wrong with my installation.
@jit(nopython=True, cache=True)#, parallel=True)
def IKinSpaceConstrained(screw_list, ee_home, ee_goal, theta_list,
        position_tolerance, rotation_tolerance, joint_mins, joint_maxs, max_iterations):
    """
    Calculates IK to a certain goal within joint rotation constraints

    Args:
        screw_list: screw list
        ee_home: home end effector position
        ee_goal: Goal Position
        theta_list: Initial thetas
        position_tolerance: Positional tolerance
        rotation_tolerance: Rotational tolerance
        joint_mins: joint minimum rotations
        joint_maxs: joint maximum rotations
        max_iterations: Maximum Iterations before failure
    Returns:
        ndarray: joint configuration
        boolean: success

    """
    ee_current = FKinSpace(ee_home, screw_list, theta_list)
    error_vec = np.dot(Adjoint(ee_current),
            se3ToVec(MatrixLog6(np.dot(TransInv(ee_current), ee_goal))))
    #print(mhp.MatrixLog6(np.dot(mhp.TransInv(ee_current), ee_goal)), "Test")
    error_bool = (np.linalg.norm(error_vec[0:3]) > position_tolerance or
            np.linalg.norm(error_vec[3:6]) > rotation_tolerance)
    #if np.isnan(error_vec).any():
    #    error_bool = True
    i = 0
    while error_bool and i < max_iterations:
        jacobian_space = JacobianSpace(screw_list, theta_list)
        inverse_jacobian_space = np.linalg.pinv(jacobian_space)
        new_theta = np.dot(inverse_jacobian_space, error_vec)
        theta_list = theta_list + new_theta
        for j in range(len(theta_list)):
            if theta_list[j] < joint_mins[j]:
                theta_list[j] = joint_mins[j]
            if theta_list[j] > joint_maxs[j]:
                theta_list[j] = joint_maxs[j];
        i = i + 1
        ee_current = FKinSpace(ee_home, screw_list, theta_list)
        error_vec = np.dot(Adjoint(ee_current),
                se3ToVec(MatrixLog6(np.dot(TransInv(ee_current), ee_goal))))
        error_bool = (np.linalg.norm(error_vec[0:3]) > position_tolerance or
                np.linalg.norm(error_vec[3:6]) > rotation_tolerance)
        #if np.isnan(error_vec).any():
        #    error_bool = True
    success = not error_bool

    return theta_list, success

@jit(nopython=True, cache=True, parallel=True)
def SPIKinSpace(bottom_transform, top_transform, bottom_joints,
        top_joints, bottom_joint_locations, top_joint_locations):
    """
    Calculates IK for a stewart platform

    Args:
        bottom_transform (ndarray): bottom plae transformation matrix
        top_transform (ndarray): top plate transformation matrix
        bottom_joints (ndarray): bottom joints initial
        top_joints (ndarray): top joints initial
        bottom_joint_locations (ndarray): bottom joints current locations space
        top_joint_locations (ndarray): top joints current locations space

    Returns:
        ndarray: leg lengths
        ndarray: bottom joint locations space
        ndarray: top joint locations space

    """
    lengths = np.zeros((6, 1))

    #Perform Inverse Kinematics
    for i in range(6):
        bottom_joint_locations[0:3, i] = TrVec(bottom_transform, bottom_joints[0:3, i])
        top_joint_locations[0:3, i] = TrVec(top_transform, top_joints[0:3, i])
        t_len = Norm(top_joint_locations[0:3, i] - bottom_joint_locations[0:3, i])
        lengths[i] = t_len

    return lengths, bottom_joint_locations, top_joint_locations

#Majority of following section adapted from work by Jak-O-Shadows, Under MIT License
@jit(nopython=True, cache=True)
def SPFKinSpaceR(bottom_transform, leg_lengths,
        top_plate_init, bottom_joints_init,
        top_joints_init, max_iterations, tol_f, tol_a, leg_ext_min):
    """
    Calculates FK for a stewart platform

    Args:
        bottom_transform (ndarray): bottom transform of the platform.
        leg_lengths (ndarray): leg lengths to calculate for
        top_plate_inith (ndarray): top plate initial guess position
        bottom_joints_init (ndarray): bottom joints local to bottom plate
        top_joints_init (ndarray): top joints local to top plate
        max_iterations (int): maximum number of iterations
        tol_f (float): leg length tolerance
        tol_a (float): plate tolerance
        leg_ext_min (float): leg length minimum

    Returns:
        ndarray: top plate position
        int: iterations to completion

    """
    current_iteration = 0
    #a = np.zeros((6))
    #a[2] = h
    top_plate_guess = top_plate_init
    while current_iteration < max_iterations:
            current_iteration += 1
            angs = np.zeros((6))
            j = 0
            top_plate_guess[3:6] = AngleMod(a[3:6])
            for i in range(3, 6):
                angs[j] = (np.cos(top_plate_guess[i]))
                angs[j+1] = (np.sin(top_plate_guess[i]))
                j+=2

            t = top_plate_guess[2]
            if t < leg_ext_min/2:
                top_plate_guess[2] = leg_ext_min/2.0
            #angs[5] = np.clip(angs[5], -np.pi/3.85, np.pi/3.85)
            #Must translate platform coordinates into base coordinate system
            #Calculate rotation matrix elements
            Rzyx = (MatrixExp3(VecToso3(top_plate_guess[3:6])))

            #disp(Rzyx, "Rzyx")
            #Hence platform sensor points with respect to the base coordinate system
            xbar = top_plate_guess[0:3] - bottom_joints_init

            #Hence orientation of platform wrt base
            uvw = np.zeros(top_joints_init.shape)
            for i in range(6):
                uvw[i, :] = np.dot(Rzyx, top_joints_init[i, :])

            iteration_length_guess = np.sum(np.square(xbar + uvw), 1)
            #Hence find value of objective function
            #The calculated lengths minus the actual length
            f = -1 * (iteration_length_guess - np.square(leg_lengths))
            sum_f = np.sum(np.abs(f))
            if sum_f < tol_f:
                #success!
                #print("Converged!")
                break

            #As using the newton-raphson matrix, need the jacobian (/hessian?) matrix
            #Using paper linked
            #https://jak-o-shadows.github.io/electronics/stewart-gough/kinematics-stewart-gough-platform.pdf
            dfda = np.zeros((6, 6))
            dfda[:, 0:3] = 2*(xbar + uvw)
            for i in range(6):
                #dfda4 is swapped with dfda6 for magic reasons!
                dfda[i, 5] = 2*(-xbar[i, 0] * uvw[i, 1] + xbar[i, 1] * uvw[i, 0]) #dfda4
                dfda[i, 4] = 2 * ((-xbar[i, 0] * angs[4] +
                        xbar[i, 1] * angs[5]) * uvw[i, 2] -
                        (top_joints_init[i, 0] * angs[2] +
                        top_joints_init[i, 1] * angs[3] * angs[1]) * xbar[i, 2]) #dfda5
                dfda[i, 3] = 2 * top_joints_init[i, 1] * (np.dot(xbar[i,:], Rzyx[:, 2])) #dfda
            #disp(dfda, "Dfda")
            #disp(np.linalg.inv(
            #       self.InverseJacobianSpace(self.gbottom_transform(), self.gtop_transform())))
            #Hence solve system for delta_{a} - The change in lengths
            top_plate_delta = np.linalg.solve(dfda, f)

            if abs(np.sum(top_plate_delta)) < tol_a:
                #print ("Small change in lengths -- converged?")
                break
            top_plate_guess = top_plate_guess + top_plate_delta
    return top_plate_guess, current_iteration

#Performs tv = transformation_matrix*vec and removes the 1
#TrVec can keep name to keep constant with modern robotics naming scheme
@jit(nopython=True, cache=True)
def TrVec(transformation_matrix, vector):
    """
    Performs tv = TM*vec and removes the 1
    Args:
        transform (ndarray): transform to operate on
        vector (ndarray): vector to multipy

    Returns:
        ndarray: vector product
    """
    vector_4 = np.ones((4))
    vector_4[0:3] = vector
    new_vec = transformation_matrix @ vector_4
    return new_vec[0:3]
