import numpy as np
from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Robot, Arm, loadArmFromURDF
from basic_robotics.plotting.vis_matplotlib import *
from basic_robotics.utilities.disp import disp
from basic_robotics.modern_robotics_numba import mr
from basic_robotics.metrology import Camera
import random
import unittest

class test_kinematics_arm(unittest.TestCase):
    def matrix_equality_assertion(self, mat_a, mat_b, num_dec = 3):
        def matrix_equality_message(a, b):
            return ('\nMatrix Equality Error\n' + disp(a, 'Matrix A', noprint=True) +
                    '\nIs Not Equal To\n' + disp(b, 'Matrix B', noprint=True))
        def matrix_shape_message(a, b):
            return ('\nMatrix Shape Error\n' + disp(a.shape, 'Matrix A', noprint=True) +
                    '\nIs Not Equal To\n' + disp(b.shape, 'Matrix B', noprint=True))
        shape_a = mat_a.shape
        shape_b = mat_b.shape
        self.assertEqual(len(shape_a), len(shape_b), matrix_shape_message(mat_a, mat_b))
        for i in range(len(shape_a)):
            self.assertEqual(shape_a[i], shape_b[i], matrix_shape_message(mat_a, mat_b))
        mat_a_flat = mat_a.flatten()
        mat_b_flat = mat_b.flatten()
        for i in range(len(mat_a_flat)):
            self.assertAlmostEqual(
                mat_a_flat[i], mat_b_flat[i], num_dec, matrix_equality_message(mat_a, mat_b))

    def setUp(self):
        Base_T = tm() # Set a transformation for the base

        # Define some link lengths
        L1 = 4.5
        L2 = 3.75
        L3 = 3.75
        W = 0.1
        # Define the transformations of each link
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
        basic_arm_link_box_dims = np.array([[W, W, L1],[L2, W, W],[L3, W, W],[W, W, W],[W, W, W],[W, W, W]]).conj().T
        basic_arm_link_mass_transforms = [None] * (len(basic_arm_screw_list) + 1)
        basic_arm_link_mass_transforms[0] = Tspace[0]

        #Set mass transforms
        for i in range(1,6):
            basic_arm_link_mass_transforms[i] = (Tspace[i-1].inv() @ Tspace[i])
        basic_arm_link_mass_transforms[6] = (Tspace[5].inv() @ basic_arm_end_effector_home)
        masses = np.array([20, 20, 20, 1, 1, 1])
        basic_arm_inertia_list = np.zeros((6,6,6))

        #Create spatial inertia matrices for links
        for i in range(6):
            basic_arm_inertia_list[i,:,:] = fsr.boxSpatialInertia(masses[i],basic_arm_link_box_dims[0,i],basic_arm_link_box_dims[1,i],basic_arm_link_box_dims[2,i])

        # Create the arm from the above paramters
        arm = Arm(Base_T,basic_arm_screw_list,basic_arm_end_effector_home,basic_arm_joint_homes,basic_arm_joint_axes)

        #ALTERNATIVELY, JUST LOAD A URDF USING THE 'loadArmFromURDF' function in basic_robotics.kinematics
        #arm.setDynamicsProperties(
         #   basic_arm_link_mass_transforms,
          #  Tspace,
           # basic_arm_inertia_list,
            #basic_arm_link_box_dims)
        arm.setJointProperties(
            np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])* -2,
            np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) * 2)
        arm.setOrigins(link_homes_global = Tspace)
        arm.setMassProperties(masses, basic_arm_link_mass_transforms, basic_arm_inertia_list)
        arm.setVisColProperties(link_dimensions = basic_arm_link_box_dims)

        self.arm = arm

    #Kinematics

    def test_kinematics_arm_thetaProtector(self):
        test_theta = np.array([2*np.pi + np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10])
        n_theta = self.arm.thetaProtector(test_theta)
        self.assertAlmostEqual(n_theta[0], self.arm.joint_maxs[0])

    def test_kinematics_arm_FK(self):
        ee_pos = self.arm.FK(np.zeros((6)))
        self.assertAlmostEqual(ee_pos[0], 7.8)
        self.assertAlmostEqual(ee_pos[2], 4.5)

        ee_pos = self.arm.FK(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]))
        self.assertAlmostEqual(ee_pos[0], 6.2408, 3)
        self.assertAlmostEqual(ee_pos[1], 3.6031, 3)
        self.assertAlmostEqual(ee_pos[2], 1.5151, 3)

    def test_kinematics_arm_FKLink(self):
        link_i = self.arm.FKLink(np.zeros((6)), 0).gTM()
        link_i_ref = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2.25],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)

        link_i = self.arm.FKLink(np.zeros((6)), 5).gTM()
        link_i_ref = np.array([[1, 0, 0, 7.75],
            [0, 1, 0, 0],
            [0, 0, 1, 4.5],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)

        link_i = self.arm.FKLink(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]), 0).gTM()
        link_i_ref = np.array([[0.86603, -0.5, 0, 0],
            [0.5, 0.86603, 0, 0],
            [0, 0, 1, 2.25],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)

        link_i = self.arm.FKLink(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]), 5).gTM()
        link_i_ref = np.array([[0.8001, -0.52446, 0.29116, 6.2008],
            [0.46194, 0.84834, 0.2587, 3.58],
            [-0.38268, -0.072487, 0.92103, 1.5342],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)

    def test_kinematics_arm_FKJoint(self):
        link_i = self.arm.FKJoint(np.zeros((6)), 0).gTM()
        link_i_ref = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)
        link_i = self.arm.FKJoint(np.zeros((6)), 1).gTM()
        link_i_ref = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4.5],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)

        link_i = self.arm.FKJoint(np.zeros((6)), 5).gTM()
        link_i_ref = np.array([[1, 0, 0, 7.8],
            [0, 1, 0, 0],
            [0, 0, 1, 4.5],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)

        link_i = self.arm.FKJoint(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]), 0).gTM()
        link_i_ref = np.array([[0.86603, -0.5, 0, 0],
            [0.5, 0.86603, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        self.matrix_equality_assertion(link_i, link_i_ref)

        link_i = self.arm.FKJoint(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]), 5).gTM()
        link_i_ref = self.arm.FK(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10])).gTM()
        self.matrix_equality_assertion(link_i, link_i_ref)

    def test_kinematics_arm_IK(self):
        random.seed(10)
        test_tm = tm([1, 1, 3, 0, 0, 0])

        ref_theta, suc = self.arm.IK(test_tm)

        self.matrix_equality_assertion(test_tm.TM , self.arm.getEEPos().TM)

        test_angs = np.array([np.pi/7, np.pi/5, -np.pi/8, np.pi/7, -np.pi/8, np.pi/10])
        test_tm_2 = self.arm.FK(test_angs)

        test_tm_2 = self.arm.FK(np.zeros(6))
        self.arm.IK(test_tm_2)

        self.matrix_equality_assertion(test_tm_2.TM, self.arm.getEEPos().TM)

        test_angs_2 = np.array([-np.pi/7, -np.pi/5, np.pi/8, -np.pi/7, np.pi/8, -np.pi/10])

        test_tm_2 = self.arm.FK(np.zeros(6))
        self.arm.IK(test_tm_2, test_angs_2)
        self.matrix_equality_assertion(test_tm_2.TM, self.arm.getEEPos().TM)

        test_tm_2 = self.arm.FK(np.zeros(6))
        self.arm.IK(test_tm_2, test_angs_2, protect=True)
        self.matrix_equality_assertion(test_tm_2.TM, self.arm.getEEPos().TM)

        test_tm_2 = self.arm.FK(np.array([-np.pi, 0, np.pi/4, -np.pi/4, 0, 0]))
        self.arm.IK(test_tm_2, np.zeros(6), protect=True)
        self.matrix_equality_assertion(test_tm_2.TM, self.arm.getEEPos().TM)

        test_tm_2 = self.arm.FK(test_angs_2)
        self.arm.IK(test_tm_2, test_angs_2 * -1, protect=True)
        self.matrix_equality_assertion(test_tm_2.TM, self.arm.getEEPos().TM)

        test_tm = tm([4, 3.999999995, 3, 0, 0, 0])

        theta, suc = self.arm.IK(test_tm, test_angs_2, protect=True)

        self.assertNotEqual(theta.flatten()[0], ref_theta.flatten()[0])

    def test_kinematics_arm_constrainedIK(self):
        random.seed(10)
        test_tm = tm([1, 1, 3, 0, 0, 0])

        ref_theta, suc = self.arm.constrainedIK(test_tm)

        self.matrix_equality_assertion(test_tm.TM , self.arm.getEEPos().TM)

        test_angs = np.array([np.pi/7, np.pi/5, -np.pi/8, np.pi/7, -np.pi/8, np.pi/10])
        test_tm_2 = self.arm.FK(test_angs)

    #def test_kinematics_arm_IKForceOptimal(self):
    #    #TODO
    #    pass

    #def test_kinematics_arm_IKMotion(self):
    #    #TODO This function doesn't work at all
    #    pass

    def test_kinematics_arm_IKFree(self):
        random.seed(10)
        init_theta = np.array([0, np.pi/4, -np.pi/4, 0, 0, np.pi/3])
        test_tm = self.arm.FK(init_theta)
        #test_tm = tm([1, 1, 3, 0, 0, 0])

        ref_theta, suc = self.arm.IKFree(test_tm, np.array([0, 0, 0, 0, 0, 0.5]), [0,1,2,5])
        self.assertTrue(suc)
        ref_theta, suc = self.arm.IKFree(test_tm, np.array([0, 0, 0, 0, 0, 0]), [0,1,2,5])
        self.assertFalse(suc)


    #Kinematics Helpers

    def test_kinematics_arm_randomPos(self):
        random.seed(10)
        end_effector_a = self.arm.randomPos()
        random.seed(10)
        end_effector_b = self.arm.randomPos()
        self.matrix_equality_assertion(end_effector_a.gTM(), end_effector_b.gTM())
        pass

    #def test_kinematics_arm_reverse(self):
    #    self.arm.reverse()
    #    self.matrix_equality_assertion(self.arm.getEEPos().gTM(), tm().gTM())

    # Motion Planning

    def test_kinematics_arm_lineTrajectory(self):
        test_theta = np.array([-np.pi/4, np.pi/4, -np.pi/4, 0, np.pi/4, 0])
        desired = self.arm.getEEPos()
        self.arm.FK(test_theta)
        traj = self.arm.lineTrajectory(desired)
        self.matrix_equality_assertion(self.arm.getEEPos().gTM(), desired.gTM())

    def test_kinematics_arm_visualServoToTarget(self):
        tg = self.arm.FK(self.arm._theta) @ tm(np.array([.5, .5, 1, 0, 0, 0]))
        theta, _ = self.arm.visualServoToTarget(tg, desired_dist=0, pose_tol = 0.02, pose_delta=0.01)
        self.matrix_equality_assertion(theta, self.arm._theta)

        cam = Camera(200, 200, 1024, 1024, 2048, 2048, 1, tm())
        test_theta = np.array([0, np.pi/4, -np.pi/4, 0, -np.pi/4, 0])
        cam.moveCamera(self.arm.FK(test_theta))
        #Attach a camera to the end link of the arm
        self.arm.addCamera(cam, tm(np.array([0, 0, 0, 0, 0, 0])))

        tg = self.arm.FK(self.arm._theta) @ tm(np.array([.5, .5, 1, 0, 0, 0]))
        self.arm.updateCams()
        theta, _ = self.arm.visualServoToTarget(tg, desired_dist=0, pose_tol = 0.02, pose_delta=0.01)
        self.assertTrue(fsr.distance(self.arm.getEEPos(), tg) < 0.02)

        #Failed to find solution
        tg = self.arm.FK(self.arm._theta) @ tm(np.array([5, .5, 1, 0, 0, 0]))
        theta, _ = self.arm.visualServoToTarget(tg, desired_dist=0, pose_tol = 0.02, pose_delta=0.01, max_iter=800)
        self.assertFalse(fsr.distance(self.arm.getEEPos(), tg) < 0.02)

        #Failed to locate Target
        tg = self.arm.FK(self.arm._theta) @ tm(np.array([5, .5, -.4, 0, 0, 0]))
        theta, _ = self.arm.visualServoToTarget(tg, desired_dist=0, pose_tol = 0.02, pose_delta=0.01)
        self.assertFalse(fsr.distance(self.arm.getEEPos(), tg) < 0.02)

        tg = self.arm.FK(self.arm._theta) @ tm(np.array([-.5, -.5, .995, 0, 0, 0]))
        self.arm.updateCams()
        theta, _ = self.arm.visualServoToTarget(tg, desired_dist=0, pose_tol = 0.02, pose_delta=0.01)
        self.assertTrue(fsr.distance(self.arm.getEEPos(), tg) < 0.02)

        tg = self.arm.FK(self.arm._theta) @ tm(np.array([-.5, -.5, 1.005, 0, 0, 0]))
        self.arm.updateCams()
        theta, _ = self.arm.visualServoToTarget(tg, desired_dist=0, pose_tol = 0.02, pose_delta=0.01)
        self.assertTrue(fsr.distance(self.arm.getEEPos(), tg) < 0.02)

    def test_kinematics_arm_PDControlToGoalEE(self):
        test_theta = np.array([-np.pi/4, np.pi/4, -np.pi/4, 0, np.pi/4, 0])
        self.arm.FK(test_theta)
        desired = self.arm.getEEPos()
        self.arm.FK(np.zeros(6))
        #disp([self.arm.getEEPos(), desired])
        traj = self.arm.PDControlToGoalEE(desired)
        sc = np.array([[-62.8433],
              [19.66],
              [21.3039],
              [11.6727],
              [22.9917],
              [11.6727]]).flatten()
        self.matrix_equality_assertion(traj, sc)

    # Get and Set

    def test_kinematics_arm_setDynamicsProperties(self):
        #TODO
        pass

    def test_kinematics_arm_setMasses(self):
        #TODO
        pass

    def test_kinematics_arm_getJointTransforms(self):
        joint_transforms = self.arm.getJointTransforms()
        self.matrix_equality_assertion(joint_transforms[-1].gTM(), self.arm.getEEPos().gTM())
        self.assertEqual(len(joint_transforms), 8)
        L1 = 4.5
        L2 = 3.75
        L3 = 3.75
        W = 0.1
        ref_poses = np.array([[0, 0, 0],[0, 0, L1],[L2, 0, L1],[L2+L3, 0, L1],[L2+L3+W, 0, L1],[L2+L3+2*W, 0, L1]]).conj().T
        for i in range(6):
            self.assertEqual(ref_poses[0,i], joint_transforms[i + 1][0])
            self.assertEqual(ref_poses[1,i], joint_transforms[i + 1][1])
            self.assertEqual(ref_poses[2,i], joint_transforms[i + 1][2])
        self.arm.FK(np.array([0, 0, 0, 0, 0, np.pi/2]))
        joint_transforms = self.arm.getJointTransforms()
        self.matrix_equality_assertion(joint_transforms[-1].gTM(), self.arm.getEEPos().gTM())
        self.arm.FK(np.array([np.pi/4, np.pi/3, np.pi/6, np.pi/8, np.pi/10, np.pi/16]))
        joint_transforms = self.arm.getJointTransforms()
        self.matrix_equality_assertion(joint_transforms[-1].gTM(), self.arm.getEEPos().gTM())


    def test_kinematics_arm_setArbitraryHome(self):
        new_home = self.arm.getEEPos() @ tm([0, 0, 1, 0, 0, 0])
        self.arm.setArbitraryHome(new_home)
        ref_ee_to_last = self.arm._eef_to_last_joint

        self.arm.restoreOriginalEE()
        self.arm.FK(np.array([0, 1, 2, 0, 1, 2]))
        new_home = self.arm.getEEPos() @ tm([0, 0, 1, 0, 0, 0])
        self.arm.FK(np.zeros(6))
        self.arm.setArbitraryHome(new_home, np.array([0, 1, 2, 0, 1, 2]))
        ref_ee_to_last_2 = self.arm._eef_to_last_joint

        self.matrix_equality_assertion(ref_ee_to_last.gTM(), ref_ee_to_last_2.gTM())

    def test_kinematics_arm_restoreOriginalEE(self):
        #TODO
        pass

    def test_kinematics_arm_getEEPos(self):
        #TODO
        pass

    def test_kinematics_arm_getScrewList(self):
        #TODO
        pass

    def test_kinematics_arm_getLinkDimensions(self):
        #TODO
        pass

    #Forces and Dynamics

    def test_kinematics_arm_velocityAtEndEffector(self):
        test_theta = np.array([[0.5236],
                      [0.3927],
                      [0],
                      [-0.3927],
                      [0],
                      [0.31416]]).flatten()
        joint_vel_ref = np.array([1, 2, 2, 3, 1, 3])
        eef_vel_ref = np.array([[2.2119],
             [6.9626],
             [-1.6497],
             [-27.9734],
             [15.2506],
             [13.4161]])
        eef_vel = self.arm.velocityAtEndEffector(joint_vel_ref, test_theta)
        self.matrix_equality_assertion(eef_vel, eef_vel_ref)


    def test_kinematics_arm_staticForces(self):
        test_theta = np.array([np.pi/16, np.pi/5, -np.pi/6, np.pi/9, -np.pi/14, np.pi/7])
        wrench_ref = np.array([[0.82906],
            [-11.5542],
            [5.223],
            [-9.8724],
            [-1.1878],
            [-1.0605]])
        tau_arm = self.arm.staticForces(wrench_ref, test_theta)
        tau_ref = np.array([[5.223],
            [33.1211],
            [8.0504],
            [0],
            [7.9936e-15],
            [3.3307e-16]])
        self.matrix_equality_assertion(tau_arm, tau_ref)

        test_theta = np.array([-1.0, 2.0, -3, 4, -5, 6])
        wrench_ref = np.array([[55.6422],
                      [28.636],
                      [0],
                      [0],
                      [0],
                      [-49.05]])
        tau_ref = np.array([[0],
                  [62.2934],
                  [138.8384],
                  [3.2241],
                  [-20.2582],
                  [-28.9437]])

        tau_arm = self.arm.staticForces(wrench_ref, test_theta)
        self.matrix_equality_assertion(tau_arm, tau_ref)

        self.arm.move(tm([1, 2, 3, 0, 0, 0]))
        wrench_ref = np.array([[-42.4578],
                      [77.686],
                      [0],
                      [0],
                      [0],
                      [-49.05]])
        tau_arm = self.arm.staticForces(wrench_ref, test_theta)
        self.matrix_equality_assertion(tau_arm, tau_ref)

        
    def test_kinematics_arm_staticForcesInv(self):
        test_theta = np.array([np.pi/16, np.pi/5, -np.pi/6, np.pi/9, -np.pi/14, np.pi/7])
        tau_ref = np.array([[5.223],
            [33.1211],
            [8.0504],
            [0],
            [7.9936e-15],
            [3.3307e-16]])
        wrench_ref = np.array([[0.82906],
            [-11.5542],
            [5.223],
            [-9.8724],
            [-1.1878],
            [-1.0605]])

        wrench_arm = self.arm.staticForcesInv(tau_ref, test_theta)

        self.matrix_equality_assertion(wrench_arm, wrench_ref)


    def test_kinematics_arm_staticForcesWithLinkMasses(self):
        arm = loadArmFromURDF('./tests/test_helpers/ur5.urdf')
        tau = arm.staticForcesWithLinkMasses(fsr.makeWrench(tm(), [0.0, 0.0, -9.81], 5.0))
        tau2 = arm.staticForces(fsr.makeWrench(tm(), [0.0, 0.0, -9.81], 5), np.zeros(6))
        self.assertEqual(tau[5], tau2[5])


    def test_kinematics_arm_inverseDynamics(self):
        #TODO
        pass

    def test_kinematics_arm_inverseDynamicsEMR(self):
            tau = self.arm.inverseDynamicsEMR(
                np.zeros(6),
                np.ones((6)) * -1,
                np.zeros(6),
                np.array([0, 0, -9.81]),
                np.zeros(6)
            )
            ref_tau = np.array([[ 3.07833333e+00],
                 [-1.69663283e+03],
                 [-4.82645333e+02],
                 [-1.33333333e-02],
                 [-1.95700000e+00],
                 [-5.00000000e-03]]).flatten()
            self.matrix_equality_assertion(tau, ref_tau)

            tau = self.arm.inverseDynamicsEMR(
                np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]),
                np.array([-.1, .2, -.3, .4, -.5, .6]),
                np.array([.01, .02, .03, .04, .05, .06]),
                np.array([0, 0, -9.81]),
                np.array([0, 0, 0, 0, 0, 150*9.81])
            )

            ref_tau = np.array([845.496,-12978.494,-6375.407, 0.000,-281.521,  0.000])
            self.matrix_equality_assertion(tau, ref_tau)

    def test_kinematics_arm_inverseDynamics(self):
        tau, A, V, vel_dot, F = self.arm.inverseDynamics(
            np.zeros(6),
            np.ones((6)) * -1,
            np.zeros(6),
            np.array([0, 0, -9.81]),
            np.zeros(6)
        )
        a_ref = np.array([[0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -1.875, -1.875, 0, -0.05, 0]])
        v_ref = np.array([[0, 0, 0, -1, -1, -2],
            [0, -1, -2, -2, -3, -3],
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0],
            [0, -1.875, -5.625, -7.55, -7.65, -7.75],
            [0, 1.875, 7.5, 11.35, 11.6, 11.9]])
        v_dot_ref = np.array([[0, -1, -2, -2, -3, -3],
            [0, 0, 0, 1, 1, 2],
            [0, 0, 0, -2, -1, -4],
            [0, 0, 3.75, 3.75, 15.2, 15.2],
            [0, 0, 0, -11.35, -11.5, -23.5],
            [9.81, 9.81, 9.81, 2.26, 2.16, -5.69]])
        F_ref = np.array([[-1.0133, -0.11333, -0.08, -0.013333, -0.01, -0.005],
            [-2809.2578, -905.7016, -59.5891, -2.9363, -0.976, 0.0033333],
            [3.0783, 2.3283, 0.82833, 0.058333, 0.021667, -0.0066667],
            [-494.5, -494.5, -419.5, -82, -55.5, -28.25],
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.3],
            [618.03, 421.83, 225.63, 29.43, 19.62, 9.81]])
        self.matrix_equality_assertion(A, a_ref)
        self.matrix_equality_assertion(V, v_ref)
        self.matrix_equality_assertion(vel_dot, v_dot_ref)
        self.matrix_equality_assertion(F, F_ref)
        tau, A, V, vel_dot, F = self.arm.inverseDynamics(
            np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]),
            np.ones((6)) * -1,
            np.zeros(6),
            np.array([0, 0, -9.81]),
            np.zeros(6)
        )
        a_ref = np.array([[0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -1.875, -1.875, 0, -0.05, 0]])
        v_ref = np.array([[0, 0.38268, 0.38268, -0.61732, -0.61732, -1.6173],
            [0, -1, -2, -1.4942, -2.4942, -2.8724],
            [-1, -0.92388, -0.92388, -1.6189, -1.6189, -0.76893],
            [0, 0, 0, 0, 0, 0],
            [0, -1.7323, -5.1968, -10.7878, -10.9497, -8.0135],
            [0, 1.875, 7.5, 7.8167, 8.0161, 11.2947]])
        v_dot_ref = np.array([[0, -0.92388, -1.8478, -1.8478, -3.4667, -3.4667],
            [0, 0, 0, 1.9118, 1.9118, 2.0977],
            [0, -0.38268, -0.76537, -2.2013, -1.584, -4.9697],
            [0, -3.7541, -0.0041245, -0.0041245, 7.8873, 7.8873],
            [0, -0.71753, -2.8701, -15.2979, -15.4872, -27.5524],
            [9.81, 9.0633, 9.0633, -4.0766, -4.2678, -7.4195]])
        F_ref = np.array([[16.6485, -0.10702, -0.076227, -0.014635, -0.011556, -0.0057778],
            [-2267.1137, -623.6241, -30.136, -0.71699, -0.23097, 0.0034962],
            [-881.8675, -636.1993, -89.1478, -3.1756, -1.0637, -0.0082828],
            [-456.7077, -630.3971, -485.8064, -89.6991, -60.5507, -30.7174],
            [-169.7981, -169.7981, -141.0969, -31.5543, -21.0818, -9.2853],
            [740.9171, 328.4785, 160.4716, 7.4749, 4.892, 5.5409]])
        self.matrix_equality_assertion(A, a_ref)
        self.matrix_equality_assertion(V, v_ref)
        self.matrix_equality_assertion(vel_dot, v_dot_ref)
        self.matrix_equality_assertion(F, F_ref)

        tau, A, V, vel_dot, F = self.arm.inverseDynamics(
            np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]),
            np.array([-.1, .2, -.3, .4, -.5, .6]),
            np.array([.01, .02, .03, .04, .05, .06]),
            np.array([0, 0, -9.81]),
            np.array([0, 0, 0, 0, 0, 150*9.81])
        )
        a_ref = np.array([[0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -1.875, -1.875, 0, -0.05, 0]])
        v_ref = np.array([[0, 0.038268, 0.038268, 0.43827, 0.43827, 1.0383],
            [0, 0.2, -0.1, -0.057033, -0.55703, -0.56797],
            [-0.1, -0.092388, -0.092388, -0.12362, -0.12362, 0.054559],
            [0, 0, 0, 0, 0, 0],
            [0, -0.17323, -0.51968, -0.50284, -0.5152, -0.66316],
            [0, -0.375, -0.5625, -0.60877, -0.57806, -0.33377]])
        v_dot_ref = np.array([[0, 0.014651, -0.013066, 0.026934, -0.034877, 0.025123],
            [0, 0.02, 0.05, -0.0053266, 0.044673, 0.022014],
            [0.01, 0.016892, 0.005412, 0.046947, -0.17219, 0.16322],
            [0, -3.7541, -3.9791, -3.9791, -4.2821, -4.2821],
            [0, 0.031673, 0.073494, -3.5329, -3.5392, -1.0007],
            [9.81, 9.0258, 8.8945, 8.3618, 8.3598, 9.4433]])
        F_ref = np.array([[83.6283, 8.147e-05, -0.00040689, 2.8632e-05, -1.6258e-05, 4.1871e-05],
            [-11734.6464, -9507.5657, -3242.6955, -352.3081, -210.7343, -73.575],
            [845.4958, 693.0015, 251.5654, -114.6736, -68.5405, 0.00027203],
            [552.8212, -168.407, -91.5044, -12.0867, -8.0801, -4.0563],
            [118.4833, 118.4833, 117.5628, -464.598, -461.3318, -0.65412],
            [1970.8971, 1851.1618, 1670.7792, 1423.8792, 1415.7378, 1480.2548]])
        self.matrix_equality_assertion(A, a_ref)
        self.matrix_equality_assertion(V, v_ref)
        self.matrix_equality_assertion(vel_dot, v_dot_ref)
        self.matrix_equality_assertion(F, F_ref)

    def test_kinematics_arm_inverseDynamicsC(self):
        param_a = np.array([[-0.99592],
                   [1.1955],
                   [0.27203],
                   [-3.0431],
                   [2.6713],
                   [np.pi/4]]).flatten()
        param_b = np.array([[-0.81133],
                   [0.49231],
                   [2.158],
                   [17.8095],
                   [-15.2309],
                   [np.pi/4]]).flatten()
        #Checkpoint 1
        refTauc = np.array([[302.3778],
                   [395.2302],
                   [327.529],
                   [4.1344],
                   [6.7104],
                   [0.17229]])
        refM = np.array([[71.747, -0.064157, -0.033075, -0.16879, 0.030752, 0.0015553],
                [-0.064157, 906.5926, 312.9486, 0.065482, 1.2044, -7.4236e-05],
                [-0.033075, 312.9486, 136.5088, 0.033304, 0.65488, -7.4236e-05],
                [-0.16879, 0.065482, 0.033304, 0.010134, 5.7506e-17, -0.0014857],
                [0.030752, 1.2044, 0.65488, 5.6379e-17, 0.028333, -9.2519e-20],
                [0.0015553, -7.4236e-05, -7.4236e-05, -0.0014857, -9.2519e-20, 0.0016667]])


        tauc, M, _ = self.arm.inverseDynamicsC(param_a, param_b, np.zeros(6), np.zeros(3), np.zeros((6,1)))
        self.matrix_equality_assertion(tauc, refTauc, 1)
        self.matrix_equality_assertion(M, refM, 1)

        param_c = np.array([[0.1],
                   [0.2],
                   [0.3],
                   [0.4],
                   [0.5],
                   [0.6]]).flatten()
        param_d = np.array([[0],
                   [0],
                   [-9.81]]).flatten()
        param_e = np.array([[76.096],
                   [48.779],
                   [0],
                   [0],
                   [0],
                   [-49.05]])


        tauc, M, _ = self.arm.inverseDynamicsC(param_a, param_b, param_c, param_d, param_e)
        refTauc = np.array([[309.4305],
                   [366.8189],
                   [464.5087],
                   [-44.9258],
                   [49.2987],
                   [76.2688]])
        self.matrix_equality_assertion(tauc, refTauc, 1)

    def test_kinematics_arm_forwardDynamicsE(self):
        theta_dot_dot, M, h, ee = self.arm.forwardDynamicsE(
            np.zeros(6),
            np.zeros(6),
            np.zeros(6)
        )
        tdd_ref = np.array([[0],
            [3.3409],
            [-4.3537],
            [0],
            [9.349],
            [0]]).flatten()
        M_ref = np.array([[925.6592, 0, 0, 0, 0, 0],
            [0, 925.6258, 323.9217, 0, 1.5483, 0],
            [0, 323.9217, 139.4217, 0, 0.79833, 0],
            [0, 0, 0, 0.005, 0, 0.0016667],
            [0, 1.5483, 0.79833, 0, 0.028333, 0],
            [0, 0, 0, 0.0016667, 0, 0.0016667]])
        h_ref = np.array([[0],
            [-1696.6395],
            [-482.652],
            [0],
            [-1.962],
            [0]])
        ee_ref = np.array([[0],
            [0],
            [0],
            [0],
            [0],
            [0]])
        self.matrix_equality_assertion(theta_dot_dot, tdd_ref)
        self.matrix_equality_assertion(M, M_ref)
        self.matrix_equality_assertion(h, h_ref)
        self.matrix_equality_assertion(ee, ee_ref)
        theta_dot_dot, M, h, ee = self.arm.forwardDynamicsE(
            np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]),
            np.array([-.1, .2, -.3, .4, -.5, .6]),
            np.array([.01, .02, .03, .04, .05, .06])
        )
        tdd_ref = np.array([[-0.00027184],
            [3.0833],
            [-4.0245],
            [-5.96],
            [9.8658],
            [42.0309]]).flatten()
        M_ref = np.array([[790.1149, -5.493e-14, -5.0935e-14, -0.0019134, -0.54742, -0.00063781],
            [-5.493e-14, 925.6258, 323.9217, 5.6845e-15, 1.4305, -5.8766e-16],
            [-5.0935e-14, 323.9217, 139.4217, 2.8714e-15, 0.73756, -3.0327e-16],
            [-0.0019134, 5.6845e-15, 2.8714e-15, 0.005, 1.1288e-17, 0.0016667],
            [-0.54742, 1.4305, 0.73756, 1.1288e-17, 0.028333, -3.6329e-17],
            [-0.00063781, -5.8766e-16, -3.0327e-16, 0.0016667, -3.6329e-17, 0.0016667]])
        h_ref = np.array([[5.6409],
            [-1564.4777],
            [-444.9009],
            [-0.00025223],
            [-1.672],
            [-0.00011842]])
        ee_ref = np.array([[0],
            [0],
            [0],
            [0],
            [0],
            [0]])
        self.matrix_equality_assertion(theta_dot_dot, tdd_ref)
        self.matrix_equality_assertion(M, M_ref)
        self.matrix_equality_assertion(h, h_ref)
        self.matrix_equality_assertion(ee, ee_ref)

    def test_kinematics_arm_forwardDynamics(self):
        #Test Derived from L11.m
        ee_pos = self.arm.FK(np.zeros((6)))
        theta_dot_dot = self.arm.forwardDynamics(
            np.zeros((6)),
            np.ones((6)) * -1,
            np.zeros((6)),
            end_effector_wrench = np.array([0, 0, 0, 0, 0, 150*9.81])
        )
        self.assertAlmostEqual(theta_dot_dot[0], -0.0033, 3)
        self.assertAlmostEqual(theta_dot_dot[1], 9.1779, 3)
        self.assertAlmostEqual(theta_dot_dot[2], -38.2955, 3)
        self.assertAlmostEqual(theta_dot_dot[3], 2.5, 3)
        self.assertAlmostEqual(theta_dot_dot[4], 11033.6127, 3)
        self.assertAlmostEqual(theta_dot_dot[5], 0.5, 3)

        #ee_pos = self.arm.FK()
        ee_pos = self.arm.FK(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]))
        theta_dot_dot = self.arm.forwardDynamics(
            np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]),
            np.ones((6)) * -1,
            np.zeros((6)),
            end_effector_wrench = np.array([0, 0, 0, 0, 0, 150*9.81])
        )
        self.assertAlmostEqual(theta_dot_dot[0], 7.233560, 0)
        self.assertAlmostEqual(theta_dot_dot[1], 6.007569, 0)
        self.assertAlmostEqual(theta_dot_dot[2], -23.72005, 0)
        self.assertAlmostEqual(theta_dot_dot[3], 5.42538, 0)
        self.assertAlmostEqual(theta_dot_dot[4], 10349.3890, 0)
        self.assertAlmostEqual(theta_dot_dot[5], 0.80946, 0)

    def test_kinematics_arm_integrateForwardDynamics(self):
        t_ref = np.array([[0],
            [5.0238e-05],
            [0.00010048],
            [0.00015071],
            [0.00020095],
            [0.00045214],
            [0.00070333],
            [0.00095452],
            [0.0012057],
            [0.0024616],
            [0.0037176],
            [0.0049735],
            [0.0062295],
            [0.012509],
            [0.018789],
            [0.025069],
            [0.031348],
            [0.045716],
            [0.060084],
            [0.074452],
            [0.08882],
            [0.10609],
            [0.12335],
            [0.14062],
            [0.15788],
            [0.17065],
            [0.18342],
            [0.19619],
            [0.20896],
            [0.22173],
            [0.2345],
            [0.24727],
            [0.26004],
            [0.27133],
            [0.28263],
            [0.29392],
            [0.30521],
            [0.3213],
            [0.33739],
            [0.35347],
            [0.36956],
            [0.38571],
            [0.40186],
            [0.41802],
            [0.43417],
            [0.44956],
            [0.46494],
            [0.48032],
            [0.49571],
            [0.51414],
            [0.53258],
            [0.55102],
            [0.56946],
            [0.58356],
            [0.59766],
            [0.61176],
            [0.62586],
            [0.63996],
            [0.65406],
            [0.66816],
            [0.68226],
            [0.69888],
            [0.71549],
            [0.73211],
            [0.74872],
            [0.76522],
            [0.78172],
            [0.79823],
            [0.81473],
            [0.82779],
            [0.84085],
            [0.8539],
            [0.86696],
            [0.88002],
            [0.89308],
            [0.90614],
            [0.9192],
            [0.93677],
            [0.95434],
            [0.97191],
            [0.98948],
            [0.99211],
            [0.99474],
            [0.99737],
            [1]]).flatten()
        t, thetadotdot = self.arm.integrateForwardDynamics(np.zeros(6), -1*np.ones(6), np.zeros(6), grav=np.zeros((3)), t_series = t_ref)
        t_ref = np.array([[0],
            [5.0238e-05],
            [0.00010048],
            [0.00015071],
            [0.00020095],
            [0.00045214],
            [0.00070333],
            [0.00095452],
            [0.0012057],
            [0.0024616],
            [0.0037176],
            [0.0049735],
            [0.0062295],
            [0.012509],
            [0.018789],
            [0.025069],
            [0.031348],
            [0.045716],
            [0.060084],
            [0.074452],
            [0.08882],
            [0.10609],
            [0.12335],
            [0.14062],
            [0.15788],
            [0.17065],
            [0.18342],
            [0.19619],
            [0.20896],
            [0.22173],
            [0.2345],
            [0.24727],
            [0.26004],
            [0.27133],
            [0.28263],
            [0.29392],
            [0.30521],
            [0.3213],
            [0.33739],
            [0.35347],
            [0.36956],
            [0.38571],
            [0.40186],
            [0.41802],
            [0.43417],
            [0.44956],
            [0.46494],
            [0.48032],
            [0.49571],
            [0.51414],
            [0.53258],
            [0.55102],
            [0.56946],
            [0.58356],
            [0.59766],
            [0.61176],
            [0.62586],
            [0.63996],
            [0.65406],
            [0.66816],
            [0.68226],
            [0.69888],
            [0.71549],
            [0.73211],
            [0.74872],
            [0.76522],
            [0.78172],
            [0.79823],
            [0.81473],
            [0.82779],
            [0.84085],
            [0.8539],
            [0.86696],
            [0.88002],
            [0.89308],
            [0.90614],
            [0.9192],
            [0.93677],
            [0.95434],
            [0.97191],
            [0.98948],
            [0.99211],
            [0.99474],
            [0.99737],
            [1]]).flatten()
        tdot_ref = np.array([[0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
            [-5.0238e-05, -5.0238e-05, -5.0238e-05, -5.0235e-05, -5.0238e-05, -5.0237e-05, -1, -1, -1, -0.99987, -1, -0.99997],
            [-0.00010048, -0.00010048, -0.00010048, -0.00010046, -0.00010048, -0.00010047, -1, -1, -1, -0.99975, -1, -0.99995],
            [-0.00015071, -0.00015071, -0.00015071, -0.00015068, -0.00015072, -0.00015071, -1, -1, -1, -0.99962, -1, -0.99992],
            [-0.00020095, -0.00020095, -0.00020095, -0.0002009, -0.00020096, -0.00020094, -1, -1, -1, -0.9995, -1, -0.9999],
            [-0.00045214, -0.00045214, -0.00045214, -0.00045188, -0.00045216, -0.00045209, -1, -1, -1, -0.99887, -1.0001, -0.99977],
            [-0.00070333, -0.00070333, -0.00070333, -0.00070271, -0.00070338, -0.00070321, -1, -1, -0.99999, -0.99824, -1.0001, -0.99965],
            [-0.00095452, -0.00095452, -0.00095451, -0.00095338, -0.0009546, -0.00095429, -1, -1, -0.99999, -0.99761, -1.0002, -0.99953],
            [-0.0012057, -0.0012057, -0.0012057, -0.0012039, -0.0012058, -0.0012053, -1, -1, -0.99998, -0.99698, -1.0002, -0.9994],
            [-0.0024617, -0.0024617, -0.0024616, -0.002454, -0.0024621, -0.0024602, -1, -1, -0.99993, -0.99381, -1.0002, -0.9988],
            [-0.0037177, -0.0037177, -0.0037174, -0.0037002, -0.0037182, -0.0037142, -1, -1, -0.99984, -0.99061, -1.0001, -0.99821],
            [-0.0049737, -0.0049737, -0.0049731, -0.0049423, -0.0049742, -0.0049676, -1.0001, -1.0001, -0.99972, -0.98739, -0.99986, -0.99765],
            [-0.0062297, -0.0062298, -0.0062286, -0.0061804, -0.0062297, -0.0062202, -1.0001, -1.0001, -0.99957, -0.98415, -0.99945, -0.99711],
            [-0.012511, -0.012512, -0.012502, -0.012309, -0.012494, -0.012474, -1.0004, -1.0006, -0.99828, -0.96745, -0.99509, -0.99481],
            [-0.018794, -0.018797, -0.018765, -0.018329, -0.01872, -0.018716, -1.0008, -1.0012, -0.99614, -0.94981, -0.98691, -0.99338],
            [-0.02508, -0.025087, -0.025011, -0.024235, -0.024881, -0.024953, -1.0014, -1.0022, -0.99314, -0.93101, -0.97495, -0.99307],
            [-0.031371, -0.031384, -0.031236, -0.030019, -0.030957, -0.031191, -1.0021, -1.0034, -0.98931, -0.91087, -0.9593, -0.99409],
            [-0.045785, -0.045828, -0.04537, -0.042745, -0.044414, -0.045522, -1.0044, -1.0073, -0.97737, -0.85876, -0.91029, -1.0026],
            [-0.060238, -0.060336, -0.059301, -0.054652, -0.057042, -0.060048, -1.0076, -1.0125, -0.9611, -0.79684, -0.84462, -1.0213],
            [-0.074743, -0.074929, -0.072968, -0.065589, -0.068616, -0.074928, -1.0116, -1.019, -0.94062, -0.72389, -0.76504, -1.0518],
            [-0.089311, -0.089624, -0.08631, -0.075394, -0.078968, -0.090337, -1.0164, -1.0268, -0.91603, -0.63922, -0.67499, -1.0952],
            [-0.10692, -0.10744, -0.10183, -0.085447, -0.089636, -0.10981, -1.0234, -1.0377, -0.88127, -0.52173, -0.55839, -1.1647],
            [-0.12466, -0.12547, -0.11671, -0.093324, -0.098258, -0.13065, -1.0316, -1.0501, -0.8411, -0.38765, -0.43942, -1.2529],
            [-0.14255, -0.14371, -0.13085, -0.098743, -0.10484, -0.15318, -1.041, -1.0639, -0.79585, -0.23819, -0.32509, -1.3589],
            [-0.16061, -0.16221, -0.14416, -0.10146, -0.10953, -0.17768, -1.0517, -1.0789, -0.74588, -0.074952, -0.22184, -1.4809],
            [-0.1741, -0.17606, -0.15344, -0.1016, -0.11193, -0.19722, -1.0605, -1.0907, -0.70612, 0.053497, -0.15593, -1.5803],
            [-0.1877, -0.19007, -0.16219, -0.10007, -0.11356, -0.21806, -1.0699, -1.1029, -0.66414, 0.18741, -0.10102, -1.6862],
            [-0.20142, -0.20423, -0.17039, -0.096799, -0.11456, -0.2403, -1.0801, -1.1155, -0.62012, 0.32575, -0.058536, -1.7974],
            [-0.21529, -0.21856, -0.17802, -0.091737, -0.11511, -0.26399, -1.091, -1.1284, -0.57422, 0.46743, -0.029401, -1.9127],
            [-0.22929, -0.23305, -0.18505, -0.084851, -0.11537, -0.28916, -1.1027, -1.1416, -0.52661, 0.61134, -0.014031, -2.0307],
            [-0.24345, -0.24771, -0.19146, -0.07612, -0.11552, -0.31585, -1.1151, -1.1549, -0.47747, 0.75626, -0.012471, -2.1499],
            [-0.25777, -0.26255, -0.19724, -0.065538, -0.11574, -0.34407, -1.1283, -1.1683, -0.42695, 0.90097, -0.024269, -2.269],
            [-0.27227, -0.27755, -0.20236, -0.053116, -0.1162, -0.37379, -1.1422, -1.1817, -0.37524, 1.0443, -0.04846, -2.3865],
            [-0.28524, -0.29097, -0.20634, -0.040615, -0.11691, -0.40132, -1.1553, -1.1935, -0.32864, 1.1689, -0.079075, -2.488],
            [-0.29837, -0.30451, -0.20978, -0.026723, -0.11801, -0.42998, -1.1689, -1.2052, -0.28134, 1.2906, -0.11712, -2.5861],
            [-0.31165, -0.31819, -0.21269, -0.011477, -0.11958, -0.45973, -1.1832, -1.2167, -0.23347, 1.4084, -0.1612, -2.6799],
            [-0.3251, -0.332, -0.21505, 0.0050732, -0.12167, -0.4905, -1.1981, -1.228, -0.18514, 1.5214, -0.2097, -2.7682],
            [-0.34455, -0.35187, -0.21747, 0.030777, -0.12562, -0.53597, -1.2206, -1.2437, -0.11574, 1.6721, -0.28299, -2.8827],
            [-0.36437, -0.372, -0.21877, 0.058788, -0.13076, -0.58315, -1.2444, -1.2587, -0.045966, 1.8077, -0.35615, -2.9805],
            [-0.38459, -0.39236, -0.21895, 0.088833, -0.13705, -0.63175, -1.2697, -1.2729, 0.023853, 1.9246, -0.42334, -3.0583],
            [-0.40522, -0.41294, -0.21801, 0.12058, -0.14433, -0.68141, -1.2964, -1.2861, 0.093393, 2.0193, -0.47865, -3.1126],
            [-0.42639, -0.43382, -0.21594, 0.1538, -0.15239, -0.73196, -1.3249, -1.2984, 0.16263, 2.0884, -0.51645, -3.14],
            [-0.44804, -0.45489, -0.21276, 0.1879, -0.16088, -0.7827, -1.355, -1.3094, 0.23094, 2.1281, -0.53071, -3.1374],
            [-0.47018, -0.47612, -0.20848, 0.22239, -0.16938, -0.83314, -1.3867, -1.319, 0.29801, 2.1361, -0.51641, -3.1025],
            [-0.49285, -0.49749, -0.20314, 0.25673, -0.1774, -0.88276, -1.4203, -1.3272, 0.36354, 2.1109, -0.46963, -3.0343],
            [-0.51496, -0.51796, -0.19708, 0.28882, -0.18408, -0.92874, -1.4539, -1.3334, 0.42423, 2.0554, -0.39234, -2.9383],
            [-0.53759, -0.53851, -0.1901, 0.31982, -0.1893, -0.97301, -1.4892, -1.3381, 0.48299, 1.9694, -0.28153, -2.8125],
            [-0.56078, -0.55912, -0.18223, 0.34926, -0.19256, -1.0151, -1.5264, -1.3409, 0.53957, 1.8535, -0.13707, -2.6576],
            [-0.58456, -0.57976, -0.17351, 0.37669, -0.19335, -1.0546, -1.5653, -1.3419, 0.59372, 1.7085, 0.03957, -2.4748],
            [-0.61387, -0.60449, -0.16199, 0.40634, -0.1904, -1.098, -1.6144, -1.3405, 0.65506, 1.4981, 0.28947, -2.2206],
            [-0.64411, -0.62917, -0.14938, 0.43173, -0.18251, -1.1363, -1.6661, -1.3359, 0.7121, 1.2483, 0.57383, -1.9288],
            [-0.67532, -0.65373, -0.13576, 0.45214, -0.16909, -1.1689, -1.7206, -1.3278, 0.7644, 0.96276, 0.88211, -1.6028],
            [-0.70757, -0.67811, -0.12122, 0.46702, -0.14987, -1.1952, -1.7777, -1.3162, 0.81154, 0.6489, 1.2007, -1.2502],
            [-0.73296, -0.69659, -0.10955, 0.47442, -0.13124, -1.2109, -1.8231, -1.3047, 0.84385, 0.39783, 1.4415, -0.97073],
            [-0.75899, -0.71489, -0.097442, 0.47826, -0.10928, -1.2226, -1.87, -1.2907, 0.87273, 0.14984, 1.6714, -0.69607],
            [-0.7857, -0.73298, -0.084953, 0.47873, -0.08419, -1.2306, -1.9182, -1.2743, 0.89805, -0.077331, 1.8827, -0.44482],
            [-0.81309, -0.75082, -0.072133, 0.47628, -0.056298, -1.2354, -1.9677, -1.2553, 0.9197, -0.26012, 2.0675, -0.24171],
            [-0.84119, -0.76837, -0.059033, 0.47171, -0.026035, -1.2378, -2.0183, -1.2336, 0.93763, -0.37076, 2.2186, -0.11584],
            [-0.87002, -0.7856, -0.045707, 0.46629, 0.0060734, -1.2391, -2.07, -1.2092, 0.95186, -0.38064, 2.3286, -0.096923],
            [-0.89957, -0.80246, -0.032206, 0.46156, 0.03941, -1.2411, -2.1225, -1.182, 0.96245, -0.27161, 2.3914, -0.20326],
            [-0.92988, -0.81892, -0.018582, 0.45922, 0.073276, -1.2455, -2.1758, -1.152, 0.96945, -0.040101, 2.4026, -0.43735],
            [-0.96656, -0.83774, -0.0024373, 0.4618, 0.11286, -1.2561, -2.2393, -1.1129, 0.9732, 0.37314, 2.3469, -0.85732],
            [-1.0043, -0.85587, 0.01373, 0.47223, 0.15089, -1.2746, -2.303, -1.0698, 0.97215, 0.89624, 2.2177, -1.3859],
            [-1.0431, -0.87326, 0.02984, 0.49193, 0.18617, -1.3024, -2.3666, -1.0227, 0.96634, 1.4838, 2.0221, -1.9728],
            [-1.0829, -0.88984, 0.045813, 0.52166, 0.21775, -1.3402, -2.4293, -0.97159, 0.95577, 2.0948, 1.7733, -2.5735],
            [-1.1235, -0.90542, 0.061466, 0.56118, 0.24474, -1.3875, -2.4901, -0.91685, 0.94056, 2.694, 1.4907, -3.151],
            [-1.1651, -0.92007, 0.07683, 0.6104, 0.26689, -1.444, -2.5486, -0.85826, 0.92069, 3.268, 1.1898, -3.6922],
            [-1.2076, -0.93373, 0.091827, 0.66881, 0.284, -1.509, -2.6041, -0.79593, 0.89622, 3.8019, 0.88609, -4.1838],
            [-1.251, -0.94632, 0.10638, 0.73561, 0.29614, -1.5818, -2.6556, -0.73004, 0.86726, 4.2824, 0.59167, -4.6153],
            [-1.2859, -0.9555, 0.11754, 0.79376, 0.30241, -1.644, -2.693, -0.67549, 0.84125, 4.6168, 0.37044, -4.9079],
            [-1.3213, -0.96396, 0.12834, 0.85597, 0.30587, -1.7097, -2.7271, -0.61897, 0.81259, 4.9028, 0.16101, -5.1516],
            [-1.3572, -0.97166, 0.13875, 0.92158, 0.30667, -1.7783, -2.7574, -0.56062, 0.78139, 5.1345, -0.0374, -5.3416],
            [-1.3933, -0.97859, 0.14874, 0.98982, 0.30493, -1.849, -2.7837, -0.50058, 0.74775, 5.307, -0.22766, -5.4747],
            [-1.4298, -0.98473, 0.15827, 1.0599, 0.30074, -1.921, -2.8055, -0.43903, 0.71178, 5.4171, -0.41418, -5.5489],
            [-1.4666, -0.99005, 0.16732, 1.131, 0.29411, -1.9937, -2.8227, -0.37616, 0.6736, 5.464, -0.60203, -5.5643],
            [-1.5035, -0.99455, 0.17586, 1.2024, 0.28498, -2.0661, -2.835, -0.31215, 0.63334, 5.4494, -0.79609, -5.5233],
            [-1.5406, -0.9982, 0.18385, 1.2731, 0.27327, -2.1377, -2.8423, -0.2472, 0.59113, 5.3778, -1.0005, -5.431],
            [-1.5906, -1.0018, 0.19372, 1.3662, 0.25315, -2.2316, -2.844, -0.15868, 0.53145, 5.2037, -1.2969, -5.2389],
            [-1.6405, -1.0038, 0.20251, 1.4556, 0.22758, -2.3215, -2.8364, -0.069328, 0.46878, 4.9614, -1.6182, -4.9895],
            [-1.6902, -1.0042, 0.21018, 1.5403, 0.19615, -2.4067, -2.8198, 0.020378, 0.40345, 4.6782, -1.9588, -4.7103],
            [-1.7395, -1.0031, 0.21668, 1.6199, 0.15867, -2.487, -2.7945, 0.10997, 0.33579, 4.3862, -2.3056, -4.433],
            [-1.7469, -1.0028, 0.21755, 1.6314, 0.15254, -2.4986, -2.79, 0.12333, 0.32549, 4.344, -2.3569, -4.3939],
            [-1.7542, -1.0024, 0.21839, 1.6428, 0.14627, -2.5101, -2.7854, 0.13668, 0.31515, 4.3025, -2.4079, -4.3557],
            [-1.7615, -1.002, 0.2192, 1.654, 0.13988, -2.5215, -2.7805, 0.15002, 0.30476, 4.2619, -2.4584, -4.3186],
            [-1.7688, -1.0016, 0.21999, 1.6652, 0.13335, -2.5328, -2.7755, 0.16333, 0.29433, 4.2222, -2.5085, -4.2827]])
        self.matrix_equality_assertion(t, t_ref)
        self.matrix_equality_assertion(thetadotdot, tdot_ref, 1)

        #TODO add another one

    def test_kinematics_arm_massMatrix(self):
        mmat = self.arm.massMatrix(np.zeros(6))
        massmat_ref = np.array([[925.6592, 0, 0, 0, 0, 0],
            [0, 925.6258, 323.9217, 0, 1.5483, 0],
            [0, 323.9217, 139.4217, 0, 0.79833, 0],
            [0, 0, 0, 0.005, 0, 0.0016667],
            [0, 1.5483, 0.79833, 0, 0.028333, 0],
            [0, 0, 0, 0.0016667, 0, 0.0016667]])
        self.matrix_equality_assertion(mmat, massmat_ref, 1)

        mmat = self.arm.massMatrix(np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]))
        massmat_ref = np.array([[790.1149, -5.493e-14, -5.0935e-14, -0.0019134, -0.54742, -0.00063781],
            [-5.493e-14, 925.6258, 323.9217, 5.6845e-15, 1.4305, -5.8766e-16],
            [-5.0935e-14, 323.9217, 139.4217, 2.8714e-15, 0.73756, -3.0327e-16],
            [-0.0019134, 5.6845e-15, 2.8714e-15, 0.005, 1.1288e-17, 0.0016667],
            [-0.54742, 1.4305, 0.73756, 1.1288e-17, 0.028333, -3.6329e-17],
            [-0.00063781, -5.8766e-16, -3.0327e-16, 0.0016667, -3.6329e-17, 0.0016667]])
        self.matrix_equality_assertion(mmat, massmat_ref, 1)

    def test_kinematics_arm_coriolisGravity(self):
        #TODO
        pass
    # Jacobian Calculations

    def test_kinematics_arm_jacobian(self):
        test_theta_1 = np.array([0, np.pi/4, -np.pi/4, 0, 0, 0])
        arm_jac = self.arm.jacobian(test_theta_1)

        ref_screws = np.array([[0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, -4.5, -4.5, 0, -4.5, 0],
            [0, 0, 0, 4.5, 0, 4.5],
            [0, 0, 3.75, 0, 7.6, 0]])
        self.matrix_equality_assertion(self.arm.screw_list, ref_screws)

        ref_jac = np.array([[0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, -4.5, -1.8483, 0, -1.8483, 0],
            [0, 0, 0, 1.8483, 0, 1.8483],
            [0, 0, 2.6517, 0, 6.5017, 0]])

        self.matrix_equality_assertion(arm_jac, ref_jac)

        test_theta_2 = np.array([np.pi/16, np.pi/5, -np.pi/6, np.pi/9, -np.pi/14, np.pi/7])
        arm_jac = self.arm.jacobian(test_theta_2)
        ref_jac = np.array([[0, -0.19509, -0.19509, 0.97541, -0.14826, 0.98724],
            [0, 0.98079, 0.98079, 0.19402, 0.92861, 0.11878],
            [1, 0, 0, -0.10453, 0.34015, 0.10605],
            [0, -4.4135, -2.2517, -0.5073, -1.3028, -0.082906],
            [0, -0.87791, -0.44789, 2.5504, -2.5702, 1.1554],
            [0, 0, 3.0338, 2.7756e-17, 6.4489, -0.5223]])

        self.matrix_equality_assertion(arm_jac, ref_jac)

        test_theta = np.array([-1, 2, -3, 4, -5, 6])
        arm_jac = self.arm.jacobian(test_theta)
        ref_jac = np.array([[0, 0.84147, 0.84147, 0.29193, -0.20594, -0.81283],
               [0, 0.5403, 0.5403, -0.45465, -0.88904, -0.077256],
               [1, 0, 0, 0.84147, -0.4089, 0.57735],
               [0, -2.4314, -0.589, 1.6006, 4.0281, 0.082062],
               [0, 3.7866, 0.91732, 1.0277, -0.77689, -3.6815],
               [0, 0, -1.5606, 3.4694e-16, -0.33964, -0.37709]])
        self.matrix_equality_assertion(arm_jac, ref_jac)



    def test_kinematics_arm_jacobianBody(self):
        test_theta = np.array([[-0.7854],
                      [0.7854],
                      [-0.7854],
                      [0],
                      [0.7854],
                      [0]]).flatten()
        jB = self.arm.jacobianBody(test_theta)
        jbref = np.array([[-0.70711, 0, 0, 0.70711, 0, 1],
                 [0, 1, 1, 0, 1, 0],
                 [0.70711, 0, 0, 0.70711, 0, 0],
                 [0, 2.7224, 2.7224, 0, 0, 0],
                 [6.6431, 0, 0, 0.14142, 0, 0],
                 [0, -6.6724, -2.9224, 0, -0.2, 0]])
        self.matrix_equality_assertion(jB, jbref)

    def test_kinematics_arm_jacobianLink(self):
        jlink = self.arm.jacobianLink(0, np.zeros((6)))
        jlink_ref = np.array([[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])
        self.matrix_equality_assertion(jlink, jlink_ref)

        jlink = self.arm.jacobianLink(4, np.zeros((6)))
        jlink_ref = np.array([[0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [7.65, 0, 0, 0, 0, 0],
            [0, -7.65, -3.9, 0, -0.05, 0]])
        self.matrix_equality_assertion(jlink, jlink_ref)

        jlink = self.arm.jacobianLink(5, np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]))
        jlink_ref = np.array([[-0.38268, 5.5511e-17, 5.5511e-17, 1, 8.3267e-17, 1],
            [-0.072487, 0.99692, 0.99692, -1.0408e-17, 0.95106, -1.0408e-17],
            [0.92103, 0.078459, 0.078459, 5.5511e-17, -0.30902, 5.5511e-17],
            [-3.3307e-16, 4.4409e-16, 0, 4.4409e-16, -4.4409e-16, 6.6613e-16],
            [7.138, 0.60806, 0.31384, 0, -0.046353, 4.4409e-16],
            [0.56177, -7.7261, -3.9877, 0, -0.14266, 1.1102e-16]])
        self.matrix_equality_assertion(jlink, jlink_ref)

        jlink = self.arm.jacobianLink(0, np.array([np.pi/6, np.pi/8, 0, -np.pi/8, 0, np.pi/10]))
        jlink_ref = np.array([[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])
        self.matrix_equality_assertion(jlink, jlink_ref)

    def test_kinematics_arm_jacobianEETrans(self):
        test_theta = np.array([[0.5236],
                      [0.3927],
                      [0],
                      [-0.3927],
                      [0],
                      [0.31416]]).flatten()
        jeeref = np.array([[0, -0.5, -0.5, 0.8001, -0.58877, 0.8001],
                  [0, 0.86603, 0.86603, 0.46194, 0.72688, 0.46194],
                  [1, 0, 0, -0.38268, -0.35355, -0.38268],
                  [-3.6031, -2.585, -1.3422, 0, -0.022969, 0],
                  [6.2408, -1.4925, -0.77493, 4.4409e-16, -0.10164, 8.8818e-16],
                  [0, -7.2063, -3.7417, 0, -0.17071, 0]])

        njee = self.arm.jacobianEETrans(test_theta)
        self.matrix_equality_assertion(jeeref, njee)

    def test_kinematics_arm_numericalJacobian(self):
        test_theta = np.array([[-0.7854],
                      [0.7854],
                      [-0.7854],
                      [0],
                      [0.7854],
                      [0]]).flatten()
        numjac = np.array([[0, 0.70711, 0.70711, 0.70711, 0.70711, 0.5],
                  [-3.9181e-14, 0.70711, 0.70711, -0.70711, 0.70711, -0.5],
                  [1, 1.1102e-16, -5.54e-14, 1.1102e-16, 1.1113e-13, -0.70711],
                  [9.5035e-14, -3.182, -1.307, 1.307, -1.307, 4.175],
                  [9.5923e-14, 3.182, 1.307, 1.307, 1.307, 4.175],
                  [0, -3.153e-13, 2.6517, 5.0577e-16, 6.5017, 3.5763e-16]])
        jacobian = np.zeros((6, self.arm.num_dof))
        temp = lambda x : self.arm.FK(x).gTM().T.flatten()
        numerical_jacobian = fsr.numericalJacobian(temp, test_theta, 0.0005)
        jst = np.array([[0.5, -0.5, -0.5, 0.5, -0.5, 0],
               [0.5, 0.5, 0.5, 0.5, 0.5, 0],
               [0, -0.70711, -0.70711, 0, -0.70711, 0],
               [0, 0, 0, 0, 0, 0],
               [-0.70711, 0, 0, 0, 0, 0.5],
               [0.70711, 0, 0, 0, 0, -0.5],
               [0, 0, 0, 1, 0, 0.70711],
               [0, 0, 0, 0, 0, 0],
               [0.5, 0.5, 0.5, -0.5, 0.5, -0.70711],
               [0.5, -0.5, -0.5, -0.5, -0.5, -0.70711],
               [0, -0.70711, -0.70711, 0, -0.70711, 0],
               [0, 0, 0, 0, 0, 0],
               [4.6974, -1.975, -0.1, 0.1, -0.1, 0],
               [4.6974, 1.975, 0.1, 0.1, 0.1, 0],
               [0, -6.6431, -3.9914, 0, -0.14142, 0],
               [0, 0, 0, 0, 0, 0]])
        nj = self.arm.numericalJacobian(test_theta)
        self.matrix_equality_assertion(numerical_jacobian, jst)
        self.matrix_equality_assertion(nj, numjac)
        self.matrix_equality_assertion(nj, self.arm.jacobian(test_theta))

    def test_kinematics_arm_getManipulability(self):
        test_theta = np.array([[-0.7854],
                      [0.7854],
                      [-0.7854],
                      [0],
                      [0.7854],
                      [0]]).flatten()
        self.arm.FK(test_theta)
        AwEig, AwEigVec, uAw, AvEig, AvEigVec, uAw = self.arm.getManipulability()
        #print(AwEig)
        #print(AwEigVec)
        #print(uAw)
        #TODO

    # Camera

    def test_kinematics_arm_addCamera(self):
        #TODO
        pass

    def test_kinematics_arm_updateCams(self):
        #TODO
        pass

    # Class Methods

    def test_kinematics_arm_move_1(self):
        random.seed(10)
        joint_transforms_0a = self.arm.getJointTransforms()
        test_theta = np.array([np.pi/5, np.pi/6, -np.pi/7, -np.pi/6, -np.pi/5, np.pi/6])
        ee_pos = self.arm.FK(test_theta)
        joint_transforms_1a = self.arm.getJointTransforms()
        self.arm.FK(np.zeros(6))

        self.arm.move(tm([5, 0, 0, 0, 0, 0]))
        joint_transforms_0b = self.arm.getJointTransforms()
        test_theta = np.array([np.pi/5, np.pi/6, -np.pi/7, -np.pi/6, -np.pi/5, np.pi/6])
        ee_pos = self.arm.FK(test_theta)
        joint_transforms_1b = self.arm.getJointTransforms()
        self.matrix_equality_assertion(self.arm._base_pos_global.gTM(), tm([5, 0, 0, 0, 0, 0]).gTM())

        for i in range(len(joint_transforms_0a)):
            self.matrix_equality_assertion(
                    (joint_transforms_0a[i] + tm([5, 0, 0, 0, 0, 0])).gTM(),
                    joint_transforms_0b[i].gTM())
            self.matrix_equality_assertion(
                    (joint_transforms_1a[i] + tm([5, 0, 0, 0, 0, 0])).gTM(),
                    joint_transforms_1b[i].gTM())

        self.arm.FK(np.zeros(6))
        self.arm.move(tm([0, 0, -5, 0, 0, 0]))
        joint_transforms_1c = self.arm.getJointTransforms()
        for i in range(len(joint_transforms_0a)):
            self.matrix_equality_assertion(
                    (joint_transforms_0a[i] + tm([0, 0, -5, 0, 0, 0])).gTM(),
                    joint_transforms_1c[i].gTM())

        random.seed(10)
        self.arm.move(tm())
        curr_ee = self.arm.getEEPos()
        self.arm.move(tm([0.5, 0.5, 0.75, 0, 0, 0]), True)
        new_ee = self.arm.getEEPos()
        self.matrix_equality_assertion(curr_ee.gTM(), new_ee.gTM())

    def test_kinematics_arm_move_2(self):
        ref_thetas = [np.array([0, 0, 0, 0, 0, 0]),
                np.array([np.pi/4, 0, 0, 0, 0, 0]),
                np.array([0, 0, np.pi/4, 0, 0, 0]),
                np.array([0, 0, 0, 0, np.pi/3, 0]),
                np.array([.1, .2, .3, .4, .5, .6]),
                np.array([.2, -.4, .6, -.8, 1, -1.2]),
                np.array([-.4, .4, -.4, .4, -.4, .4])]
        ref_joint_lists = [] 
        ref_joint_locals = []
        ref_joint_homes = []
        for theta in ref_thetas:
            self.arm.FK(theta)
            ref_joint_lists.append(self.arm.getJointTransforms())
            new_local = []
            for i in range(len(ref_joint_lists[-1])-1):
                new_local.append(fsr.globalToLocal(ref_joint_lists[-1][i], ref_joint_lists[-1][i+1]))
            ref_joint_locals.append(new_local)
            ref_joint_homes.append(self.arm._joint_homes_global.copy())
        base_poses = [tm(), tm([0, 1, 0, 0, 0, 0]), tm([1, 0, 0, 0, 0, 0]), tm([.1, -2.4, .5, 0, 0, 0]),
                tm([.1, .2, .3, .4, .5, .6]), tm([.1, -.2, .3, -.4, .5, -.6])]
        for base_pose in base_poses:
            self.arm.move(base_pose)
            #disp(base_pose, 'Base Pos')
            for i in range(len(ref_thetas)):
                #disp(ref_thetas[i])
                #disp(ref_joint_homes[i], 'ref joint homes global')
                self.arm.FK(ref_thetas[i])
                joint_poses = self.arm.getJointTransforms()
                alt_joint_list = [fsr.globalToLocal(base_pose, x) for x in joint_poses]
                new_local = []
                for j in range(len(joint_poses)-1):
                    new_local.append(fsr.localToGlobal(joint_poses[i], joint_poses[i+1]))    
                #disp(ref_joint_lists[i], 'global ref')
                #disp(alt_joint_list, 'global test')
                #disp(ref_joint_locals[i], 'local ref')
                #disp(new_local, 'local test')
                for j in range(len(joint_poses)):
                    self.matrix_equality_assertion(ref_joint_lists[i][j].gTM(), alt_joint_list[j].gTM())

    def test_kinematics_arm_loadArmFromURDF_UR5(self):
        arm = loadArmFromURDF('./tests/test_helpers/ur5.urdf')
        #Parameters taken from Modern Roboticss Textbook for UR5
        M01 = tm(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]))
        M12 = tm(np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]))
        M23 = tm(np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]))
        M34 = tm(np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]))
        M45 = tm(np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]))
        M56 = tm(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]))
        M67 = tm(np.array([[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]))
        G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
        G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
        G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
        G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
        Glist = [G1, G2, G3, G4, G5, G6]
        Mlist = [M01, M12, M23, M34, M45, M56, M67]
        Slist = np.array([[0,         0,         0,         0,        0,        0],
                 [0,         1,         1,         1,        0,        1],
                 [1,         0,         0,         0,       -1,        0],
                 [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
                 [0,         0,         0,         0,  0.81725,        0],
                 [0,         0,     0.425,   0.81725,        0,  0.81725]])

        self.matrix_equality_assertion(arm.screw_list, Slist, 2)

        Mlist_Aug = [Mlist[0]]
        for m in Mlist[1:]:
            Mlist_Aug.append(Mlist_Aug[-1] @ m)

        for i in range(6):
            self.matrix_equality_assertion(arm._link_homes_global[i+1].gTM(), Mlist_Aug[i].gTM())

        self.matrix_equality_assertion(arm.getEEPos().gTM(), Mlist_Aug[-1].gTM())
        test_theta = np.array([np.pi/5, np.pi/2, np.pi/3, -np.pi/5, -np.pi/3, np.pi/8])
        ee_pos_mr = tm(mr.FKinSpace(Mlist_Aug[-1].gTM(), Slist, test_theta))
        arm.FK(test_theta)
        self.matrix_equality_assertion(arm.getEEPos().gTM(), ee_pos_mr.gTM())

        test_theta_2 = np.array([np.pi/5, np.pi/6, -np.pi/7, -np.pi/6, -np.pi/5, np.pi/6])
        ee_pos = arm.FK(test_theta_2)

        thetas_ik, success = arm.IK(ee_pos)

        self.assertTrue(success)
        self.matrix_equality_assertion(thetas_ik, test_theta_2, 2)

        arm.FK(np.array([np.pi/4, np.pi/3, np.pi/6, np.pi/8, np.pi/10, np.pi/16]))
        joint_transforms = arm.getJointTransforms()
        self.assertEqual(8, len(joint_transforms))
        self.matrix_equality_assertion(joint_transforms[-1].gTM(), arm.getEEPos().gTM())



    def test_kinematics_arm_loadArmFromURDF_IRB2400(self):
        arm = loadArmFromURDF('./tests/test_helpers/irb_2400.urdf')
        # https://library.e.abb.com/public/4057a4faf6814f138012ca658469f6f9/3HAC042195%20PS%20IRB%202400-en.pdf

        ref = tm([.855 +.085, 0, 1.455, 0, 0, 0])
        self.matrix_equality_assertion(ref.gTM(), arm.getEEPos().gTM(), 2)
        test_ang = np.array([0, 0, fsr.deg2Rad(-60), 0, 0, 0])
        arm.FK(test_ang)
        res = arm.getEEPos() @ tm([-.085, 0, 0, 0, 0, 0])

        self.assertAlmostEqual(res[0], .360, 2)
        self.assertAlmostEqual(res[2], 2.041, 2)

        test_ang = np.array([0, 0, fsr.deg2Rad(65), 0, 0, 0])
        arm.FK(test_ang)
        res = arm.getEEPos() @ tm([-.085, 0, 0, 0, 0, 0])

        self.assertAlmostEqual(res[0], .541, 2)
        self.assertAlmostEqual(res[2], .693, 2)

        test_ang = np.array([0, fsr.deg2Rad(110), fsr.deg2Rad(-60), 0, 0, 0])
        arm.FK(test_ang)
        res = arm.getEEPos() @ tm([-.085, 0, 0, 0, 0, 0])

        self.assertAlmostEqual(res[0], 1.351, 2)
        self.assertAlmostEqual(res[2], -.118, 2)

        test_theta_2 = np.array([np.pi/5, np.pi/6, -np.pi/7, -np.pi/6, -np.pi/5, np.pi/6])
        ee_pos = arm.FK(test_theta_2)

        thetas_ik, success = arm.IK(ee_pos)

        self.assertTrue(success)
        self.matrix_equality_assertion(thetas_ik, test_theta_2, 2)

    def test_kinematics_arm_loadArmFromURDF_Puma560(self):
        arm = loadArmFromURDF('./tests/test_helpers/puma_560.urdf')
        #https://i.stack.imgur.com/SjXiJ.png
        ref = tm([0.4318, -0.1397, 0.18288, np.pi, 0, 0])
        self.matrix_equality_assertion(ref.gTM(), arm.getEEPos().gTM(), 2)
        arm.FK(np.array([np.pi/4, np.pi/3, np.pi/6, np.pi/8, np.pi/10, np.pi/16]))
        joint_transforms = arm.getJointTransforms()
        self.assertEqual(7, len(joint_transforms))
        self.matrix_equality_assertion(joint_transforms[-1].gTM(), arm.getEEPos().gTM())

    def test_kinematics_arm_jacobianRelationships(self):
        test_theta = np.array([-1, 2, -3, 4, -5, 6])
        self.arm.FK(test_theta)
        jac_space_1 = self.arm.jacobian()
        jac_body_1 = self.arm.jacobianBody()
        
        t_bs = self.arm.getEEPos()
        jac_body_2 = t_bs.inv().adjoint() @ jac_space_1
        jac_space_2 = t_bs.adjoint() @ jac_body_1
        self.matrix_equality_assertion(jac_space_1, jac_space_2)
        self.matrix_equality_assertion(jac_body_1, jac_body_2)

        self.arm.move(tm([1, 2, 3, 0, 0, 0]))
        self.arm.FK(test_theta)
        jac_space_3 = self.arm.jacobian()
        jac_body_3 = self.arm.jacobianBody()
        
        t_bs = self.arm.getEEPos()
        jac_body_4 = t_bs.inv().adjoint() @ jac_space_3
        jac_space_4 = t_bs.adjoint() @ jac_body_3
        self.matrix_equality_assertion(jac_space_3, jac_space_4)
        self.matrix_equality_assertion(jac_body_3, jac_body_4)
        self.matrix_equality_assertion(jac_body_1, jac_body_3)

        inverse_translated = self.arm.inverseJacobianBody()
        inverse_actual = np.linalg.inv(self.arm.jacobianBody())
        self.matrix_equality_assertion(inverse_actual, inverse_translated)

 
