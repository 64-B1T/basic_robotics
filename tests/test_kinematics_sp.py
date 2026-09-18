import unittest
from unittest.mock import patch
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from basic_robotics.general import tm, fsr, fmr, Wrench
from basic_robotics.metrology.virtual_vision import Camera
import json
import os
from basic_robotics.kinematics import loadSP, makeSP
from basic_robotics.utilities.disp import disp
class test_kinematics_sp(unittest.TestCase):

    def setUp(self):
        self.goals = [
            tm([0.25, 0.25, 1.1, np.pi/8, 0, 0]),
            tm([0.2, 0.3, 1.2, 0, np.pi/8, 0]),
            tm([0, .1, 1.2, 0, 0, np.pi/5]),
            tm([0, -.2, 1.3, 0, -np.pi/8, 0]),
            tm([-.2, -.2, 1.15, 0, 0, 0]),
            tm([-.3, 0, 1.1, 0, 0, 0]),
            tm([-.1, .1, 1.1, np.pi/8, np.pi/8, 0]),
            tm([.1, .2, 1.1, np.pi/16, np.pi/16, np.pi/16]),
            tm([-.1, -.1, 1.2, -np.pi/10, 0, np.pi/10]),
            tm([.05, .05, 1.2, 0, 0, np.pi/6])
        ]
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

        self.sp = loadSP('sp_test_data.json', '')
        self.sp.validation_settings = [1, 1, 1, 1]
        #self.sp.IK(top_plate_pos = tm([0, 0, 1.2, 0, 0, 0]))
        #Delete file
        os.remove('sp_test_data.json')

        basic_sp = {
           "Name":"Basic SP",
           "Type":"SP",
           "BottomPlate":{
              "Thickness":0.0,
              "JointRadius":0.075,
              "JointSpacing":6,
              "Mass": 6
           },
           "TopPlate":{
              "Thickness":0.0,
              "JointRadius":0.045,
              "JointSpacing":6,
              "Mass": 1
           },
           "Actuators":{
              "MinExtension":0.001,
              "MaxExtension":1,
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
              "RestHeight":0.25,
              "Spin":30
           }
        }

        basic_sp_string = json.dumps(basic_sp)
        with open ('sp_test_data.json', 'w') as outfile:
            outfile.write(basic_sp_string)

        self.sp2 = loadSP('sp_test_data.json', '')
        self.sp2.IK(top_plate_pos = tm([0, 0, 0.25, 0, 0, 0]))
        #Delete file
        os.remove('sp_test_data.json')

    def matrix_equality_assertion(self, mat_a, mat_b, num_dec = 3, eq = True):
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
            if eq:
                self.assertAlmostEqual(
                    mat_a_flat[i], mat_b_flat[i], num_dec, matrix_equality_message(mat_a, mat_b))
            else:
                self.assertNotAlmostEqual(
                    mat_a_flat[i], mat_b_flat[i], num_dec, matrix_equality_message(mat_a, mat_b))

    def test_kinematics_sp_initialization(self):
        lens = self.sp.getLens()
        self.matrix_equality_assertion(lens, np.ones((6,1)) * (self.sp.leg_ext_max + self.sp.leg_ext_min)/2)

    def test_kinematics_sp_setMasses_default_top_plate_mass(self):
        # With top_plate_mass left at its default (0), it falls back to
        # plate_mass_general rather than staying 0.
        self.sp.setMasses(3.5, 0.9, 0.5)
        self.assertEqual(self.sp._bottom_plate_mass, 3.5)
        self.assertEqual(self.sp._top_plate_mass, 3.5)

    def test_kinematics_sp_setMaxPlateRotation(self):
        self.sp.setMaxPlateRotation(0.5)
        self.assertAlmostEqual(self.sp.plate_rotation_limit, np.cos(0.5))
        self.sp.setMaxPlateRotation(45, degrees=True)
        self.assertAlmostEqual(self.sp.plate_rotation_limit, np.cos(fsr.deg2Rad(45)))

    def test_kinematics_sp_getTopJoints(self):
        top_joints = self.sp.getTopJoints()
        self.assertEqual(top_joints.shape[1], 6)

    def test_kinematics_sp_getCurrentLocalTransform(self):
        self.sp.IK(tm([0, 0, 1.2, 0, 0, 0]))
        local_transform = self.sp.getCurrentLocalTransform()
        self.assertIsInstance(local_transform, tm)

    def test_kinematics_sp_draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        try:
            self.sp.draw(ax)
        finally:
            plt.close(fig)

    def test_kinematics_sp_getActuatorLoc(self):
        self.sp.IK(tm([0, 0, 1.2, 0, 0, 0]))
        bottom_loc_1 = self.sp.getActuatorLoc(0, 'b')
        mid_loc_1 = self.sp.getActuatorLoc(0, 'm')
        top_loc_1 = self.sp.getActuatorLoc(0, 't')

        bottom_loc_2 = self.sp.getActuatorLoc(1, 'b')
        mid_loc_2 = self.sp.getActuatorLoc(1, 'm')
        top_loc_2 = self.sp.getActuatorLoc(1, 't')

        self.assertEqual(bottom_loc_1[2], bottom_loc_2[2])
        self.assertEqual(mid_loc_1[2], mid_loc_2[2])
        self.assertEqual(top_loc_1[2], top_loc_2[2])

        test_tm = tm([ 0.779347, 0.096168, 0.259536, 0.000000, 0.000000, 0.000000 ])
        self.matrix_equality_assertion(bottom_loc_2.gTM(), test_tm.gTM())
        

    def test_kinematics_sp_spinCustom(self):
        bj1 = self.sp.getBottomJoints()[0:3,0]
        bj1tm = tm([bj1, np.zeros(3)])
        bj0 = bj1tm.copy()
        rot_ang = np.pi/4
        self.sp.spinCustom(rot_ang)
        bj2 = self.sp.getBottomJoints()[0:3,0]
        bj2tm = tm([bj2, np.zeros(3)])

        ang_detected = fsr.angleBetween(bj1tm, self.sp.getBottomT(), bj2tm)
        self.assertAlmostEqual(ang_detected, rot_ang, 1)

        bj1 = self.sp.getBottomJoints()[0:3,0]
        bj1tm = tm([bj1, np.zeros(3)])
        rot_ang = np.pi/9
        self.sp.spinCustom(rot_ang)
        bj2 = self.sp.getBottomJoints()[0:3,0]
        bj2tm = tm([bj2, np.zeros(3)])

        ang_detected = fsr.angleBetween(bj1tm, self.sp.getBottomT(), bj2tm)
        self.assertAlmostEqual(ang_detected, rot_ang, 1)

        self.sp.spinCustom(fsr.rad2Deg(- np.pi/9 - np.pi/4), True)

        bj2 = self.sp.getBottomJoints()[0:3,0]
        bj2tm = tm([bj2, np.zeros(3)])

        ang_detected = fsr.angleBetween(bj0, self.sp.getBottomT(), bj2tm)
        self.assertAlmostEqual(ang_detected, 0, 1)


    def test_kinematics_sp_IK(self):
        for goal in self.goals:
            self.sp.IK(top_plate_pos = goal)
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)
            self.assertTrue(self.sp.validate(True))

    def test_kinematics_sp_FKDefault(self):
        for j in range(10):
            goal = self.goals[j]
            self.sp.IK(top_plate_pos = goal)
            lens = self.sp.getLens().flatten()
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)
            self.sp.IK(top_plate_pos = tm([0, 0, 1.2, 0, 0, 0]))
            self.assertTrue(self.sp.validate(True))
            self.sp.FK(lens)
            #print(result_top)
            result_lens = self.sp.getLens().flatten()
            for i in range(6):
                self.assertAlmostEqual(lens[i], result_lens[i], 2)
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)


    def test_kinematics_sp_FKSolve(self):
        top_init = self.sp.getTopT()
        for j in range(10):
            goal = self.goals[j]
            self.sp.IK(top_plate_pos = goal)
            lens = self.sp.getLens().flatten()
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)
            self.sp.IK(top_plate_pos = tm([0, 0, 1.2, 0, 0, 0]))
            self.assertTrue(self.sp.validate(True))
            self.sp.FK(lens, fk_mode = 0)
            #print(result_top)
            result_lens = self.sp.getLens().flatten()
            for i in range(6):
                self.assertAlmostEqual(lens[i], result_lens[i], 2)
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)
        for i in range(0,3):
            random.seed(10)
            goal = self.goals[i]
            goal[2] = goal[2] - self.sp._nominal_height
            self.sp.IK(top_init, goal)
            if not self.sp.validate(donothing=True):
                continue
            lens = self.sp.getLens().flatten()
            result_bottom = self.sp.getBottomT()
            self.matrix_equality_assertion(result_bottom.gTM(), goal.gTM())
            self.sp.IK(top_init, bottom_plate_pos= tm([0, 0, 0, 0, 0, 0]))
            self.assertTrue(self.sp.validate(True))
            self.sp.FK(lens, top_init, fk_mode = 0, reverse=True)

            result_lens = self.sp.getLens().flatten()
            for i in range(6):
                self.assertAlmostEqual(lens[i], result_lens[i], 1)
            result_top = self.sp.getTopT()
            result_bottom = self.sp.getBottomT()
            #disp([result_bottom, goal])
            self.matrix_equality_assertion(result_bottom.gTM(), goal.gTM(), 2)

            #TODO Might need more tests

    def test_kinematics_sp_FKRaphson(self):
        for j in range(10):
            goal = self.goals[j]
            self.sp.IK(top_plate_pos = goal)
            lens = self.sp.getLens().flatten()
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)
            self.sp.IK(top_plate_pos = tm([0, 0, 1.2, 0, 0, 0]))
            self.assertTrue(self.sp.validate(True))
            self.sp.FK(lens)
            #print(result_top)
            result_lens = self.sp.getLens().flatten()
            for i in range(6):
                self.assertAlmostEqual(lens[i], result_lens[i], 2)
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)

    def test_kinematics_sp_fixUpsideDown(self):
        test_tm = tm([0, .1, -.5, 0, np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        l_old = lens.copy()
        self.assertFalse(self.sp._continuousTranslationConstraint())
        self.sp._fixUpsideDown()
        lens2 = self.sp.getLens()
        for i in range(6):
            self.assertEqual(l_old[i], lens2[i])
        self.assertTrue(self.sp._continuousTranslationConstraint())

        test_tm = tm([-.1, -.1, -.6, np.pi/8, -np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        l_old = lens.copy()
        self.assertFalse(self.sp._continuousTranslationConstraint())
        self.sp._fixUpsideDown()
        lens2 = self.sp.getLens()
        for i in range(6):
            self.assertEqual(l_old[i], lens2[i])
        self.assertTrue(self.sp._continuousTranslationConstraint())
        #print(self.sp.getTopT())

    def test_kinematics_sp_FK_with_explicit_plate_pos_fixes_upside_down(self):
        test_tm = tm([0, .1, -.5, 0, np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos=test_tm, protect=True)
        self.assertFalse(self.sp._continuousTranslationConstraint())

        bottom = self.sp.getBottomT()
        # The FK solver itself tends to converge on an already-valid
        # (non-inverted) solution from these leg lengths, which never
        # actually exercises FK()'s own internal _fixUpsideDown() repair
        # path. Force that path deterministically by making the very next
        # constraint check report "inverted" once, regardless of the real
        # geometry, and confirm FK() calls the repair in response.
        with patch.object(type(self.sp), '_continuousTranslationConstraint',
                side_effect=[False, True, True, True]) as mocked_check, \
                patch.object(type(self.sp), '_fixUpsideDown') as mocked_fix:
            self.sp.FK(lens, plate_pos=bottom, protect=True)
            mocked_fix.assert_called_once()

        self.assertTrue(self.sp._continuousTranslationConstraint())

    def test_kinematics_sp_FKRaphson_falls_back_to_FKSolve_on_failure(self):
        lens = self.sp.getLens()
        with patch('basic_robotics.kinematics.sp_model.fmr.SPFKinSpaceR',
                side_effect=RuntimeError('forced failure')):
            bottom, top = self.sp._FKRaphson(lens, self.sp.getBottomT(), protect=True)
        self.assertIsInstance(bottom, tm)
        self.assertIsInstance(top, tm)
        self.assertGreater(self.sp.fail_count, 0)

    def test_kinematics_sp_lengthCorrectiveAction_boosts_short_legs(self):
        # min leg (0.7) is below leg_ext_min (0.75), but boosting every leg
        # by the shortfall keeps the max leg comfortably under leg_ext_max -
        # this is the "Boost" corrective path (as opposed to Rescale/Subtract).
        self.sp.lengths = np.array([0.7, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.sp.validation_error = ""

        self.sp._lengthCorrectiveAction()

        self.assertIn("Boost", self.sp.validation_error)
        self.assertGreaterEqual(min(self.sp.lengths), self.sp.leg_ext_min)

    def test_kinematics_sp_loadSP_infers_actuator_cog_from_geometry(self):
        # With InferActuatorCOG set to anything other than 1, loadSP()
        # derives motor/shaft gravity centers from actuator extension
        # geometry instead of reading them directly from the json.
        basic_sp = {
           "Name": "COG Inference SP", "Type": "SP",
           "BottomPlate": {"Thickness": 0.1, "JointRadius": 0.9, "JointSpacing": 9, "Mass": 6},
           "TopPlate": {"Thickness": 0.16, "JointRadius": 0.3, "JointSpacing": 25, "Mass": 1},
           "Actuators": {"MinExtension": 0.75, "MaxExtension": 1.5, "MotorMass": 0.5,
               "ShaftMass": 0.9, "ForceLimit": 800, "MotorCOGD": 0.2, "ShaftCOGD": 0.2},
           "Drawing": {"TopRadius": 1, "BottomRadius": 1, "ShaftRadius": 0.1, "MotorRadius": 0.2},
           "Settings": {"MaxAngleDev": 55, "GenerateActuators": 0, "IgnoreRestHeight": 1,
               "UseSpin": 0, "AssignMasses": 1, "InferActuatorCOG": 0},
           "Params": {"RestHeight": 1.2, "Spin": 30}
        }
        with open('sp_test_data_infer_cog.json', 'w') as outfile:
            json.dump(basic_sp, outfile)
        try:
            sp = loadSP('sp_test_data_infer_cog.json', '')
        finally:
            os.remove('sp_test_data_infer_cog.json')

        expected_cog = 1/4 * (0.75 + 1.5) / 2
        self.assertAlmostEqual(sp._act_motor_grav_center, expected_cog)
        self.assertAlmostEqual(sp._act_shaft_grav_center, expected_cog)

    def test_kinematics_sp_validateLegs(self):
        test_tm = tm([0, .1, -.5, 0, np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        self.assertTrue(self.sp.validateLegs(donothing=True))

        test_tm = tm([0, .1, -1.5, 0, np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        self.assertFalse(self.sp.validateLegs(donothing=True))
        self.assertTrue(self.sp.validateLegs())

    def test_kinematics_sp_validateContinuousTranslation(self):
        self.sp.validation_settings[1] = True
        test_tm = tm([-.1, -.1, -.6, np.pi/8, -np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        self.assertFalse(self.sp.validateContinuousTranslation(donothing=True))
        self.assertTrue(self.sp.validateContinuousTranslation(donothing=False))
        test_tm = tm([-.1, -.1, .6, np.pi/8, -np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        self.assertTrue(self.sp.validateContinuousTranslation())
        test_tm = tm([-.1, -.1, .6, np.pi/8, -np.pi/8, np.pi/2])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        #Even though it's upside down, it's still in front of the prior plate
        self.assertTrue(self.sp.validateContinuousTranslation())

    def test_kinematics_sp_plateRotationConstraint(self):
        self.sp.validation_settings[1] = True
        test_tm = tm([-.1, -.1, .6, np.pi/8, -np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        self.assertTrue(self.sp._plateRotationConstraint())
        test_tm = tm([-.1, -.1, .6, 0, 0, np.arccos(self.sp.plate_rotation_limit)+.01])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        self.assertFalse(self.sp._plateRotationConstraint())
        test_tm = tm([-.1, -.1, .6, 0, 0, np.arccos(self.sp.plate_rotation_limit)-.01])
        lens, valid = self.sp.IK(top_plate_pos = test_tm, protect=True)
        self.assertTrue(self.sp._plateRotationConstraint())

    def test_kinematics_sp_interiorAnglesConstraint_false_on_nan(self):
        with patch.object(type(self.sp), 'getJointAnglesFromNorm',
                return_value=np.array([0.1, np.nan, 0.2, 0.1, 0.1, 0.1])):
            self.assertFalse(self.sp._interiorAnglesConstraint())

    def test_kinematics_sp_validateInteriorAngles(self):
        self.sp.validation_settings[2] = True
        test_tm = tm([0, .1, -.5, 0, np.pi/8, 0])
        lens, valid = self.sp.IK(top_plate_pos=test_tm, protect=True)
        self.assertFalse(self.sp.validateInteriorAngles(donothing=True))
        self.assertTrue(self.sp.validateInteriorAngles())

    def test_kinematics_sp_validatePlateRotation(self):
        self.sp.validation_settings[3] = True
        test_tm = tm([-.1, -.1, .6, 0, 0, np.arccos(self.sp.plate_rotation_limit)+.01])
        lens, valid = self.sp.IK(top_plate_pos=test_tm, protect=True)
        self.assertFalse(self.sp.validatePlateRotation(donothing=True))
        self.assertTrue(self.sp.validatePlateRotation())

    def test_kinematics_sp_inverseJacobianSpace(self):
        invjacref = np.array([[-0.0038071, -0.072643, -0.010357, -0.20747, -0.1274, 0.96991],
                     [0.0038306, -0.073092, 0.0092824, -0.18595, 0.11419, 0.9759],
                     [0.065068, 0.033154, -0.0096382, 0.19991, -0.10929, 0.9737],
                     [0.061128, 0.039697, 0.010604, -0.015843, -0.2352, 0.97182],
                     [-0.061103, 0.039681, -0.010677, -0.015952, 0.23681, 0.97142],
                     [-0.064602, 0.032917, 0.010821, 0.22444, 0.1227, 0.96673]])
        invjac = self.sp2.inverseJacobian(tm([0, 0, .25, np.pi/8, 0, 0]))
        self.matrix_equality_assertion(invjac, invjacref)
        invjacref = np.array([[-0.0037225, -0.07103, 0.023601, 0.056149, 0.31217, 0.94837],
                     [0.0034975, -0.066736, 0.033698, -0.088068, 0.44531, 0.89103],
                     [0.062153, 0.031669, -0.026452, 0.26762, 0.25164, 0.93008],
                     [0.060905, 0.039552, -0.018364, 0.23241, 0.091689, 0.96829],
                     [-0.051976, 0.033754, -0.0082299, 0.20883, 0.52305, 0.82632],
                     [-0.053467, 0.027243, 0.010527, 0.38984, 0.45593, 0.80009]])
        invjac = self.sp2.inverseJacobian(tm([0.05, 0.1, .25, np.pi/8, -np.pi/7, np.pi/6]))
        self.matrix_equality_assertion(invjac, invjacref)

    def test_kinematics_sp_staticForces(self):
        force_sets = []
        wrench_sets = [] 
        for i in range(10):
            goal = self.goals[i]
            self.sp.IK(top_plate_pos = goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            wrench_sets.append(wrench)
            forces = self.sp.staticForces(wrench)
            force_sets.append(forces)
        self.sp.move(tm([1, 2, 3, 0, 0, 0]))
        for i in range(10):
            goal = self.sp.getBasePos() @ self.goals[i]
            self.sp.IK(top_plate_pos = goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            self.matrix_equality_assertion(wrench[0:2], wrench_sets[i][0:2], eq=False)
            forces = self.sp.staticForces(wrench)
            self.matrix_equality_assertion(forces, force_sets[i])

        self.sp.move(tm([0, 0, 0, 0, np.pi/4, np.pi/6]))
        force_sets = []
        wrench_sets = [] 
        for i in range(10):
            goal = self.goals[i]
            self.sp.IK(top_plate_pos = self.sp.getBasePos() @ goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            wrench_sets.append(wrench)
            forces = self.sp.staticForces(wrench)
            force_sets.append(forces)
        self.sp.move(tm([1, 2, 3, 0, np.pi/4, np.pi/6]))
        for i in range(10):
            goal = self.sp.getBasePos() @ self.goals[i]
            self.sp.IK(top_plate_pos = goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            self.matrix_equality_assertion(wrench[0:2], wrench_sets[i][0:2], eq=False)
            forces = self.sp.staticForces(wrench)
            self.matrix_equality_assertion(forces, force_sets[i])
        
    def test_kinematics_sp_staticForcesInv(self):
        force_sets = []
        for i in range(10):
            goal = self.goals[i]
            self.sp.IK(top_plate_pos = goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            forces = self.sp.staticForces(wrench)
            force_sets.append(forces)
            res_wrench = self.sp.staticForcesInv(forces)
            self.matrix_equality_assertion(wrench, res_wrench)
        self.sp.move(tm([1, 2, 3, 0, 0, 0]))
        for i in range(10):
            goal = self.sp.getBasePos() @ self.goals[i]
            self.sp.IK(top_plate_pos = goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            forces = self.sp.staticForces(wrench)
            res_wrench = self.sp.staticForcesInv(forces)
            self.matrix_equality_assertion(wrench, res_wrench)

        self.sp.move(tm([0, 0, 0, 0, np.pi/4, np.pi/6]))
        force_sets = []
        for i in range(10):
            goal = self.goals[i]
            self.sp.IK(top_plate_pos = self.sp.getBasePos() @ goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            forces = self.sp.staticForces(wrench)
            res_wrench = self.sp.staticForcesInv(forces)
            self.matrix_equality_assertion(wrench, res_wrench)
        self.sp.move(tm([1, 2, 3, 0, np.pi/4, np.pi/6]))
        for i in range(10):
            goal = self.sp.getBasePos() @ self.goals[i]
            self.sp.IK(top_plate_pos = goal)
            wrench = fsr.makeWrench(self.sp.getEEPos(), self.sp.grav, 5)
            forces = self.sp.staticForces(wrench)
            res_wrench = self.sp.staticForcesInv(forces)
            self.matrix_equality_assertion(wrench, res_wrench)

    def test_kinematics_sp_getJointAnglesFromNorm(self):
        joint_angles_init = self.sp.getJointAnglesFromNorm()
        self.matrix_equality_assertion(joint_angles_init, np.zeros(12))
        lens = self.sp.getLens()
        self.matrix_equality_assertion(lens, np.ones((6,1)) * (self.sp.leg_ext_max + self.sp.leg_ext_min)/2)

        self.sp.IK(self.sp.getTopT() @ tm([0, 0, .1, 0, 0, 0]))
        joint_angles = self.sp.getJointAnglesFromNorm()
        self.matrix_equality_assertion(joint_angles[0:6], joint_angles[6:12])

        self.sp.IK(tm([0, 0, 1.2, 0, 0, np.pi/6]))
        joint_angles = self.sp.getJointAnglesFromNorm()
        self.assertAlmostEqual(joint_angles[0], joint_angles[2])
        self.assertAlmostEqual(joint_angles[1], joint_angles[3])
        self.assertAlmostEqual(joint_angles[0+6], joint_angles[2+6])
        self.assertAlmostEqual(joint_angles[1+6], joint_angles[3+6])

        self.sp.IK(tm([0, 0, 1.2, 0, np.pi/6, 0]))
        joint_angles = self.sp.getJointAnglesFromNorm()
        self.assertAlmostEqual(joint_angles[0], joint_angles[1])
        self.assertAlmostEqual(joint_angles[3], joint_angles[4])
        self.assertAlmostEqual(joint_angles[0+6], joint_angles[1+6])
        self.assertAlmostEqual(joint_angles[3+6], joint_angles[4+6])

    def test_kinematics_sp_getJointAnglesFromVertical(self):
        joint_angles_init = self.sp.getJointAnglesFromVertical()
        self.assertAlmostEqual(joint_angles_init[0], .684, 2)
        #TODO actually come up some useful things for this. 

    def test_kinematics_sp_randomPos(self):
        np.random.seed(10)
        pos = self.sp.randomPos()
        ref_tm = tm([ -0.242650, -0.225996, 1.045733, 0.336068, -0.433146, 0.565761 ])
        self.matrix_equality_assertion(pos.gTM(), ref_tm.gTM())

    def test_kinematics_sp_sumActuatorWrenches(self):
        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)

        self.sp.IK(tm([.1, .2, 1.2, 0, np.pi/6, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)


        self.sp.move(tm([1, 2, 3, 0, 0, 0]))
        self.sp.IK(tm([0, 0, 1.2, 0, 0, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)

        self.sp.IK(tm([.1, .2, 1.2, 0, np.pi/6, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        bottom_wrench = self.sp.sumActuatorWrenches()
        self.matrix_equality_assertion(in_wrench, -1 * bottom_wrench)

    def test_kinematics_sp_componentForces(self):
        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))

        self.sp.IK(tm([.1, .2, 1.2, 0, np.pi/6, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))


        self.sp.move(tm([1, 2, 3, 0, 0, 0]))
        self.sp.IK(tm([0, 0, 1.2, 0, 0, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))

        self.sp.IK(tm([.1, .2, 1.2, 0, np.pi/6, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT() @ tm([1, 2, 0, 0, 0, 0]), self.sp.grav, 5)
        self.sp.staticForces(in_wrench)
        vertical, horizontal = self.sp.componentForces()
        for i in range(6):
            self.assertAlmostEqual(
                    np.sqrt(vertical[i]**2 + horizontal[i]**2), abs(self.sp._last_tau[i, 0]))

    def test_kinematics_sp_carryMassCalc(self):
        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        out_tau, out_wr = self.sp.carryMassCalc(in_wrench)
        out_wr_ref = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5 + 
                6*self.sp._act_shaft_mass + 6*self.sp._act_motor_mass + 
                self.sp._top_plate_mass + self.sp._bottom_plate_mass)
        self.matrix_equality_assertion(out_wr_ref, out_wr)

        self.sp.move(tm([1, 2, 3, 0, 0, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        out_tau, out_wr = self.sp.carryMassCalc(in_wrench)
        out_wr_ref = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5 + 
                6*self.sp._act_shaft_mass + 6*self.sp._act_motor_mass + 
                self.sp._top_plate_mass + self.sp._bottom_plate_mass)
        self.matrix_equality_assertion(out_wr_ref, out_wr)

        self.sp.IK(tm([.1, .4, 1.2, 0, np.pi/6, 0]))

        in_wrench = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5)
        out_tau, out_wr = self.sp.carryMassCalc(in_wrench)
        out_wr_ref = fsr.makeWrench(self.sp.getTopT(), self.sp.grav, 5 + 
                6*self.sp._act_shaft_mass + 6*self.sp._act_motor_mass + 
                self.sp._top_plate_mass + self.sp._bottom_plate_mass)
        self.matrix_equality_assertion(out_wr_ref, out_wr)

    def test_kinematics_sp_carryMassCalcBody(self):
        in_wrench = fsr.makeWrench(tm(), self.sp.grav, 5)
        out_tau, out_wr = self.sp.carryMassCalc(in_wrench)
        out_wr_ref = fsr.makeWrench(tm(), self.sp.grav, 5 + 
                6*self.sp._act_shaft_mass + 6*self.sp._act_motor_mass + 
                self.sp._top_plate_mass + self.sp._bottom_plate_mass)
        self.matrix_equality_assertion(out_wr_ref, out_wr)

        self.sp.move(tm([1, 2, 3, 0, 0, 0]))

        in_wrench = fsr.makeWrench(tm(), self.sp.grav, 5)
        out_tau, out_wr = self.sp.carryMassCalcBody(in_wrench)
        out_wr_ref = fsr.makeWrench(tm(), self.sp.grav, 5 + 
                6*self.sp._act_shaft_mass + 6*self.sp._act_motor_mass + 
                self.sp._top_plate_mass + self.sp._bottom_plate_mass)
        self.matrix_equality_assertion(out_wr_ref, out_wr.changeFrame(self.sp.getTopT()))

        self.sp.IK(tm([.1, .4, 1.2, 0, np.pi/6, 0]))

        in_wrench = fsr.makeWrench(tm(), self.sp.grav, 5)
        out_tau, out_wr = self.sp.carryMassCalcBody(in_wrench)
        out_wr_ref = fsr.makeWrench(tm(), self.sp.grav, 5 + 
                6*self.sp._act_shaft_mass + 6*self.sp._act_motor_mass + 
                self.sp._top_plate_mass + self.sp._bottom_plate_mass)
        self.matrix_equality_assertion(out_wr_ref, out_wr)

    def test_kinematics_sp_sp2(self):
        self.assertEqual(self.sp2.getTopT()[2], .25)
        for i in range(6):
            self.assertAlmostEqual(self.sp2.lengths[i][0], 0.257259498458319, 2)
        bjref = np.array([[0.074897, 0.074897, -0.034049, -0.040848, -0.040848, -0.034049],
                 [-0.0039252, 0.0039252, 0.066825, 0.0629, -0.0629, -0.066825],
                 [0, 0, 0, 0, 0, 0]])
        self.matrix_equality_assertion(bjref, self.sp2._bottom_joints_space)
        self.sp2.IK(tm([0, 0, .25, 0, 0, 0]))
        tjref = np.array([[0.024509, 0.024509, 0.02043, -0.044938, -0.044938, 0.02043],
               [-0.03774, 0.03774, 0.040095, 0.0023551, -0.0023551, -0.040095],
               [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
        self.matrix_equality_assertion(tjref, self.sp2._top_joints_space)

        invjac_ref = np.array([[-0.0038144, -0.072784, -0.010614, -0.19587, -0.13144, 0.97178],
                      [0.0038144, -0.072784, 0.010614, -0.19587, 0.13144, 0.97178],
                      [0.06494, 0.033088, -0.010614, 0.21177, -0.1039, 0.97178],
                      [0.061125, 0.039695, 0.010614, -0.0159, -0.23535, 0.97178],
                      [-0.061125, 0.039695, -0.010614, -0.0159, 0.23535, 0.97178],
                      [-0.06494, 0.033088, 0.010614, 0.21177, 0.1039, 0.97178]])
        invjac = self.sp2.inverseJacobian(tm([0, 0, .25, 0, 0, 0]), tm(), protect=False)
        self.matrix_equality_assertion(invjac_ref, invjac)
        topWee = np.array([[302.0785],
                  [-318.9032],
                  [-344.1081],
                  [-33.6172],
                  [16.7924],
                  [327.2833]])
        spwr = np.array([[-49.05],
                [0],
                [0],
                [0],
                [0],
                [-49.05]])
        tau = self.sp2.staticForces(spwr)
        self.matrix_equality_assertion(topWee, tau)
        tau2 = self.sp2.staticForcesBody(spwr)
        self.matrix_equality_assertion(topWee, tau2)

        self.sp2.IK(tm([0, 0, .25, 0, np.pi/8, 0]))
        tau_ref = np.array([[22.4804],
                   [6.4132],
                   [-665.124],
                   [295.4656],
                   [-310.8661],
                   [605.1336]])
        tau3 = self.sp2.staticForcesBody(spwr)
        self.matrix_equality_assertion(tau_ref, tau3)
        tau_ref = np.array([[295.6284],
                   [-312.5117],
                   [-346.6045],
                   [-31.5055],
                   [14.818],
                   [329.6894]])
        tau4 = self.sp2.staticForces(spwr)
        self.matrix_equality_assertion(tau_ref, tau4)

        self.sp2.move(tm([1, 2, 0, 0, 0, 0]))
        tau_ref = np.array([[22.4804],
                   [6.4132],
                   [-665.124],
                   [295.4656],
                   [-310.8661],
                   [605.1336]])
        tau3 = self.sp2.staticForcesBody(spwr)
        self.matrix_equality_assertion(tau_ref, tau3)
        spwr = np.array([[-147.15],
                [49.05],
                [0],
                [0],
                [0],
                [-49.05]])
        tau_ref = np.array([[295.6284],
                   [-312.5117],
                   [-346.6045],
                   [-31.5055],
                   [14.818],
                   [329.6894]])
        tau4 = self.sp2.staticForces(spwr)
        self.matrix_equality_assertion(tau_ref, tau4)

    def test_kinematics_sp_makeSP(self):
        spnew, bottom, top = makeSP(0.075, 0.045, 6, tm(), .25, 1, 0)
        spnew.setMasses(6, 0.9, 0.5, top_plate_mass = 1)
        spnew.setMaxAngleDev(55)
        spnew.leg_ext_min = self.sp2.leg_ext_min
        spnew.leg_ext_max = self.sp2.leg_ext_max
        self.matrix_equality_assertion(spnew._bottom_joints_local, self.sp2._bottom_joints_local)
        self.matrix_equality_assertion(spnew._top_joints_local, self.sp2._top_joints_local)
        self.matrix_equality_assertion(spnew._bottom_joints_space, self.sp2._bottom_joints_space)
        self.matrix_equality_assertion(spnew._top_joints_space, self.sp2._top_joints_space)
        self.matrix_equality_assertion(spnew.getBottomT().gTM(), self.sp2.getBottomT().gTM())
        self.matrix_equality_assertion(spnew.getTopT().gTM(), self.sp2.getTopT().gTM())

        self.sp2.move(tm([1, 2, 0, 0, 0, 0]))
        self.sp2.IK(tm([1, 2, .25, 0, np.pi/8, 0]))

        spnew.move(tm([1, 2, 0, 0, 0, 0]))
        spnew.IK(tm([1, 2, .25, 0, np.pi/8, 0]))

        self.matrix_equality_assertion(spnew._bottom_joints_local, self.sp2._bottom_joints_local)
        self.matrix_equality_assertion(spnew._top_joints_local, self.sp2._top_joints_local)
        self.matrix_equality_assertion(spnew.getBottomT().gTM(), self.sp2.getBottomT().gTM())
        self.matrix_equality_assertion(spnew.getTopT().gTM(), self.sp2.getTopT().gTM())
        self.matrix_equality_assertion(spnew._bottom_joints_space, self.sp2._bottom_joints_space)
        self.matrix_equality_assertion(spnew._top_joints_space, self.sp2._top_joints_space)

    def test_kinematics_sp_makeSP_inverted_rotation(self):
        # rot=-1 swaps which set of joint angles is used for the top vs
        # bottom plate ("creates an invert platform if flipped").
        spnew, bottom, top = makeSP(0.075, 0.045, 6, tm(), .25, -1, 0)
        self.assertIsNotNone(spnew)
        self.assertEqual(spnew._bottom_joints_local.shape, (3, 6))

    def test_kinematics_sp_loadSP_inverted_rotation(self):
        basic_sp = {
           "Name": "Inverted SP", "Type": "SP",
           "BottomPlate": {"Thickness": 0.1, "JointRadius": 0.9, "JointSpacing": 9, "Mass": 6},
           "TopPlate": {"Thickness": 0.16, "JointRadius": 0.3, "JointSpacing": 25, "Mass": 1},
           "Actuators": {"MinExtension": 0.75, "MaxExtension": 1.5, "MotorMass": 0.5,
               "ShaftMass": 0.9, "ForceLimit": 800, "MotorCOGD": 0.2, "ShaftCOGD": 0.2},
           "Drawing": {"TopRadius": 1, "BottomRadius": 1, "ShaftRadius": 0.1, "MotorRadius": 0.2},
           "Settings": {"MaxAngleDev": 55, "GenerateActuators": 0, "IgnoreRestHeight": 1,
               "UseSpin": 0, "AssignMasses": 1, "InferActuatorCOG": 1},
           "Params": {"RestHeight": 1.2, "Spin": 30}
        }
        with open('sp_test_data_inverted.json', 'w') as outfile:
            json.dump(basic_sp, outfile)
        try:
            sp = loadSP('sp_test_data_inverted.json', '', altRot=-1)
        finally:
            os.remove('sp_test_data_inverted.json')

        self.assertIsNotNone(sp)
        self.assertEqual(sp._bottom_joints_local.shape, (3, 6))


    # Parity additions: motion/config utilities

    def test_kinematics_sp_setNames(self):
        self.assertEqual(len(self.sp.leg_names), 6)
        self.sp.setNames('MySP', ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertEqual(self.sp.name, 'MySP')
        self.assertEqual(self.sp.leg_names, ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_kinematics_sp_setJointProperties(self):
        self.assertTrue(np.all(np.isinf(self.sp.max_leg_vels)))
        vels = np.ones(6) * 2
        accels = np.ones(6) * 3
        effort = np.ones(6) * 800
        self.sp.setJointProperties(max_vels = vels, max_effort = effort, max_accels = accels)
        self.matrix_equality_assertion(self.sp.max_leg_vels, vels)
        self.matrix_equality_assertion(self.sp.max_leg_accels, accels)
        self.matrix_equality_assertion(self.sp.max_leg_effort, effort)

    def test_kinematics_sp_timeParametrizePath(self):
        self.sp.setJointProperties(max_vels = np.ones(6), max_accels = np.ones(6) * 2)
        path = [np.copy(self.sp.lengths).flatten(), np.copy(self.sp.lengths).flatten() + 0.05]
        traj = self.sp.timeParametrizePath(path)
        self.assertGreater(traj.duration, 0)
        self.matrix_equality_assertion(traj.position(0.0), path[0])
        self.matrix_equality_assertion(traj.position(traj.duration), path[1])

    def test_kinematics_sp_lineTrajectory(self):
        target = self.sp.getTopT() @ tm([0.02, -0.01, 0, 0, 0, 0])
        leg_lengths_list = self.sp.lineTrajectory(target)
        self.assertGreater(len(leg_lengths_list), 0)
        self.matrix_equality_assertion(self.sp.getTopT().gTM(), target.gTM())

    def test_kinematics_sp_lineTrajectory_no_execute_restores_lengths(self):
        start_lengths = np.copy(self.sp.lengths)
        target = self.sp.getTopT() @ tm([0.02, 0, 0, 0, 0, 0])
        leg_lengths_list = self.sp.lineTrajectory(target, execute = False)
        self.assertGreater(len(leg_lengths_list), 0)
        self.matrix_equality_assertion(self.sp.lengths, start_lengths)

    def test_kinematics_sp_reverse_round_trip(self):
        bottom_before = self.sp.getBottomT().gTM().copy()
        top_before = self.sp.getTopT().gTM().copy()
        motor_mass_before = self.sp._act_motor_mass
        shaft_mass_before = self.sp._act_shaft_mass
        motor_cog_before = self.sp._act_motor_grav_center
        shaft_cog_before = self.sp._act_shaft_grav_center

        self.sp.reverse()
        # After a single reverse, what used to be the top plate is now fixed.
        self.matrix_equality_assertion(self.sp.getBottomT().gTM(), top_before)
        self.matrix_equality_assertion(self.sp.getTopT().gTM(), bottom_before)

        self.sp.reverse()
        self.matrix_equality_assertion(self.sp.getBottomT().gTM(), bottom_before)
        self.matrix_equality_assertion(self.sp.getTopT().gTM(), top_before)
        self.assertAlmostEqual(self.sp._act_motor_mass, motor_mass_before)
        self.assertAlmostEqual(self.sp._act_shaft_mass, shaft_mass_before)
        self.assertAlmostEqual(self.sp._act_motor_grav_center, motor_cog_before)
        self.assertAlmostEqual(self.sp._act_shaft_grav_center, shaft_cog_before)

    def test_kinematics_sp_reverse_swaps_actuator_cog(self):
        motor_mass_before = self.sp._act_motor_mass
        shaft_mass_before = self.sp._act_shaft_mass
        self.sp.reverse()
        self.assertAlmostEqual(self.sp._act_motor_mass, shaft_mass_before)
        self.assertAlmostEqual(self.sp._act_shaft_mass, motor_mass_before)

    def test_kinematics_sp_move_stationary(self):
        top_before = self.sp.getTopT().gTM().copy()
        new_base = tm([0.3, -0.1, 0.05, 0, 0, 0.1])
        self.sp.move(new_base, stationary = True)
        self.matrix_equality_assertion(self.sp.getTopT().gTM(), top_before)
        self.matrix_equality_assertion(self.sp.getBottomT().gTM(), new_base.gTM())

    # Parity additions: Jacobian extras

    def test_kinematics_sp_getManipulability(self):
        result = self.sp.getManipulability()
        self.assertEqual(len(result), 6)

    def test_kinematics_sp_numericalJacobian(self):
        self.sp.IK(tm([0.02, 0.01, 1.26, 0.05, 0.02, 0.01]))
        numerical = self.sp.numericalJacobian()
        analytic = self.sp.inverseJacobian()
        self.matrix_equality_assertion(numerical, analytic, num_dec = 2)

    # Parity additions: camera / visual servoing

    def test_kinematics_sp_addCamera_and_updateCams(self):
        cam = Camera(200, 200, 1024, 1024, 2048, 2048, 1, tm())
        self.sp.addCamera(cam, tm())
        self.assertEqual(len(self.sp.cameras), 1)
        self.sp.updateCams()

    def test_kinematics_sp_visualServoToTarget_no_camera(self):
        lengths, path = self.sp.visualServoToTarget(tm([0, 0, 2, 0, 0, 0]))
        self.assertEqual(path, [])

    def test_kinematics_sp_visualServoToTarget(self):
        cam = Camera(200, 200, 1024, 1024, 2048, 2048, 1, tm())
        self.sp.addCamera(cam, tm())
        target = self.sp.getTopT() @ tm([0, 0, 1, 0, 0, 0])
        lengths, path = self.sp.visualServoToTarget(target, max_iter = 50)
        self.assertGreater(len(path), 0)

    # Parity additions: dynamics

    def test_kinematics_sp_inverseDynamics_matches_static_carryMassCalc(self):
        # With zero platform acceleration, inverseDynamics must reduce exactly
        # to the existing quasi-static carryMassCalc actuator forces.
        wrench = fsr.makeWrench(self.sp.getTopT(), 5.0, self.sp.grav / fmr.Norm(self.sp.grav))
        tau_static, _ = self.sp.carryMassCalc(wrench)
        tau_dynamic = self.sp.inverseDynamics(
                np.zeros(3), np.zeros(3), np.zeros(3), top_plate_wrench = wrench)
        self.matrix_equality_assertion(
                np.asarray(tau_static).flatten(), np.asarray(tau_dynamic).flatten())

    def test_kinematics_sp_massMatrix_affine_decomposition(self):
        # inverseDynamics must be exactly affine in [angular_accel; linear_accel]:
        # tau == M @ accel + coriolisGravity(w).
        w = np.array([0.2, -0.1, 0.05])
        accel = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.05])
        M = self.sp.massMatrix(w)
        h = np.asarray(self.sp.coriolisGravity(w)).flatten()
        tau_affine = M @ accel + h
        tau_direct = np.asarray(
                self.sp.inverseDynamics(w, accel[0:3], accel[3:6])).flatten()
        self.matrix_equality_assertion(tau_affine, tau_direct)

    def test_kinematics_sp_forwardDynamics_inverts_inverseDynamics(self):
        w = np.array([0.1, 0.05, -0.2])
        accel = np.array([0.05, -0.1, 0.2, 0.3, -0.2, 0.1])
        tau = np.asarray(self.sp.inverseDynamics(w, accel[0:3], accel[3:6])).flatten()
        recovered = self.sp.forwardDynamics(tau, w)
        self.matrix_equality_assertion(recovered, accel, num_dec = 5)

    def test_kinematics_sp_setTopPlateInertia_adds_euler_moment(self):
        # Isolate the platform's own Euler-equation moment by zeroing leg masses,
        # which would otherwise also pick up alpha-dependent inertial forces.
        self.sp.setMasses(self.sp._bottom_plate_mass, 0.0, 0.0,
                top_plate_mass = self.sp._top_plate_mass)
        inertia = np.diag([0.01, 0.02, 0.03])
        self.sp.setTopPlateInertia(inertia)
        alpha = np.array([1.0, 0, 0])
        tau0 = np.asarray(self.sp.inverseDynamics(np.zeros(3), np.zeros(3), np.zeros(3))).flatten()
        tau1 = np.asarray(self.sp.inverseDynamics(np.zeros(3), alpha, np.zeros(3))).flatten()
        rotation = self.sp.getTopT().gRot()
        inertia_space = rotation @ inertia @ rotation.T
        expected_extra = self.sp.jacobian().T @ np.hstack((inertia_space @ alpha, np.zeros(3)))
        self.matrix_equality_assertion(tau1 - tau0, expected_extra)

    def test_kinematics_sp_integrateForwardDynamics_free_fall(self):
        # With zero actuator force and no leg masses, the top plate origin
        # must fall exactly as a point mass under gravity: z(t) = z0 + 0.5*g*t^2.
        self.sp.setMasses(self.sp._bottom_plate_mass, 0.0, 0.0,
                top_plate_mass = self.sp._top_plate_mass)
        z0 = self.sp.getTopT().gPos().flatten()[2]
        times, poses, angular_vels = self.sp.integrateForwardDynamics(
                np.zeros(3), np.zeros(6), dt = 0.05, n_steps = 20)
        z = np.array([p.gPos().flatten()[2] for p in poses])
        expected_z = z0 + 0.5 * self.sp.grav[2] * times ** 2
        self.matrix_equality_assertion(z, expected_z, num_dec = 5)


if __name__ == '__main__':
    unittest.main()
