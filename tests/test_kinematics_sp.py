import unittest
import numpy as np
from basic_robotics.general import tm, fsr, fmr
import json
import os
from basic_robotics.kinematics import loadSP
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
        self.sp.IK(top_plate_pos = tm([0, 0, 1.2, 0, 0, 0]))
        #Delete file
        os.remove('sp_test_data.json')

    def test_kinematics_sp_getActuatorLoc(self):
        pass #TODO

    def test_kinematics_sp_spinCustom(self):
        pass #TODO

    def test_kinematics_sp_IK(self):
        for goal in self.goals:
            self.sp.IK(top_plate_pos = goal)
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)
            self.assertTrue(self.sp.validate(True))

    def test_kinematics_sp_IKHelper(self):
        pass #TODO

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
        for j in range(10):
            goal = self.goals[j]
            self.sp.IK(top_plate_pos = goal)
            lens = self.sp.getLens().flatten()
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)
            self.sp.IK(top_plate_pos = tm([0, 0, 1.2, 0, 0, 0]))
            self.assertTrue(self.sp.validate(True))
            self.sp.FKSolve(lens)
            #print(result_top)
            result_lens = self.sp.getLens().flatten()
            for i in range(6):
                self.assertAlmostEqual(lens[i], result_lens[i], 2)
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)

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
            self.sp.FKRaphson(lens)
            #print(result_top)
            result_lens = self.sp.getLens().flatten()
            for i in range(6):
                self.assertAlmostEqual(lens[i], result_lens[i], 2)
            result_top = self.sp.getTopT()
            for i in range(6):
                self.assertAlmostEqual(result_top[i], goal[i], 3)

    def test_kinematics_sp_lambdaTopPlateReorientation(self):
        pass #TODO

    def test_kinematics_sp_reorientTopPlate(self):
        pass #TODO

    def test_kinematics_sp_fixUpsideDown(self):
        pass #TODO

    def test_kinematics_sp_validateLegs(self):
        pass #TODO

    def test_kinematics_sp_validateContinuousTranslation(self):
        pass #TODO

    def test_kinematics_sp_validateInteriorAngles(self):
        pass #TODO

    def test_kinematics_sp_validatePlateRotation(self):
        pass #TODO

    def test_kinematics_sp_validate(self):
        pass #TODO

    def test_kinematics_sp_plateRotationConstraint(self):
        pass #TODO

    def test_kinematics_sp_legLengthConstraint(self):
        pass #TODO

    def test_kinematics_sp_rescaleLegLengths(self):
        pass #TODO

    def test_kinematics_sp_addLegsToMinimum(self):
        pass #TODO

    def test_kinematics_sp_subLegsToMaximum(self):
        pass #TODO

    def test_kinematics_sp_lengthCorrectiveAction(self):
        pass #TODO

    def test_kinematics_sp_continuousTranslationConstraint(self):
        pass #TODO

    def test_kinematics_sp_continuousTranslationCorrectiveAction(self):
        pass #TODO

    def test_kinematics_sp_getJointAnglesFromNorm(self):
        pass #TODO

    def test_kinematics_sp_getJointAnglesFromVertical(self):
        pass #TODO

    def test_kinematics_sp_componentForces(self):
        pass #TODO

    def test_kinematics_sp_bottomTopCheck(self):
        pass #TODO

    def test_kinematics_sp_jacobianSpace(self):
        pass #TODO

    def test_kinematics_sp_inverseJacobianSpace(self):
        pass #TODO

    def test_kinematics_sp_altInverseJacobianSpace(self):
        pass #TODO

    def test_kinematics_sp_carryMassCalc(self):
        pass #TODO

    def test_kinematics_sp_carryMassCalcLocal(self):
        pass #TODO

    def test_kinematics_sp_measureForcesAtEENew(self):
        pass #TODO

    def test_kinematics_sp_carryMassCalcUp(self):
        pass #TODO

    def test_kinematics_sp_measureForcesFromWrenchEE(self):
        pass #TODO

    def test_kinematics_sp_measureForcesFromBottomEE(self):
        pass #TODO

    def test_kinematics_sp_wrenchBottomFromMeasuredForces(self):
        pass #TODO

    def test_kinematics_sp_sumActuatorWrenches(self):
        pass #TODO

    def test_kinematics_sp_move(self):
        pass #TODO

if __name__ == '__main__':
    unittest.main()
