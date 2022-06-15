"""Holding file for Stewart Platform specific functions, notably the SP class."""
import json

import numpy as np
import scipy as sci
from sqlalchemy import true

from ..general import Wrench, fmr, fsr, tm
from ..plotting.Draw import DrawSP
from ..utilities.disp import disp
from .robot_model import Robot


class SP(Robot):
    """Models a stewart platform."""

    #Conventions:
    #Filenames:  snake_case
    #Variables: snake_case
    #Functions: camelCase
    #ClassNames: CapsCase
    #Docstring: Google

    def __init__(self, bottom_joints : 'np.ndarray[float]', top_joints : 'np.ndarray[float]', 
            bT : tm , tT : tm, leg_ext_min : float, leg_ext_max : float, 
            bottom_plate_thickness : float, top_plate_thickness : float, name : str) -> 'SP':
        """
        Initialize a new Stewart Platform Object.

        Args:
            bottom_joints (ndarray): Bottom joint positions of the stewart platform
            top_joints (ndarray): Top joint positions of the stewart platform
            bT (tm): bottom plate position
            tT (tm): top plate position
            leg_ext_min (float): minimum leg ext limit
            leg_ext_max (float): maximum leg ext limit
            bottom_plate_thickness (float): bottom plate thickness
            top_plate_thickness (float): top plate thickness
            name (string): name of the sp
        Returns:
            SP: sp model object

        """
        super().__init__(name)

        #State Variables
        self._bottom_joints_local = np.copy(bottom_joints)
        self._top_joints_local = np.copy(top_joints)
        self._bottom_joints_space = np.zeros((3, 6))
        self._top_joints_space = np.zeros((3, 6))
        self._base_pos_global = bT.copy()
        self._end_effector_pos_global = tT.copy()
        self._current_plate_transform_local = tm()
        self._last_tau =  np.zeros(6)
        
        #Initialization variables / static 
        self._bottom_joints_init = self._bottom_joints_local.conj().transpose()
        self._top_joints_init = self._top_joints_local.conj().transpose()
        self._top_joint_angles_init = [None] * 6
        self._bottom_joint_angles_init = [None] * 6
        
        #Debug
        self._leg_ext_safety = .001
        self.debug = 0

        #Physical Parameters
        self.bottom_plate_thickness = bottom_plate_thickness
        self.top_plate_thickness = top_plate_thickness
        self._plate_thickness_avg = (self.top_plate_thickness + self.bottom_plate_thickness) / 2
        if leg_ext_min == 0:
            self.leg_ext_min = 0
            self.leg_ext_max = 2
        self.leg_ext_min = leg_ext_min
        self.leg_ext_max = leg_ext_max

        #Reserve Val
        self._nominal_height = fsr.distance(bT, tT)
        self._nominal_plate_transform = tm([0, 0, self._nominal_height, 0, 0, 0])

        self._aux_inits = [
            self._nominal_plate_transform,
            self._nominal_plate_transform @ tm([np.pi/8, 0, 0]),
            self._nominal_plate_transform @ tm([0, -np.pi/8, 0]),
            self._nominal_plate_transform @ tm([-np.pi/8, 0, np.pi/8]),
            self._nominal_plate_transform @ tm([.25, .25, -.25, np.pi/10, 0, .001])
        ]

        #Drawing Characteristics
        self._outer_top_radius = 0
        self._outer_bottom_radius = 0
        self._act_shaft_radius = 0
        self._act_motor_radius = 0

        #Mass values from bottom mass, top mass, and actuator portion masses can be set directly.
        self._bottom_plate_mass = 0
        self._top_plate_mass = 0
        self._act_shaft_mass = 0
        self._act_motor_mass = 0
        self._act_shaft_grav_center = 0
        self._act_motor_grav_center = 0

        #Tolerances and Limits
        self.joint_deflection_max = 140/2*np.pi/180#2*np.pi/5
        self.plate_rotation_limit = np.cos(60*np.pi/180)

        #Newton Settings
        self._tol_f = 1e-5/2
        self._tol_a = 1e-5/2
        self._max_iterations = 1e4

        #Errors and Counts
        self.fail_count = 0
        self.validation_settings = [1, 0, 0, 1]
        self.fk_mode = 1
        self.validation_error = ""

        self.IK(top_plate_pos = tT, bottom_plate_pos = bT, protect = True)

        #disp([bT, tT])

        for i in range(6):
            self._bottom_joint_angles_init[i] = fsr.globalToLocal(
                self.getBottomT(),
                tm([self._top_joints_space.T[i][0], self._top_joints_space.T[i][1],
                self._top_joints_space.T[i][2], 0, 0, 0]))
            self._top_joint_angles_init[i] = fsr.globalToLocal(
                self.getTopT(),
                tm([self._bottom_joints_space.T[i][0], self._bottom_joints_space.T[i][1],
                self._bottom_joints_space.T[i][2], 0, 0, 0]))

        t1 = fsr.globalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self._top_joints_space[0, 0],
            self._top_joints_space[1, 0],
            self._top_joints_space[2, 0], 0, 0, 0]))
        t2 = fsr.globalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self._top_joints_space[0, 2],
            self._top_joints_space[1, 2],
            self._top_joints_space[2, 2], 0, 0, 0]))
        t3 = fsr.globalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self._top_joints_space[0, 4],
            self._top_joints_space[1, 4],
            self._top_joints_space[2, 4], 0, 0, 0]))
        self.reorients = [t1, t2, t3]

    """
    Getters and Setters
    """
    def setDrawingParameters(self, top_plate_radius : float = None, 
            bottom_plate_radius : float = None, 
            act_shaft_radius : float = None, 
            act_motor_radius : float = None) -> None:
        """
        Set drawing parameters for stewart platform.

        Primarily used by 3JS visualizer. 
        Values set here (such as plate radius) have no effect on kinematics.
        Args:
            top_plate_radius (float, optional): Top plate radius (m). Defaults to None.
            bottom_plate_radius (float, optional): Bottom plate radius (m). Defaults to None.
            act_shaft_radius (float, optional): Actuator shaft radius (m). Defaults to None.
            act_motor_radius (float, optional): Actuator motor radius (m). Defaults to None.
        """        
        if top_plate_radius is not None: 
            self._outer_top_radius = top_plate_radius
        if bottom_plate_radius is not None: 
            self._outer_bottom_radius = bottom_plate_radius
        if act_shaft_radius is not None:
            self._act_shaft_radius = act_shaft_radius 
        if act_motor_radius is not None: 
            self._act_motor_radius = act_motor_radius

    def setMasses(self, plate_mass_general : float, 
        act_shaft_mass : float, act_motor_mass : float, 
        grav : 'np.ndarray[float]' = np.array([0, 0, -9.81]), top_plate_mass : float = 0.0) -> None:
        """
        Set SP Masses and gravity vector.

        share plates, these weights are halved with respect to end plates
        Args:
            plate_mass_general (float): mass of bottom plate (both if top is not specified) (kg)
            act_shaft_mass (float): mass of actuator shaft (kg)
            act_motor_mass (float): mass of actuator motor (kg)
            grav (np.ndarray[float]):  [Optional, default 9.81] acceleration due to gravity
            top_plate_mass (float): [Optional, default 0] top plate mass (kg)
        """
        self._bottom_plate_mass = plate_mass_general
        if top_plate_mass != 0:
            self._top_plate_mass = top_plate_mass
        else:
            self._top_plate_mass = plate_mass_general
        self.setGrav(grav)
        self._act_shaft_mass = act_shaft_mass
        self._act_motor_mass = act_motor_mass

    def setCOG(self, motor_grav_center : float, shaft_grav_center : float) -> None:
        """
        Set the centers of gravity for actuator components.

        Args:
            motor_grav_center (float): distance from top of actuator to actuator shaft COG
            shaft_grav_center (float): distance from bottom of actuator to actuator motor COG
        """
        self._act_shaft_grav_center = shaft_grav_center
        self._act_motor_grav_center = motor_grav_center

    def setMaxAngleDev(self, max_angle_dev : float = 1.0, degrees : bool = True) -> None:
        """
        Set the maximum angle joints can deflect before failure.

        Args:
            max_angle_dev (float, Optional): maximum deflection angle. Default 1.0 radians.
            degrees (bool, Optional): use degrees instead. Defaults to true.
        """
        if degrees:
            max_angle_dev = fsr.deg2Rad(max_angle_dev)
        self.joint_deflection_max = max_angle_dev

    def setMaxPlateRotation(self, 
            max_plate_rotation : float = 1.25, degrees : bool = False) -> True:
        """
        Set the maximum angle the plate can rotate before failure.

        Args:
            max_plate_rotation (Float): Maximum angle before plate rotation failure. Default 1.25 radians.
            degrees (bool, Optional): use degrees instead. Defaults to true.
        """
        if degrees:
            max_plate_rotation = fsr.deg2Rad(max_plate_rotation)
        self.plate_rotation_limit = np.cos(max_plate_rotation)

    def getBottomJoints(self) -> 'np.ndarray[float]':
        """
        Get the bottom joint positions in space.

        Does not return orientation, only position.

        Returns:
            ndarray(Float): bottom joint positions
        """
        return self._bottom_joints_space

    def getTopJoints(self) -> 'np.ndarray[float]':
        """
        Get the top joint positions in space.

        Does not return orientation, only position.
        Returns:
            ndarray(Float): top joint positions in space
        """
        return self._top_joints_space

    def getCurrentLocalTransform(self) -> tm:
        """
        Get the current local transform from the bottom plate to the top plate.

        Returns:
            tm: Top plate relative to bottom plate
        """
        return self._current_plate_transform_local.copy()

    def getLens(self) -> 'np.ndarray[float]':
        """
        Get Actuator lengths in meters (joint to joint).

        Returns:
            ndarray(Float): Actuator lengths in meters.
        """
        return self.lengths.copy()

    def getTopT(self) -> float:
        """
        Return the global frame transform of the top plate.

        Synonymous with getEEPos(), and will be removed eventuallly.

        Returns:
            tm: top plate transform in space frame
        """
        return self._end_effector_pos_global.copy()

    def getBottomT(self) -> float:
        """
        Return the global frame transform of the bottom plate.

        Synonymous with getBasePos() and will be removed eventually.

        Returns:
            tm: bottom plate transform in space frame
        """
        return self._base_pos_global.copy()

    def getActuatorLoc(self, num : int, type  : str = 'm') -> tm:
        """
        Return the position (no orientation) of a specified actuator in the global (space) frame.

        Takes in an actuator number and a type.
        m for actuator midpoint
        b for actuator motor position
        t for actuator top position

        Args:
            num (Int): number of actuator to return
            type (Str): property of actuator to return

        Returns:
           tm: location of desired component of an actuator in the global frame.
        """
        pos = 0
        if type == 'm':
            pos = np.array([(self._bottom_joints_space[0, num] + self._top_joints_space[0, num])/2,
                (self._bottom_joints_space[1, num] + self._top_joints_space[1, num])/2,
                (self._bottom_joints_space[2, num] + self._top_joints_space[2, num])/2])
        bottom_act_joint = tm([self._bottom_joints_space[0, num],
            self._bottom_joints_space[1, num], self._bottom_joints_space[2, num], 0, 0, 0])
        top_act_joint = tm([self._top_joints_space[0, num],
            self._top_joints_space[1, num], self._top_joints_space[2, num], 0, 0, 0])
        if type == 'b':
            #return fsr.adjustRotationToMidpoint(bottom_act_joint, bottom_act_joint,
            #   top_act_joint, mode = 1) @ tm([0, 0, self._act_motor_grav_center, 0, 0, 0])
            return fsr.getUnitVec(bottom_act_joint,
                top_act_joint, self._act_motor_grav_center)
        if type == 't':
            #return fsr.adjustRotationToMidpoint(top_act_joint, top_act_joint, bottom_act_joint,
            #   mode = 1) @ tm([0, 0, self._act_shaft_grav_center, 0, 0, 0])
            return fsr.getUnitVec(top_act_joint,
                bottom_act_joint, self._act_shaft_grav_center)
        new_position = tm([pos[0], pos[1], pos[2], 0, 0, 0])
        return new_position

    def getJointAnglesFromNorm(self) -> 'np.ndarray[float]':
        """
        Return the angular deviation of each angle socket from its nominal position in radians.

        Returns:
            ndarray(Float): Angular deviation from home of each joint socket
        """
        delta_angles_top = np.zeros((6))
        delta_angles_bottom = np.zeros((6))
        bottom_plate_transform = self.getBottomT()
        top_plate_transform = self.getTopT()
        for i in range(6):

                top_joint_i = tm([
                    self._top_joints_space.T[i][0],
                    self._top_joints_space.T[i][1],
                    self._top_joints_space.T[i][2],
                    top_plate_transform[3],
                    top_plate_transform[4],
                    top_plate_transform[5]])

                bottom_joint_i = tm([
                    self._bottom_joints_space.T[i][0],
                    self._bottom_joints_space.T[i][1],
                    self._bottom_joints_space.T[i][2],
                    bottom_plate_transform[3],
                    bottom_plate_transform[4],
                    bottom_plate_transform[5]])

                #We have the relative positions to the top plate
                #   of the bottom joints (bottom angles) in home pose
                #We have the relative positions to the bottom plate of
                #   the top joints (bottom_joint_angles_init) in home pose
                bottom_to_top_local_home = self._bottom_joint_angles_init[i].copy()
                top_to_bottom_local_home = self._top_joint_angles_init[i].copy()

                #We acquire the current relative (local positions of each)
                bottom_to_top_local = fsr.globalToLocal(self.getBottomT(), top_joint_i)
                top_to_bottom_local = fsr.globalToLocal(self.getTopT(), bottom_joint_i)

                #We acquire the base positions of each joint
                bottom_to_bottom_local = fsr.globalToLocal(self.getBottomT(), bottom_joint_i)
                top_to_top_local = fsr.globalToLocal(self.getTopT(), top_joint_i)

                delta_angles_bottom[i] = fsr.angleBetween(
                    bottom_to_top_local,
                    bottom_to_bottom_local,
                    bottom_to_top_local_home)
                    
                delta_angles_top[i] = fsr.angleBetween(
                    top_to_bottom_local,
                    top_to_top_local,
                    top_to_bottom_local_home)

            #DeltAnglesA are the Angles From Norm Bottom
            #DeltAnglesB are the Angles from Norm TOp
        return np.hstack((delta_angles_bottom, delta_angles_top))

    def getJointAnglesFromVertical(self) -> tuple['np.ndarray[float]', 'np.ndarray[float]']:
        """
        Calculate joint angles from vertical at each joint.

        Returns:
            ndarray(Float): top joints from vertical (downward)
            ndarray(Float): bottom joints from vertical (upward)
        """
        top_down = np.zeros((6))
        bottom_up = np.zeros((6))
        for i in range(6):
            top_joints_temp = self._top_joints_space[:, i].copy().flatten()
            top_joints_temp[2] = top_joints_temp[2] - 1
            bottom_joints_temp = self._bottom_joints_space[:, i].copy().flatten()
            bottom_joints_temp[2] = bottom_joints_temp[2] + 1
            angle = fsr.angleBetween(
                self._bottom_joints_space[:, i],
                self._top_joints_space[:, i],
                top_joints_temp)
            angle_up = fsr.angleBetween(
                self._top_joints_space[:, i],
                self._bottom_joints_space[:, i],
                bottom_joints_temp)
            top_down[i] = angle
            bottom_up[i] = angle_up
        return np.hstack((top_down, bottom_up))

    """ 
    Kinematics 
    """

    def randomPos(self, max_attempts = 100, min_deviation = np.pi/10):
        """
        Generate a random configuration.

        Args:
            max_attempts (int, optional): Maximum attempts before giving up. Defaults to 100.
            min_deviation (float, optional): Minimum acceptable angular deviation (avoids 'home-like' poses). Defaults to np.pi/10.
        Returns:
            tm: new top pose in global space
        """        
        done = False 
        attempt = 0
        while not done and attempt < max_attempts:
            leg_lengths = np.random.uniform(
                    self.leg_ext_min + self._leg_ext_safety, 
                    self.leg_ext_max - self._leg_ext_safety, ((6)))
            self.FK(leg_lengths)
            if (self.validate() == True and 
                    fmr.Norm(self._current_plate_transform_local[3:6]) > min_deviation):
                done = true
            attempt+=1
        return self.getTopT()


    def spinCustom(self, rot : float, degrees : bool = False) -> None:
        """
        Rotate platform by radian amount while maintaining orientation in global space.

        This function is useful for setting an initial coordinate alignment with the platform relative
        to global space that is not generated by default upon loading the platform.
        This function is NOT absolute, instead relative, so to undo a rotation, one would 
        Need to apply the inverse rotation (e.g. -pi/4 to counteract a pi/4 rotation.)
        Args:
            rot (Float): rotation in radians (clockwise)
            degrees (bool) : use degrees instead of radians
        """
        if degrees:
            rot = fsr.deg2Rad(rot)
        old_base_pos = self.getBottomT()
        self.move(tm())
        top_joints_copy = self._top_joints_space.copy()
        bottom_joints_copy = self._bottom_joints_space.copy()
        top_joints_origin_copy = self._top_joints_local[2, 0:6]
        bottom_joints_origin_copy = self._bottom_joints_local[2, 0:6]
        rotation_transform = tm([0, 0, 0, 0, 0, rot])
        self.move(rotation_transform)
        top_joints_space_new = self._top_joints_space.copy()
        bottom_joints_space_new = self._bottom_joints_space.copy()
        top_joints_copy[0:2, 0:6] = top_joints_space_new[0:2, 0:6]
        bottom_joints_copy[0:2, 0:6] = bottom_joints_space_new[0:2, 0:6]
        bottom_joints_copy[2, 0:6] = bottom_joints_origin_copy
        top_joints_copy[2, 0:6] = top_joints_origin_copy
        self.move(tm())
        self._bottom_joints_local = bottom_joints_copy
        self._top_joints_local = top_joints_copy
        self._bottom_joints_space = bottom_joints_space_new
        self._top_joints_space = top_joints_space_new
        self.move(old_base_pos)


    def IK(self, top_plate_pos : tm = None, bottom_plate_pos : tm = None, 
            protect : bool = False) -> tuple['np.ndarray[float]' , bool]:
        """
        Calculate inverse kinematics for given goals.

        Args:
            top_plate_pos (tm): top plate position
            bottom_plate_pos (tm): bottom plate position
            protect (bool): If true, bypass any safeties
        Returns:
            ndarray(float): leg lengths
            bool: validity of pose
        """
        bottom_plate_pos, top_plate_pos = self._bottomTopCheck(bottom_plate_pos, top_plate_pos)

        leg_lengths, bottom_plate_pos, top_plate_pos = self._IKHelper(
            top_plate_pos, bottom_plate_pos)
        #Determine current transform

        self._setPlatePos(bottom_plate_pos, top_plate_pos)

        #Ensure a valid position
        valid = True
        if not protect:
            valid = self.validate()
        return leg_lengths, valid

    def FK(self, L, plate_pos = None, reverse = False, 
            protect = False, fk_mode = None) -> tuple[tm, bool]:
        """
        Calculate Forward Kinematics for desired leg lengths (joint to joint).

        Args:
            L (ndarray(Float)): Goal leg lengths
            plate_pos (tm): Fixed plate location
            reverse (Bool): Boolean to reverse action. If true, treat the top plate as stationary.
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tm: top plate configuration
            Bool: validity

        """
        if fk_mode is None: 
            fk_mode = self.fk_mode
        if plate_pos is None or reverse:
            #If no bottom pose is supplied, use the last known.
            bottom_plate_pos = self.getBottomT()
            if reverse: 
                plate_pos = self.getTopT()
        else:
            bottom_plate_pos = plate_pos
        #FK host function, calls subfunctions depedning on the value of fk_mode
        #return self.FKSciRaphson(L, bottom_plate_pos, reverse, protect)
        #bottom_plate_pos, n = self._applyPlateTransform(bottom_plate_pos = bottom_plate_pos)
        if fk_mode == 0:
            bottom, top = self._FKSolve(L, bottom_plate_pos, protect)
        else:
            bottom, top = self._FKRaphson(L, bottom_plate_pos, protect)

        if not self._continuousTranslationConstraint():
            if self.debug:# pragma: no cover
                disp("FK Resulted In Inverted Plate Alignment. Repairing...")
            #self.IK(top_plate_pos = self.getBottomT() @ tm([0, 0, self._nominal_height, 0, 0, 0]))
            #self.FK(L, protect = True)
            self._fixUpsideDown()
        self._current_plate_transform_local = fsr.globalToLocal(bottom, top)
        #self._undoPlateTransform(bottom, top)

        valid = True
        if not protect:
            valid = self.validate()
        
        if reverse:
            self.IK(plate_pos, plate_pos @ self._current_plate_transform_local.inv())

        return top, valid

    """ 
    Validation and Corrective Actions
    """

    def validate(self, donothing :bool = False, validation_limit : int = 4) -> bool:
        """
        Validate the current configuration of the stewart platform.

        Perform corrective action if necessary.
        Args:
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it
            validation_limit (Int): number at which to stop validating. For internal use.
        Returns:
            Bool: Validity of configuration
        """
        valid = True #innocent until proven INVALID
        #if self.debug:
        #    disp("Validating")
        #First check to make sure leg lengths are not exceeding limit points
        if fsr.distance(self.getTopT(), self.getBottomT()) > 2 * self._nominal_height:
            valid = False

        if validation_limit > 0: valid = self.validateLegs(valid, donothing)
        if validation_limit > 1: valid = self.validateContinuousTranslation(valid, donothing)
        if validation_limit > 2: valid = self.validateInteriorAngles(valid, donothing)
        if validation_limit > 3: valid = self.validatePlateRotation(valid, donothing)

        if valid:
            self.validation_error = ""

        return valid

    def validateLegs(self, valid : bool = True, donothing : bool = False) -> bool:
        """
        Validate leg lengths against leg bounds and perform corrective action if necessary.

        Args:
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it
        Returns:
            Bool: Validity of configuration
        """
        if self.validation_settings[0]:
            temp_valid = self._legLengthConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Leg Length Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:# pragma: no cover
                    disp("Executing Length Corrective Action...")
                self._lengthCorrectiveAction()
                valid = self.validate(True, 1)
        return valid

    def validateContinuousTranslation(self, valid : bool = True, donothing :bool = False) -> bool:
        """
        Validate that the top plate has positive Z in bottom plate's local frame.

        There is no situation where the top plate should be 'underneath' the bottom.
        Performs corrective action if necessary.
        Args:
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it
        Returns:
            Bool: Validity of configuration
        """
        if self.validation_settings[1]:
            temp_valid = self._continuousTranslationConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Platform Inversion Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:# pragma: no cover
                    disp("Executing Continuous Translation Corrective Action...")
                self._continuousTranslationCorrectiveAction()
                valid = self.validate(True, 2)
        return valid

    def validateInteriorAngles(self, valid : bool = True, donothing :bool = False) -> bool:
        """
        Validate that the interior angles of the legs from normal are within limits.

        Perform corrective action if necessary.
        Args:
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it
        Returns:
            Bool: Validity of configuration
        """
        if self.validation_settings[2]:
            temp_valid = self._interiorAnglesConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Interior Angles Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:# pragma: no cover
                    disp("Executing Interior Angles Corrective Action...")
                self.IK(top_plate_pos = self.getBottomT() @
                    self._nominal_plate_transform, 
                    bottom_plate_pos = self.getBottomT(), protect = True)
                valid = self.validate(True, 3)
        return valid

    def validatePlateRotation(self, valid : bool = True, donothing :bool = False) -> bool:
        """
        Validate that plate rotation is not exceeding bounds.

        Perform corrective action if necessary. 
        Args:
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it
        Returns:
            Bool: Validity of configuration
        """
        if self.validation_settings[3]:
            temp_valid = self._plateRotationConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Plate Tilt/Rotate Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:# pragma: no cover
                    disp("Executing Plate Rotation Corrective Action By Resetting Platform")
                #disp(self._nominal_plate_transform)
                self.IK(top_plate_pos = self.getBottomT() @ self._nominal_plate_transform, 
                        bottom_plate_pos = self.getBottomT(), protect = True)
                valid = self.validate(True, 4)
        return valid

    """
    Jacobian Functions
    """

    def inverseJacobian(self, top_plate_pos : tm = None, 
            bottom_plate_pos : tm = None, protect : bool = True) -> 'np.ndarray[float]':
        """
        Calculate Inverse Jacobian for stewart platform. Optionally use top and bottom transforms.

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            ndarray(Float): Inverse Jacobian for current configuration
        """
        #Ensure everything is kosher with the plates
        bottom_plate_pos, top_plate_pos = self._bottomTopCheck(bottom_plate_pos, top_plate_pos)

        #Store old values
        old_bottom_plate_transform = self.getBottomT()
        old_top_plate_transform = self.getTopT()

        #Perform IK on bottom and top
        self.IK(top_plate_pos = top_plate_pos, bottom_plate_pos = bottom_plate_pos, protect = protect)

        #Create Jacobian
        inverse_jacobian_transpose = np.zeros((6, 6))
        for i in range(6):
            #todo check sign on nim,
            ni = fmr.Normalize(self._top_joints_space[:, i]-self._bottom_joints_space[:, i])
             #Reverse for upward forces?
            qi = self._bottom_joints_space[:, i]
            col = np.hstack((np.cross(qi, ni), ni))
            inverse_jacobian_transpose[:, i] = col
        inverse_jacobian = inverse_jacobian_transpose.T

        #Restore original Values
        self.IK(top_plate_pos = old_top_plate_transform, 
                bottom_plate_pos = old_bottom_plate_transform, protect = protect)
        return inverse_jacobian

    """ 
    Force Calculations
    """

    def carryMassCalc(self, twrench : Wrench, 
            protect : bool = False) -> tuple['np.ndarray[float]', Wrench]:
        """
        Calculate the forces on each actuator given actuator masses, plate masses, and a wrench on the end effector in the global frame.

        Use this over the body frame equivalent in most cases,
        Args:
            twrench (ndarray(Float)): input wrench for configuration
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True
        Returns:
            ndarray(Float): forces in Newtons for each leg
        """
        wrench = twrench.copy()
        wrench = wrench + fsr.makeWrench(self.getTopT(),
            self._top_plate_mass, self.grav)
        
        for i in range(6):
            #print(self.getActuatorLoc(i, 't'))
            wrench += fsr.makeWrench(self.getActuatorLoc(i, 't'),
                self._act_shaft_mass, self.grav)
        tau = self.staticForces(wrench, protect = protect)
        for i in range(6):
            wrench += fsr.makeWrench(self.getActuatorLoc(i, 'b'),
                self._act_motor_mass, self.grav)
        wrench = wrench + fsr.makeWrench(self.getBottomT(),
            self._bottom_plate_mass, self.grav)
        return tau, wrench

    def carryMassCalcBody(self, twrench : Wrench, 
            protect : bool = False) -> tuple['np.ndarray[float]', Wrench]:
        """
        Calculate the forces on each actuator given actuator masses, plate masses, and a wrench in the end effector frame.

        Args:
            twrench (ndarray(Float)): input wrench for configuration
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True
        Returns:
            ndarray(Float): forces in Newtons for each leg
        """
        #We will here assume that the wrench is in the local frame of the top platform.
        wrench = twrench.copy()
        wrench = wrench + fsr.makeWrench(tm(), self._top_plate_mass, self.grav)
        wrench_local_frame = fsr.transformWrenchFrame(wrench, self.getTopT(), self.getBottomT())
        for i in range(6):
            wrench_local_frame += fsr.makeWrench(fsr.globalToLocal(self.getBottomT(), 
                    self.getActuatorLoc(i, 't')), self._act_shaft_mass, self.grav, self.getBottomT())
        tau = self.staticForcesBody(wrench.copy().changeFrame(self.getTopT()), protect = protect)
        for i in range(6):
            wrench_local_frame += fsr.makeWrench(fsr.globalToLocal(self.getBottomT(),
                     self.getActuatorLoc(i, 'b')), self._act_motor_mass, self.grav, self.getBottomT())
        wrench_local_frame = wrench_local_frame + fsr.makeWrench(tm(),
            self._bottom_plate_mass, self.grav, self.getBottomT())
        return tau, wrench_local_frame

    def componentForces(self, forces : 
            'np.ndarray[float]' = None) -> tuple['np.ndarray[float]', 'np.ndarray[float]']:
        """
        Calculate force components for given leg forces.

        Args:
            forces (ndarray(Float)): force exerted through each leg in Newtons.
        Returns:
            ndarray(Float): vertical components of forces
            ndarray(Float): horizontal components of forces
        """
        if forces is None:
            forces = self._last_tau
        vertical_components = np.zeros((6))
        horizontal_components = np.zeros((6))
        for i in range(6):
            top_joint = self._top_joints_space[:, i].copy().flatten()
            top_joint[2] = 0
            angle = fsr.angleBetween(
                self._bottom_joints_space[:, i],
                self._top_joints_space[:, i],
                top_joint)
            vertical_force = forces[i] * np.sin(angle)
            horizontal_force = forces[i] * np.cos(angle)
            vertical_components[i] = vertical_force
            horizontal_components[i] = horizontal_force
        return vertical_components, horizontal_components

    def sumActuatorWrenches(self, forces : 'np.ndarray[float]' = None) -> Wrench:
        """
        Sum all actuator wrenches to produce bottom wrench.

        Args:
            forces (ndarray(Float)): leg forces in Newtons
        Returns:
            ndarray(Float): bottom plate wrench
        """
        if forces is None:
            forces = self._last_tau
        wrench = fsr.makeWrench(self.getBasePos(), 0, self.grav/fmr.Norm(self.grav))
        for i in range(6):
            unit_vector = fmr.Normalize(self._bottom_joints_space[:, i]-self._top_joints_space[:, i])
            wrench += fsr.makeWrench(self._top_joints_space[:, i], float(forces[i]), unit_vector)
        #wrench = fsr.transformWrenchFrame(wrench, tm(), self.getTopT())
        return wrench

    
    """
    Public Helper Functions
    """

    def move(self, new_pos : tm, protect : bool = False) -> None:
        """
        Move entire stewart platform to another location and orientation.

        Args:
            new_pos (tm): New base transform to move to
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True
        """
        #Moves the base of the stewart platform to a new location


        self._current_plate_transform_local = fsr.globalToLocal(self.getBottomT(), self.getTopT())
        self._base_pos_global = new_pos.copy()
        self.IK(
            top_plate_pos = fsr.localToGlobal(self.getBottomT(),
                    self._current_plate_transform_local),
            protect = protect)

    def draw(self, ax) -> None:
        """
        Draw this SP in the Matplotlib environemnt.

        Args:
            ax: axis object
        """
        DrawSP(self, ax)

    """
    Internal Functions, Grouped by Type
    """

    """Kinematic Helpers"""
    def _bottomTopCheck(self, bottom_plate_pos : tm, top_plate_pos :tm) -> tuple[tm, tm]:
        """
        Ensure bottom and top plate arguments are not null.

        If a null argument is passed, populate from current SP state before returning.
        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
        Returns:
            tm: bottomm plate transformation in space frame
            tm: top plate transformation in space frame
        """
        if bottom_plate_pos == None:
            bottom_plate_pos = self.getBottomT()
        if top_plate_pos == None:
            top_plate_pos = self.getTopT()
        return bottom_plate_pos, top_plate_pos

    def _setPlatePos(self, bottom_plate_pos : tm, top_plate_pos : tm) -> None:
        """
        Set plate positions.

        Meant to be called internally only.
        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
        """
        if bottom_plate_pos is not None:
            self._base_pos_global = bottom_plate_pos
        if top_plate_pos is not None:
            self._end_effector_pos_global = top_plate_pos

    
    def _IKHelper(self, top_plate_pos : tm = None, 
            bottom_plate_pos : tm = None) -> tuple['np.ndarray[float]', tm, tm]:
        """
        Calculate Inverse Kinematics for a single stewart plaform.

        Takes in bottom plate transform, top plate transform, protection paramter, and direction
        This function is meant to be called internally only.

        Args:
            top_plate_pos (tm): top plate position
            bottom_plate_pos (tm): bottom plate position
        Returns:
            ndarray(Float): lengths of legs in meters
            tm: bottom plate position new
            tm: top plate position new
        """
        #If not supplied paramters, draw from stored values
        bottom_plate_pos, top_plate_pos = self._bottomTopCheck(
                bottom_plate_pos, top_plate_pos)
        #Check for excessive rotation
        #Poses which would be valid by leg length
        #But would result in singularity
        #Set bottom and top transforms

        #Call the IK method from the JIT numba file (FASER HIGH PER)
        #Shoulda just called it HiPer FASER. Darn.
        self.lengths, self._bottom_joints_space, self._top_joints_space = fmr.SPIKinSpace(
                bottom_plate_pos.gTM(),
                top_plate_pos.gTM(),
                self._bottom_joints_local,
                self._top_joints_local,
                self._bottom_joints_space,
                 self._top_joints_space)
        self._current_plate_transform_local = fsr.globalToLocal(
                bottom_plate_pos, top_plate_pos)
        return np.copy(self.lengths), bottom_plate_pos, top_plate_pos

    def _FKSolve(self, L : 'np.ndarray[float]', plate_pos : tm = None, 
            protect : bool = False) -> tuple[tm, tm]:
        """
        Solve FK using an older version of python solver, no jacobian used.
        
        Takes in length list, optionally bottom position, reverse parameter, and protection
        plate_pos refers to the the bottom plate position (stationary plate)
        If reversed, the parameter plate_pos refers to the top plate which is then stationary.

        Args:
            L (ndarray(Float)): Goal leg lengths
            bottom_plate_pos (tm): bottom plate transformation in space frame
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tm: bottom plate transform
            tm: top plate transform
        """
        #Do SPFK with scipy inbuilt solvers. Way less speedy o
        #Or accurate than Raphson, but much simpler to look at
        L = L.reshape((6, 1))
        self.lengths = L.reshape((6, 1)).copy()
        #jac = lambda x : self.inverseJacobian(top_plate_pos = x)

        #Slightly different if the platform is supposed to be "reversed"

        #Find top pose that produces the desired leg lengths.
        fk = lambda x : (self._IKHelper(tm(x), plate_pos)[0] - L).reshape((6))

        #solres = sci.optimize.fmin(fkprime, self.getTopT().TAA, disp=True)
        init = self.getTopT().TAA
        found_sol = True
        solres = sci.optimize.fsolve(fk, init)
        sol = tm(solres)
        sol.angleMod()
        sol.TMtoTAA()
        self.IK(top_plate_pos = sol, bottom_plate_pos = plate_pos, protect = True)
        nLens = self.getLens()
        for j in range(6):
            if abs(abs(L[j]) - abs(nLens[j])) > 0.00001 or not self.validate(True):
                return self._FKRaphson(L, plate_pos, protect)
        #If not "Protected" from recursion, call IK.
        if not protect:
            self.IK(protect = True)
        return plate_pos, sol


    def _FKRaphson(self, L : 'np.ndarray[float]', 
            bottom_plate_pos : tm = None, protect : bool = False) -> tuple[tm, tm]:
        """
        Solve FK using Newton Raphson method.

        This method is much more stable than FKSolve. 
        Adapted from the work done by
        http://jak-o-shadows.github.io/electronics/stewart-gough/stewart-gough.html
        Args:
            L (ndarray(Float)): Goal leg lengths
            bottom_plate_pos (tm): bottom plate transformation in space frame
            reverse (Bool): Boolean to reverse action. If true, treat the top plate as stationary.
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tm: bottom plate transform
            tm: top plate transform
        """
        if self.debug:# pragma: no cover
            disp("Starting Raphson FK")
        #^Look here for the original code and paper describing how this works.
        success = True
        L = L.reshape((6))
        self.lengths = L.copy()

        bottom_plate_pos_backup = bottom_plate_pos.copy()
        bottom_plate_pos = np.eye(4)
        iteration = 0

        try:
            #ap = (fsr.localToGlobal(tm([0, 0, self._nominal_height, 0, 0, 0]), tm()))
            ap = (fsr.localToGlobal(self._current_plate_transform_local, tm())).gTAA().reshape((6))
            attempt = np.zeros((6), dtype=float)
            for i in range(6):
                attempt[i] = ap[i]

            #Call the actual algorithm from the high performance faser library
            #Pass in initial lengths, guess, bottom and top plate positions,
            #max iterations, tolerances, and minimum leg lengths
            attempt, iteration = fmr.SPFKinSpaceR(L, attempt,
                self._bottom_joints_init, self._top_joints_init,
                self._max_iterations, self._tol_f, self._tol_a, self.leg_ext_min)

            #If the algorithm failed, try again, but this time set initial position to neutral
            if iteration == self._max_iterations:
                for i in range(6):
                    attempt = self._aux_inits[i].TAA.flatten()
                    #attempt[2] = self._nominal_height
                    attempt, iteration = fmr.SPFKinSpaceR(L, attempt,
                        self._bottom_joints_init, self._top_joints_init,
                        self._max_iterations, self._tol_f, self._tol_a, self.leg_ext_min)
                    if iteration == self._max_iterations:
                        if self.debug:# pragma: no cover
                            print("Raphson Failed to Converge")
                        self.fail_count += 1
                        self.IK(
                                top_plate_pos = (bottom_plate_pos_backup @ 
                                self._nominal_plate_transform), 
                                bottom_plate_pos = bottom_plate_pos_backup, protect = True)
                        return self.getBottomT(), self.getTopT()
                    else:
                        break

            #Otherwise return the calculated end effector position
            #coords =tm(bottom_plate_pos_backup @ fsr.TAAtoTM(a.reshape((6, 1))))
            coords = bottom_plate_pos_backup @ tm(attempt)

            self._IKHelper(coords, bottom_plate_pos_backup)
            #self._base_pos_global = bottom_plate_pos_backup
            #@ tm([0, 0, self.bottom_plate_thickness, 0, 0, 0])
            #self._end_effector_pos_global = coords #@ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
            self._setPlatePos(bottom_plate_pos_backup, coords)
            if self.debug:# pragma: no cover
                disp("Returning from Raphson FK")
            return bottom_plate_pos_backup, coords
        except Exception as e:

            if self.debug:# pragma: no cover
                disp("Raphson FK Failed due to: " + str(e))
            self.fail_count+=1
            return self._FKSolve(L, bottom_plate_pos_backup, protect)

    """
    Validation and Corrective Action Helpers
    """
    
    """Corrective Actions"""
    def _lambdaTopPlateReorientation(self, stopt : tm) -> 'np.ndarray[float]':
        """
        Return distance of top plate to reorientation reference points.

        Only used as an assistance function for fixing plate alignment
        Meant to be called internally only.
        Args:
            stopt (tm): top transform in space frame.
        Returns:
            ndarray(Float): distances array
        """
        reorient_helper_1 = fsr.localToGlobal(stopt, self.reorients[0])
        reorient_helper_2 = fsr.localToGlobal(stopt, self.reorients[1])
        reorient_helper_3 = fsr.localToGlobal(stopt, self.reorients[2])

        d1 = fsr.distance(reorient_helper_1,
            tm([self._top_joints_space[0, 0],
            self._top_joints_space[1, 0],
            self._top_joints_space[2, 0], 0, 0, 0]))
        d2 = fsr.distance(reorient_helper_2,
            tm([self._top_joints_space[0, 2],
            self._top_joints_space[1, 2],
            self._top_joints_space[2, 2], 0, 0, 0]))
        d3 = fsr.distance(reorient_helper_3,
            tm([self._top_joints_space[0, 4],
            self._top_joints_space[1, 4],
            self._top_joints_space[2, 4], 0, 0, 0]))
        return np.array([d1 , d2 , d3])

    def _fixUpsideDown(self) -> None:
        """
        Correct position and orientation of top plate when inversed due to computational error.

        In situations where the top plate is inverted underneath
        the bottom plate, yet lengths are valid,
        This function can be used to mirror all the joint locations and "fix" the resultant problem
        Meant to be called internally only.
        """
        for num in range(6):
            newTJ = fsr.mirror(self.getBottomT() @
                tm([0, 0, -self.bottom_plate_thickness, 0, 0, 0]),
                tm([self._top_joints_space[0, num],
                self._top_joints_space[1, num],
                self._top_joints_space[2, num], 0, 0, 0]))
            self._top_joints_space[0, num] = newTJ[0]
            self._top_joints_space[1, num] = newTJ[1]
            self._top_joints_space[2, num] = newTJ[2]
            self.lengths[num] = fsr.distance(
                self._top_joints_space[:, num], self._bottom_joints_space[:, num])
        top_true = fsr.mirror(self.getBottomT() @ tm([0, 0, -self.bottom_plate_thickness, 0, 0, 0]),
            self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]))
        top_true[3:6] = self.getTopT()[3:6] * -1
        self._end_effector_pos_global = top_true @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
        top_true = self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0])
        res = lambda x : self._lambdaTopPlateReorientation(
            tm([top_true[0], top_true[1], top_true[2], x[0], x[1], x[2]]))
        x_init = self.getTopT()[3:6].flatten()
        solution = sci.optimize.fsolve(res, x_init)
        top_true[3:6] = solution
        self._end_effector_pos_global = top_true @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])

    def _rescaleLegLengths(self, current_leg_min : float, current_leg_max : float) -> None:
        """
        Rescale leg lengths to meet bounds.

        Meant to be called internally only.
        Args:
            current_leg_min (Float): current minimum leg length (may be invalid)
            current_leg_max (Float): current maximum leg length (may be invalid)
        """
        for i in range(6):
            self.lengths[i] = ((self.lengths[i]-current_leg_min)/
                (current_leg_max-current_leg_min) *
                (min(self.leg_ext_max, current_leg_max) -
                max(self.leg_ext_min, current_leg_min)) +
                max(self.leg_ext_min, current_leg_min))

    def _addLegsToMinimum(self, current_leg_min : float) -> None:
        """
        Add the difference of the shortest leg to minimum bound to all legs.

        Designed to preserve end effector orientation.
        Meant to be called internally only.
        Args:
            current_leg_min (Float):  current minimum leg length (may be invalid)
        """
        boost_amount = ((self.leg_ext_min - current_leg_min) + self._leg_ext_safety)
        if self.debug: # pragma: no cover
            print("Boost Amount: " + str(boost_amount))
        self.lengths += boost_amount

    def _subLegsToMaximum(self, current_leg_max : float) -> None:
        """
        Subtract the difference of the maximum leg to maximum bound from all legs.

        Designed to preserve end effector orientation.
        Meant to be called internally only.
        Args:
            current_leg_max (Float): current maximum leg length (may be invalid)
        """
        #print([current_leg_max, self.leg_ext_max, current_leg_min,
        #    self.leg_ext_min, current_leg_max -
        #    (current_leg_max - self.leg_ext_max + self._leg_ext_safety)])
        sub_amount = ((current_leg_max - self.leg_ext_max) + self._leg_ext_safety)
        self.lengths -= sub_amount

    def _lengthCorrectiveAction(self) -> None:
        """
        Attempt to correct leg lengths that are out-of-bounds.

        Will frequently result in a home-like position
        Meant to be called internally only.
        """
        if self.debug:# pragma: no cover
            disp(self.lengths, "Lengths Pre Correction")
            disp(self.lengths[np.where(self.lengths > self.leg_ext_max)], "over max")
            disp(self.lengths[np.where(self.lengths < self.leg_ext_min)], "below min")

        current_leg_min = min(self.lengths)
        current_leg_max = max(self.lengths)

        #for i in range(6):
        #    self.lengths[i] = ((self.lengths[i]-current_leg_min)/
        #    (current_leg_max-current_leg_min) *
        #    (min(self.leg_ext_max, current_leg_max) -
        #    max(self.leg_ext_min, current_leg_min)) +
        #    max(self.leg_ext_min, current_leg_min))
        if current_leg_min < self.leg_ext_min and current_leg_max > self.leg_ext_max:
            self._rescaleLegLengths(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Rescale, "
        elif (current_leg_min < self.leg_ext_min and
            current_leg_max + (self.leg_ext_min - current_leg_min) +
            self._leg_ext_safety < self.leg_ext_max):
            self._addLegsToMinimum(current_leg_min)
            self.validation_error+= " CMethod: Boost, "
        elif (current_leg_max > self.leg_ext_max and
            current_leg_min - (current_leg_max - self.leg_ext_max) -
            self._leg_ext_safety > self.leg_ext_min):
            self.validation_error+= " CMethod: Subract, "
            self._subLegsToMaximum(current_leg_max)
        else:
            self._rescaleLegLengths(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Unknown Rescale, "

        #self.lengths[np.where(self.lengths > self.leg_ext_max)] = self.leg_ext_max
        #self.lengths[np.where(self.lengths < self.leg_ext_min)] = self.leg_ext_min
        if self.debug:# pragma: no cover
            disp(self.lengths, "Corrected Lengths")
        #disp("HEre's what happened")
        self.FK(self.lengths.copy(), protect = True)
        #print(self.lengths)

    def _continuousTranslationCorrectiveAction(self) -> None:
        """
        Reset to home position.

        Meant to be called internally only.
        """
        self.IK(top_plate_pos = self.getBottomT() @ self._nominal_plate_transform, protect = True)

    """Constraints"""

    def _plateRotationConstraint(self) -> bool:
        """
        Check validity of current plate rotation against constraint.

        Meant to be called internally only.
        Returns:
            Bool: Validity of configuration
        """
        valid = True
        for i in range(3):
            if self._current_plate_transform_local.gTM()[i, i] <= self.plate_rotation_limit - .0001:
                if self.debug:# pragma: no cover
                    disp(self._current_plate_transform_local.gTM(), "Erroneous TM")
                    print([self._current_plate_transform_local.gTM()[i, i],
                        self.plate_rotation_limit])
                valid = False
        return valid

    def _legLengthConstraint(self) -> bool:
        """
        Check validity of current leg lengths against constraint.

        Meant to be called internally only.
        Returns:
            Bool: Validity of configuration

        """
        valid = True
        if(np.any(self.lengths < self.leg_ext_min) or np.any(self.lengths > self.leg_ext_max)):
            valid = False
        return valid

    def _continuousTranslationConstraint(self) -> bool:
        """
        Validate that the top plate has positive Z in bottom plate's local frame.

        Takes no corrective action.
        Meant to be called internally only.
        Returns:
            Bool: Validity at configuration

        """
        valid = True
        bot = self.getBottomT()
        for i in range(6):
            if fsr.globalToLocal(self.getBottomT(), self.getTopT())[2] < 0:
                valid = False
        return valid

    def _interiorAnglesConstraint(self) -> bool:
        """
        Validate that the interior angles of the legs from normal are valid against constraint.

        Takes no corrective action.
        Meant to be called internally only.
        Returns:
            Bool: Validity at configuration
        """
        angles = abs(self.getJointAnglesFromNorm())
        if(np.any(np.isnan(angles))):
            return False
        if(np.any(angles > self.joint_deflection_max)):
            return False
        return True

    def printOutOfDateFunction(self, old_name, use_name):   # pragma: no cover
        """
        Print an OOD notice for an old function.

        Args:
            old_name (String): Description of parameter `old_name`.
            use_name (String): Description of parameter `use_name`.
        """
        print(old_name + " is deprecated. Please use " + use_name + " instead.")

    def SetMasses(self, plateMass, actuatorTop, actuatorBottom, grav = 9.81, tPlateMass = 0):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("SetMasses", "setMasses")
        return self.setMasses(plateMass, actuatorTop, actuatorBottom, grav, tPlateMass)
    def SetGrav(self, grav = 9.81):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("SetGrav", "setGrav")
        return self.setGrav(grav)
    def SetCOG(self, motor_grav_center, shaft_grav_center):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("SetCOG", "setCOG")
        return self.setCOG(motor_grav_center, shaft_grav_center)
    def SetAngleDev(self, MaxAngleDev = 55):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("SetAngleDev", "setMaxAngleDev")
        return self.setMaxAngleDev(MaxAngleDev)
    def SetPlateAngleDev(self, MaxPlateDev = 60):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("SetPlateAngleDev", "setMaxPlateRotation")
        return self.setMaxPlateRotation(MaxPlateDev)
    def setPlatePos(self, bottomT, topT):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("setPlatePos", "_setPlatePos")
        return self._setPlatePos(bottomT, topT)
    def gLens(self):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("gLens", "getLens")
        return self.getLens()
    def gtopT(self):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("gtopT", "getTopT")
        return self.getTopT()
    def gbottomT(self):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("gbottomT", "getBottomT")
        return self.getBottomT()
    def GetActuatorUnit(self, p1, p2, dist):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("GetActuatorUnit", "fsr.getUnitVec")
        return fsr.getUnitVec(p1, p2, dist)
    def SpinCustom(self, rot):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("SpinCustom", "spinCustom")
        return self.spinCustom(rot)
    def LambdaRTP(self, stopt):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("LambdaRTP", "_lambdaTopPlateReorientation")
        return self._lambdaTopPlateReorientation(stopt)
    def _resclLegs(self, cMin, cMax):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("_resclLegs", "rescaleLegLengths")
        return self._rescaleLegLengths(cMin, cMax)
    def _addLegs(self, cMin, cMax):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("_addLegs", "addLegsToMinimum")
        return self._addLegsToMinimum(cMin, cMax)
    def _subLegs(self, cMin, cMax):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("_subLegs", "subLegsToMaximum")
        return self._subLegsToMaximum(cMin, cMax)
    def AngleFromNorm(self):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("AngleFromNorm", "getJointAnglesFromNorm")
        return self.getJointAnglesFromNorm()
    def AngleFromVertical(self):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("AngleFromVertical", "getJointAnglesFromVertical")
        return self.getJointAnglesFromVertical()
    def JacobianSpace(self, bottomT = None, topT = None):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("JacobianSpace", "jacobian")
        return self.jacobian(bottomT, topT)
    def InverseJacobianSpace(self, bottomT = None, topT = None, protect = True):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("InverseJacobianSpace", "inverseJacobian")
        return self.inverseJacobian(topT, bottomT, protect)
    def CarryMassCalc(self, twrench, protect = False):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("CarryMassCalc", "carryMassCalc")
        return self.carryMassCalc(twrench, protect)
    def CarryMassCalcNew(self, twrench, protect = False):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("CarryMassCalcNew", "carryMassCalcLocal")
        return self.carryMassCalcLocal(twrench, protect)
    def MeasureForcesAtEENew(self, wrench, protect = False):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("MeasureForcesAtEENew", "measureForcesAtEENew")
        return self.staticForces(wrench, protect)
    def CarryMassCalcUp(self, twrench, protect = False):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("CarryMassCalcUp", "carryMassCalcUp")
        return self.carryMassCalcUp(twrench, protect)
    def MeasureForcesFromBottomEE(self, bottomT = np.zeros((1)) ,  # pragma: no cover
        topT = np.zeros((1)), topWEE = np.zeros((1)), protect = True):
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("MeasureForcesFromBottomEE", "measureForcesFromBottomEE")
        return self.measureForcesFromBottomEE(bottomT, topT, topWEE, protect)

    def WrenchBottomFromMeasuredForces(self, bottomT, topT, tau):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction(
            "WrenchBottomFromMeasuredForces", "wrenchBottomFromMeasuredForces")
        return self.wrenchBottomFromMeasuredForces(bottomT, topT, tau)
    def SumActuatorWrenches(self, forces = None):  # pragma: no cover
        """Return Deprecation Notice and Function Call. Don't Use."""
        self.printOutOfDateFunction("SumActuatorWrenches", "sumActuatorWrenches")
        return self.sumActuatorWrenches(forces)

def loadSP(fname : str, file_directory : str = "../robot_definitions/", 
        baseloc : tm = None, altRot : float = 1) -> 'SP':
    """
    Load A Stewart Platform Object from a json file.

    Args:
        fname (String): file name of the sp config
        file_directory (String): optional directory, defaults to robot_defintions
        baseloc (tm): Base location.
        altRot (Float): alternate relative plate rotation.
    Returns:
        SP: SP object
    """
    #print(fname)
    #print(file_directory)
    total_name = file_directory + fname
    #print(total_name)
    with open(total_name, "r") as sp_file:
        sp_data = json.load(sp_file)
    bot_radius = sp_data["BottomPlate"]["JointRadius"] #Radius of Ball Joint Circle in Meters
    top_radius = sp_data["TopPlate"]["JointRadius"]
    bot_joint_spacing = sp_data["BottomPlate"]["JointSpacing"] #Spacing in Degrees
    top_joint_spacing = sp_data["TopPlate"]["JointSpacing"]
    bot_thickness = sp_data["BottomPlate"]["Thickness"]
    top_thickness = sp_data["TopPlate"]["Thickness"]
    outer_top_radius = sp_data["Drawing"]["TopRadius"]
    outer_bottom_radius = sp_data["Drawing"]["BottomRadius"]
    act_shaft_radius = sp_data["Drawing"]["ShaftRadius"]
    act_motor_radius = sp_data["Drawing"]["MotorRadius"]
    actuator_shaft_mass = 0
    actuator_motor_mass = 0
    plate_top_mass = 0
    plate_bot_mass = 0
    motor_grav_center = 0
    shaft_grav_center = 0
    name = sp_data["Name"]
    actuator_min = sp_data["Actuators"]["MinExtension"] #meters
    actuator_max = sp_data["Actuators"]["MaxExtension"]
    force_lim = sp_data["Actuators"]["ForceLimit"]
    max_dev = sp_data["Settings"]["MaxAngleDev"]
    if sp_data["Settings"]["AssignMasses"] == 1:
        actuator_motor_mass = sp_data["Actuators"]["MotorMass"]
        actuator_shaft_mass = sp_data["Actuators"]["ShaftMass"]
        plate_top_mass = sp_data["TopPlate"]["Mass"]
        plate_bot_mass = sp_data["BottomPlate"]["Mass"]
        if sp_data["Settings"]["InferActuatorCOG"] == 1:
            motor_grav_center = sp_data["Actuators"]["MotorCOGD"]
            shaft_grav_center = sp_data["Actuators"]["ShaftCOGD"]
        else:
            inferred_cog = 1/4 * (actuator_min+actuator_max)/2
            actuator_shaft_mass = inferred_cog
            motor_grav_center = inferred_cog
    if baseloc == None:
        baseloc = tm()


    newsp = newSP(bot_radius, top_radius, bot_joint_spacing, top_joint_spacing,
        bot_thickness, top_thickness, actuator_shaft_mass, actuator_motor_mass, plate_top_mass,
        plate_bot_mass, motor_grav_center, shaft_grav_center,
        actuator_min, actuator_max, baseloc, name, altRot)

    newsp.setDrawingParameters(
        outer_top_radius,
        outer_bottom_radius,
        act_shaft_radius,
        act_motor_radius)
    newsp.setMaxAngleDev(max_dev)
    newsp.force_limit = force_lim

    return newsp

def newSP(bottom_radius, top_radius, bJointSpace, tJointSpace,
    bottom_plate_thickness, top_plate_thickness, actuator_shaft_mass,
    actuator_motor_mass, plate_top_mass, plate_bot_mass, motor_grav_center,
    shaft_grav_center, actuator_min, actuator_max, base_location, name, rot = 1):
    """
    Build a new SP, called usually by a constructor.

    Args:
        bottom_radius (Float): Bottom plate Radius (m)
        top_radius (Float): Top plate Radius (m)
        bJointSpace (ndarray(Float)): bottom joints space locations
        tJointSpace (ndarray(Float)): top joints space locations
        bottom_plate_thickness (Float): bottom plate thickness (m)
        top_plate_thickness (Float): top plate thickness (m)
        actuator_shaft_mass (Float): Actuator shaft (moving portion) mass Kg
        actuator_motor_mass (Float): Actuator motor (stationary portion) mass Kg
        plate_top_mass (Float): top plate mass (Kg)
        plate_bot_mass (Float):  bottom plate mass (Kg)
        motor_grav_center (Float): Actuator motor inline COG distance from joint
        shaft_grav_center (Float): Actuator shaft inline CG distance from top joint
        actuator_min (Float): Actuator length when fully retracted
        actuator_max (Float): Actuator length when fully extended
        base_location (tm): Base transform
        name (String): Name of the SP
        rot (Float): Rotation parameter

    Returns:
        SP: SP object
    """
    bottom_gap = bJointSpace / 2 * np.pi / 180
    top_gap = tJointSpace / 2 * np.pi / 180

    bottom_joint_gap = 120 * np.pi / 180 #Angle of seperation between joint clusters
    top_joint_gap = 60 * np.pi / 180 #Offset in rotation of the top plate versus the bottom plate

    bangles = np.array([
        -bottom_gap, bottom_gap,
        bottom_joint_gap-bottom_gap,
        bottom_joint_gap+bottom_gap,
        2*bottom_joint_gap-bottom_gap,
        2*bottom_joint_gap+bottom_gap])
    tangles = np.array([
        -top_joint_gap+top_gap,
        top_joint_gap-top_gap,
        top_joint_gap+top_gap,
        top_joint_gap+bottom_joint_gap-top_gap,
        top_joint_gap+bottom_joint_gap+top_gap,
        -top_joint_gap-top_gap])
    if rot == -1:
        tangles = np.array([
            -bottom_gap, bottom_gap,
            bottom_joint_gap-bottom_gap,
            bottom_joint_gap+bottom_gap,
            2*bottom_joint_gap-bottom_gap,
            2*bottom_joint_gap+bottom_gap])
        bangles = np.array([
            -top_joint_gap+top_gap,
            top_joint_gap-top_gap,
            top_joint_gap+top_gap,
            top_joint_gap+bottom_joint_gap-top_gap,
            top_joint_gap+bottom_joint_gap+top_gap,
            -top_joint_gap-top_gap])

    S = fmr.ScrewToAxis(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0).reshape((6, 1))

    Mb = tm(np.array([bottom_radius, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #how far from the bottom plate origin should clusters be generated
    Mt = tm(np.array([top_radius, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #Same thing for the top

    bj = np.zeros((3, 6)) #Pre allocate arrays
    tj = np.zeros((3, 6))

    for i in range(0, 6):
        bji = fsr.transformFromTwist(bangles[i] * S) @ Mb
        tji = fsr.transformFromTwist(tangles[i] * S) @ Mt
        bj[0:3, i] = bji[0:3].reshape((3))
        tj[0:3, i] = tji[0:3].reshape((3))
        bj[2, i] = bottom_plate_thickness
        tj[2, i] = -top_plate_thickness

    bottom = base_location.copy()
    tentative_height = midHeightEstimate(
        actuator_min, actuator_max, bj, tj, bottom_plate_thickness, top_plate_thickness)
    if rot == -1:
        tentative_height = midHeightEstimate(
            actuator_min, actuator_max, tj, bj, bottom_plate_thickness, top_plate_thickness)
    top = bottom @ tm(np.array([0.0, 0.0, tentative_height, 0.0, 0.0, 0.0]))

    newsp = SP(bj, tj, bottom, top,
        actuator_min, actuator_max,
        bottom_plate_thickness, top_plate_thickness, name)
    newsp.setDrawingParameters(top_radius, bottom_radius, .1, .2)
    newsp.setMasses(
        plate_bot_mass,
        actuator_shaft_mass,
        actuator_motor_mass,
        top_plate_mass = plate_top_mass)
    newsp.setCOG(motor_grav_center, shaft_grav_center)

    return newsp

def makeSP(bRad, tRad, spacing, baseT,
    platOffset, rot = 1, plate_thickness_avg = 0, altRot = 0):
    """
    Load a SP without need of a file.

    Deprecated in favor of loading SPs from a JSON file.
    Args:
        bRad (Float): bottom plate radius
        tRad (Float): top plate radius
        spacing (Float): joint spacing (deg)
        baseT (tm):base transform
        platOffset (Float): platform offset height
        rot (Float): creates an invert platform if flipped
        plate_thickness_avg (Float): plate thickness
        altRot (Float): rotational offset

    Returns:
        SP: Stewart platform object

    """
    gapS = spacing/2*np.pi/180 #Angle between cluster joints
    bottom_joint_gap = 120*np.pi/180 #Angle of seperation between joint clusters
    top_joint_gap = 60*np.pi/180 #Offset in rotation of the top plate versus the bottom plate
    bangles = np.array([
        -gapS,
        gapS,
        bottom_joint_gap-gapS,
        bottom_joint_gap+gapS,
        2*bottom_joint_gap-gapS,
        2*bottom_joint_gap+gapS]) + altRot * np.pi/180
    tangles = np.array([
        -top_joint_gap+gapS,
        top_joint_gap-gapS,
        top_joint_gap+gapS,
        top_joint_gap+bottom_joint_gap-gapS,
        top_joint_gap+bottom_joint_gap+gapS,
        -top_joint_gap-gapS])+ altRot * np.pi/180
    if rot == -1:
        tangles = np.array([
            -gapS, gapS,
            bottom_joint_gap-gapS,
            bottom_joint_gap+gapS,
            2*bottom_joint_gap-gapS,
            2*bottom_joint_gap+gapS])+ altRot * np.pi/180
        bangles = np.array([
            -top_joint_gap+gapS,
            top_joint_gap-gapS,
            top_joint_gap+gapS,
            top_joint_gap+bottom_joint_gap-gapS,
            top_joint_gap+bottom_joint_gap+gapS,
            -top_joint_gap-gapS])+ altRot * np.pi/180

    S = fmr.ScrewToAxis(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0).reshape((6, 1))

    Mb = tm(np.array([bRad, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #how far from the bottom plate origin should clusters be generated
    Mt = tm(np.array([tRad, 0.0, 0.0, 0.0, 0.0, 0.0]))
     #Same thing for the top

    bj = np.zeros((3, 6)) #Pre allocate arrays
    tj = np.zeros((3, 6))

    #Generate position vectors (XYZ) for top and bottom joint locations
    for i in range(0, 6):
        bji = fsr.transformFromTwist(bangles[i] * S) @ Mb
        tji = fsr.transformFromTwist(tangles[i] * S) @ Mt
        bj[0:3, i] = bji[0:3].reshape((3))
        tj[0:3, i] = tji[0:3].reshape((3))
        bj[2, i] = plate_thickness_avg/2
        tj[2, i] = -plate_thickness_avg/2

    #if rot == -1:
    #    disp(bj, "Prechange")
#
#        rotby = TAAtoTM(np.array([0, 0, 0, 0, 0, np.pi/3]))
#        for i in range(6):
#            bj[0:3, i] = TMtoTAA(rotby @
#                TAAtoTM(np.array([bj[0, i], bj[1, i], bj[2, i], 0, 0, 0])))[0:3].reshape((3))
#            tj[0:3, i] = TMtoTAA(rotby @
#                TAAtoTM(np.array([tj[0, i], tj[1, i], tj[2, i], 0, 0, 0])))[0:3].reshape((3))
#        disp(bj, "postchange")
    bottom = baseT.copy()
    #Generate top position at offset from the bottom position
    top = bottom @ tm(np.array([0.0, 0.0, platOffset, 0.0, 0.0, 0.0]))
    sp = SP(bj, tj, bottom, top, 0, 1, plate_thickness_avg, plate_thickness_avg, 'sp')

    return sp, bottom, top
#Helpers
def midHeightEstimate(leg_ext_min, leg_ext_max, bj, tj, bth, tth):
    """
    Calculate an estimate of the resting height of a stewart plaform.

    Args:
        leg_ext_min (float): minimum leg extension
        leg_ext_max (float): maximum leg extension
        bj (array(float)): bottom joints
        tj (array(float)): top joints
        bth (tm):bottom plate thickness
        tth (tm): top plate thickness

    Returns:
        Float: Description of returned object.

    """
    leg_half_extension = (leg_ext_min + leg_ext_max) / 2 
    dist_joints_xy = (tj[0, 0] - bj[0, 0])**2 + (tj[1, 0] - bj[1, 0])**2

    hest = np.sqrt(leg_half_extension**2 - dist_joints_xy) + bth + tth 
    return hest

