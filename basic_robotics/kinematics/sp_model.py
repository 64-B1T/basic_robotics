from ..general import tm, fmr, fsr
from ..utilities.disp import disp
import numpy as np
import scipy as sci
import scipy.linalg as ling
import copy
import json

class SP:
    #Conventions:
    #Filenames:  snake_case
    #Variables: snake_case
    #Functions: camelCase
    #ClassNames: CapsCase
    #Docstring: Google
    def __init__(self, bottom_joints, top_joints, bT, tT, leg_ext_min,
        leg_ext_max, bottom_plate_thickness, top_plate_thickness, name):
        """
        Initializes a new Stewart Platform Object

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
        self.bottom_joints = np.copy(bottom_joints)
        self.top_joints = np.copy(top_joints)
        self.bottom_joints_init = self.bottom_joints.conj().transpose()
        self.top_joints_init = self.top_joints.conj().transpose()
        self.bottom_plate_pos = bT.copy()
        self.top_plate_pos = tT.copy()
        self.bottom_joints_space = np.zeros((3, 6))
        self.top_joints_space = np.zeros((3, 6))
        self.current_plate_transform_local = tm()

        #Debug
        self.leg_ext_safety = .001
        self.debug = 0

        #Physical Parameters
        self.bottom_plate_thickness = bottom_plate_thickness
        self.top_plate_thickness = top_plate_thickness
        if leg_ext_min == 0:
            self.leg_ext_min = 0
            self.leg_ext_max = 2
        self.leg_ext_min = leg_ext_min
        self.leg_ext_max = leg_ext_max

        #Reserve Val
        self.nominal_height = fsr.distance(bT, tT)
        self.nominal_plate_transform = tm([0, 0, self.nominal_height, 0, 0, 0])

        self.aux_inits = [
            self.nominal_plate_transform,
            self.nominal_plate_transform @ tm([np.pi/8, 0, 0]),
            self.nominal_plate_transform @ tm([0, -np.pi/8, 0]),
            self.nominal_plate_transform @ tm([-np.pi/8, 0, np.pi/8]),
            self.nominal_plate_transform @ tm([.25, .25, -.25, np.pi/10, 0, .001])
        ]

        #Drawing Characteristics
        self.outer_top_radius = 0
        self.outer_bottom_radius = 0
        self.act_shaft_radius = 0
        self.act_motor_radius = 0

        #Empty array indicates these values haven't been populated yet
        self.leg_forces =  np.zeros(1)
        self.top_plate_wrench =  np.zeros(1)
        self.bottom_plate_wrench =  np.zeros(1)

        #Mass values from bottom mass, top mass, and actuator portion masses can be set directly.
        self.bottom_plate_mass = 0
        self.top_plate_mass = 0
        self.act_shaft_mass = 0
        self.act_motor_mass = 0
        self.act_shaft_newton_force = 0
        self.act_motor_newton_force = 0
        self.top_plate_newton_force = 0
        self.bottom_plate_newton_force = 0
        self.grav = 9.81
        self.dir = np.array([0, 0, -1])
        self.act_shaft_grav_center = 0
        self.act_motor_grav_center = 0
        self.force_limit= 0

        #Tolerances and Limits
        self.joint_deflection_max = 140/2*np.pi/180#2*np.pi/5
        self.plate_rotation_limit = np.cos(60*np.pi/180)

        #Newton Settings
        self.tol_f = 1e-5/2
        self.tol_a = 1e-5/2
        self.max_iterations = 1e4

        #Errors and Counts
        self.fail_count = 0
        self.validation_settings = [1, 0, 0, 1]
        self.fk_mode = 1
        self.validation_error = ""

        self.IK(bT, tT, protect = True)

        self.bottom_joint_angles_init = self.top_joints_space.T.copy()
        self.bottom_joint_angles = self.bottom_joints_space.T.copy()

        self.bottom_joint_angles_init = [None] * 6
        self.bottom_joint_angles = [None] * 6
        for i in range(6):
            self.bottom_joint_angles_init[i] = fsr.globalToLocal(self.getBottomT(),
                tm([self.top_joints_space.T[i][0], self.top_joints_space.T[i][1],
                self.top_joints_space.T[i][2], 0, 0, 0]))
            self.bottom_joint_angles[i] = fsr.globalToLocal(self.getTopT(),
                tm([self.bottom_joints_space.T[i][0], self.bottom_joints_space.T[i][1],
                self.bottom_joints_space.T[i][2], 0, 0, 0]))

        t1 = fsr.globalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self.top_joints_space[0, 0],
            self.top_joints_space[1, 0],
            self.top_joints_space[2, 0], 0, 0, 0]))
        t2 = fsr.globalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self.top_joints_space[0, 2],
            self.top_joints_space[1, 2],
            self.top_joints_space[2, 2], 0, 0, 0]))
        t3 = fsr.globalToLocal(self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]),
            tm([self.top_joints_space[0, 4],
            self.top_joints_space[1, 4],
            self.top_joints_space[2, 4], 0, 0, 0]))
        self.reorients = [t1, t2, t3]


        #Compatibility
        self.plate_thickness_avg = (self.top_plate_thickness + self.bottom_plate_thickness) / 2
        self.nominal_plate_transform = tm([0, 0, self.plate_thickness_avg, 0, 0, 0])

        #Validation Settings


    """
       _____      _   _                                     _    _____      _   _
      / ____|    | | | |                    /\             | |  / ____|    | | | |
     | |  __  ___| |_| |_ ___ _ __ ___     /  \   _ __   __| | | (___   ___| |_| |_ ___ _ __ ___
     | | |_ |/ _ \ __| __/ _ \ '__/ __|   / /\ \ | '_ \ / _` |  \___ \ / _ \ __| __/ _ \ '__/ __|
     | |__| |  __/ |_| ||  __/ |  \__ \  / ____ \| | | | (_| |  ____) |  __/ |_| ||  __/ |  \__ \
      \_____|\___|\__|\__\___|_|  |___/ /_/    \_\_| |_|\__,_| |_____/ \___|\__|\__\___|_|  |___/

    """
    def setMasses(self, plate_mass_general, act_shaft_mass,
        act_motor_mass, grav=9.81, top_plate_mass=0):
        """
        Set masses for each SP in the Assembler, note that because central platforms
        share plates, these weights are halved with respect to end plates
        Args:
            plate_mass_general (float): mass of bottom plate (both if top is not specified) (kg)
            act_shaft_mass (float): mass of actuator shaft (kg)
            act_motor_mass (float): mass of actuator motor (kg)
            grav (float):  [Optional, default 9.81] acceleration due to gravity
            top_plate_mass (float): [Optional, default 0] top plate mass (kg)
        """
        self.bottom_plate_mass = plate_mass_general
        if top_plate_mass != 0:
            self.top_plate_mass = top_plate_mass
        else:
            self.top_plate_mass = plate_mass_general
        self.setGrav(grav)
        self.act_shaft_mass = act_shaft_mass
        self.act_motor_mass = act_motor_mass
        self.act_motor_newton_force = self.act_motor_mass * self.grav
        self.act_shaft_newton_force = self.act_shaft_mass * self.grav
        self.top_plate_newton_force = self.top_plate_mass * self.grav
        self.bottom_plate_newton_force = self.bottom_plate_mass * self.grav

    def setGrav(self, grav=9.81):
        """
        Sets Gravity
        Args:
            grav (float): Acceleration due to gravity
        Returns:
            None: None
        """
        self.grav = grav

    def setCOG(self, motor_grav_center, shaft_grav_center):
        """
        Sets the centers of gravity for actuator components
        Args:
            motor_grav_center (float): distance from top of actuator to actuator shaft COG
            shaft_grav_center (float): distance from bottom of actuator to actuator motor COG
        """
        self.act_shaft_grav_center = shaft_grav_center
        self.act_motor_grav_center = motor_grav_center

    def setMaxAngleDev(self, max_angle_dev=55):
        """
        Set the maximum angle joints can deflect before failure
        Args:
            max_angle_dev (float): maximum deflection angle (degrees)
        """
        self.joint_deflection_max = max_angle_dev*np.pi/180

    def setMaxPlateRotation(self, max_plate_rotation=60):
        """
        Set the maximum angle the plate can rotate before failure

        Args:
            max_plate_rotation (Float): Maximum angle before plate rotation failure (degrees)
        """
        self.plate_rotation_limit = np.cos(max_plate_rotation * np.pi / 180)

    def setDrawingDimensions(self, outer_top_radius,
        outer_bottom_radius, act_shaft_radius, act_motor_radius):
        """
        Set Drawing Dimensions
        Args:
            outer_top_radius (Float): Description of parameter `outer_top_radius`.
            outer_bottom_radius (Float): Description of parameter `outer_bottom_radius`.
            act_shaft_radius (Float): Description of parameter `act_shaft_radius`.
            act_motor_radius (Float): Description of parameter `act_motor_radius`.
        """
        self.outer_top_radius = outer_top_radius
        self.outer_bottom_radius = outer_bottom_radius
        self.act_shaft_radius = act_shaft_radius
        self.act_motor_radius = act_motor_radius

    def setPlatePos(self, bottom_plate_pos, top_plate_pos):
        """
        Set plate positions. called internally
        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
        """
        if bottom_plate_pos is not None:
            self.bottom_plate_pos = bottom_plate_pos
        if top_plate_pos is not None:
            self.top_plate_pos = top_plate_pos

    def getBottomJoints(self):
        """
        get the bottom joint positions in space. Not orientations
        Returns:
            ndarray(Float): bottom joint positions

        """
        return self.bottom_joints_space

    def getTopJoints(self):
        """
        get the top joint positions in space. Not orientations
        Returns:
            ndarray(Float): top joint positions in space

        """
        return self.top_joints_space

    def getCurrentLocalTransform(self):
        """
        Get the current local transform between bottom and top plate
        Returns:
            tm: Top plate relative to bottom plate

        """
        return self.current_plate_transform_local

    def getLegForces(self):
        """
        Return calculated leg forces
        Returns:
            ndarray(Float): Leg forces (N)
        """
        return self.leg_forces

    def getLens(self):
        """
        Get Leg Lengths
        Returns:
            ndarray(Float): Leg Lengths
        """
        return self.lengths.copy()

    def getTopT(self):
        """
        Return the transform of the top plate
        Returns:
            tm: top plate transform in space frame
        """
        return self.top_plate_pos.copy()

    def getBottomT(self):
        """
        Return the transform of the bottom plate
        Returns:
            tm: bottom plate transform in space frame
        """
        return self.bottom_plate_pos.copy()

    def getActuatorLoc(self, num, type = 'm'):
        """
        Returns the position of a specified actuator. Takes in an actuator number and a type.
        m for actuator midpoint
        b for actuator motor position
        t for actuator top position

        Args:
            num (Int): number of actuator to return
            type (Char): property of actuator to return

        Returns:
            ndarray(Float): location of desired point
        """
        pos = 0
        if type == 'm':
            pos = np.array([(self.bottom_joints_space[0, num] + self.top_joints_space[0, num])/2,
                (self.bottom_joints_space[1, num] + self.top_joints_space[1, num])/2,
                (self.bottom_joints_space[2, num] + self.top_joints_space[2, num])/2])
        bottom_act_joint = tm([self.bottom_joints_space[0, num],
            self.bottom_joints_space[1, num], self.bottom_joints_space[2, num], 0, 0, 0])
        top_act_joint = tm([self.top_joints_space[0, num],
            self.top_joints_space[1, num], self.top_joints_space[2, num], 0, 0, 0])
        if type == 'b':
            #return fsr.adjustRotationToMidpoint(bottom_act_joint, bottom_act_joint,
            #   top_act_joint, mode = 1) @ tm([0, 0, self.act_motor_grav_center, 0, 0, 0])
            return fsr.getUnitVec(bottom_act_joint,
                top_act_joint, self.act_motor_grav_center)
        if type == 't':
            #return fsr.adjustRotationToMidpoint(top_act_joint, top_act_joint, bottom_act_joint,
            #   mode = 1) @ tm([0, 0, self.act_shaft_grav_center, 0, 0, 0])
            return fsr.getUnitVec(top_act_joint,
                bottom_act_joint, self.act_shaft_grav_center)
        new_position = tm([pos[0], pos[1], pos[2], 0, 0, 0])
        return new_position

    def spinCustom(self, rot):
        """
        Rotates plate to meet desired transform
        Args:
            rot (Float): rotation in radians
        """
        old_base_pos = self.getBottomT()
        self.move(tm())
        current_top_pos = self.getTopT()
        top_joints_copy = self.top_joints_space.copy()
        bottom_joints_copy = self.bottom_joints_space.copy()
        top_joints_origin_copy = self.top_joints[2, 0:6]
        bottom_joints_origin_copy = self.bottom_joints[2, 0:6]
        rotation_transform = tm([0, 0, 0, 0, 0, rot * np.pi / 180])
        self.move(rotation_transform)
        top_joints_space_new = self.top_joints_space.copy()
        bottom_joints_space_new = self.bottom_joints_space.copy()
        top_joints_copy[0:2, 0:6] = top_joints_space_new[0:2, 0:6]
        bottom_joints_copy[0:2, 0:6] = bottom_joints_space_new[0:2, 0:6]
        bottom_joints_copy[2, 0:6] = bottom_joints_origin_copy
        top_joints_copy[2, 0:6] = top_joints_origin_copy
        self.move(tm())
        self.bottom_joints = bottom_joints_copy
        self.top_joints = top_joints_copy
        self.bottom_joints_space = bottom_joints_space_new
        self.top_joints_space = top_joints_space_new
        self.move(old_base_pos)


    def IK(self, bottom_plate_pos=None, top_plate_pos=None, protect=False):
        """
        Calculate inverse kinematics for given goals
        Args:
            bottom_plate_pos (tm): bottom plate position
            top_plate_pos (tm): top plate position
            protect (Bool): If true, bypass any safeties
        Returns:
            ndarray(Float): leg lengths
            Bool: validity of pose

        """

        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)

        leg_lengths, bottom_plate_pos, top_plate_pos = self.IKHelper(
            bottom_plate_pos, top_plate_pos, protect)
        #Determine current transform

        self.bottom_plate_pos = bottom_plate_pos.copy()
        self.top_plate_pos = top_plate_pos.copy()

        #Ensure a valid position
        valid = True
        if not protect:
            valid = self.validate()
        return leg_lengths, valid

    def IKHelper(self, bottom_plate_pos=None, top_plate_pos=None, protect=False):
        """
        Calculates Inverse Kinematics for a single stewart plaform.
        Takes in bottom plate transform, top plate transform, protection paramter, and direction

        Args:
            bottom_plate_pos (tm): bottom plate position
            top_plate_pos (tm): top plate position
            protect (Bool): If true, bypass any safeties

        Returns:
            ndarray(Float): lengths of legs in meters
            tm: bottom plate position new
            tm: top plate position new
        """
        #If not supplied paramters, draw from stored values
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(
                bottom_plate_pos, top_plate_pos)
        #Check for excessive rotation
        #Poses which would be valid by leg length
        #But would result in singularity
        #Set bottom and top transforms
        #self.bottom_plate_pos = bottom_plate_pos
        #self.top_plate_pos = top_plate_pos

        #Call the IK method from the JIT numba file (FASER HIGH PER)
        #Shoulda just called it HiPer FASER. Darn.
        self.lengths, self.bottom_joints_space, self.top_joints_space = fmr.SPIKinSpace(
                bottom_plate_pos.gTM(),
                top_plate_pos.gTM(),
                self.bottom_joints,
                self.top_joints,
                self.bottom_joints_space,
                 self.top_joints_space)
        self.current_plate_transform_local = fsr.globalToLocal(
                bottom_plate_pos, top_plate_pos)
        return np.copy(self.lengths), bottom_plate_pos, top_plate_pos

    def FK(self, L, bottom_plate_pos =None, reverse = False, protect=False):
        """
        Calculate Forward Kinematics for desired leg lengths

        Args:
            L (ndarray(Float)): Goal leg lengths
            bottom_plate_pos (tm): bottom plate position
            reverse (Bool): Boolean to reverse action. If true, treat the top plate as stationary.
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tm: top plate configuration
            Bool: validity

        """
        #FK host function, calls subfunctions depedning on the value of fk_mode
        #return self.FKSciRaphson(L, bottom_plate_pos, reverse, protect)
        #bottom_plate_pos, n = self._applyPlateTransform(bottom_plate_pos = bottom_plate_pos)
        if self.fk_mode == 0:
            bottom, top = self.FKSolve(L, bottom_plate_pos, reverse, protect)
        else:
            bottom, top = self.FKRaphson(L, bottom_plate_pos, reverse, protect)

        if not self.continuousTranslationConstraint():
            if self.debug:
                disp("FK Resulted In Inverted Plate Alignment. Repairing...")
            #self.IK(top_plate_pos = self.getBottomT() @ tm([0, 0, self.nominal_height, 0, 0, 0]))
            #self.FK(L, protect = True)
            self.fixUpsideDown()
        self.current_plate_transform_local = fsr.globalToLocal(bottom, top)
        #self._undoPlateTransform(bottom, top)
        valid = True
        if not protect:
            valid = self.validate()
        return top, valid

    def FKSolve(self, L, bottom_plate_pos=None, reverse=False, protect=False):
        """
        Older version of python solver, no jacobian used. Takes in length list,
        optionally bottom position, reverse parameter, and protection

        Args:
            L (ndarray(Float)): Goal leg lengths
            bottom_plate_pos (tm): bottom plate transformation in space frame
            reverse (Bool): Boolean to reverse action. If true, treat the top plate as stationary.
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tm: bottom plate transform
            tm: top plate transform
        """
        #Do SPFK with scipy inbuilt solvers. Way less speedy o
        #Or accurate than Raphson, but much simpler to look at
        L = L.reshape((6, 1))
        self.lengths = L.reshape((6, 1)).copy()
        #jac = lambda x : self.inverseJacobianSpace(top_plate_pos = x)

        #Slightly different if the platform is supposed to be "reversed"
        if reverse:
            if bottom_plate_pos == None:
                top_plate_pos = self.getTopT()
            else:
                top_plate_pos = bottom_plate_pos
            fk = lambda x : (self.IK(tm(x), top_plate_pos, protect = True) - L).reshape((6))
            sol = tm(sci.optimize.fsolve(fk, self.getTopT().gTAA()))
            #self.top_plate_pos = bottom_plate_pos
        else:
            #General calls will go here.
            if bottom_plate_pos == None:
                #If no bottom pose is supplied, use the last known.
                bottom_plate_pos = self.getBottomT()
            #Find top pose that produces the desired leg lengths.
            fk = lambda x : (self.IKHelper(bottom_plate_pos, tm(x),
                protect = True)[0] - L).reshape((6))

            #solres = sci.optimize.fmin(fkprime, self.getTopT().TAA, disp=True)
            init = self.getTopT().TAA
            found_sol = True
            solres = sci.optimize.fsolve(fk, init)
            sol = tm(solres)
            sol.angleMod()
            sol.TMtoTAA()
            self.IK(bottom_plate_pos, sol, True)
            nLens = self.getLens()
            for j in range(6):
                if abs(abs(L[j]) - abs(nLens[j])) > 0.00001 or not self.validate(True):
                    return self.FKRaphson(L, bottom_plate_pos, reverse, protect)
        #If not "Protected" from recursion, call IK.
        if not protect:
            self.IK(protect = True)
        return bottom_plate_pos, sol


    def FKRaphson(self, L, bottom_plate_pos =None, reverse=False, protect=False):
        """
        FK Solver
        Adapted from the work done by
        #http://jak-o-shadows.github.io/electronics/stewart-gough/stewart-gough.html
        Args:
            L (ndarray(Float)): Goal leg lengths
            bottom_plate_pos (tm): bottom plate transformation in space frame
            reverse (Bool): Boolean to reverse action. If true, treat the top plate as stationary.
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tm: bottom plate transform
            tm: top plate transform
        """
        if self.debug:
            disp("Starting Raphson FK")
        #^Look here for the original code and paper describing how this works.
        if bottom_plate_pos == None:
            bottom_plate_pos = self.getBottomT()
        success = True
        L = L.reshape((6))
        self.lengths = L.copy()

        bottom_plate_pos_backup = bottom_plate_pos.copy()
        bottom_plate_pos = np.eye(4)
        iteration = 0

        #Initial Guess Position
        #a = fsr.TMtoTAA(bottom_plate_pos @
        #   fsr.TM([0, 0, self.nominal_height, 0, 0, 0])).reshape((6))
        #disp(a, "Attempt")
        try:
            #ap = (fsr.localToGlobal(tm([0, 0, self.nominal_height, 0, 0, 0]), tm()))
            ap = (fsr.localToGlobal(self.current_plate_transform_local, tm())).gTAA().reshape((6))
            attempt = np.zeros((6), dtype=float)
            for i in range(6):
                attempt[i] = ap[i]

            #Call the actual algorithm from the high performance faser library
            #Pass in initial lengths, guess, bottom and top plate positions,
            #max iterations, tolerances, and minimum leg lengths
            attempt, iteration = fmr.SPFKinSpaceR(bottom_plate_pos, L, attempt,
                self.bottom_joints_init, self.top_joints_init,
                self.max_iterations, self.tol_f, self.tol_a, self.leg_ext_min)

            #If the algorithm failed, try again, but this time set initial position to neutral
            if iteration == self.max_iterations:
                for i in range(6):
                    attempt = self.aux_inits[i].TAA.flatten()
                    #attempt[2] = self.nominal_height
                    attempt, iteration = fmr.SPFKinSpaceR(bottom_plate_pos, L, attempt,
                        self.bottom_joints_init, self.top_joints_init,
                        self.max_iterations, self.tol_f, self.tol_a, self.leg_ext_min)
                    if iteration == self.max_iterations:
                        if self.debug:
                            print("Raphson Failed to Converge")
                        self.fail_count += 1
                        self.IK(bottom_plate_pos_backup,
                            bottom_plate_pos_backup @ self.nominal_plate_transform, protect = True)
                        return self.getBottomT(), self.getTopT()
                    else:
                        break

            #Otherwise return the calculated end effector position
            #coords =tm(bottom_plate_pos_backup @ fsr.TAAtoTM(a.reshape((6, 1))))
            coords = bottom_plate_pos_backup @ tm(attempt)

            # @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])

            #Disabling these cause unknown issues so far.
            #self.bottom_plate_pos = bottom_plate_pos_backup
            #self.top_plate_pos = coords


            self.IKHelper(bottom_plate_pos_backup, coords, protect = True)
            self.bottom_plate_pos = bottom_plate_pos_backup
            #@ tm([0, 0, self.bottom_plate_thickness, 0, 0, 0])
            self.top_plate_pos = coords #@ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
            if self.debug:
                disp("Returning from Raphson FK")
            return bottom_plate_pos_backup, tm(coords)
        except Exception as e:

            if self.debug:
                disp("Raphson FK Failed due to: " + str(e))
            self.fail_count+=1
            return self.FKSolve(L, bottom_plate_pos_backup, reverse, protect)


    def lambdaTopPlateReorientation(self, stopt):
        """
        Only used as an assistance function for fixing plate alignment

        Args:
            stopt (tm): top transform in space frame.

        Returns:
            ndarray(Float): distances array

        """
        reorient_helper_1 = fsr.localToGlobal(stopt, self.reorients[0])
        reorient_helper_2 = fsr.localToGlobal(stopt, self.reorients[1])
        reorient_helper_3 = fsr.localToGlobal(stopt, self.reorients[2])

        d1 = fsr.distance(reorient_helper_1,
            tm([self.top_joints_space[0, 0],
            self.top_joints_space[1, 0],
            self.top_joints_space[2, 0], 0, 0, 0]))
        d2 = fsr.distance(reorient_helper_2,
            tm([self.top_joints_space[0, 2],
            self.top_joints_space[1, 2],
            self.top_joints_space[2, 2], 0, 0, 0]))
        d3 = fsr.distance(reorient_helper_3,
            tm([self.top_joints_space[0, 4],
            self.top_joints_space[1, 4],
            self.top_joints_space[2, 4], 0, 0, 0]))
        return np.array([d1 , d2 , d3])

    def reorientTopPlate(self):
        """
        Subfunction of fixUpsideDown,
        responsible for orienting the top plate transform after mirroring
        """
        top_true = self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0])
        res = lambda x : self.lambdaTopPlateReorientation(
            tm([top_true[0], top_true[1], top_true[2], x[0], x[1], x[2]]))
        x_init = self.getTopT()[3:6].flatten()
        solution = sci.optimize.fsolve(res, x_init)
        top_true[3:6] = solution
        self.top_plate_pos = top_true @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
        #disp(self.lambdaTopPlateReorientation(self.getTopT() @
        #   tm([0, 0, -self.top_plate_thickness, 0, 0, 0])))


    def fixUpsideDown(self):
        """
        In situations where the top plate is inverted underneath
        the bottom plate, yet lengths are valid,
        This function can be used to mirror all the joint locations and "fix" the resultant problem
        """
        #print(self.getTopT())
        #print('fixing upside down!')
        for num in range(6):
            #reversable = fsr.globalToLocal(tm([self.top_joints_space[0, num],
            #    self.top_joints_space[1, num], self.top_joints_space[2, num], 0, 0, 0]),
            #    tm([self.bottom_joints_space[0, num],
            #    self.bottom_joints_space[1, num],
            #    self.bottom_joints_space[2, num], 0, 0, 0]))
            #newTJ = tm([self.bottom_joints_space[0, num],
            #    self.bottom_joints_space[1, num],
            #    self.bottom_joints_space[2, num], 0, 0, 0]) @ reversable
            newTJ = fsr.mirror(self.getBottomT() @
                tm([0, 0, -self.bottom_plate_thickness, 0, 0, 0]),
                tm([self.top_joints_space[0, num],
                self.top_joints_space[1, num],
                self.top_joints_space[2, num], 0, 0, 0]))
            self.top_joints_space[0, num] = newTJ[0]
            self.top_joints_space[1, num] = newTJ[1]
            self.top_joints_space[2, num] = newTJ[2]
            self.lengths[num] = fsr.distance(
                self.top_joints_space[:, num], self.bottom_joints_space[:, num])
        top_true = fsr.mirror(self.getBottomT() @ tm([0, 0, -self.bottom_plate_thickness, 0, 0, 0]),
            self.getTopT() @ tm([0, 0, -self.top_plate_thickness, 0, 0, 0]))
        top_true[3:6] = self.getTopT()[3:6] * -1
        self.top_plate_pos = top_true @ tm([0, 0, self.top_plate_thickness, 0, 0, 0])
        self.reorientTopPlate()

    def validateLegs(self, valid = True, donothing = False):
        """
        Validates leg lengths against leg minimums and maximums
        Args:
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it

        Returns:
            Bool: Validity of configuration

        """
        if self.validation_settings[0]:
            temp_valid = self.legLengthConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Leg Length Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Length Corrective Action...")
                self.lengthCorrectiveAction()
                valid = self.validate(True, 1)
        return valid

    def validateContinuousTranslation(self, valid=True, donothing = False):
        """
        Ensures that the top plate is always locally above the bottom plate
        Args:
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it

        Returns:
            Bool: Validity of configuration

        """
        if self.validation_settings[1]:
            temp_valid = self.continuousTranslationConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Platform Inversion Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Continuous Translation Corrective Action...")
                self.continuousTranslationCorrectiveAction()
                valid = self.validate(True, 2)
        return valid

    def validateInteriorAngles(self, valid = True, donothing = False):
        """
        Ensures that interior angles do not violate angular limits
        Args:f
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it

        Returns:
            Bool: Validity of configuration

        """
        if self.validation_settings[2]:
            temp_valid = self.interiorAnglesConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Interior Angles Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Interior Angles Corrective Action...")
                self.IK(self.getBottomT(), self.getBottomT() @
                    self.nominal_plate_transform, protect = True)
                valid = self.validate(True, 3)
        return valid

    def validatePlateRotation(self, valid = True, donothing = False):
        """
        Ensures plate rotation does not validate limits
        Args:
            valid (Bool): whether to start the validator with an assumption of prior validity
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it

        Returns:
            Bool: Validity of configuration

        """
        if self.validation_settings[3]:
            temp_valid = self.plateRotationConstraint()
            valid = valid and temp_valid
            if not temp_valid:
                self.validation_error += "Plate Tilt/Rotate Constraint Violated "
            if not temp_valid and not donothing:
                if self.debug:
                    disp("Executing Plate Rotation Corrective Action By Resetting Platform")
                #disp(self.nominal_plate_transform)
                self.IK(self.getBottomT(),(self.getBottomT() @
                    self.nominal_plate_transform), protect = True)
                valid = self.validate(True, 4)
        return valid

    def validate(self, donothing = False, validation_limit = 4):
        """
        Validate the current configuration of the stewart platform
        Args:
            donothing (Bool): If set to true, even if an invalid configuration is detected,
                will not attempt to correct it
            validation_limit (Int): Description of parameter `validation_limit`.

        Returns:
            Bool: Validity of configuration

        """
        valid = True #innocent until proven INVALID
        #if self.debug:
        #    disp("Validating")
        #First check to make sure leg lengths are not exceeding limit points
        if fsr.distance(self.getTopT(), self.getBottomT()) > 2 * self.nominal_height:
            valid = False

        if validation_limit > 0: valid = self.validateLegs(valid, donothing)
        if validation_limit > 1: valid = self.validateContinuousTranslation(valid, donothing)
        if validation_limit > 2: valid = self.validateInteriorAngles(valid, donothing)
        if validation_limit > 3: valid = self.validatePlateRotation(valid, donothing)

        if valid:
            self.validation_error = ""

        return valid

    def plateRotationConstraint(self):
        """
        Constraint for plate rotations. Assesses validity
        Returns:
            Bool: Validity of configuration

        """
        valid = True
        for i in range(3):
            if self.current_plate_transform_local.gTM()[i, i] <= self.plate_rotation_limit - .0001:
                if self.debug:
                    disp(self.current_plate_transform_local.gTM(), "Erroneous TM")
                    print([self.current_plate_transform_local.gTM()[i, i],
                        self.plate_rotation_limit])
                valid = False
        return valid

    def legLengthConstraint(self):
        """
        Evaluate Leg Length Limitations of Stewart Platform
        Returns:
            Bool: Validity of configuration

        """
        valid = True
        if(np.any(self.lengths < self.leg_ext_min) or np.any(self.lengths > self.leg_ext_max)):
            valid = False
        return valid

    def rescaleLegLengths(self, current_leg_min, current_leg_max):
        """
        Rescale leg lengths to meet minimums
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

    def addLegsToMinimum(self, current_leg_min, current_leg_max):
        """
        Adds the difference to the leg below minimum to preserve end effector orientation
        Args:
            current_leg_min (Float):  current minimum leg length (may be invalid)
            current_leg_max (Float): current maximum leg length (may be invalid)
        """
        boostamt = ((self.leg_ext_min-current_leg_min)+self.leg_ext_safety)
        if self.debug:
            print("Boost Amount: " + str(boostamt))
        self.lengths += boostamt

    def subLegsToMaximum(self, current_leg_min, current_leg_max):
        """
        Subtracts the difference to the leg above maximum to preserve end effector orientation
        Args:
            current_leg_min (Float):  current minimum leg length (may be invalid)
            current_leg_max (Float): current maximum leg length (may be invalid)
        """
        #print([current_leg_max, self.leg_ext_max, current_leg_min,
        #    self.leg_ext_min, current_leg_max -
        #    (current_leg_max - self.leg_ext_max + self.leg_ext_safety)])
        self.lengths -= ((current_leg_max - self.leg_ext_max)+self.leg_ext_safety)
        #print(self.lengths)

    def lengthCorrectiveAction(self):
        """
        Make an attempt to correct leg lengths that are out of bounds.
        Will frequently result in a home-like position
        """
        if self.debug:
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
            self.rescaleLegLengths(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Rescale, "
        elif (current_leg_min < self.leg_ext_min and
            current_leg_max + (self.leg_ext_min - current_leg_min) +
            self.leg_ext_safety < self.leg_ext_max):
            self.addLegsToMinimum(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Boost, "
        elif (current_leg_max > self.leg_ext_max and
            current_leg_min - (current_leg_max - self.leg_ext_max) -
            self.leg_ext_safety > self.leg_ext_min):
            self.validation_error+= " CMethod: Subract, "
            self.subLegsToMaximum(current_leg_min, current_leg_max)
        else:
            self.rescaleLegLengths(current_leg_min, current_leg_max)
            self.validation_error+= " CMethod: Unknown Rescale, "

        #self.lengths[np.where(self.lengths > self.leg_ext_max)] = self.leg_ext_max
        #self.lengths[np.where(self.lengths < self.leg_ext_min)] = self.leg_ext_min
        if self.debug:
            disp(self.lengths, "Corrected Lengths")
        #disp("HEre's what happened")
        self.FK(self.lengths.copy(), protect = True)
        #print(self.lengths)

    def continuousTranslationConstraint(self):
        """
        Ensure that the plate is above the prior

        Returns:
            Bool: Validity at configuration

        """
        valid = True
        bot = self.getBottomT()
        for i in range(6):
            if fsr.globalToLocal(self.getBottomT(), self.getTopT())[2] < 0:
                valid = False
        return valid

    def continuousTranslationCorrectiveAction(self):
        """
        Resets to home position
        """
        self.IK(top_plate_pos = self.getBottomT() @ self.nominal_plate_transform, protect = True)

    def interiorAnglesConstraint(self):
        """
        Ensures no invalid internal angles
        Returns:
            Bool: Validity at configuration
        """
        angles = abs(self.getJointAnglesFromNorm())
        if(np.any(np.isnan(angles))):
            return False
        if(np.any(angles > self.joint_deflection_max)):
            return False
        return True

    def getJointAnglesFromNorm(self):
        """
        Returns the angular deviation of each angle socket from its nominal position in radians

        Returns:
            ndarray(Float): Angular deviation from home of each joint socket

        """
        delta_angles_top = np.zeros((6))
        delta_angles_bottom = np.zeros((6))
        bottom_plate_transform = self.getBottomT()
        top_plate_transform = self.getTopT()
        for i in range(6):

                top_joint_i = tm([
                    self.top_joints_space.T[i][0],
                    self.top_joints_space.T[i][1],
                    self.top_joints_space.T[i][2],
                    top_plate_transform[3],
                    top_plate_transform[4],
                    top_plate_transform[5]])
                bottom_joint_i = tm([
                    self.bottom_joints_space.T[i][0],
                    self.bottom_joints_space.T[i][1],
                    self.bottom_joints_space.T[i][2],
                    bottom_plate_transform[3],
                    bottom_plate_transform[4],
                    bottom_plate_transform[5]])

                #We have the relative positions to the top plate
                #   of the bottom joints (bottom angles) in home pose
                #We have the relative positions to the bottom plate of
                #   the top joints (bottom_joint_angles_init) in home pose
                bottom_to_top_local_home = self.bottom_joint_angles_init[i].copy()
                top_to_bottom_local_home = self.bottom_joint_angles[i].copy()

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

    def getJointAnglesFromVertical(self):
        """
        Calculate joint angles from vertical at each joint
        Returns:
            ndarray(Float): top joints from vertical (downward)
            ndarray(Float): bottom joints from vertical (upward)

        """
        top_down = np.zeros((6))
        bottom_up = np.zeros((6))
        for i in range(6):
            top_joints_temp = self.top_joints_space[:, i].copy().flatten()
            top_joints_temp[2] = 0
            bottom_joints_temp = self.bottom_joints_space[:, i].copy().flatten()
            bottom_joints_temp[2] = bottom_joints_temp[2] + 1
            angle = fsr.angleBetween(
                self.bottom_joints_space[:, i],
                self.top_joints_space[:, i],
                top_joints_temp)
            angle_up = fsr.angleBetween(
                self.top_joints_space[:, i],
                self.bottom_joints_space[:, i],
                bottom_joints_temp)
            top_down[i] = angle
            bottom_up[i] = angle_up
        return top_down, bottom_up

    """
      ______                                        _   _____                              _
     |  ____|                                      | | |  __ \                            (_)
     | |__ ___  _ __ ___ ___  ___    __ _ _ __   __| | | |  | |_   _ _ __   __ _ _ __ ___  _  ___ ___
     |  __/ _ \| '__/ __/ _ \/ __|  / _` | '_ \ / _` | | |  | | | | | '_ \ / _` | '_ ` _ \| |/ __/ __|
     | | | (_) | | | (_|  __/\__ \ | (_| | | | | (_| | | |__| | |_| | | | | (_| | | | | | | | (__\__ \
     |_|  \___/|_|  \___\___||___/  \__,_|_| |_|\__,_| |_____/ \__, |_| |_|\__,_|_| |_| |_|_|\___|___/
                                                                __/ |
                                                               |___/
    """
    def componentForces(self, tau):
        """
        Calculate force components for given leg forces
        Args:
            tau (ndarray(Float)): force exerted through each leg in Newtons.

        Returns:
            ndarray(Float): vertical components of forces
            ndarray(Float): horizontal components of forces

        """
        vertical_components = np.zeros((6))
        horizontal_components = np.zeros((6))
        for i in range(6):
            top_joint = self.top_joints_space[:, i].copy().flatten()
            top_joint[2] = 0
            angle = fsr.angleBetween(
                self.bottom_joints_space[:, i],
                self.top_joints_space[:, i],
                top_joint)
            vertical_force = tau[i] * np.sin(angle)
            horizontal_force = tau[i] * np.cos(angle)
            vertical_components[i] = vertical_force
            horizontal_components[i] = horizontal_force
        return vertical_components, horizontal_components

    def bottomTopCheck(self, bottom_plate_pos, top_plate_pos):
        """
        Checks to make sure that a bottom and top provided are not null

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

    def jacobianSpace(self, bottom_plate_pos = None, top_plate_pos = None):
        """
        Calculates space jacobian for stewart platform. Takes in bottom transform and top transform

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame

        Returns:
            ndarray(Float): Jacobian for current configuration

        """
        #If not supplied paramters, draw from stored values
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)
        #Just invert the inverted
        inverse_jacobian = self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos)
        return ling.pinv(inverse_jacobian)


    def inverseJacobianSpace(self, bottom_plate_pos = None, top_plate_pos = None, protect = True):
        """
        Calculates Inverse Jacobian for stewart platform. Takes in bottom and top transforms

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            ndarray(Float): Inverse Jacobian for current configuration

        """
        #Ensure everything is kosher with the plates
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)

        #Store old values
        old_bottom_plate_transform = self.getBottomT()
        old_top_plate_transform = self.getTopT()

        #Perform IK on bottom and top
        self.IK(bottom_plate_pos, top_plate_pos, protect = protect)

        #Create Jacobian
        inverse_jacobian_transpose = np.zeros((6, 6))
        for i in range(6):
            #todo check sign on nim,
            ni = fmr.Normalize(self.top_joints_space[:, i]-self.bottom_joints_space[:, i])
             #Reverse for upward forces?
            qi = self.bottom_joints_space[:, i]
            col = np.hstack((np.cross(qi, ni), ni))
            inverse_jacobian_transpose[:, i] = col
        inverse_jacobian = inverse_jacobian_transpose.T

        #Restore original Values
        self.IK(old_bottom_plate_transform, old_top_plate_transform, protect = protect)
        return inverse_jacobian

    #Returns Top Down Jacobian instead of Bottom Up
    def altInverseJacobianSpace(self,
        bottom_plate_pos = None, top_plate_pos = None, protect = True):
        """
        Returns top down jacobian instead of bottom up

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            ndarray(Float): top down Jacobian Space

        """
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)
        old_bottom_plate_transform = copy.copy(bottom_plate_pos)
        old_top_plate_transform = copy.copy(top_plate_pos)
        self.IK(bottom_plate_pos, top_plate_pos)
        inverse_jacobian_transpose = np.zeros((6, 6))
        for i in range(6):
            ni = fmr.Normalize(self.bottom_joints_space[:, i]-self.top_joints_space[:, i])
            qi = self.top_joints_space[:, i]
            inverse_jacobian_transpose[:, i] = np.hstack((np.cross(qi, ni), ni))
        inverse_jacobian = inverse_jacobian_transpose.conj().transpose()

        self.IKHelper(old_bottom_plate_transform, old_top_plate_transform)

        return inverse_jacobian

    #Adds in actuator and plate forces, useful for finding forces on a full stack assembler
    def carryMassCalc(self, twrench, protect=False):
        """
        Calculates the forces on each leg given their masses,
        masses of plates, and a wrench on the end effector.
        Use this over Local in most cases
        Args:
            twrench (ndarray(Float)): input wrench for configuration
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            ndarray(Float): forces in Newtons for each leg

        """
        wrench = twrench.copy()
        wrench = wrench + fsr.makeWrench(self.getTopT(),
            self.top_plate_newton_force, self.dir)
        tau = self.measureForcesFromWrenchEE(self.getBottomT(),
            self.getTopT(), wrench, protect = protect)
        for i in range(6):
            #print(self.getActuatorLoc(i, 't'))
            wrench += fsr.makeWrench(self.getActuatorLoc(i, 't'),
                self.act_shaft_newton_force, self.dir)
            wrench += fsr.makeWrench(self.getActuatorLoc(i, 'b'),
                self.act_motor_newton_force, self.dir)
        wrench = wrench + fsr.makeWrench(self.getBottomT(),
            self.bottom_plate_newton_force, self.dir)
        return tau, wrench

    def carryMassCalcLocal(self, twrench, protect = False):
        """
        Perform force mass calculations in local frame
        Args:
            twrench (ndarray(Float)): input wrench for configuration
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            ndarray(Float): forces in Newtons for each leg

        """
        #We will here assume that the wrench is in the local frame of the top platform.
        wrench = twrench.copy()
        wrench = wrench + fsr.makeWrench(tm(), self.top_plate_newton_force, self.dir)
        tau = self.measureForcesAtEENew(wrench, protect = protect)
        wrench_local_frame = fsr.transformWrenchFrame(wrench, self.getTopT(), self.getBottomT())

        for i in range(6):
            #print(self.getActuatorLoc(i, 't'))
            #The following representations are equivalent.
            wrench_local_frame += fsr.makeWrench(fsr.globalToLocal(self.getActuatorLoc(i, 't'),
                self.getBottomT()), self.act_shaft_newton_force, self.dir)
            wrench_local_frame += fsr.makeWrench(fsr.globalToLocal(self.getActuatorLoc(i, 'b'),
                self.getBottomT()), self.act_motor_newton_force, self.dir)
            #wrench_local_frame += fsr.transformWrenchFrame(fsr.makeWrench(tm(),
            #    self.act_shaft_newton_force, self.dir),
            #   self.getActuatorLoc(i, 't'), self.getBottomT())
            #wrench_local_frame += fsr.transformWrenchFrame(fsr.makeWrench(tm(),
            #    self.act_motor_newton_force, self.dir),
            #   self.getActuatorLoc(i, 'b'), self.getBottomT())
        wrench_local_frame = wrench_local_frame + fsr.makeWrench(tm(),
            self.bottom_plate_newton_force, self.dir)
        return tau, wrench_local_frame

    def measureForcesAtEENew(self, wrench, protect = False):
        """
        Measure forces based on end effector wrench
        Args:
            wrench (ndarray(Float)): Description of parameter `wrench`.
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            ndarray(Float): forces in Newtons for each leg

        """
        jacobian_space = ling.pinv(
            self.inverseJacobianSpace(self.getBottomT(), self.getTopT(), protect = protect))
        tau = jacobian_space.T @ wrench
        self.leg_forces = tau
        return tau

    def carryMassCalcUp(self, twrench, protect = False):
        """
        Carry masses from bottom up
        Args:
            twrench (ndarray(Float)): input wrench for configuration
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            ndarray(Float): forces in Newtons for each leg
            ndarray(Float): wrench to carry

        """
        wrench = twrench.copy()
        wrench = wrench + fsr.makeWrench(self.getBottomT(),
            self.bottom_plate_mass * self.grav, np.array([0, 0, -1]))
        tau = self.measureForcesFromBottomEE(
            self.getBottomT(), self.getTopT(), wrench, protect = protect)
        for i in range(6):
            wrench += fsr.makeWrench(
                self.getActuatorLoc(i, 't'), self.act_shaft_mass * self.grav, np.array([0, 0, -1]))
            wrench += fsr.makeWrench(
                self.getActuatorLoc(i, 'b'), self.act_motor_mass * self.grav, np.array([0, 0, -1]))
        wrench = wrench + fsr.makeWrench(
            self.getTopT(), self.top_plate_mass * self.grav, np.array([0, 0, -1]))
        return tau, wrench

    #Get Force wrench from the End Effector Force
    def measureForcesFromWrenchEE(self, bottom_plate_pos = np.zeros((1)),
        top_plate_pos = np.zeros((1)), top_plate_wrench = np.zeros((1)), protect = True):
        """
        Calculates forces on legs given end effector wrench

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
            top_plate_wrench (ndarray(Float)): input wrench for configuration
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tau: forces in Newtons for each leg

        """
        bottom_plate_pos, top_plate_pos = self.bottomTopCheck(bottom_plate_pos, top_plate_pos)
        if top_plate_wrench.size < 6:
            disp("Please Enter a Wrench")
        #top_wrench = fmr.Adjoint(ling.inv(top_plate_pos)).conj().transpose() @ top_plate_wrench
        #Modern Robotics 3.95 Fb = Ad(Tba)^T * Fa
        #top_wrench = top_plate_pos.inv().Adjoint().T @ top_plate_wrench
        top_wrench = fsr.transformWrenchFrame(top_plate_wrench, tm(), top_plate_pos)
        jacobian_space = ling.pinv(
            self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos, protect = protect))
        tau = jacobian_space.T @ top_wrench
        self.leg_forces = tau
        return tau

    def measureForcesFromBottomEE(self, bottom_plate_pos = np.zeros((1)),
        top_plate_pos = np.zeros((1)), top_plate_wrench = np.zeros((1)), protect = True):
        """
        Calculates forces on legs given end effector wrench

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
            top_plate_wrench (ndarray(Float)): input wrench for configuration
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True

        Returns:
            tau: forces in Newtons for each leg

        """
        bottom_plate_pos, top_plate_pos = self._bttomTopCheck(bottom_plate_pos, top_plate_pos)
        if top_plate_wrench.size < 6:
            disp("Please Enter a Wrench")
        #top_wrench = fmr.Adjoint(ling.inv(top_plate_pos)).conj().transpose() @ top_plate_wrench
        bottom_wrench = bottom_plate_pos.inv().Adjoint().T @ top_plate_wrench
        jacobian_space = ling.pinv(
            self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos, protect = protect))
        tau = jacobian_space.T @ bottom_wrench
        self.leg_forces = tau
        return tau

    def wrenchEEFromMeasuredForces(self, bottom_plate_pos, top_plate_pos, tau):
        """
        Calculates wrench on end effector from leg forces

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
            tau (ndarray(Float)): force exerted through each leg in Newtons.

        Returns:
            ndarray(Float): top plate wrench
            ndarray(Float): top wrench (local)
            ndarray(Float): jacobian

        """
        self.leg_forces = tau
        jacobian_space = ling.pinv(self.inverseJacobianSpace(bottom_plate_pos, top_plate_pos))
        top_wrench = ling.inv(jacobian_space.conj().transpose()) @ tau
        #self.top_plate_wrench = fmr.Adjoint(top_plate_pos).conj().transpose() @ top_wrench
        self.top_plate_wrench = top_plate_pos.Adjoint().conj().transpose() @ top_wrench
        return self.top_plate_wrench, top_wrench, jacobian_space

    def wrenchBottomFromMeasuredForces(self, bottom_plate_pos, top_plate_pos, tau):
        """
        Unused. Calculates wrench on the bottom plate from leg forces

        Args:
            bottom_plate_pos (tm): bottom plate transformation in space frame
            top_plate_pos (tm): top plate transformation in space frame
            tau (ndarray(Float)): force exerted through each leg in Newtons.

        Returns:
            ndarray(Float): bottom plate wrench
            ndarray(Float): bottom wrench (local)
            ndarray(Float): jacobian

        """
        self.leg_forces = tau
        jacobian_space = ling.pinv(self.altInverseJacobianSpace(bottom_plate_pos, top_plate_pos))
        bottom_wrench = ling.inv(jacobian_space.conj().transpose()) @ tau
        #self.bottom_plate_wrench = fmr.Adjoint(bottom_plate_pos).conj().transpose() @ bottom_wrench
        self.bottom_plate_wrench = bottom_plate_pos.Adjoint().conj().transpose() @ bottom_wrench
        return self.bottom_plate_wrench, bottom_wrench, jacobian_space

    def sumActuatorWrenches(self, forces = None):
        """
        Sum all actuator wrenches to produce bottom wrench
        Args:
            forces (ndarray(Float)): leg forces in Newtons

        Returns:
            ndarray(Float): bottom plate wrench

        """
        if forces is None:
            forces = self.leg_forces

        wrench = fsr.makeWrench(tm(), 0, [0, 0, -1])
        for i in range(6):
            unit_vector = fmr.Normalize(self.bottom_joints_space[:, i]-self.top_joints_space[:, i])
            wrench += fsr.makeWrench(self.top_joints_space[:, i], float(forces[i]), unit_vector)
        #wrench = fsr.transformWrenchFrame(wrench, tm(), self.getTopT())
        return wrench


    def move(self, T, protect = False):
        """
        Move entire Assembler Stack to another location and orientation
        This function and syntax are shared between all kinematic structures.
        Args:
            T (tm): New base transform to move to
            protect (Bool): Boolean to bypass error detection and correction. Bypass if True
        """
        #Moves the base of the stewart platform to a new location


        self.current_plate_transform_local = fsr.globalToLocal(self.getBottomT(), self.getTopT())
        self.bottom_plate_pos = T.copy()
        self.IK(
            top_plate_pos = fsr.localToGlobal(self.getBottomT(),
                    self.current_plate_transform_local),
            protect = protect)

    def printOutOfDateFunction(self, old_name, use_name):   # pragma: no cover
        """
        Prints an old function with an OOD notice
        Args:
            old_name (String): Description of parameter `old_name`.
            use_name (String): Description of parameter `use_name`.
        """
        print(old_name + " is deprecated. Please use " + use_name + " instead.")

    def SetMasses(self, plateMass, actuatorTop, actuatorBottom, grav = 9.81, tPlateMass = 0):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetMasses", "setMasses")
        return self.setMasses(plateMass, actuatorTop, actuatorBottom, grav, tPlateMass)
    def SetGrav(self, grav = 9.81):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetGrav", "setGrav")
        return self.setGrav(grav)
    def SetCOG(self, motor_grav_center, shaft_grav_center):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetCOG", "setCOG")
        return self.setCOG(motor_grav_center, shaft_grav_center)
    def SetAngleDev(self, MaxAngleDev = 55):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetAngleDev", "setMaxAngleDev")
        return self.setMaxAngleDev(MaxAngleDev)
    def SetPlateAngleDev(self, MaxPlateDev = 60):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetPlateAngleDev", "setMaxPlateRotation")
        return self.setMaxPlateRotation(MaxPlateDev)
    def SetDrawingDimensions(self, OuterTopRad, OuterBotRad, ShaftRad, MotorRad):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SetDrawingDimensions", "setDrawingDimensions")
        return self.setDrawingDimensions( OuterTopRad, OuterBotRad, ShaftRad, MotorRad)
    def _setPlatePos(self, bottomT, topT):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_setPlatePos", "setPlatePos")
        return self.setPlatePos(bottomT, topT)
    def gLens(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("gLens", "getLens")
        return self.getLens()
    def gtopT(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("gtopT", "getTopT")
        return self.getTopT()
    def gbottomT(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("gbottomT", "getBottomT")
        return self.getBottomT()
    def GetActuatorUnit(self, p1, p2, dist):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("GetActuatorUnit", "fsr.getUnitVec")
        return fsr.getUnitVec(p1, p2, dist)
    def SpinCustom(self, rot):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SpinCustom", "spinCustom")
        return self.spinCustom(rot)
    def LambdaRTP(self, stopt):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("LambdaRTP", "lambdaTopPlateReorientation")
        return self.lambdaTopPlateReorientation(stopt)
    def ReorientTopPlate(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("ReorientTopPlate", "reorientTopPlate")
        return self.reorientTopPlate()
    def _legLengthConstraint(self, donothing):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_legLengthConstraint", "legLengthConstraint")
        return self.legLengthConstraint()
    def _resclLegs(self, cMin, cMax):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_resclLegs", "rescaleLegLengths")
        return self.rescaleLegLengths(cMin, cMax)
    def _addLegs(self, cMin, cMax):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_addLegs", "addLegsToMinimum")
        return self.addLegsToMinimum(cMin, cMax)
    def _subLegs(self, cMin, cMax):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_subLegs", "subLegsToMaximum")
        return self.subLegsToMaximum(cMin, cMax)
    def _lengthCorrectiveAction(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_lengthCorrectiveAction", "lengthCorrectiveAction")
        return self.lengthCorrectiveAction()
    def _continuousTranslationConstraint(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction(
            "_continuousTranslationConstraint", "continuousTranslationConstraint")
        return self.continuousTranslationConstraint()
    def _continuousTranslationCorrectiveAction(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction(
            "_continuousTranslationCorrectiveAction", "continuousTranslationCorrectiveAction")
        return self.continuousTranslationCorrectiveAction()
    def _interiorAnglesConstraint(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_interiorAnglesConstraint", "interiorAnglesConstraint")
        return self.interiorAnglesConstraint()
    def AngleFromNorm(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("AngleFromNorm", "getJointAnglesFromNorm")
        return self.getJointAnglesFromNorm()
    def AngleFromVertical(self):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("AngleFromVertical", "getJointAnglesFromVertical")
        return self.getJointAnglesFromVertical()
    def _bottomTopCheck(self, bottomT, topT):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("_bottomTopCheck", "bottomTopCheck")
        return self.bottomTopCheck(bottomT, topT)
    def JacobianSpace(self, bottomT = None, topT = None):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("JacobianSpace", "jacobianSpace")
        return self.jacobianSpace(bottomT, topT)
    def InverseJacobianSpace(self, bottomT = None, topT = None, protect = True):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("InverseJacobianSpace", "inverseJacobianSpace")
        return self.inverseJacobianSpace(bottomT, topT, protect)
    def AltInverseJacobianSpace(self, bottomT = None, topT = None, protect = True):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("AltInverseJacobianSpace", "altInverseJacobianSpace")
        return self.altInverseJacobianSpace(bottomT, topT, protect)
    def CarryMassCalc(self, twrench, protect = False):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("CarryMassCalc", "carryMassCalc")
        return self.carryMassCalc(twrench, protect)
    def CarryMassCalcNew(self, twrench, protect = False):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("CarryMassCalcNew", "carryMassCalcLocal")
        return self.carryMassCalcLocal(twrench, protect)
    def MeasureForcesAtEENew(self, wrench, protect = False):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("MeasureForcesAtEENew", "measureForcesAtEENew")
        return self.measureForcesAtEENew(wrench, protect)
    def CarryMassCalcUp(self, twrench, protect = False):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("CarryMassCalcUp", "carryMassCalcUp")
        return self.carryMassCalcUp(twrench, protect)
    def MeasureForcesFromWrenchEE(self, bottomT = np.zeros((1)) ,  # pragma: no cover
        topT = np.zeros((1)), topWEE = np.zeros((1)), protect = True):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("MeasureForcesFromWrenchEE", "measureForcesFromWrenchEE")
        return self.measureForcesFromWrenchEE(bottomT, topT, topWEE, protect)
    def MeasureForcesFromBottomEE(self, bottomT = np.zeros((1)) ,  # pragma: no cover
        topT = np.zeros((1)), topWEE = np.zeros((1)), protect = True):
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("MeasureForcesFromBottomEE", "measureForcesFromBottomEE")
        return self.measureForcesFromBottomEE(bottomT, topT, topWEE, protect)
    def WrenchEEFromMeasuredForces(self, bottomT, topT, tau):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("WrenchEEFromMeasuredForces", "wrenchEEFromMeasuredForces")
        return self.wrenchEEFromMeasuredForces(bottomT, topT, tau)
    def WrenchBottomFromMeasuredForces(self, bottomT, topT, tau):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction(
            "WrenchBottomFromMeasuredForces", "wrenchBottomFromMeasuredForces")
        return self.wrenchBottomFromMeasuredForces(bottomT, topT, tau)
    def SumActuatorWrenches(self, forces = None):  # pragma: no cover
        """
        Deprecated. Don't Use
        """
        self.printOutOfDateFunction("SumActuatorWrenches", "sumActuatorWrenches")
        return self.sumActuatorWrenches(forces)

def loadSP(fname, file_directory = "../robot_definitions/", baseloc = None, altRot = 1):
    """
    Loads A Stewart Platform Object froma  file

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

    newsp.setDrawingDimensions(
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
    Builds a new SP, called usually by a constructor
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
        actuator_min, actuator_max, bj, bottom_plate_thickness, top_plate_thickness)
    if rot == -1:
        tentative_height = midHeightEstimate(
            actuator_min, actuator_max, tj, bottom_plate_thickness, top_plate_thickness)
    top = bottom @ tm(np.array([0.0, 0.0, tentative_height, 0.0, 0.0, 0.0]))

    newsp = SP(bj, tj, bottom, top,
        actuator_min, actuator_max,
        bottom_plate_thickness, top_plate_thickness, name)
    newsp.setMasses(
        plate_bot_mass,
        actuator_shaft_mass,
        actuator_motor_mass,
        top_plate_mass = plate_top_mass)
    newsp.setCOG(motor_grav_center, shaft_grav_center)

    return newsp
def makeSP(bRad, tRad, spacing, baseT,
    platOffset, rot = -1, plate_thickness_avg = 0, altRot = 0):
    """
    Largely deprecated in favor of Loading SP objects from json

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

    disp(bangles, "bangles")
    disp(tangles, "tangles")
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
    sp.bRad = bRad
    sp.tRad = tRad

    return sp, bottom, top
#Helpers
def midHeightEstimate(leg_ext_min, leg_ext_max, bj, bth, tth):
    """
    Calculates an estimate of thee resting height of a stewart plaform
    Args:
        leg_ext_min (float): minimum leg extension
        leg_ext_max (float): maximum leg extension
        bj (array(float)): bottom joints
        bth (tm):bottom plate thickness
        tth (tm): top plate thickness

    Returns:
        Float: Description of returned object.

    """
    s1 = (leg_ext_min + leg_ext_max) / 2
    d1 = fsr.distance(tm([bj[0, 0], bj[1, 0], bj[2, 0], 0, 0, 0]),
            tm([bj[0, 1], bj[1, 1], bj[2, 1], 0, 0, 0]))
    hest = (np.sqrt(s1 ** 2 - d1 **2)) + bth + tth
    return hest
