"""Holding file for Robot class, which is the superclass of sp_model and arm_model."""
from ..general import fsr, tm, Wrench
import numpy as np

class Robot:
    """
    Models a Robot.

    Provides interfaces for a standard set of functions, and makes best guesses at some implementations.
    Most functions will work if either jacobian or inverseJacobian are defined in child class.
    """

    def __init__(self, name : str = "Robot") -> 'Robot':
        """
        Generate new Robot instance.

        Args:
            name (str, optional): Name of the robot instance. Defaults to "Robot".

        Returns:
            Robot: Robot instance.
        """
        self.name = name
        self._end_effector_pos_global = None
        self._base_pos_global = None
        self.grav = np.array([0, 0, -9.81])
        self._last_tau = None
        self.cameras = []

    def getActuatorForces(self) -> 'np.ndarray[float]':
        """
        Return a copy of the last calculated actuator forces/torques depending on the robot type.

        If the type of the robot is parallel/joints are linear, then it will be forces N.
        If the type of the robot is serial/joints are rotary, then it will be torque N*m.

        Returns:
            last_tau (np.ndarray[float]) : last calculated joint/actuator forces.
        """
        return self._last_tau.copy()

    def getEEPos(self) -> tm:
        """
        Return a copy of the current global end effector position.

        Returns:
            tm: global frame end effector position
        """
        return self._end_effector_pos_global.copy()
    
    def getBasePos(self) -> tm:
        """
        Return a copy of the current global base position.

        Returns:
            tm: global base position
        """
        return self._base_pos_global.copy()
    
    def getGrav(self) -> 'np.ndarray[float]':
        """
        Get a copy of the current applied gravity vector.

        Returns:
            np.ndarray[Float] : gravity vector
        """
        return self.grav.copy()
    
    def setGrav(self, new_grav : 'np.ndarray[float]'= np.array([0, 0, -9.81])) -> None:
        """
        Set gravity vector.

        Defaults to earth gravity (-9.81 ms^2 in negative Z)
        Args:
            grav (float): Acceleration due to gravity
        Returns:
            None: None
        """
        self.grav = new_grav

    def FK(self,  *args, **kwargs):
        """
        Calculate Forward Kinematics of a robot.
        
        This function must be implemented in child class.
        """        
        pass

    def IK(self, *args, **kwargs):
        """
        Calculate Inverse Kinematics of a robot.
        
        This function must be implemented in child class.
        """
        pass

    def randomPos(self):
        """
        Generate a random configuration through Forward Kinematics.
        
        This function must be implemented in child class.
        """
        pass

    def velocityAtEndEffector(self, 
            joint_vels : 'np.ndarray[float]', *args, **kwargs) -> 'np.ndarray[float]':
        """
        Calculate end effector velocity twist given a set of joint velocities.

        Source for Open Chains: Modern Robotics 5.1.6
        Source for Parallel Robots: NASA Memorandum 107585 p18 (36)
        Args:
            joint_vels (np.ndarray[float]) : joint velocities
        Returns:
            np.ndarray[float]: end effector velocity twist.
        """
        end_effector_vel = self.jacobian(*args, **kwargs) @ joint_vels.reshape((len(joint_vels), 1))
        return end_effector_vel

    def velocityAtJoints(self, end_effector_twist, *args, **kwargs) -> 'np.ndarray[float]':
        """
        Calculate joint velocities from end effector velocity twist.

        Source for Open Chains: Modern Robotics 5.1.6
        Source for Parallel Robots: NASA Memorandum 107585 p18 (37)
        Args:
            end_effector_twist (np.ndarray[float]) : end effector velocity twist
        Returns:
            np.ndarray[float]: joint velocities
        """
        joint_velocities = self.inverseJacobian(*args, **kwargs) @ end_effector_twist.reshape((6, 1))
        return joint_velocities

    def staticForces(self, eef_wrench : Wrench, *args, **kwargs) -> 'np.ndarray[float]':
        """
        Calculate actuator forces for a static application of a wrench in global space.

        Args:
            eef_wrench (Wrench): Wrench in global frame to apply at end effector

        Returns:
            np.ndarray(Float): Forces on each actuator of the robot
        """        
        self._last_tau = self.jacobian(*args, **kwargs).T @ eef_wrench
        return self._last_tau.copy()

    def staticForcesInv(self, forces : 'np.ndarray[float]',  *args,**kwargs) -> Wrench:
        """
        Calculate the wrench acting on the end effector of the robot in the space frame given joint/actuator forces.

        Args:
            forces (np.ndarray(Float)): joint forces/torques in Newtons.

        Returns:
            Wrench: Space Frame Acting Wrench
        """
        self._last_tau = forces
        return Wrench(np.linalg.pinv(self.jacobian(*args, **kwargs).T) @ forces)

    def staticForcesBody(self, eef_wrench : Wrench,  *args,**kwargs) -> 'np.ndarray[float]':
        """
        Calculate actuator forces for a static application of a wrench in body frame.

        Args:
            eef_wrench (Wrench): Wrench in body frame to apply at end effector

        Returns:
            np.ndarray(Float): Forces on each actuator of the robot
        """   
        self._last_tau =  self.jacobianBody(*args, **kwargs).T @ eef_wrench
        return self._last_tau.copy()

    def staticForcesInvBody(self, forces : 'np.ndarray[float]',  *args,**kwargs) -> Wrench:
        """
        Calculate the wrench acting on the end effector of the robot in the body/EE frame given joint/actuator forces.

        Args:
            forces (np.ndarray(Float)): joint forces/torques in Newtons.

        Returns:
            Wrench: Body Frame Acting Wrench
        """
        self._last_tau = forces
        return Wrench(np.linalg.pinv(self.jacobianBody(*args, **kwargs).T) @ forces)

    # Either jacobian or inverseJacobian must be defined in child class.
    def jacobian(self, *args, **kwargs) -> 'np.ndarray[float]':
        """
        Calculate Space Jacobian for given configuration.

        Returns:
            jacobian
        """
        return np.linalg.pinv(self.inverseJacobian(*args, **kwargs))

    # Either jacobian or inverseJacobian must be defined in child class.
    def inverseJacobian(self, *args, **kwargs) -> 'np.ndarray[float]':
        """
        Calculate Inverse Space Jacobian for given configuration.

        Returns:
            jacobian
        """
        return np.linalg.pinv(self.jacobian(*args, **kwargs))

    def jacobianBody(self, *args, **kwargs) -> 'np.ndarray[float]':
        """
        Calculate Body (EE Frame) Jacobian for given configuration.

        Returns:
            body jacobian
        """
        return self._end_effector_pos_global.inv().adjoint() @ self.jacobian(*args, **kwargs)

    def inverseJacobianBody(self, *args, **kwargs) -> 'np.ndarray[float]':
        """
        Inverse Body (EE Frame) Jacobian for given configuration.

        Returns:
            jacobian
        """
        #Why both inverses? One will have to be overridden in any robot configuration
        return np.linalg.pinv(self.jacobianBody(*args, **kwargs))

    def addCamera(self, cam, end_effector_to_cam : tm) -> None:
        """
        Add a camera to the robot, mounted at a fixed offset from the end effector.

        Args:
            cam: camera object
            end_effector_to_cam (tm): end effector to camera transform
        """
        cam.moveCamera(self.getEEPos() @ end_effector_to_cam)
        img, _, _ = cam.getPhoto(self.getEEPos() @ tm([0, 0, 1, 0, 0, 0]))
        camL = [cam, end_effector_to_cam, img]
        self.cameras.append(camL)

    def updateCams(self) -> None:
        """Update camera locations to follow the current end effector pose."""
        for i in range(len(self.cameras)):
            self.cameras[i][0].moveCamera(self.getEEPos() @ self.cameras[i][1])

    def _servoStep(self, pose : tm):
        """
        Drive the robot to the given end effector pose during visual servoing.

        This function must be implemented in child class.
        Args:
            pose (tm): target end effector pose for this servo step
        Returns:
            actuator state (joint angles, leg lengths, etc) after moving to pose
        """
        pass

    def _servoNoSolutionState(self):
        """
        Return the actuator state to report when visual servoing cannot proceed.

        This function must be implemented in child class.
        """
        pass

    def visualServoToTarget(self, target : tm, pixel_tol : int = 2, desired_dist : float = 1.0,
            pose_delta : float = 0.1, pose_tol : float = 0.2, max_iter : int = 1000,
            cam_ind : int = 0):
        """
        Perform visual servoing to a target using a virtual camera mounted on the robot.

        Args:
            target (tm): Object of interest to move towards.
            pixel_tol (int, optional): Pixel Tolerance. Defaults to 2.
            desired_dist (float, optional): Desired distance to object in meters. Defaults to 1.0.
            pose_delta (float, optional): Distance in meters to move the end effector per step.
                Defaults to 0.1.
            pose_tol (float, optional): Pose tolerance in meters. Defaults to 0.2.
            max_iter (int, optional): Maximum iterations before giving up. Defaults to 1000.
            cam_ind (int, optional): Virtual camera to use. Defaults to 0.

        Returns:
            state: final actuator state (joint angles, leg lengths, etc)
            state_list (list): actuator state at each step along the servo trajectory.
        """
        if len(self.cameras) == 0:
            print('NO CAMERA CONNECTED')
            return self._servoNoSolutionState(), []
        at_target = False
        done = False
        start_pos = self.getEEPos()
        state_list = []
        j = 0
        while not (at_target and done):
            pose_adjust = tm()
            at_target = True
            done = True
            img, _, suc = self.cameras[cam_ind][0].getPhoto(target)
            if not suc:
                print('Failed to locate Target')
                return self._servoNoSolutionState(), []
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
                d = fsr.distance(self.getEEPos(), target)
                if d < desired_dist - pose_tol:
                    done = False
                    pose_adjust[2] = -.01
                if d > desired_dist + pose_tol:
                    done = False
                    pose_adjust[2] = .01
            start_pos = start_pos @ pose_adjust
            state = self._servoStep(start_pos)
            state_list.append(state)
            self.updateCams()
            j = j + 1
            if j > max_iter:
                print('Failed to find solution, max iterations')
                return self._servoNoSolutionState(), []
        return state, state_list

    def _motionLimits(self):
        """
        Return the (max_vels, max_accels, max_jerks) actuator motion limits to use
        as defaults in timeParametrizePath().

        This function must be implemented in child class.
        """
        pass

    def timeParametrizePath(self, path, max_vels : 'np.ndarray[float]' = None,
            max_accels : 'np.ndarray[float]' = None,
            max_jerks : 'np.ndarray[float]' = None) -> 'JointTrajectory':
        """
        Convert a sequence of actuator-space waypoints into a velocity/acceleration-limited
        (and, if max_jerks is given, jerk-limited) time-parametrized JointTrajectory.

        By default this uses the robot's own max_vels/max_accels/max_jerks (see
        setJointProperties); pass any of the three explicitly to override them for this
        call without changing the robot's configured limits.

        Args:
            path (list[np.ndarray[float]]): at least two actuator-space waypoints
            max_vels (np.ndarray[float], optional): overrides the robot's configured limit
            max_accels (np.ndarray[float], optional): overrides the robot's configured limit
            max_jerks (np.ndarray[float], optional): overrides the robot's configured limit. A
                fully-infinite value (the default when unset) disables jerk
                limiting rather than raising an error.

        Returns:
            JointTrajectory: retimed trajectory; see JointTrajectory.sample()
        """
        from ..path_planning.trajectory import JointTrajectory
        default_vels, default_accels, default_jerks = self._motionLimits()
        v = default_vels if max_vels is None else max_vels
        a = default_accels if max_accels is None else max_accels
        j = default_jerks if max_jerks is None else max_jerks
        if j is not None and np.all(~np.isfinite(np.atleast_1d(j))):
            j = None
        return JointTrajectory(path, v, a, j)

    def move(self, new_pos : tm) -> None:
        """
        Move to a new position.
        
        This function must be implemented in child class.
        Args: 
            new_pos (tm) : new position to move to
        """
        pass

    def draw(self, *args, **kwargs) -> None:
        """
        Draw the robot.
        
        This function must be implemented in child class.
        """
        pass
