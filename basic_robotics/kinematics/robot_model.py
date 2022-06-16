"""Holding file for Robot class, which is the superclass of sp_model and arm_model."""
from ..general import tm, Wrench
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
