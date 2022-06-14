from ..general import tm, fmr, fsr, Wrench
import numpy as np
import scipy as sci

class Robot:

    def __init__(self, name = "Robot"):
        self.name = "Robot"
        self._end_effector_pos_global = None 
        self._base_pos_global = None
        self.grav = np.array([0, 0, -9.81])
        self._last_tau = None

    def getEEPos(self):
        return self._end_effector_pos_global.copy()
    
    def getBasePos(self):
        return self._base_pos_global.copy()
    
    def getGrav(self):
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
        """Calculate Forward Kinematics of a robot."""        
        pass

    def IK(self, *args, **kwargs):
        """Calculate Inverse Kinematics of a robot."""
        pass

    def randomPos(self):
        """Generate a random configuration through Forward Kinematics."""
        pass

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

    def jacobian(self, *args, **kwargs) -> 'np.ndarray[float]':
        """
        Calculate Space Jacobian for given configuration.

        Returns:
            jacobian
        """
        return np.linalg.pinv(self.inverseJacobian(*args, **kwargs))

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
        Inverse Body Jacobian for given configuration.

        Returns:
            jacobian
        """
        #Why both inverses? One will have to be overridden in any robot configuration
        return np.linalg.pinv(self.jacobianBody(*args, **kwargs))

    def move(self, new_pos : tm) -> None:
        """
        Move to a new position.
        
        Args: 
            new_pos (tm) : new position to move to
        """
        pass

    def draw(self):
        pass
