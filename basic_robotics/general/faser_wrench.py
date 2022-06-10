"""Model a wrench."""

import numpy as np 
from .faser_transform import tm
from .basic_helpers import globalToLocal
from .faser_screw import Screw

class Wrench(Screw):
    """
    Model a wrench.

    Returns:
        Wrench: the wrench to be modelled.
    """

    def __init__(self, force : 'np.ndarray[float]',
            position_applied : tm = None, frame_applied : tm = None):
        """Create a new Wrench.

        Args:
            force (np.ndarray[float]): force vector (could be gravity * mass), units are N
            position_applied (tm, optional): If not specified,
                    assumes acting on origin in frame applied
            frame_applied (tm, optional): If not specified, assumes origin.
        """  
        self.position_applied = position_applied
        if position_applied is None:
            self.position_applied = tm()
        if isinstance(force, Screw):
            super().__init__(force.data, force.frame_applied)
        elif len(force) == 3:
            t_wren = np.cross(self.position_applied[0:3].reshape((3)), force)
            # Calculate moment based on position and action
            super().__init__(np.array(
                    [t_wren[0], t_wren[1], t_wren[2],
                    force[0], force[1], force[2]]).reshape((6, 1)), frame_applied)
                    # Create Complete Wrench
        elif len(force.shape) == 2:
            super().__init__(force, frame_applied)
        else:
            super().__init__(force.reshape((6,1)), frame_applied)

    def getMoment(self) -> 'np.ndarray[float]':
        """Get the moment component of the wrench.

        Returns:
            np.ndarray[float]: moment of the wrench.
        """    
        return self.data[0:3].copy()

    def getForce(self) -> 'np.ndarray[float]':
        """Get the force component of the wrench.

        Returns:
            np.ndarray[float]: force component
        """ 
        return self.data[3:6].copy()

    def changeFrame(self, new_frame : tm, old_frame : tm = None):
        """Change a wrench from frame A to frame B.

        Args:
            new_frame (tm): New base frame of reference.
            old_frame (tm, optional): Old base frame of reference.

        MR 3.97 Fb = [Ad(T(a->b))]^T(Fa)

        Returns:
            Wrench: reference to self, for compatibility and code-golfing reasons.
        """     
        if old_frame is None:
            old_frame = self.frame_applied
        if old_frame == new_frame:
            return self
        self.frame_applied = new_frame
        frame_transition = globalToLocal(old_frame, new_frame)
        self.data = frame_transition.adjoint().T @ self.data
        return self # Compatibility

    def copy(self):
        """Get a copy of this Wrench.

        Returns:
            Wrench: Wrench copy
        """        
        new_wrench = Wrench(self.data.copy(), self.position_applied.copy(), self.frame_applied.copy())
        return new_wrench

    def _wrenchConverter(self, possible_screw):
        """
        If a value is a Screw, convert it to a Wrench.

        This function exists in order to reduce code duplication, letting Screw superclass 
        Do most of the heavy lifting.
        """
        if isinstance(possible_screw, Screw):
            return Wrench(possible_screw, self.frame_applied) # It was a screw
        return possible_screw # Not a screw

    def __abs__(self):
        """Inherited from Super."""
        return Wrench(super().__abs__())

    def __add__(self, other_object):
        """Inherited from Super."""
        return self._wrenchConverter(super().__add__(other_object))

    def __sub__(self, other_object):
        """Inherited from Super."""
        return self._wrenchConverter(super().__sub__(other_object))

    def __mul__(self, other_object):
        """Inherited from Super."""
        return self._wrenchConverter(super().__mul__(other_object))

    def __rmul__(self, other_object):
        """Inherited from Super."""
        return self._wrenchConverter(super().__rmul__(other_object))

    def __truediv__(self, other_object):
        """Inherited from Super."""
        return self._wrenchConverter(super().__truediv__(other_object))

    def __eq__(self, other_object):
        """Inherited from Super."""
        if not isinstance(other_object, Wrench):
            return False
        return super().__eq__(other_object)
