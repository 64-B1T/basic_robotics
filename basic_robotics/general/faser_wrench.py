"""Model a wrench."""

import numpy as np 
from .faser_transform import tm
from .basic_helpers import globalToLocal

class Wrench:
    """
    Model a wrench.

    Returns:
        Wrench: the wrench to be modelled.
    """

    __array_ufunc__ = None
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
        self.frame_applied = frame_applied
        if position_applied is None:
            self.position_applied = tm()
        if self.frame_applied is None:
            self.frame_applied = tm()
        if len(force) == 3:
            t_wren = np.cross(self.position_applied[0:3].reshape((3)), force)
            # Calculate moment based on position and action
            self.wrench_arr = np.array(
                    [t_wren[0], t_wren[1], t_wren[2],
                    force[0], force[1], force[2]]).reshape((6, 1))
                    # Create Complete Wrench
        elif len(force.shape) == 2:
            self.wrench_arr = force
        else:
            self.wrench_arr = force.reshape((6,1))

    def getMoment(self) -> 'np.ndarray[float]':
        """Get the moment component of the wrench.

        Returns:
            np.ndarray[float]: moment of the wrench.
        """    
        return self.wrench_arr[0:3].copy()

    def getForce(self) -> 'np.ndarray[float]':
        """Get the force component of the wrench.

        Returns:
            np.ndarray[float]: force component
        """ 
        return self.wrench_arr[3:6].copy()

    def flatten(self) -> 'np.ndarray[float]':
        """Get a flattened representation of the wrench.

        Returns:
            np.ndarray[float]: flattened wrench.
        """
        return self.wrench_arr.copy().flatten()

    def changeFrame(self, new_frame : tm, old_frame : tm = None):
        """Change a wrench from frame A to frame B.

        Args:
            new_frame (tm): New base frame of reference.
            old_frame (tm, optional): Old base frame of reference.

        Returns:
            Wrench: reference to self, for compatibility and code-golfing reasons.
        """        
        if old_frame is None:
            old_frame = self.frame_applied
        elif old_frame == new_frame:
            return self
        new_frame = globalToLocal(old_frame, new_frame)
        self.frame_applied = new_frame
        self.wrench_arr = new_frame.adjoint().T @ self.wrench_arr
        return self # Compatibility

    def copy(self):
        """Get a copy of this Wrench.

        Returns:
            Wrench: Wrench copy
        """        
        new_wrench = Wrench(self.wrench_arr.copy(), self.position_applied.copy(), self.frame_applied.copy())
        return new_wrench


    def __getitem__(self, ind : int) -> 'np.ndarray[float]':
        """
        Get an indexed slice of the wrench.

        Args:
            ind: slice
        Returns:
            wrench slice
        """
        if isinstance(ind, slice):
            return self.wrench_arr[ind]
        else:
            return self.wrench_arr[ind, 0]

    def __setitem__(self, ind, val):
        """
        Set an indexed slice of the wrench representation.

        Args:
            ind: slice
            val: value(s)
        """
        if isinstance(val, np.ndarray) and val.shape == ((3, 1)):
            self.wrench_arr[ind] = val
        else:
            self.wrench_arr[ind, 0] = val


    def __floordiv__(self, other_object):
        """
        PERFORM RIGHT MATRIX DIVISION IF A IS A MATRIX.

        Otherwise performs elementwise floor division
        Args:
            other_object : object to divide by
        Returns:
            floor (or matrix right division) result
        """
        return self.wrench_arr // other_object

    def __abs__(self):
        """
        Return absolute valued variant of wrench.

        Returns:
            |wrench|
        """
        new_wrench = self.copy()
        new_wrench.wrench_arr = abs(new_wrench.wrench_arr)
        return new_wrench

    def __sum__(self):
        """
        Return sum of the values of the TAA representation.

        Returns:
            sum(TAA)
        """
        return sum(self.wrench_arr.flatten())

    def __add__(self, other_object):
        """
        Add a value to the wrench.

        In the case of adding two Wrench objects together, the frame of the left most wrench takes
        precedence. For example A in frame 1 + B in frame 2 = C in frame 1
        If adding a 6x1 array, it will return a new Wrench object, with the array added
        to the original wrench.

        Args:
            other_object : object to add to the wrench
        Returns:
            this + other_object
        """
        if isinstance(other_object, Wrench):
            if other_object.frame_applied == self.frame_applied:
                return Wrench(self.wrench_arr + other_object.wrench_arr,
                        self.position_applied, self.frame_applied)
            else:
                local_frame_other = other_object.copy().changeFrame(self.frame_applied)
                return Wrench(self.wrench_arr + local_frame_other.wrench_arr,
                        self.position_applied, self.frame_applied)
        else:
            if isinstance(other_object, np.ndarray) and len(other_object) == 6:
                return Wrench(self.wrench_arr + other_object.reshape((6,1)),
                        self.position_applied, self.frame_applied)
        return self.wrench_arr + other_object

    def __sub__(self, other_object):
        """
        Subract a value from the wrench.

        In the case of subradcting two Wrench objects together, the frame of the left most wrench
        takes precedence. For example A in frame 1 - B in frame 2 = C in frame 1
        If adding a 6x1 array, it will return a new Wrench object, with the array subracted
        to the original wrench.

        Args:
            other_object : object to subract from the wrench
        Returns:
            this - other_object
        """
        if isinstance(other_object, Wrench):
            if other_object.frame_applied == self.frame_applied:
                return Wrench(self.wrench_arr - other_object.wrench_arr,
                        self.position_applied, self.frame_applied)
            else:
                local_frame_other = other_object.copy().changeFrame(self.frame_applied)
                return Wrench(self.wrench_arr - local_frame_other.wrench_arr,
                        self.position_applied, self.frame_applied)
        else:
            if isinstance(other_object, np.ndarray) and len(other_object) == 6:
                return Wrench(self.wrench_arr - other_object.reshape((6,1)),
                        self.position_applied, self.frame_applied)
        return self.wrench_arr - other_object

    def __matmul__(self, other_object):
        """
        Perform matrix multiplication on the wrench component.

        Accepts anything compatibile with matrix multiplication
        Args:
            other_object : thing to multiply by
        Returns:
            TM @ a
        """
        return self.wrench_arr @ other_object

    def __rmatmul__(self, other_object):
        """
        Perform right matrix multiplication on the wrench component.

        Accepts anything compatibile with matrix multiplication
        Args:
            other_object : thing to multiply by
        Returns:
            a @ TM
        """
        return other_object @ self.wrench_arr

    def __mul__(self, other_object):
        """
        Multiplie by a matrix or scalar.

        Args:
            other_object : other value
        Returns:
            TM * a
        """
        if isinstance(other_object, int) or isinstance(other_object, float):
            return Wrench(self.wrench_arr * other_object, self.position_applied, self.frame_applied)
        return self.wrench_arr * other_object

    def __rmul__(self, other_object):
        """
        Perform right multiplication by a matrix or scalar.

        Args:
            other_object : thing to multiply by
        Returns:
            a * TM
        """
        if isinstance(other_object, int) or isinstance(other_object, float):
            return Wrench(other_object * self.wrench_arr, self.position_applied, self.frame_applied)
        return other_object * self.wrench_arr

    def __truediv__(self, other_object):
        """
        Perform elementwise division across a TAA object.

        Args:
            other_object : value to divide by
        Returns
            tm: new tm with requested division
        """
        #Divide Elementwise from TAA
        if isinstance(other_object, int) or isinstance(other_object, float):
            return Wrench(self.wrench_arr / other_object, self.position_applied, self.frame_applied)
        return self.wrench_arr / other_object

    def __eq__(self, other_object):
        """
        Check for equality with another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if not isinstance(other_object, tm):
            return False
        if (np.allclose(self.wrench_arr,other_object.wrench_arr, rtol=0, atol=1e-8) and
                self.frame_applied == other_object.frame_applied):
            return True # Floats Are Evil
        return False # RTOL Being Zero means only Absolute Tolerance of 1e-8
        # So at an RTOL 1e-8, a maximum m deviation of 10 nanometers
        # So at an RTOL 1e-9, a maximum r deviation of 5.7e-7 degrees

    def __gt__(self, other_object):
        """
        Check for greater than another tm.
        
        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, Wrench):
            return self.wrench_arr > other_object.wrench_arr
        return self.wrench_arr > other_object

    def __lt__(self, other_object):
        """
        Check for less than another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, Wrench):
            return self.wrench_arr < other_object.wrench_arr
        return self.wrench_arr < other_object

    def __le__(self, other_object):
        """
        Check for less than or equa to another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, Wrench):
            return self.wrench_arr <= other_object.wrench_arr
        return self.wrench_arr <= other_object

    def __ge__(self, other_object):
        """
        Check for greater than or equal to another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, Wrench):
            return self.wrench_arr >= other_object.wrench_arr
        return self.wrench_arr >= other_object

    def __ne__(self, other_object):
        """
        Check for not equality with another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        return not self.__eq__(other_object)

    def __str__(self, dlen=6):
        """
        Create a string from a tm object.

        Returns:
            String: representation of transform
        """
        fst = '%.' + str(dlen) + 'f'
        return ("[ " + fst % (self.wrench_arr[0, 0]) + ", "+ fst % (self.wrench_arr[1, 0]) +
         ", "+ fst % (self.wrench_arr[2, 0]) + ", "+ fst % (self.wrench_arr[3, 0]) +
         ", "+ fst % (self.wrench_arr[4, 0]) + ", "+ fst % (self.wrench_arr[5, 0])+ " ]")
