"""Model a Screw."""

import numpy as np 
from .faser_transform import tm
from .basic_helpers import globalToLocal
from . import faser_high_performance as mr

class Screw:
    """
    Model a Screw.

    Emulates a Screw based on content in Modern Robotics Chapter 3.

    Returns:
        Screw : Screw object inheriting np ndarray functionality
    """

    __array_ufunc__ = None

    def __init__(self, data : 'np.ndarray[float]' = np.zeros((6,1)), frame_applied : tm = None):
        """
        Create a new Screw.

        Emulates a Screw based on content in Modern Robotics Chapter 3.

        Args:
            data (np.ndarray[Float]) : data representing a screw, of shape ((6,1))
            frame_applied (tm, optional): If not specified, assumes origin.
        """
        if not data.shape == ((6,1)):
            self.data = data.reshape((6,1))
        else:
            self.data = data
        self.frame_applied = frame_applied
        if self.frame_applied is None:
            self.frame_applied = tm()
        self.shape = ((6,1)) # For numpy compatibility

    def copy(self) -> 'Screw':
        """
        Get a copy of this Screw.

        Returns:
            Copy (Screw): Screw copy
        """        
        new_array_base = Screw(self.data.copy(), self.frame_applied.copy())
        return new_array_base

    def flatten(self) -> 'np.ndarray[float]':
        """
        Get a flattened representation of the screw.

        Returns:
            flat_screw (np.ndarray[float]): flattened screw.
        """
        return self.data.copy().flatten()

    
    def _setFrame(self, new_frame):
        """
        Set a frame without changing it, called as a private method for Screw and subclasses.

        Args:
            new_frame (tm): New frame to use
        """
        self.frame_applied = new_frame.copy()

    def changeFrame(self, new_frame : tm, old_frame : tm = None) -> 'Screw':
        """
        Change a screw from one frame to another.

        MR3.85 Sb = [Ad(T(b->a))](Sa)

        Args:
            new_frame (tm): New base frame of reference.
            old_frame (tm, optional): Old base frame of reference.

        Returns:
            screw : reference to self, for compatibility and code-golfing reasons.
        """
        if old_frame is None:
            old_frame = self.frame_applied
        if old_frame == new_frame:
            return self
        self._setFrame(new_frame)
        frame_transition = globalToLocal(new_frame, old_frame)
        self.data = frame_transition.adjoint() @ self.data
        return self # Compatibility

    def getData(self) -> 'np.ndarray[float]':
        """
        Get the data of this Screw for raw computation.

        Returns:
            data (np.ndarray[float]) : raw data of the Screw
        """
        return self.data.copy()

    def getPitch(self) -> 'float':
        """
        Get the pitch of a Screw.
        
        Equal to magnitude of angular component / magnitude of linear component

        Returns:
            pitch (float) : Screw pitch.
        """
        return mr.Norm(self.data[0:3,0])/mr.Norm(self.data[3:6,0])

    def cross(self, other_screw : 'Screw') -> 'Screw':
        """
        Return cross product of this screw and another screw.

        Returns (Screw): New crossed screw
        """
        if not self.frame_applied == other_screw.frame_applied:
            other_screw = other_screw.copy().changeFrame(self.frame_applied)
        nt = np.cross(self.data[0:3,0], other_screw.data[0:3,0])
        nb = (np.cross(self.data[0:3,0], other_screw.data[3:6,0]) + 
                np.cross(self.data[3:6,0], other_screw.data[0:3,0]))
        return Screw(np.hstack((nt, nb)).reshape((6,1)), self.frame_applied.copy())

    def dot(self, other_screw : 'Screw') -> 'Screw':
        """
        Return dot product of this screw and another screw.

        Returns (Screw): New crossed screw
        """
        if not self.frame_applied == other_screw.frame_applied:
            other_screw = other_screw.copy().changeFrame(self.frame_applied)
        nt = np.dot(self.data[0:3,0], other_screw.data[0:3,0])
        nb = (np.dot(self.data[0:3,0], other_screw.data[3:6,0]) + 
                np.dot(self.data[3:6,0], other_screw.data[0:3,0]))
        return np.hstack((nt, nb))

    def dualScalarMultiply(self, dual_scalar : list) -> 'Screw':
        """
        Perform multiplication of screw by a dual scalar.
        
        Returns (Screw) : New scaled screw
        """
        nt = dual_scalar[0] * self.data[0:3]
        nb = dual_scalar[1] * self.data[0:3] + dual_scalar[0] * self.data[3:6]
        return Screw(np.vstack((nt, nb)), self.frame_applied.copy())

    def reshape(self, new_shape) -> 'np.ndarray[float]':
        """
        Reshape the data field for legacy code support.

        Args:
            new_shape ((int, int)): New shape

        Returns:
            np.ndarray[float]: Data reshaped according to user.
        """
        return self.data.copy().reshape(new_shape)

    def __sum__(self):
        """
        Return sum of the values of the screw.

        Returns:
            sum_screw (float) : sum of the screw values
        """
        return sum(self.data.flatten())

    def __array__(self) -> 'np.ndarray[float]':
        """
        Return the internal data array, useful for declaring larger arrays.

        Returns:
            np.ndarray[float]: Internal data of the screw
        """        
        return self.data

    def __getitem__(self, ind : int) -> 'np.ndarray[float]':
        """
        Get an indexed slice of the Screw.

        Args:
            ind: slice
        Returns:
            Screw slice
        """
        if isinstance(ind, slice):
            return self.data[ind]
        else:
            return self.data[ind, 0]

    def __setitem__(self, ind : int, val : float) -> None:
        """
        Set an indexed slice of the Screw representation.

        Args:
            ind: slice
            val: value(s)
        """
        if isinstance(val, np.ndarray) and val.shape == ((3, 1)):
            self.data[ind] = val
        else:
            self.data[ind, 0] = val

    def __abs__(self) -> 'Screw':
        """
        Return absolute valued variant of Screw.

        Returns:
            |Screw|
        """
        new_base = self.copy()
        new_base.data = abs(new_base.data)
        return new_base
    

    def __add__(self, other_object):
        """
        Add a value to the Screw.

        In the case of adding two Screw objects together, the frame of the left most Screw takes
        precedence. For example A in frame 1 + B in frame 2 = C in frame 1
        If adding a 6x1 array, it will return a new Screw object, with the array added
        to the original Screw.

        Args:
            other_object : object to add to the Screw
        Returns:
            this + other_object
        """
        if isinstance(other_object, Screw):
            if other_object.frame_applied == self.frame_applied:
                return Screw(self.data + other_object.data, self.frame_applied)
            else:
                local_frame_other = other_object.copy().changeFrame(self.frame_applied)
                return Screw(self.data + local_frame_other.data, self.frame_applied.copy())
        if isinstance(other_object, np.ndarray) and len(other_object) == 6:
            return Screw(self.data + other_object.reshape((6,1)), self.frame_applied.copy())
        return self.data + other_object

    def __radd__(self, other_object):
        """
        Add value on left hand side.
        
        Args: 
            other_object : object to add on left hand side
        Returns:
            new_object : result of addition, either a screw or matrix.
        """
        return self.__add__(other_object)

    def __sub__(self, other_object):
        """
        Subract a value from the Screw.

        In the case of subradcting two Screw objects together, the frame of the left most Screw
        takes precedence. For example A in frame 1 - B in frame 2 = C in frame 1
        If adding a 6x1 array, it will return a new Screw object, with the array subracted
        to the original Screw.

        Args:
            other_object : object to subract from the Screw
        Returns:
            this - other_object
        """
        if isinstance(other_object, Screw):
            if other_object.frame_applied == self.frame_applied:
                return Screw(self.data- other_object.data, self.frame_applied.copy())
            else:
                local_frame_other = other_object.copy().changeFrame(self.frame_applied)
                return Screw(self.data- local_frame_other.data, self.frame_applied.copy())
        if isinstance(other_object, np.ndarray) and len(other_object) == 6:
            return Screw(self.data - other_object.reshape((6,1)), self.frame_applied.copy())
        return self.data + other_object

    def __rsub__(self, other_object):
        """
        Subract a value on left hand side.
        
        Args: 
            other_object : object to subract on left hand side.
        Returns:
            new_obj
        """
        if isinstance(other_object, Screw):
            return Screw(other_object.data - self.data, self.frame_applied.copy())
        if isinstance(other_object, np.ndarray) and len(other_object) == 6:
            return Screw(other_object.reshape((6,1)) - self.data, self.frame_applied.copy())
        return self.data + other_object

    def __matmul__(self, other_object):
        """
        Perform matrix multiplication on the Screw component.

        Accepts anything compatible with matrix multiplication. 
        If the second argument is a Screw, performs a screw dot product.

        Args:
            other_object : thing to multiply by
        Returns:
            TM @ a
        """
        if isinstance(other_object, Screw):
            return self.dot(other_object)
        return self.data @ other_object

    def __rmatmul__(self, other_object):
        """
        Perform right matrix multiplication on the Screw component.

        Accepts anything compatibile with matrix multiplication
        Args:
            other_object : thing to multiply by
        Returns:
            a @ TM
        """
        return other_object @ self.data

    def __mul__(self, other_object):
        """
        Multiply Screw * x.

        If x is a number, performs scalar multiplcation on all values.
        If x is a Screw, perform screw cross product and yield a new screw.
        If x is a dual scalar, perform dual scalar multiplication.
        If x is something else, multiply W by x and hope for the best.

        Args:
            other_object : other value
        Returns:
            Screw * a
        """
        if isinstance(other_object, int) or isinstance(other_object, float):
            return Screw(self.data * other_object, self.frame_applied.copy())
        if isinstance(other_object, Screw):
            return self.cross(other_object)
        if isinstance(other_object, (np.ndarray, list)) and len(other_object) == 2:
            return self.dualScalarMultiply(other_object)
        return self.data * other_object

    def __rmul__(self, other_object):
        """
        Perform right multiplication by a matrix or scalar.

        Args:
            other_object : thing to multiply by
        Returns:
            a * TM
        """
        if isinstance(other_object, int) or isinstance(other_object, float):
            return Screw(other_object * self.data, self.frame_applied.copy())
        return other_object * self.data

    def __truediv__(self, other_object):
        """
        Perform elementwise division across a TAA object.

        Args:
            other_object : value to divide by
        Returns
            tm: new tm with requested division
        """
        #Divide Elementwise from TAA
        if isinstance(other_object, (int, float)):
            return Screw(self.data / other_object, self.frame_applied.copy())
        return self.data / other_object

    def __rtruediv__(self, other_object, fill=0):
        """
        Perform elementwise division on the right side.

        Args:
            other_object : value to divide by
        Returns
            t
        """
        if isinstance(other_object, (int, float)):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = other_object / self.data 
                if np.isscalar( result ):
                    return Screw(result, self.frame_applied.copy())
                else:
                    result[ ~ np.isfinite( result )] = fill
                    return Screw(result, self.frame_applied.copy())
        return other_object / self.data

    
    def __floordiv__(self, other_object) :
        """
        Perform elementwise floor division.
        
        Args:
            other_object : object to divide by
        Returns:
            floor (or matrix right division) result
        """
        if isinstance(other_object, (int, float)):
            return Screw(self.data // other_object, self.frame_applied.copy())
        return self.data // other_object
    
    def __rfloordiv__(self, other_object): 
        """
        Perform elementwise floor division on right side.

        Args:
            other_object : object to divide by
        Returns:
            floor (or matrix right division) result
        """
        if isinstance(other_object, (int, float)):
            return Screw(other_object // self.data, self.frame_applied.copy())
        return self.other_object // self.data

    def __eq__(self, other_object):
        """
        Check for equality with another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if not isinstance(other_object, Screw):
            return False
        if not self.frame_applied == other_object.frame_applied:
            return False
        if (np.allclose(self.data,other_object.data, rtol=0, atol=1e-8)):
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
        if isinstance(other_object, Screw):
            return self.data > other_object.data
        return self.data > other_object

    def __lt__(self, other_object):
        """
        Check for less than another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, Screw):
            return self.data < other_object.data
        return self.data < other_object

    def __le__(self, other_object):
        """
        Check for less than or equa to another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, Screw):
            return self.data <= other_object.data
        return self.data <= other_object

    def __ge__(self, other_object):
        """
        Check for greater than or equal to another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, Screw):
            return self.data >= other_object.data
        return self.data >= other_object

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
        return ("[ " + fst % (self.data[0, 0]) + ", "+ fst % (self.data[1, 0]) +
         ", "+ fst % (self.data[2, 0]) + ", "+ fst % (self.data[3, 0]) +
         ", "+ fst % (self.data[4, 0]) + ", "+ fst % (self.data[5, 0])+ " ]")
