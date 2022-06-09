"""Implement Transformations."""

from ..modern_robotics_numba import modern_high_performance as mr
import numpy as np
from scipy.spatial.transform import Rotation as R
import traceback

class tm:
    """
    Class to represent and manipulate 3d transformations easily.

    Utilizes Euler Angles and 4x4 Transformation Matrices
    """

    def __init__(self, initializer_array = np.eye((4)), rpy=False):
        """
        Initialize a new Transformation object.

        If no initializer_array is supplied, an identity transform is returned
        initializer_array can be of the following types, with different behaviors
            1) tm: a copy of the tm object is returned
            2) len(3) list: a tm object representing an XYZ rotation is returned
            3) len(3) list with rpy True: a tm representing a ZYX(RPY) rotation is returned
            3) len(6) list: a tm object representing an XYZ translation and XYZ rotation is returned
            4) len(6) list with rpy True: a tm object representing an XYZ translation and ZYX(RPY) rotation is returned
            4) len(7) list: a tm object representing an XYZ translation and a quaternion is returned
            5) 4x4 transformation matrix: a tm object representing the given transform is returned

        Args:
            initializer_array: Optional - data to generate new transformation
        """
        if hasattr(initializer_array, 'TM'):
            #Returns a copy of the tm object
            self.TM = initializer_array.TM.copy()
            self.TAA = initializer_array.TAA.copy()
            return
        init_arr_len = len(initializer_array)
        if isinstance(initializer_array, list):
            #Generates tm from list
            if init_arr_len == 6:
                # Generate rotation and translation from 6dof list
                return self.from6DOF(initializer_array, rpy)
            elif init_arr_len == 7:
                # Generate Rotation and translation from 7dof list (quaternion)
                return self.from7DOF(initializer_array)
            elif init_arr_len == 3:
                # Generate rotation from 3DOF list
                return self.from3DOF(initializer_array, rpy)
        else:
            if init_arr_len == 6:
                # Generate rotation adn translation from 6dof array
                return self.from6DOF(initializer_array.flatten(), rpy)
            elif init_arr_len == 7:
                # Generate rotation and translation from 7dof array (quaternion)
                return self.from7DOF(initializer_array.flatten())
            elif init_arr_len == 3:
                # Generate rotation from 3dof array
                return self.from3DOF(initializer_array.flatten(), rpy)
            elif (len(initializer_array) == 1):
                if isinstance(initializer_array, np.ndarray):
                    if isinstance(initializer_array[0], tm):
                        self.TM = initializer_array[0].TM.copy()
                        self.TMtoTAA()
                        return
            else:
                self.transformSqueezedCopy(initializer_array)
                self.TMtoTAA()
                return

    def from3DOF(self, initializer_array, rpy):
        """
        Initialize from a 3DOF array.

        Args:
            initializer_array (type): Description of parameter `initializer_array`.
            rpy (type): Description of parameter `rpy`.

        Returns:
            type: Description of returned object.
        """
        if rpy == False:
            self.TAA = np.array([0, 0, 0,
                    initializer_array[0],
                    initializer_array[1],
                    initializer_array[2]], dtype=float)
        else:
            temp_init =  tm([0, 0, 0, initializer_array[0], 0, 0])
            temp_init = temp_init @ tm([0, 0, 0, 0, initializer_array[1], 0])
            self.TAA = (temp_init @ tm([0, 0, 0, 0, 0, initializer_array[2]])).gTAA()
        self.TAAtoTM()


    def from6DOF(self, initializer_array, rpy):
        """Short summary.

        Args:
            initializer_array (type): Description of parameter `initializer_array`.
            rpy (type): Description of parameter `rpy`.

        Returns:
            type: Description of returned object.
        """
        if rpy == False:
            self.TAA = np.array([initializer_array[0],
                    initializer_array[1],
                    initializer_array[2],
                    initializer_array[3],
                    initializer_array[4],
                    initializer_array[5]], dtype=float)
        else:
            temp_init =  tm([0, 0, 0, initializer_array[3],0, 0])
            temp_init = temp_init @ tm([0, 0, 0, 0, initializer_array[4], 0])
            temp_init = temp_init @ tm([0, 0, 0, 0, 0, initializer_array[5]])
            #temp_init = temp_init @ tm([0, 0, 0, 0, initializer_array[4], 0])
            #temp_init = temp_init @ tm([0, 0, 0, initializer_array[3],0, 0])
            self.TAA = np.array([
                    initializer_array[0],
                    initializer_array[1],
                    initializer_array[2],
                    temp_init[3],
                    temp_init[4],
                    temp_init[5]], dtype=float)
        self.TAAtoTM()
        return

    def from7DOF(self, initializer_array):
        """Short summary.

        Args:
            initializer_array (type): Description of parameter `initializer_array`.

        Returns:
            type: Description of returned object.

        """
        self.TAA = np.array([initializer_array[0],
                initializer_array[1],
                initializer_array[2],
                0,
                0,
                0], dtype=float) #Why this? Preseve compatibility with lists
        self.TAAtoTM()
        self.setQuat(initializer_array[3:])

    def spawnNew(self, init):
        """
        Spawns a new TM object. Useful when tm is not speifically imported.

        Args:
            init: initialization argument
        Returns:
            tm: tm object
        """
        return tm(init)

    def transformSqueezedCopy(self, transform_matrix):
        """
        In cases of dimension troubles, squeeze out the extra dimension.

        Args:
            TM: input 4x4 matrix
        Returns:
            TM: return 4x4 matrix
        """
        self.TM = np.eye((4), dtype=float)
        for i in range(4):
            for j in range(4):
                self.TM[i, j] = transform_matrix[i, j]
        return self.TM

    def getQuat(self):
        """
        Return quaternion.

        Returns:
            quaternion from euler
        """
        return R.from_matrix(self.TM[0:3, 0:3]).as_quat()

    def setQuat(self, quaternion):
        """
        Set TAA from quaternion.

        Args:
            quat: input quaternion
        """
        self.TM[0:3, 0:3] = R.from_quat(quaternion).as_matrix()
        self.TMtoTAA()

    def angleMod(self):
        """Truncate excessive rotation."""
        refresh = 0
        for i in range(3, 6):
            if abs(self.TAA[i, 0]) > 2 * np.pi:
                refresh = 1
                self.TAA[i, 0] = self.TAA[i, 0] % (np.pi)
        if refresh == 1:
            self.TAAtoTM()

    def tripleUnit(self, lv=1):
        """
        Return XYZ unit vectors based on current position.

        Args:
            lv: length to magnify by
        Returns:
            vec1: vector X
            vec2: vector Y
            vec3: vector Z
        """
        xvec = np.zeros((6, 1))
        yvec = np.zeros((6, 1))
        zvec = np.zeros((6, 1))

        xvec[0:3, 0] = self.TM[0:3, 0]
        yvec[0:3, 0] = self.TM[0:3, 1]
        zvec[0:3, 0] = self.TM[0:3, 2]

        vec1 = tm(self.TAA + xvec*lv)
        vec2 = tm(self.TAA + yvec*lv)
        vec3 = tm(self.TAA + zvec*lv)

        return vec1, vec2, vec3

    def TAAtoTM(self):
        """Convert the TAA representation to TM representation to update the object."""
        self.TAA = self.TAA.reshape((6, 1))
        mres = mr.MatrixExp3(mr.VecToso3(self.TAA[3:6].flatten()))
        self.TM = np.vstack((np.hstack((mres, self.TAA[0:3])), np.array([0, 0, 0, 1])))


    def TMtoTAA(self):
        """Convert the TM representation to TAA representation and updates the object."""
        rotation, transformation =  mr.TransToRp(self.TM)
        rotationAA = mr.so3ToVec(mr.MatrixLog3(rotation))
        self.TAA = np.vstack((transformation.reshape((3, 1)), (rotationAA.reshape((3, 1)))))

    #Modern Robotics Ports
    def adjoint(self):
        """
        Return the adjoint representation of the 4x4 transformation matrix.

        Returns:
            Adjoint
        """
        return mr.Adjoint(self.TM)

    def exp6(self):
        """
        Return the matrix exponential of the TAA.

        Returns:
            Matrix exponential
        """
        return mr.MatrixExp6(mr.VecTose3(self.TAA))

    def gRot(self):
        """
        Return the rotation matrix component of the 4x4 transformation matrix.

        Returns:
            3x3 rotation matrix
        """
        return self.TM[0:3, 0:3].copy()

    def gTAA(self):
        """
        Return the TAA representation of the tm object.

        Returns:
            TAA
        """
        return np.copy(self.TAA)

    def gTM(self):
        """
        Return the 4x4 transformation matrix representation of the tm object.

        Returns:
            4x4 tm
        """
        return np.copy(self.TM)

    def gPos(self):
        """
        Return the len(3) XYZ position of the transforamtion.

        Returns:
            position
        """
        return self.TAA[0:3]

    def sTM(self, TM):
        """
        Set the 4x4 transformation matrix and updates the object.

        Args:
            TM: 4x4 transformation matrix to be set
        """
        self.TM = TM
        self.TMtoTAA()

    def sTAA(self, TAA):
        """
        Set the TAA version of the object and updates.

        Args:
            TAA: New TAA to be set
        """
        self.TAA = TAA
        self.TAAtoTM()
        #self.AngleMod()
    #Regular Transpose
    def T(self):
        """
        Return the transpose version of the tm.

        Returns:
            tm.T
        """
        TM = self.TM.T
        return tm(TM)

    #Conjugate Transpose
    def cT(self):
        """
        Return the conjugate transpose if transpose doesn't work, use this.

        Returns:
            TM.conj().T
        """
        TM = self.TM.conj().T
        return tm(TM)

    def inv(self):
        """
        Return the inverse of the transform.

        Returns:
            tm^-1
        """
        #Regular Inverse
        TM = mr.TransInv(self.TM)
        return tm(TM)

    def pinv(self):
        """
        Return the pseudoinverse of the transform.

        Returns:
            psudo inverse
        """
        #Psuedo Inverse
        TM = np.linalg.pinv(self.TM)
        return tm(TM)

    def copy(self):
        """
        Return a deep copy of the object.

        Returns:
            copy
        """
        copy = tm()
        copy.TM = np.copy(self.TM)
        copy.TAA = np.copy(self.TAA)
        return copy

    def set(self, ind, val):
        """
        Set a specific index of the TAA to a value and then update.

        Args:
            ind: index to set
            val: val to set
        Returns:
            new version of self reference
        """
        self.TAA[ind] = val
        self.TAAtoTM()
        return self

    def approx(self, nplaces=10):
        """
        Round self to 10 decimal places.

        Returns:
            rounded TAA representation
        """
        return np.around(self.TAA, nplaces)

    #FLOOR DIVIDE IS OVERRIDDEN TO PERFORM MATRIX RIGHT DIVISION

    #OVERLOADED FUNCTIONS
    def __getitem__(self, ind):
        """
        Get an indexed slice of the TAA representation.

        Args:
            ind: slice
        Returns:
            Slice
        """
        if isinstance(ind, slice):
            return self.TAA[ind]
        else:
            return self.TAA[ind, 0]

    def __setitem__(self, ind, val):
        """
        Set an indexed slice of the TAA representation and updates.

        Args:
            ind: slice
            val: value(s)
        """
        if isinstance(val, np.ndarray) and val.shape == ((3, 1)):
            self.TAA[ind] = val
        else:
            self.TAA[ind, 0] = val
        self.TAAtoTM()

    def __floordiv__(self, other_object):
        """
        PERFORM RIGHT MATRIX DIVISION IF A IS A MATRIX.

        Otherwise performs elementwise floor division
        Args:
            other_object : object to divide by
        Returns:
            floor (or matrix right division) result
        """
        if isinstance(other_object, tm):
            return tm(np.linalg.lstsq(other_object.gTM().T,
                self.gTM().T, rcond=None)[0].T)
        elif isinstance(other_object, np.ndarray):
            return tm(np.linalg.lstsq(other_object.T, self.gTM().T, rcond=None)[0].T)
        else:
            return tm(self.TAA // other_object)

    def __abs__(self):
        """
        Return absolute valued variant of transform.

        Returns:
            |TAA|
        """
        return tm(abs(self.TAA))

    def __sum__(self):
        """
        Return sum of the values of the TAA representation.

        Returns:
            sum(TAA)
        """
        return sum(self.TAA)

    def __add__(self, other_object):
        """
        Add a value to a TM.

        If it is another TM, adds the two TAA representations together and updates
        If it is an array of 6 values, adds to the TAA and updates
        If it is a scalar, adds the scalar to each element of the TAA and updates
        Args:
            other_object : other value
        Returns:
            this + a
        """
        if isinstance(other_object, tm):
            return tm(self.TAA + other_object.TAA)
        else:
            if isinstance(other_object, np.ndarray):
                if len(other_object) == 6:
                    return tm(self.TAA + other_object.reshape((6,1)))
                else:
                    return self.TAA + other_object
            else:
                return tm(self.TAA + other_object)

    def __sub__(self, other_object):
        """
        Subract a value from a TM.

        If it is another TM, subs a from TAA and updates
        If it is an array of 6 values, subs a from TAA and updates
        If it is a scalar, subs the scalar from each element of the TAA and updates
        Args:
            other_object : other value
        Returns:
            this - a
        """
        if isinstance(other_object, tm):
            return tm(self.TAA - other_object.TAA)
        else:
            if isinstance(other_object, np.ndarray):
                if len(other_object) == 6:
                    return tm(self.TAA - other_object.reshape((6, 1)))
                else:
                    return self.TAA - other_object
            else:
                return tm(self.TAA - other_object)

    def __matmul__(self, other_object):
        """
        Perform matrix multiplication on 4x4 TAA objects.

        Accepts either another tm, or a matrix
        Args:
            other_object : thing to multiply by
        Returns:
            TM * a
        """
        if isinstance(other_object, tm):
            return tm(self.TM @ other_object.TM)
        else:
            if isinstance(other_object, np.ndarray):
                return tm(self.TM @ other_object)

    def __rmatmul__(self, other_object):
        """
        Perform right matrix multiplication on 4x4 TAA objects.

        Accepts either another tm, or a matrix
        Args:
            other_object : thing to multiply by
        Returns:
            a * TM
        """
        if isinstance(other_object, tm):
            return tm(other_object.TM @ self.TM)
        else:
            if isinstance(other_object, np.ndarray):
                return tm(other_object @ self.TM)
            return tm(other_object * self.TAA)

    def __mul__(self, other_object):
        """
        Multiply by a matrix or scalar.

        Args:
            other_object : other value
        Returns:
            TM * a
        """
        if isinstance(other_object, tm):
            return tm(self.TM @ other_object.TM)
        else:
            if isinstance(other_object, np.ndarray):
                return tm(self.TM * other_object)
            return tm(self.TAA * other_object)

    def __rmul__(self, other_object):
        """
        Perform right multiplication by a matrix or scalar.

        Args:
            other_object : thing to multiply by
        Returns:
            a * TM
        """
        if isinstance(other_object, tm):
            return tm(other_object.TM @ self.TM)
        else:
            if isinstance(other_object, np.ndarray):
                return tm(other_object * self.TM)
            return tm(other_object * self.TAA)

    def __truediv__(self, other_object):
        """
        Perform elementwise division across a TAA object.

        Args:
            other_object : value to divide by
        Returns
            tm: new tm with requested division
        """
        #Divide Elementwise from TAA
        return tm(self.TAA / other_object)

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
        return np.allclose(self.TAA,other_object.TAA, rtol=0, atol=1e-8) #Why AllClose?
        # Floats Are Evil
         # RTOL Being Zero means only Absolute Tolerance of 1e-8
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
        if isinstance(other_object, tm):
            if np.all(self.TAA > other_object.TAA):
                return True
        else:
            if np.all(self.TAA > other_object):
                return True
        return False

    def __lt__(self, other_object):
        """
        Check for less than another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if isinstance(other_object, tm):
            if np.all(self.TAA < other_object.TAA):
                return True
        else:
            if np.all(self.TAA < other_object):
                return True
        return False

    def __le__(self, other_object):
        """
        Check for less than or equa to another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if self.__lt__(other_object) or self.__eq__(other_object):
            return True
        return False

    def __ge__(self, other_object):
        """
        Check for greater than or equal to another tm.

        Args:
            other_object : Another object
        Returns:
            boolean
        """
        if self.__gt__(other_object) or self.__eq__(other_object):
            return True
        return False

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
        return ("[ " + fst % (self.TAA[0, 0]) + ", "+ fst % (self.TAA[1, 0]) +
         ", "+ fst % (self.TAA[2, 0]) + ", "+ fst % (self.TAA[3, 0]) +
         ", "+ fst % (self.TAA[4, 0]) + ", "+ fst % (self.TAA[5, 0])+ " ]")

