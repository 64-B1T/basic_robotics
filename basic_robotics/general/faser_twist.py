"""Model a Twist, a Screw representing a spatial velocity."""

import numpy as np
from . import faser_high_performance as mr
from .faser_transform import tm
from .faser_screw import Screw

class Twist(Screw):
    """
    Model a Twist.

    A Twist is a Screw that represents an instantaneous spatial velocity (linear and
    angular), and can be converted to and from homogeneous transforms and se(3) matrix
    form.
    """

    def __init__(self, twist_data, frame_applied = None):
        """
        Create a new Twist.

        Args:
            twist_data (np.ndarray[float]): data representing a twist, of shape (6,1).
            frame_applied (tm, optional): If not specified, assumes origin.
        """
        super().__init__(twist_data, frame_applied)

    @classmethod
    def fromTM(cls, twist_transform, frame_applied = None):
        """
        Build a Twist from a homogeneous transform by taking its matrix logarithm.

        Args:
            twist_transform (tm): Homogeneous transform to convert into a twist.
            frame_applied (tm, optional): Frame the resulting twist is expressed in. Defaults to None.

        Returns:
            Twist: New Twist equivalent to the given transform.
        """
        transform_skew = mr.MatrixLog6(twist_transform.TM)
        return cls(mr.se3ToVec(transform_skew), frame_applied)

    def toTM(self):
        """
        Convert this Twist to a homogeneous transform via the matrix exponential.

        Returns:
            tm: Transform obtained by exponentiating this twist's se(3) representation.
        """
        tms = mr.VecTose3(self.flatten())
        tmr = mr.MatrixExp6(tms)
        return tm(tmr)

    def twistMatrix(self):
        """
        Build the 4x4 se(3) matrix representation of this twist.

        Returns:
            np.ndarray[float]: 4x4 matrix with the skew-symmetric angular velocity
            in the upper-left 3x3 block and the linear velocity in the top of the
            last column.
        """
        data = self.data.flatten()
        return np.array([[0, -data[5], data[4], data[0]],
                         [data[5], 0, -data[3], data[1]],
                         [-data[4], data[3], 0, data[2]],
                         [0.0, 0.0, 0.0, 0.0]])

    def toScrew(self):
        """
        Convert this Twist to its equivalent Screw axis representation.

        Handles the pure-rotation case (zero linear component) by returning a Screw
        aligned with the angular velocity axis and infinite pitch, and otherwise
        derives the screw axis point from the normalized twist.

        Returns:
            Screw: Screw axis (angular direction and axis point) equivalent to this twist.
        """
        if (mr.Norm(self.data[0:3])) == 0:
            w = mr.Normalize(self.data[3:6])
            th = mr.Norm(self.data[3:6])[0]
            q = np.array([0, 0, 0]).reshape((3, 1))
            h = np.inf
        else:
            unit_twist = self.data/mr.Norm(self.data[0:3])
            w = unit_twist[0:3].reshape((3))
            v = unit_twist[3:6].reshape((3))
            th = mr.Norm(self.data[0:3])[0]
            q = np.cross(w, v)
            h = v.T @ w
        return Screw(np.hstack((w, q)).reshape((6,1)), self.frame_applied.copy())
        #return (w.reshape((3,1)), th, q.reshape((3,1)), h)

