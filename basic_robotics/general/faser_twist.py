import numpy as np
from . import faser_high_performance as mr
from .faser_transform import tm
from .faser_screw import Screw

class Twist(Screw):

    def __init__(self, twist_data, frame_applied = None):
        super().__init__(twist_data, frame_applied)
    
    @classmethod
    def fromTM(self, twist_transform, frame_applied = None):
        transform_skew = mr.MatrixLog6(twist_transform.TM)
        super().__init__(mr.se3ToVec(transform_skew), frame_applied)

    def toTM(self):
        tms = mr.VecTose3(self.flatten())
        tmr = mr.MatrixExp6(tms)
        return tm(tmr)

    def toScrew(self):
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

