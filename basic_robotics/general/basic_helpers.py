"""Subcomponent of faser_general, placed here to avoid circular imports."""
import numpy as np
from . import faser_high_performance as mr
from .faser_transform import tm

#TRANSFORMATION MATRIX MANIPULATIONS
def TAAtoTM(taa_format):
    """
    Convert Translation Axis Angle to Transformation Matrix.

    Args:
        taa_format (ndarray): TAA representation of given transformation.
    Returns:
        transformation_matrix: 4x4 transformation matrix representation
    """
    taa_format = taa_format.reshape((6))
    mres = mr.MatrixExp3(mr.VecToso3(taa_format[3:6]))
    #return mr.RpToTrans(mres, transaa[0:3])
    taa_format = taa_format.reshape((6, 1))
    transform = np.vstack((np.hstack((mres, taa_format[0:3])), np.array([0, 0, 0, 1])))
    #print(tm)
    return transform

def TMtoTAA(transformation_matrix):
    """
    Convert a 4x4 transformation matrix to TAA representation.

    Args:
        transformation_matrix: transformation matrix to be converted
    Returns:
        TAA representation
    """
    rotation_matrix, position =  mr.TransToRp(transformation_matrix)
    rotation_array = mr.so3ToVec(mr.MatrixLog3(rotation_matrix))
    return np.vstack((position.reshape((3, 1)), angleMod(rotation_array.reshape((3, 1)))))

#Change of Frames
def localToGlobal(reference, rel):
    """
    Convert a transform in a local frame to the global frame.

    Args:
        reference (temp): Transform of frame A to frame B
        rel (tm): Transform of object 1 in frame B

    Returns:
        tm: Transform of object 1 in frame A

    """
    return tm(mr.LocalToGlobal(reference.gTAA(), rel.gTAA()))

def globalToLocal(reference, rel):
    """
    Convert a transform in a global frame to a local frame.

    Args:
        reference (tm): Transform of frame A to frame B
        rel (tm): Transform of object 1 in frame A
    Returns:
        tm: Transform of object 1 in frame B
    """
    return tm(mr.GlobalToLocal(reference.gTAA(), rel.gTAA()))

#ANGLE HELPERS
def deg2Rad(deg):
    """
    Convert degrees to radians.

    Args:
        deg (float): measure of angles in degrees
    Returns:
        float: measure of angles in radians
    """
    return deg * np.pi / 180

def rad2Deg(rad):
    """
    Convert radians to degrees.

    Args:
        rad (float): measure of angles in radians
    Returns:
        float: measure of angles in degrees
    """
    return rad * 180 / np.pi

def angleMod(rad):
    """
    Cut angles in radians such that they don't exceed 2pi absolute.
    
    Args:
        rad (float): angle or angles
    Returns:
        float: cut down angle or angles
    """
    if isinstance(rad, tm):
        rad.angleMod()
        return rad
    if np.size(rad) == 1:
        if abs(rad) > 2 * np.pi:
            rad = rad % (2 * np.pi)
        return rad
    if np.size(rad) == 6:
        for i in range(3, 6):
            if abs(rad[i]) > 2 * np.pi:
                rad[i] = rad[i] % (2 * np.pi)
        return rad
    for i in range(np.size(rad)):
        if abs(rad[i]) > 2 * np.pi:
            rad[i] = rad[i] % (2 * np.pi)
    return rad