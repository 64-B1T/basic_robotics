import math
from . import faser_high_performance as mr
import numpy as np
import scipy as sci
import scipy.linalg as ling
from .faser_transform import tm

#TRANSFORMATION MATRIX MANIPULATIONS
def TAAtoTM(taa_format):
    """
    Converts Translation Axis Angle to Transformation Matrix
    Args:
        taa_format (ndarray): TAA representation of given transformation.
    Returns:
        transformation_matrix: 4x4 transformation matrix representation
    """
    taa_format = taa_format.reshape((6))
    mres = mr.MatrixExp3(mr.VecToso3(taa_format[3:6]))
    #return mr.RpToTrans(mres, transaa[0:3])
    taa_format = taa_format.reshape((6, 1))
    tm = np.vstack((np.hstack((mres, taa_format[0:3])), np.array([0, 0, 0, 1])))
    #print(tm)
    return tm

def TMtoTAA(transformation_matrix):
    """
    Converts a 4x4 transformation matrix to TAA representation
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
    Converts a transform in a local frame to the global frame

    Args:
        reference (temp): Transform of frame A to frame B
        rel (tm): Transform of object 1 in frame B

    Returns:
        tm: Transform of object 1 in frame A

    """
    return tm(mr.LocalToGlobal(reference.gTAA(), rel.gTAA()))

def globalToLocal(reference, rel):
    """
    Convert a transform in a global frame to a local frame

    Args:
        reference (tm): Transform of frame A to frame B
        rel (tm): Transform of object 1 in frame A
    Returns:
        tm: Transform of object 1 in frame B
    """
    return tm(mr.GlobalToLocal(reference.gTAA(), rel.gTAA()))

#Transformation Matrix Group Functions
def planeFromThreePoints(ref_point_1, ref_point_2, ref_point_3):
    """
    Creates the equation of a plane from three points
    Args:
        ref_point_1: tm or vector for point 1
        ref_point_2: tm or vector for point 2
        ref_point_3: tm or vector for point 3
    Returns:
        a, b, c, d: equation cooficients of a plane
    """
    p1 = np.array(ref_point_1[0:3]).flatten()
    p2 = np.array(ref_point_2[0:3]).flatten()
    p3 = np.array(ref_point_3[0:3]).flatten()

    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane

    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d

def planePointsFromTransform(ref_point_1):
    """
    Create plane TM points from one Transform (using unit vectors)
    Args:
        ref_point_1: transform to place plane on
    Returns:
        a, b, c: Plane basis points
    """
    a, b, _ = ref_point_1.tripleUnit()
    return ref_point_1, a, b

def mirror(origin, mirror_point):
    """
    Mirrors a point about a plane
    Args:
        origin: mirror plane (XY local to that tm)
        mirror_point: tm describing point to be mirrored over plane
    Returns:
        mirrored Point
    """
    t1, t2, t3 = planePointsFromTransform(origin)
    a, b, c, d = planeFromThreePoints(t1, t2, t3)
    x1 = mirror_point[0]
    y1 = mirror_point[1]
    z1 = mirror_point[2]
    k = (-a * x1 - b * y1 - c * z1 - d) / float((a * a + b * b + c * c))
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    x3 = 2 * x2-x1
    y3 = 2 * y2-y1
    z3 = 2 * z2-z1
    return tm([x3, y3, z3, 0, 0, 0])

def adjustRotationToMidpoint(active_point, ref_point_1, ref_point_2, mode = 0):
    """
    Applies the midpoint transform of reference points 1 and 2 to an active point

    Args:
        active_point (tm): Point to be modified
        ref_point_1 (tm): origin point for midpoint calculation
        ref_point_2 (tm): goal point for midpoint calculation
        mode (int): Mode of midpoint calculation. 0, TMMIDPOINT. 1, ROTFROMVEC

    Returns:
        tm: Modified version of active_point with orientation of vector 1 -> 2

    """
    modified_point = active_point.copy()
    if mode != 1:
        t_mid = tmInterpMidpoint(ref_point_1, ref_point_2)
        modified_point[3:6] = t_mid[3:6]
    else:
        modified_point[3:6] = rotationFromVector(ref_point_1, ref_point_2)[3:6]
    return modified_point

def tmAvgMidpoint(ref_point_1, ref_point_2):
    """
    Simplest version of a midpoint calculation. Simply the average of two positions

    Args:
        ref_point_1 (tm): position 1
        ref_point_2 (tm): position 2
    Returns:
        tm: midpoint average of positions 1 and 2
    """
    return (ref_point_1 + ref_point_2)/2

def tmInterpMidpoint(ref_point_1, ref_point_2):
    """
    Better version of midpoint calculation

    Position is stil average of positions 1 and 2
    but rotation is calculated as a proper interpolation

    Args:
        ref_point_1 (tm): position 1
        ref_point_2 (tm): position 2
    Returns:
        tm: midpoint of positions 1 and 2
    """
    taar = np.zeros((6, 1))
    taar[0:3] = (ref_point_1[0:3] + ref_point_2[0:3])/2
    R1 = mr.MatrixExp3(mr.VecToso3(ref_point_1[3:6].reshape((3))))
    R2 = mr.MatrixExp3(mr.VecToso3(ref_point_2[3:6].reshape((3))))
    Re = (R1 @ (R2.conj().T)).conj().T
    Re2 = mr.MatrixExp3(mr.VecToso3(mr.so3ToVec(mr.MatrixLog3((Re)/2))))
    rmid = Re2 @ R1
    taar[3:6] = mr.so3ToVec(mr.MatrixLog3((rmid))).reshape((3, 1))
    return tm(taar)

def getSurfaceNormal(tri, object_center = None):
    u = tri[1] - tri[0]
    v = tri[2] - tri[0]
    x = u[1] * v[2] - u[2] * v[1]
    y = u[2] * v[0] - u[0] * v[2]
    z = u[0] * v[1] - u[1] * v[0]
    center = tm([(tri[0][0] + tri[1][0] + tri[2][0])/3,
        (tri[0][1] + tri[1][1] + tri[2][1])/3,
        (tri[0][2] + tri[1][2] + tri[2][2])/3, 0, 0, 0])
    nvec = np.array([x, y, z])
    unit_vec = nvec/np.linalg.norm(nvec)
    unit_out = tm([unit_vec[0], unit_vec[1], unit_vec[2], 0, 0, 0])

    if object_center is not None:
        d1 = distance(center @ unit_out, object_center)
        d2 = distance(center @ (-1 * (unit_out.copy())), object_center)
        if d2 > d1:
            unit_out = -1 * unit_out
    return center, unit_out

#Rotations/Viewers
def rotationFromVector(ref_point_1, ref_point_2):
    """
    Reorients ref_point_1 such that its z axis is pointing towards ref_point_2

    Args:
        ref_point_1 (tm): position 1
        ref_point_2 (tm): position 2
    Returns:
        tm: ref_point_1 where z points to position 2
    """
    d = distance(ref_point_1, ref_point_2)
    res = lambda x : distance(
        tm([ref_point_1[0], ref_point_1[1], ref_point_1[2], x[0], x[1], ref_point_1[5]]) @ tm([0, 0, d, 0, 0, 0]),
        ref_point_2)
    x0 = np.array([ref_point_1[3], ref_point_1[4]])
    xs = sci.optimize.fmin(res, x0, disp=False)
    ref_point_1[3] = xs[0]
    ref_point_1[4] = xs[1]
    return ref_point_1

def lookAt(ref_point_1, ref_point_2):
    """
    Alternate version of RotFromVec, however rotation *about* z axis may be more random
    Does not depend on sci optimize fmin

    Args:
        ref_point_1 (tm): position 1
        ref_point_2 (tm): position 2
    Returns:
        tm: point at ref_point_1 where z points to position 2
    """
    upa = (ref_point_1 @ tm([-1, 0, 0, 0, 0, 0]))
    up = upa[0:3].flatten()
    va = ref_point_1[0:3].flatten()
    vb = ref_point_2[0:3].flatten()
    zax = mr.Normalize(vb-va)
    xax = mr.Normalize(np.cross(up, zax))
    yax = np.cross(zax, xax)
    R2 = np.eye(4)
    R2[0:3, 0:3] = np.array([xax, yax, zax]).T
    R2[0:3, 3] = va
    ttm = tm(R2)
    return ttm

    #Error and Distance Functions
def poseError(ref_point_1, ref_point_2):
    """
    Provides absolute error between two transformations

    Args:
        ref_point_1 (tm): Reference point 1
        ref_point_2 (tm): Reference point 2

    Returns:
        tm: absolute error

    """
    return abs(ref_point_1 - ref_point_2)

def geometricError(ref_point_1, ref_point_2):
    """
    Provides geometric error between two points

    Args:
        ref_point_1 (tm): Reference point 1
        ref_point_2 (tm): Reference point 2

    Returns:
        tm: geometric error

    """
    return globalToLocal(ref_point_2, ref_point_1)

def distance(ref_point_1, ref_point_2):
    """
    Calculates straight line distance between two points (2d or 3d (tm))

    Args:
        ref_point_1 (tm): Reference point 1
        ref_point_2 (tm): Reference point 2

    Returns:
        float: distance between points 1 and 2

    """
    try:
        d = math.sqrt((ref_point_2[0] - ref_point_1[0])**2 + (ref_point_2[1] - ref_point_1[1])**2 + (ref_point_2[2] - ref_point_1[2])**2)
    except:
        d = math.sqrt((ref_point_2[0] - ref_point_1[0])**2 + (ref_point_2[1] - ref_point_1[1])**2)
    return d

def arcDistance(ref_point_1, ref_point_2):
    """
    Calculates the arc distance between two points
    (magnitude average of geometric error)

    Args:
        ref_point_1 (tm): Reference point 1
        ref_point_2 (tm): Reference point 2

    Returns:
        float: arc distance between two points

    """
    geo_error = globalToLocal(ref_point_1, ref_point_2)
    d = math.sqrt(geo_error[0]**2 + geo_error[1]**2 + geo_error[2]**2 + geo_error[3]**2 +geo_error[4]**2 + geo_error[5]**2)
    return d

#Gap Closure
def closeLinearGap(origin_point, goal_point, delta):
    """
    Close linear gap between two points by delta amount

    Args:
        origin_point (tm): Current point in trajectory
        goal_point (tm): Goal to advance towards
        delta (float): Amount to advance

    Returns:
        tm: new, closer position to goal

    """
    origin_to_goal = goal_point - origin_point
    #normalize
    return_transform = np.zeros((6, 1))
    var = math.sqrt(origin_to_goal[0]**2 + origin_to_goal[1]**2 + origin_to_goal[2]**2)
    #print(var, "var")
    if var == 0:
        var = 0
    for i in range(6):
        return_transform[i] = origin_point.TAA[i] + (origin_to_goal[i] / var) * delta
    #xf = origin_point @ TAAtoTM(return_transform)

    return tm(return_transform)

def closeArcGap(origin_point, goal_point, delta):
    """
    Closes gap to goal using arc method instead of linear

    Args:
        origin_point (tm): Current point in trajectory
        goal_point (tm): Goal to advance towards
        delta (float): Amount to advance

    Returns:
        tm: new, closer position to goal

    """
    origin_to_goal = goal_point - origin_point
    #normalize
    return_transform = np.zeros((6, 1))
    var = math.sqrt(origin_to_goal[0]**2 + origin_to_goal[1]**2 + origin_to_goal[2]**2)
    #print(var, "var")
    if var == 0:
        var = 0
    for i in range(6):
        return_transform[i] = (origin_to_goal[i] / var) * delta
    xf = origin_point @ TAAtoTM(return_transform)

    return xf

def IKPath(initial, goal, steps):
    """
    Creates a simple oath from origin to goal witha  given number of steps

    Args:
        initial (tm): Initial Position
        goal (tm): Goal position
        steps (int): number of steps to take

    Returns:
        [tm]: list of transforms representing trajectory

    """
    delta = (goal.gTAA() - initial.gTAA())/steps
    pose_list = []
    for i in range(steps):
        pos = tm(initial.gTAA() + delta * i)
        pose_list.append(pos)
    return pose_list


#ANGLE HELPERS
def deg2Rad(deg):
    """
    Convert degrees to radians
    Args:
        deg (float): measure of angles in degrees
    Returns:
        float: measure of angles in radians
    """
    return deg * np.pi / 180

def rad2Deg(rad):
    """
    Converts radians to degrees
    Args:
        rad (float): measure of angles in radians
    Returns:
        float: measure of angles in degrees
    """
    return rad * 180 / np.pi

def angleMod(rad):
    """
    Cuts angles in radians such that they don't exceed 2pi absolute
    Args:
        rad (float): angle or angles
    Returns:
        float: cut down angle or angles
    """
    if isinstance(rad, tm):
        return rad.AngleMod();
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

def angleBetween(ref_point_1, ref_point_2, ref_point_3):
    """
    Calculates the interior angle between points 1, 2, and 3
    Args:
        ref_point_1 (tm): Reference point 1
        ref_point_2 (tm): Reference point 2
        ref_point_3 (tm): Reference point 3
    Returns:
        float: Angle between points 1, 2, and 3 in 3D-Space
    """
    v1 = np.array([ref_point_1[0]-ref_point_2[0], ref_point_1[1]-ref_point_2[1], ref_point_1[2] - ref_point_2[2]])
    #v1n = mr.mr.Normalize(v1)
    v1n = np.linalg.norm(v1)
    v2 = np.array([ref_point_3[0]-ref_point_2[0], ref_point_3[1]-ref_point_2[1], ref_point_3[2] - ref_point_2[2]])
    #v2n = mr.mr.Normalize(v2)
    v2n = np.linalg.norm(v2)
    res = np.clip(np.dot(v1, v2)/(v1n*v2n), -1, 1)
    #res = np.clip(np.dot(np.squeeze(v1n), np.squeeze(v2n)), -1, 1)
    res = AngleMod(math.acos(res))
    return res

#Wrench Operations
def makeWrench(position_applied, force, force_direction_vector):
    """
    Generates a new wrench
    Args:
        position_applied: relative position ation of wrench
        force: magnitude of force applied (or mass if force_direction_vector is a gravity vector)
        force_direction_vector: unit vector to apply force (or gravity)
    Returns:
        wrench
    """
    forcev = np.array(force_direction_vector) * force #Force vector (negative Z)
    t_wren = np.cross(position_applied[0:3].reshape((3)), forcev) #Calculate moment based on position and action
    wrench = np.array([t_wren[0], t_wren[1], t_wren[2], forcev[0], forcev[1], forcev[2]]).reshape((6, 1)) #Create Complete Wrench
    return wrench

def transformWrenchFrame(wrench, old_wrench_frame, new_wrench_frame):
    """
    Translates one wrench frame to another
    Args:
        wrench: original wrench to be translated
        old_wrench_frame: the original frame that the wrench was in (tm)
        new_wrench_frame: the new frame that the wrench *should* be in (tm)
    Returns:
        new Wrench in the frame of new_wrench_frame
    """
    ref = globalToLocal(old_wrench_frame, new_wrench_frame)
    return  ref.adjoint().T @ wrench

#Twists
def twistToScrew(input_twist):
    """
    Converts a twist to a screw
    Args:
        input_twist (ndarray): Twist
    Returns:
        ndarray: Screw representing twist
    """
    if (mr.Norm(input_twist[0:3])) == 0:
        w = mr.Normalize(input_twist[0:6])
        th = mr.Norm(input_twist[3:6])
        q = np.array([0, 0, 0]).reshape((3, 1))
        h = np.inf
    else:
        unit_twist = input_twist/mr.Norm(input_twist[0:3])
        w = unit_twist[0:3].reshape((3))
        v = unit_twist[3:6].reshape((3))
        th = mr.Norm(input_twist[0:3])
        q = np.cross(w, v)
        h = (v.reshape((3, 1)) @ w.reshape((1, 3)))
    return (w, th, q, h)

def normalizeTwist(twist):
    """
    Normalize a Twist
    Args:
        tw (ndarray): Input twist
    Returns:
        ndarray: Normalized Twist
    """
    if mr.Norm(twist[0:3]) > 0:
        twist_norm = twist/mr.Norm(twist[0:3])
    else:
        twist_norm = twist/mr.Norm(twist[3:6])
    return twist_norm

def twistFromTransform(input_transform):
    """
    Creates twist from transform (tm)
    Args:
        input_transform (tm): input transform
    Returns:
        ndarray: twist representing transform
    """
    transform_skew = mr.MatrixLog6(input_transform.TM)
    return mr.se3ToVec(transform_skew)

def transformFromTwist(input_twist):
    """
    Converts a twist to a transformation matrix
    Args:
        input_twist (ndarray): Input twist to be transformed
    Returns:
        tm: Transform represented by twist
    """
    input_twist = input_twist.reshape((6))
    #print(tw)
    tms = mr.VecTose3(input_twist)
    tms = delMini(tms)
    tmr = mr.MatrixExp6(tms)
    return tm(tmr)

def transformByVector(transform, vec):
    """
    Performs tv = TM*vec and removes the 1
    Args:
        transform (tm): transform to operate on
        vec (ndarray): vector to multipy

    Returns:
        ndarray: vector product
    """

    transform_matrix = transform.TM
    b = np.array([1.0])
    n = np.concatenate((vec, b))
    trvh = transform_matrix @ n
    return trvh[0:3]

#def RotationAroundVector(w, theta):
#    r = np.identity(3)+math.sin(theta) * rp.skew(w)+(1-math.cos(theta)) * rp.skew(w) @ rp.skew(w)
#    return r

# Unit Vectors
def fiboSphere(num_points):
    """
    Create Fibonacci points on the surface of a sphere
    #https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    Args:
        num_points: number of points
    Returns:
        xyzcoords: points in cartesian coordinates
    """
    indices = np.arange(0, num_points, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    xyzcoords = np.array([x, y, z]).T
    return xyzcoords

def unitSphere(num_points):
    """
    Generates a "unit sphere" with an approximate number of points
    numActual = round(num_points)^2
    Args:
        num_points: Approximate number of points to collect
    Returns:
        xyzcoords: points in cartesian coordinates
        azel: coords in azimuth/elevation notation
    """
    xyzcoords = []
    azel = []
    azr = round(math.sqrt(num_points))
    elr = round(math.sqrt(num_points))
    inc = np.pi * 2 / azr
    incb = 2 / elr
    a = 0
    e = -1
    for i in range(azr + 1):
        arccos_e = np.arccos(np.clip(e, -1.0, 1.0))
        sin_arccos_e = np.sin(arccos_e)
        for j in range(elr + 1):
            x = np.cos(a) * sin_arccos_e
            y = np.sin(a) * sin_arccos_e
            z = np.cos(arccos_e)
            xyzcoords.append([x, y, z])
            azel.append([a, e])
            a = a + inc
        e = e + incb
        a = -1

    return xyzcoords, azel


def getUnitVec(ref_point_1, ref_point_2, distance = 1.0):
    """
    Returns a vector of a given length pointed from point 1 to point 2
    Args:
        ref_point_1 (tm): Reference point 1
        ref_point_2 (tm): Reference point 2
        distance (Float): length of the returned vector. Defaults to 1
    Returns:
        tm: transform representing vector
    """
    v1 = np.array([ref_point_1[0], ref_point_1[1], ref_point_1[2]])
    unit_b = (np.array([ref_point_2[0], ref_point_2[1], ref_point_2[2]]) - v1)
    unit = unit_b / ling.norm(unit_b)
    pos = v1 + (unit * distance)
    return tm([pos[0], pos[1], pos[2], 0, 0, 0])

#Jacobians
def chainJacobian(screws, theta):
    """
    Chain Jacobian
    Args:
        Screws: screw list
        theta: theta to evaluate at
    Returns:
        jac: chain jacobian
    """
    jac = np.zeros((6, np.size(theta)))
    T = np.eye(4)
    jac[0:6, 0] = screws[0:6, 0]

    for i in range(1, np.size(theta)):
        T = T * TransformFromTwist(theta[i-1]*screws[1:6, i-1])
        jac[0:6, i] = mr.Adjoint(T)*screws[0:6, i]
    return jac

def numericalJacobian(f, x0, h):
    """
    Calculates a numerical jacobian
    Args:
        f: function handle (FK)
        x0: initial value
        h: delta value
    Returns:
        dfdx: numerical Jacobian
    """
    x0p = np.copy(x0)
    x0p[0] = x0p[0] + h
    x0m = np.copy(x0)
    x0m[0] = x0m[0] - h
    dfdx = (f(x0p)-f(x0m))/(2*h)

    for i in range(1, x0.size):
        x0p =  np.copy(x0)
        x0p[i] = x0p[i] + h
        x0m =  np.copy(x0)
        x0m[i] = x0m[i] - h
        #Conversion paused here. continue evalutation
        dfdx=np.concatenate((dfdx,(f(x0p)-f(x0m))/(2*h)), axis = 0)
    dfdx=dfdx.conj().T
    f(x0)

    return dfdx

#Misc
def boxSpatialInertia(m, l, w, h):
    """
    Calculates spatial inertial properties of a box

    Args:
        m (float): mass of box
        l (float): length of box
        w (float): width of box
        h (float): height of box
    Returns:
        ndarray: spatial inertia matrix of box
    """
    Ixx = m*(w*w+h*h)/12
    Iyy = m*(l*l+h*h)/12
    Izz = m*(w*w+l*l)/12
    Ib = np.diag((Ixx, Iyy, Izz))

    Gbox = np.vstack((np.hstack((Ib, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), m*np.identity((3))))))
    return Gbox


def delMini(arr):
    """
    Deletes subarrays of dimension 1
    Requires 2d array
    Args:
        arr: array to prune
    Returns:
        newarr: pruned array
    """

    s = arr.shape
    newarr = np.zeros((s))
    for i in range(s[0]):
        for j in range(s[1]):
            newarr[i, j] = arr[i, j]
    return newarr

def setElements(data, inds, vals):
    """
    Sets the elements in data specified by inds with the values in vals

    Args:
        data (ndarray): data to edit
        inds (ndarray): indexes of data to access
        vals (ndarray): new values to insert into the data
    Returns:
        ndarray: modified data
    """
    res = data.copy()
    for i in range(len(inds)):
        res[inds[i]] = vals[i]
    #res[inds] = vals
    #for i in range (0,(inds.size-1)):
    #    res[inds[i]] = vals[i]
    return res

# DEPRECATED FUNCTION HANDLES
import traceback
def LocalToGlobal(reference, rel):
    """Deprecation notice function. Please use indicated correct function"""
    print(LocalToGlobal.__name__ + ' is deprecated, use ' + localToGlobal.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return localToGlobal(reference, rel)

def GlobalToLocal(reference, rel):
    """Deprecation notice function. Please use indicated correct function"""
    print(GlobalToLocal.__name__ + ' is deprecated, use ' + globalToLocal.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return globalToLocal(reference, rel)

def PlaneFrom3Tms(ref_point_1, ref_point_2, ref_point_3):
    """Deprecation notice function. Please use indicated correct function"""
    print(PlaneFrom3Tms.__name__ + ' is deprecated, use ' + planeFromThreePoints.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return planeFromThreePoints(ref_point_1, ref_point_2, ref_point_3)

def PlaneTMSFromOne(ref_point_1):
    """Deprecation notice function. Please use indicated correct function"""
    print(PlaneTMSFromOne.__name__ + ' is deprecated, use ' + planePointsFromTransform.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return planePointsFromTransform(ref_point_1)

def Mirror(origin, mirror_plane):
    """Deprecation notice function. Please use indicated correct function"""
    print(Mirror.__name__ + ' is deprecated, use ' + mirror.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return mirror(origin, mirror_plane)

def TMMidRotAdjust(active_point, ref_point_1, ref_point_2, mode = 0):
    """Deprecation notice function. Please use indicated correct function"""
    print(TMMidRotAdjust.__name__ + ' is deprecated, use ' + adjustRotationToMidpoint.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return adjustRotationToMidpoint(active_point, ref_point_1, ref_point_2, mode = 0)

def TMMidPointEx(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(TMMidPointEx.__name__ + ' is deprecated, use ' + tmAvgMidpoint.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return tmAvgMidpoint(ref_point_1, ref_point_2)

def TMMidPoint(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(TMMidPoint.__name__ + ' is deprecated, use ' + tmInterpMidpoint.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return tmInterpMidpoint(ref_point_1, ref_point_2)

def RotFromVec(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(RotFromVec.__name__ + ' is deprecated, use ' + rotationFromVector.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return rotationFromVector(ref_point_1, ref_point_2)

def lookat(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(lookat.__name__ + ' is deprecated, use ' + lookAt.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return lookAt(ref_point_1, ref_point_2)

def Error(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(Error.__name__ + ' is deprecated, use ' + poseError.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return poseError(ref_point_1, ref_point_2)

def GeometricError(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(GeometricError.__name__ + ' is deprecated, use ' + geometricError.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return geometricError(ref_point_1, ref_point_2)

def Distance(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(Distance.__name__ + ' is deprecated, use ' + distance.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return distance(ref_point_1, ref_point_2)

def ArcDistance(ref_point_1, ref_point_2):
    """Deprecation notice function. Please use indicated correct function"""
    print(ArcDistance.__name__ + ' is deprecated, use ' + arcDistance.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return arcDistance(ref_point_1, ref_point_2)

def CloseGap(origin_point, goal_point, delta):
    """Deprecation notice function. Please use indicated correct function"""
    print(CloseGap.__name__ + ' is deprecated, use ' + closeLinearGap.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return closeLinearGap(origin_point, goal_point, delta)

def ArcGap(origin_point, goal_point, delta):
    """Deprecation notice function. Please use indicated correct function"""
    print(ArcGap.__name__ + ' is deprecated, use ' + closeArcGap.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return closeArcGap(origin_point, goal_point, delta)

def Deg2Rad(deg):
    """Deprecation notice function. Please use indicated correct function"""
    print(Deg2Rad.__name__ + ' is deprecated, use ' + deg2Rad.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return deg2Rad(deg)

def Rad2Deg(rad):
    """Deprecation notice function. Please use indicated correct function"""
    print(Rad2Deg.__name__ + ' is deprecated, use ' + rad2Deg.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return rad2Deg(rad)

def AngleMod(rad):
    """Deprecation notice function. Please use indicated correct function"""
    print(AngleMod.__name__ + ' is deprecated, use ' + angleMod.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return angleMod(rad)

def AngleBetween(ref_point_1, ref_point_2, ref_point_3):
    """Deprecation notice function. Please use indicated correct function"""
    print(AngleBetween.__name__ + ' is deprecated, use ' + angleBetween.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return angleBetween(ref_point_1, ref_point_2, ref_point_3)

def GenForceWrench(position_applied, force, force_direction_vector):
    """Deprecation notice function. Please use indicated correct function"""
    print(GenForceWrench.__name__ + ' is deprecated, use ' + makeWrench.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return makeWrench(position_applied, force, force_direction_vector)

def TransformWrenchFrame(wrench, old_wrench_frame, new_wrench_frame):
    """Deprecation notice function. Please use indicated correct function"""
    print(TransformWrenchFrame.__name__ + ' is deprecated, use ' + transformWrenchFrame.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return transformWrenchFrame(wrench, old_wrench_frame, new_wrench_frame)

def TwistToScrew(input_twist):
    """Deprecation notice function. Please use indicated correct function"""
    print(TwistToScrew.__name__ + ' is deprecated, use ' + twistToScrew.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return twistToScrew(input_twist)

def NormalizeTwist(twist):
    """Deprecation notice function. Please use indicated correct function"""
    print(NormalizeTwist.__name__ + ' is deprecated, use ' + normalizeTwist.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return normalizeTwist(twist)

def TwistFromTransform(input_transform):
    """Deprecation notice function. Please use indicated correct function"""
    print(TwistFromTransform.__name__ + ' is deprecated, use ' + twistFromTransform.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return twistFromTransform(input_transform)

def TransformFromTwist(input_twist):
    """Deprecation notice function. Please use indicated correct function"""
    print(TransformFromTwist.__name__ + ' is deprecated, use ' + transformFromTwist.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return transformFromTwist(input_twist)

def TrVec(transform, vec):
    """Deprecation notice function. Please use indicated correct function"""
    print(TrVec.__name__ + ' is deprecated, use ' + transformByVector.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return transformByVector(transform, vec)

def ChainJacobian(screws, theta):
    """Deprecation notice function. Please use indicated correct function"""
    print(ChainJacobian.__name__ + ' is deprecated, use ' + chainJacobian.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return chainJacobian(screws, theta)

def NumJac(f, x0, h):
    """Deprecation notice function. Please use indicated correct function"""
    print(NumJac.__name__ + ' is deprecated, use ' + numericalJacobian.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return numericalJacobian(f, x0, h)

def BoxSpatialInertia(m, l, w, h):
    """Deprecation notice function. Please use indicated correct function"""
    print(BoxSpatialInertia.__name__ + ' is deprecated, use ' + boxSpatialInertia.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return boxSpatialInertia(m, l, w, h)

def SetElements(data, inds, vals):
    """Deprecation notice function. Please use indicated correct function"""
    print(SetElements.__name__ + ' is deprecated, use ' + setElements.__name__ + ' instead')
    traceback.print_stack(limit=2)
    return setElements(data, inds, vals)
