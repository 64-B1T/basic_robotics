# General Math Functions

## Features
### TM Library
- Abstracted handling and display of transformation matrices.
- Fully overloaded operators for addition, multiplication, etc
- Generate transforms from a variety of input types
- Easily extract positions and rotations in a variety of formats

### FASER High Performance
- Provides kinematic extensions to the faser_robotics_kinematics library
- Accelerated with Numba, and extends [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics).

### FASER (fsr)
- Catchall functions for manipulation of data elsewhere in FASER system
- Simple trajectory generation
- Position interpolation

## Per-Function Usage
### General Functions
#### TAAtoTM(taa_format):

Converts Translation Axis Angle to Transformation Matrix
Args:
- taa_format (ndarray): TAA representation of given transformation.

Returns:
- transformation_matrix: 4x4 transformation matrix representation


#### TMtoTAA(transformation_matrix):

Converts a 4x4 transformation matrix to TAA representation
Args:
- transformation_matrix: transformation matrix to be converted

Returns:
- TAA representation


#### localToGlobal(reference, rel):

Converts a transform in a local frame to the global frame

Args:
- reference (temp): Transform of frame A to frame B
- rel (tm): Transform of object 1 in frame B

Returns:
- tm: Transform of object 1 in frame A



#### globalToLocal(reference, rel):

Convert a transform in a global frame to a local frame

Args:
- reference (tm): Transform of frame A to frame B
- rel (tm): Transform of object 1 in frame A

Returns:
- tm: Transform of object 1 in frame B


#Transformation Matrix Group Functions
#### planeFromThreePoints(ref_point_1, ref_point_2, ref_point_3):

Creates the equation of a plane from three points
Args:
- ref_point_1: tm or vector for point 1
- ref_point_2: tm or vector for point 2
- ref_point_3: tm or vector for point 3

Returns:
- a, b, c, d: equation cooficients of a plane


#### planePointsFromTransform(ref_point_1):

Create plane TM points from one Transform (using unit vectors)
Args:
- ref_point_1: transform to place plane on

Returns:
- a, b, c: Plane basis points


#### mirror(origin, mirror_plane):

Mirrors a point about a plane
Args:
- origin: point to be mirrored
- mirror_plane: tm describing plane to mirror over

Returns:
- mirrored Point


#### adjustRotationToMidpoint(active_point, ref_point_1, ref_point_2, mode = 0):

Applies the midpoint transform of reference points 1 and 2 to an active point

Args:
- active_point (tm): Point to be modified
- ref_point_1 (tm): origin point for midpoint calculation
- ref_point_2 (tm): goal point for midpoint calculation
- mode (int): Mode of midpoint calculation. 0, TMMIDPOINT. 1, ROTFROMVEC

Returns:
- tm: Modified version of active_point with orientation of vector 1 -> 2



#### tmAvgMidpoint(ref_point_1, ref_point_2):

Simplest version of a midpoint calculation. Simply the average of two positions

Args:
- ref_point_1 (tm): position 1
- ref_point_2 (tm): position 2

Returns:
- tm: midpoint average of positions 1 and 2


#### tmInterpMidpoint(ref_point_1, ref_point_2):

Better version of midpoint calculation

Position is stil average of positions 1 and 2
but rotation is calculated as a proper interpolation

Args:
- ref_point_1 (tm): position 1
- ref_point_2 (tm): position 2

Returns:
- tm: midpoint of positions 1 and 2


#### rotationFromVector(ref_point_1, ref_point_2):

Reorients ref_point_1 such that its z axis is pointing towards ref_point_2

Args:
- ref_point_1 (tm): position 1
- ref_point_2 (tm): position 2

Returns:
- tm: ref_point_1 where z points to position 2


#### lookAt(ref_point_1, ref_point_2):

Alternate version of RotFromVec, however rotation *about* z axis may be more random
Does not depend on sci optimize fmin

Args:
- ref_point_1 (tm): position 1
- ref_point_2 (tm): position 2

Returns:
- tm: point at ref_point_1 where z points to position 2


#### poseError(ref_point_1, ref_point_2):

Provides absolute error between two transformations

Args:
- ref_point_1 (tm): Reference point 1
- ref_point_2 (tm): Reference point 2

Returns:
- tm: absolute error


#### geometricError(ref_point_1, ref_point_2):

Provides geometric error between two points

Args:
- ref_point_1 (tm): Reference point 1
- ref_point_2 (tm): Reference point 2

Returns:
- tm: geometric error



#### distance(ref_point_1, ref_point_2):

Calculates straight line distance between two points (2d or 3d (tm))

Args:
- ref_point_1 (tm): Reference point 1
- ref_point_2 (tm): Reference point 2

Returns:
- float: distance between points 1 and 2


#### arcDistance(ref_point_1, ref_point_2):

Calculates the arc distance between two points
(magnitude average of geometric error)

Args:
- ref_point_1 (tm): Reference point 1
- ref_point_2 (tm): Reference point 2

Returns:
- float: arc distance between two points



#### closeLinearGap(origin_point, goal_point, delta):

Close linear gap between two points by delta amount

Args:
- origin_point (tm): Current point in trajectory
- goal_point (tm): Goal to advance towards
- delta (float): Amount to advance

Returns:
- tm: new, closer position to goal



#### closeArcGap(origin_point, goal_point, delta):

Closes gap to goal using arc method instead of linear

Args:
- origin_point (tm): Current point in trajectory
- goal_point (tm): Goal to advance towards
- delta (float): Amount to advance

Returns:
- tm: new, closer position to goal



#### IKPath(initial, goal, steps):

Creates a simple oath from origin to goal witha  given number of steps

Args:
- initial (tm): Initial Position
- goal (tm): Goal position
- steps (int): number of steps to take

Returns:
- [tm]: list of transforms representing trajectory



#### deg2Rad(deg):

Convert degrees to radians
Args:
- deg (float): measure of angles in degrees

Returns:
- float: measure of angles in radians


#### rad2Deg(rad):

Converts radians to degrees
Args:
- rad (float): measure of angles in radians

Returns:
- float: measure of angles in degrees


#### angleMod(rad):

Cuts angles in radians such that they don't exceed 2pi absolute
Args:
- rad (float): angle or angles

Returns:
- float: cut down angle or angles


#### angleBetween(ref_point_1, ref_point_2, ref_point_3):

Calculates the interior angle between points 1, 2, and 3
Args:
- ref_point_1 (tm): Reference point 1
- ref_point_2 (tm): Reference point 2
- ref_point_3 (tm): Reference point 3

Returns:
- float: Angle between points 1, 2, and 3 in 3D-Space


#Wrench Operations
#### makeWrench(position_applied, force, force_direction_vector):

Generates a new wrench
Args:
- position_applied: relative position ation of wrench
- force: magnitude of force applied (or mass if force_direction_vector is a gravity vector)
- force_direction_vector: unit vector to apply force (or gravity)

Returns:
- wrench


#### transformWrenchFrame(wrench, old_wrench_frame, new_wrench_frame):

Translates one wrench frame to another
Args:
- wrench: original wrench to be translated
- old_wrench_frame: the original frame that the wrench was in (tm)
- new_wrench_frame: the new frame that the wrench *should* be in (tm)

Returns:
- new Wrench in the frame of new_wrench_frame


#### twistToScrew(input_twist):

Converts a twist to a screw
Args:
- input_twist (ndarray): Twist

Returns:
- ndarray: Screw representing twist


#### normalizeTwist(twist):

Normalize a Twist
Args:
- tw (ndarray): Input twist

Returns:
- ndarray: Normalized Twist


#### twistFromTransform(input_transform):

Creates twist from transform (tm)
Args:
- input_transform (tm): input transform

Returns:
- ndarray: twist representing transform


#### transformFromTwist(input_twist):

Converts a twist to a transformation matrix
Args:
- input_twist (ndarray): Input twist to be transformed

Returns:
- tm: Transform represented by twist


#### transformByVector(transform, vec):

Performs tv = TM*vec and removes the 1
Args:
- transform (tm): transform to operate on
- vec (ndarray): vector to multipy

Returns:
- ndarray: vector product


#### fiboSphere(num_points):

Create Fibonacci points on the surface of a sphere
#https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
Args:
- num_points: number of points

Returns:
- xyzcoords: points in cartesian coordinates


#### unitSphere(num_points):

Generates a "unit sphere" with an approximate number of points
numActual = round(num_points)^2
Args:
- num_points: Approximate number of points to collect

Returns:
- xyzcoords: points in cartesian coordinates
- azel: coords in azimuth/elevation notation


#### getUnitVec(ref_point_1, ref_point_2, distance = 1.0):

Returns a vector of a given length pointed from point 1 to point 2
Args:
- ref_point_1 (tm): Reference point 1
- ref_point_2 (tm): Reference point 2
- distance (Float): length of the returned vector. Defaults to 1

Returns:
- tm: transform representing vector


#### chainJacobian(screws, theta):

Chain Jacobian
Args:
- Screws: screw list
- theta: theta to evaluate at

Returns:
- jac: chain jacobian


#### numericalJacobian(f, x0, h):

Calculates a numerical jacobian
Args:
- f: function handle (FK)
- x0: initial value
- h: delta value

Returns:
- dfdx: numerical Jacobian



#### boxSpatialInertia(m, l, w, h):

Calculates spatial inertial properties of a box

Args:
- m (float): mass of box
- l (float): length of box
- w (float): width of box
- h (float): height of box

Returns:
- ndarray: spatial inertia matrix of box


#### delMini(arr):

Deletes subarrays of dimension 1
Requires 2d array
Args:
- arr: array to prune

Returns:
- newarr: pruned array


#### setElements(data, inds, vals):

Sets the elements in data specified by inds with the values in vals

Args:
- data (ndarray): data to edit
- inds (ndarray): indexes of data to access
- vals (ndarray): new values to insert into the data

Returns:
- ndarray: modified data


### Faser High Performance (fmr)
