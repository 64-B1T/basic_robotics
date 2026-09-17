# Basic Robotics

Basic Robotics is divided into several different component libraries:
 - basic_robotics.general - Basic robotics math, transforms, helper functions
 - basic_robotics.interfaces - Communications skeletons (Serial, UDP, ROS1/ROS2, OPC UA) for physical robots and sensors
 - basic_robotics.kinematics - Serial and Parallel robot kinematics classes
 - basic_robotics.modern_robotics_numba - Numba enabled version of NxRLab's Modern Robotics
 - basic_robotics.path_planning - RRT* path planning variations and spatial indexing
 - basic_robotics.plotting - Plotting functions for visualizing robots, in matplotlib or a three.js web client
 - basic_robotics.collisions - Fast Collision Manager for robotics and obstacles
 - basic_robotics.metrology - Virtual camera and vision simulation tools for estimating object poses from viewed scenes
 - basic_robotics.filtering - Generic Low Pass, Kalman, Extended Kalman, and Particle filters that work over tm objects, numpy arrays, and raw floats
 - basic_robotics.utilities - Terminal Displays and Logging
 - basic_robotics.workspace - Reachability and manipulability analysis, visualization, and a scriptable command line front end

## General Functionality
FASER Interfaces is a general toolkit for functions used elswehere in other related repositories, and feautres tm: a transformation library, FASER: a catchall repository for useful functions, and Faser High Performance, an extension of [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)'s robotics toolkit.
### Features
#### TM Library
- Abstracted handling and display of transformation matrices.
- Fully overloaded operators for addition, multiplication, etc
- Generate transforms from a variety of input types
- Easily extract positions and rotations in a variety of formats

#### FASER High Performance
- Provides kinematic extensions to the faser_robotics_kinematics library
- Accelerated with Numba, and extends [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics).

#### FASER (fsr)
- Catchall functions for manipulation of data elsewhere in FASER system
- Simple trajectory generation
- Position interpolation

#### Screws, Wrenches, and Twists
- `Screw`: base representation of a screw axis, per Modern Robotics Chapter 3
- `Wrench`: a `Screw` representing a force/torque applied at a position, with frame transformation support
- `Twist`: a `Screw` representing an instantaneous spatial velocity, convertible to/from transforms and se(3) matrices

### Usage
Usage Examples:

#### Transformation Library
```python
import numpy as np
from basic_robotics.general import tm

# Import the disp library to properly view instances of the transformations
from basic_robotics.utilities.disp import disp

#The transformation library allows for seamless usage of rotation matrices and other forms of rotation information encoding.
identity_matrix = tm()
disp(identity_matrix, 'identity') #This is just zeros in TAA format.

#Let's create a few more.
trans_x_2m = tm([2, 0, 0, 0, 0, 0]) #Translations can be created with a list (Xm, Ym, Zm, Xrad, Yrad, Zrad)
trans_y_4m = tm(np.array([0, 4, 0, 0, 0, 0])) # Translations can be created with a numpy array
rot_z_90 = tm([0, 0, 0, 0, 0, np.pi/2]) # Rotations can be declared in radians
trans_z_2m_neg = tm(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, -2], [0, 0, 0, 1]]))
# Transformations can be created from rotation matrices

trans_x_2m_quat = tm([2, 0, 0, 0, 0, 0, 1]) # Transformations can even be declared with quaternions

list_of_transforms = [trans_x_2m, trans_y_4m, rot_z_90]
disp(list_of_transforms, 'transform list') # List of transforms will be displayed in columns

#Operations
new_simple_transform = trans_x_2m + trans_y_4m #Additon is element-wise on TAA form
new_transform_multiplied = trans_x_2m @ trans_y_4m #Transformation matrix multiplication uses '@'
new_double_transform = trans_x_2m * 2 # Multiplication by a scalar is elementwise

#And more visible in the function list documentation

#And more visible in the function list documentation
```

#### Screws, Wrenches, and Twists
```python
import numpy as np
from basic_robotics.general import tm, Wrench, Twist

# A 10N force applied 0.5m along X from the origin creates a moment about Y
lever_arm = tm([0.5, 0, 0, 0, 0, 0])
force = Wrench(np.array([0, 0, 10]), position_applied=lever_arm)
print(force.getMoment(), force.getForce())

# Twists can be built directly from a spatial velocity vector, or from a transform
spin_in_place = Twist(np.array([0, 0, 1, 0, 0, 0]))
twist_from_motion = Twist.fromTM(tm([0.1, 0, 0, 0, 0, np.pi / 8]))

# And converted back to a transform via the matrix exponential
resulting_transform = twist_from_motion.toTM()
```

## Interfaces
FASER Interfaces is a simple toolkit for communicating with robots or sensors through a common interface, used generally to bridge basic_robotics to hardware and other software stacks.

### Features
- Generalized "Comms" implementation and communications supervisor (`CommsObject`, `Comms`)
- Standard interface for Serial and UDP
- Bridging to ROS1 and ROS2 topics (requires `rospy` or `rclpy`)
- Bridging to OPC Unified Automation servers (requires `python-opcua`)


### Usage

```python
from basic_robotics.interfaces import Comms

comms = Comms()
comms.newComPort('arm_serial', 'Serial', port='COM3', baud=115200)
comms.newComPort('sensor_udp', 'UDP')

comms.openAll()

serial_port = comms.getCom('arm_serial')
serial_port.sendData('home\n')
response = serial_port.getData()

comms.closeAll()
```
ROS1/ROS2 and OPC UA bridges follow a similar pattern; see `basic_robotics.interfaces.ros_bridge` and `basic_robotics.interfaces.opc_bridge` respectively, and require `rospy`/`rclpy` or `python-opcua` to be installed.

## Kinematics
FASER Robotics Kinematics is a toolbox for kinematics, statics, and dynamics of Stewart Platforms and Serial Manipulators, largely based on [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics). It is part of a set of related robotics repositories hosted here.

### Features

#### Stewart Platforms:
- Forward and Inverse Kinematics
- Static Analysis and Force Calculations
- Custom Configurations
- Error detection and correction

#### Serial Manipulators:
- Forward and Inverse Kinematics
- Static Analysis and Force Calculations
- Custom Configurations
- Error detection and correction
- Dynamics Analysis
- Visual Servoing and Path Planning

### Usage

#### Stewart Platform Example
```python
import json
import os
import matplotlib.pyplot as plt

import basic_robotics #Import the General Library
from basic_robotics.general import tm #Import transformation library
from basic_robotics.utilities.disp import disp
# SP Tests
disp("Beginning SP Test")
from basic_robotics.kinematics import loadSP
from basic_robotics.plotting.vis_matplotlib import DrawSP

basic_sp = {
    "Name":"Basic SP","Type":"SP","BottomPlate":{"Thickness":0.1,"JointRadius":0.9,"JointSpacing":9,"Mass": 6},
    "TopPlate":{"Thickness":0.16,"JointRadius":0.3,"JointSpacing":25,"Mass": 1},
    "Actuators":{"MinExtension":0.75,"MaxExtension":1.5,"MotorMass":0.5,"ShaftMass":0.9,"ForceLimit": 800,"MotorCOGD":0.2,"ShaftCOGD":0.2},
    "Drawing":{"TopRadius":1,"BottomRadius":1,"ShaftRadius": 0.1,"MotorRadius": 0.2},
    "Settings":{"MaxAngleDev":55,"GenerateActuators":0,"IgnoreRestHeight":1,"UseSpin":0,"AssignMasses":1,"InferActuatorCOG":1},
    "Params":{"RestHeight":1.2,"Spin":30}}

basic_sp_string = json.dumps(basic_sp)
with open ('sp_test_data.json', 'w') as outfile:
    outfile.write(basic_sp_string)

sp_model = loadSP('sp_test_data.json', '')
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,2)

#Delete file
os.remove('sp_test_data.json')
DrawSP(sp_model, ax)

plt.show()
```

#### Loading an Arm The Easy Way
```python
from basic_robotics.kinematics import loadArmFromURDF

#Load the Arm
new_arm = loadArmFromURDF('some_example_robot.urdf')
#You're Done!
```

#### Loading an Arm The Hard Way
```python
from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Arm
from basic_robotics.plotting.vis_matplotlib import *

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlim3d(-3,3)
ax.set_ylim3d(-3,3)
ax.set_zlim3d(0,6)
Base_T = tm() # Set a transformation for the base

        # Define some link lengths
L1 = 3
L2 = 3
L3 = 3
W = 0.1
# Define the transformations of each link
Tspace = [tm(np.array([[0],[0],[L1/2],[0],[0],[0]])),
    tm(np.array([[L2/2],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+(L3/2)],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+L3+(W/2)],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+L3+W+(W/2)],[0],[L1],[0],[0],[0]])),
    tm(np.array([[L2+L3+W+W+(W/2)],[0],[L1],[0],[0],[0]]))]
basic_arm_end_effector_home = fsr.TAAtoTM(np.array([[L2+L3+W+W+W],[0],[L1],[0],[0],[0]]))

basic_arm_joint_axes = np.array([[0, 0, 1],[0, 1, 0],[0, 1, 0],[1, 0, 0],[0, 1, 0],[1, 0, 0]]).conj().T
basic_arm_joint_homes = np.array([[0, 0, 0],[0, 0, L1],[L2, 0, L1],[L2+L3, 0, L1],[L2+L3+W, 0, L1],[L2+L3+2*W, 0, L1]]).conj().T
basic_arm_screw_list = np.zeros((6,6))

#Create the screw list
for i in range(0,6):
    basic_arm_screw_list[0:6,i] = np.hstack((basic_arm_joint_axes[0:3,i],np.cross(basic_arm_joint_homes[0:3,i],basic_arm_joint_axes[0:3,i])))

#Input some basic dimensions
basic_arm_link_box_dims = np.array([[W, W, L1],[L2, W, W],[L3, W, W],[W, W, W],[W, W, W],[W, W, W]]).conj().T
basic_arm_link_mass_transforms = [None] * (len(basic_arm_screw_list) + 1)
basic_arm_link_mass_transforms[0] = Tspace[0]

#Set mass transforms
for i in range(1,6):
    basic_arm_link_mass_transforms[i] = (Tspace[i-1].inv() @ Tspace[i])
basic_arm_link_mass_transforms[6] = (Tspace[5].inv() @ basic_arm_end_effector_home)
masses = np.array([20, 20, 20, 1, 1, 1])
basic_arm_inertia_list = np.zeros((6,6,6))

#Create spatial inertia matrices for links
for i in range(6):
    basic_arm_inertia_list[i,:,:] = fsr.boxSpatialInertia(masses[i],basic_arm_link_box_dims[0,i],basic_arm_link_box_dims[1,i],basic_arm_link_box_dims[2,i])

# Create the arm from the above paramters
arm = Arm(Base_T,basic_arm_screw_list,basic_arm_end_effector_home,basic_arm_joint_homes,basic_arm_joint_axes)

#ALTERNATIVELY, JUST LOAD A URDF USING THE 'loadArmFromURDF' function in basic_robotics.kinematics
arm.setJointProperties(
    np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) * -2,
    np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) * 2)
arm.setOrigins(link_homes_global = Tspace)
arm.setMassProperties(masses, basic_arm_link_mass_transforms, basic_arm_inertia_list)
arm.setVisColProperties(link_dimensions = basic_arm_link_box_dims)

#Draw the arm at a couple positions
#DrawArm(arm, ax, jdia = .3)
goal = arm.FK(np.array([np.pi/2, np.pi/4, -np.pi/4+.1, 0, 0, 0]))
DrawArm(arm, ax, jdia = .3)
plt.show()
```
## Modern Robotics - Numba
`basic_robotics.modern_robotics_numba` is a Numba-accelerated Python reimplementation of the code library accompanying [_Modern Robotics:
Mechanics, Planning, and Control_](http://modernrobotics.org) (Kevin Lynch
and Frank Park, Cambridge University Press 2017), imported throughout basic_robotics as `mr`. Several of its rigid-body motion, kinematics, dynamics, and trajectory-generation functions are JIT-compiled with `numba.jit` for speed; the original library (which this is based on) is also available in MATLAB and Mathematica, but only the Python/Numba port ships here.

Each function has a commented section above it explaining the inputs required for its use as well as an example of how it can be used and what the output will be. Functions are organized according to the chapter in which they are introduced in the book. Basic functions, such as functions to calculate the magnitude of a vector, normalize a vector, test if the value is near zero, and perform matrix operations such as multiplication and inverses, are not documented here.

The primary purpose of the provided software is to be easy to read and educational, reinforcing the concepts in the book. The code is optimized neither for efficiency nor robustness.

### Usage
```python
import numpy as np
from basic_robotics.modern_robotics_numba import mr

# Matrix logarithm/exponential of a homogeneous transform, per Chapter 3
transform = np.array([[1., 0, 0, 1], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]])
se3_form = mr.MatrixLog6(transform)
reconstructed = mr.MatrixExp6(se3_form)
```

## Path Planning
FASER Path planning is a toolbox for using RRT* to plan paths quickly through adverse terrain in a generalized sense (compatible with a wide variety of robotic tools)
### Features
- RRT* generation for various configurations
- Fake terrain generation
- Collision detection and obstacle avoidance
- Bindable functions for advanced tuning
- Dual Path RRT* for quicker solution finding
- Trajectory time-parametrization: turn a geometric path into a time-stamped trajectory
  respecting velocity/acceleration limits (trapezoidal), or velocity/acceleration/jerk
  limits (S-curve), for joint-space paths (`JointTrajectory`) or `tm` Cartesian paths
  (`CartesianTrajectory`)
### Usage

#### Simple Path Generation
```python
import random
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.path_planning import RRTStar, PathNode
from basic_robotics.plotting.vis_matplotlib import *

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,2)
arm = loadArmFromURDF('tests/test_helpers/irb_2400.urdf')

#Generate an RRT* instance
init = arm.getEEPos()
rrt = RRTStar(init)
rrt.addObstruction([0.5, 0.5, 0.7], [1.2, 1.2, 1.2]) # Add some random obstructions
rrt.addObstruction([0.5, 0.5, -2], [1, 1, 0.5])
DrawArm(arm, ax)

goal = arm.FK(np.array([np.pi/3, np.pi/3, -np.pi/8, np.pi/10, -np.pi/4, np.pi/5]))
arm.FK(np.zeros(6))

DrawObstructions(rrt.obstructions, ax) #Draw the obstructions for visulization

#Find a path through the environment
random.seed(10)
traj = rrt.findPathGeneral(
lambda: rrt.generalGenerateTree( # Generate a general RRT* tree using:
    lambda : PathNode(arm.randomPos()), # Random generation for path nodes
    lambda x, y : rrt.distance(x, y), # Distance between nodes as a cost
    lambda x, y : rrt.armObstruction(arm,x,y)), #Basic rtree collision detection as obstruction checking
goal) # Goal position

DrawRRTPath(traj, ax, 'green') # Draw the finalized path


plt.show() #Show the plot
```

#### Trajectory Time-Parametrization
```python
import numpy as np
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.path_planning import JointTrajectory, CartesianTrajectory
from basic_robotics.general import tm

arm = loadArmFromURDF('tests/test_helpers/irb_2400.urdf')
arm.setJointProperties(max_vels=np.ones(6), max_accels=np.ones(6) * 2)

# A geometric joint-space path (e.g. IK-solved along an RRT* path) has no timing -
# retime it into a velocity/acceleration-limited trajectory using the arm's own limits.
path = [np.zeros(6), np.array([0.3, -0.2, 0.1, 0.0, 0.2, -0.1])]
traj = arm.timeParametrizePath(path)  # add max_jerks=... for a jerk-limited S-curve
times, positions, velocities, accelerations = traj.sample(dt=0.01)

# Cartesian (tm) paths can be retimed the same way, interpolating along the
# screw motion between waypoints instead of straight-line joint deltas.
cart_traj = CartesianTrajectory([tm(), tm([0.5, 0, 0, 0, 0, 0])], v_max=0.5, a_max=1.0)
pose_at_half_second = cart_traj.position(0.5)
```

## Plotting
FASER Plotting is a toolbox for drawing robots and how they fit together, either offline with matplotlib or live in a browser using a bundled three.js viewer.
### Features
- Animate videos using matplotlib frames
- Plot various primary shapes, transforms, and wrenches
- Plot FASER Robots (`Arm`, `SP`) with `vis_matplotlib`
- A three.js web client/server (`vis_3js_client`, `vis_3js_server`) for interactive browser-based visualization of Arms (`ArmPlot`) and Stewart Platforms (`SPPlot`) via a `DrawClient`

### Usage

#### Matplotlib
```python
import matplotlib.pyplot as plt
from basic_robotics.general import tm
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.plotting.vis_matplotlib import DrawArm, DrawAxes, DrawRectangle

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(0, 3)

arm = loadArmFromURDF('tests/test_helpers/irb_2400.urdf')
DrawArm(arm, ax, jdia=.3)
DrawAxes(tm(), 0.5, ax) # Draw the world frame
DrawRectangle(tm([0.5, 0, 0.1, 0, 0, 0]), [0.2, 0.2, 0.2], ax, c='red')

plt.show()
```

#### Three.js Web Client
```python
from basic_robotics.plotting.vis_3js_client import DrawClient, ArmPlot
from basic_robotics.kinematics import loadArmFromURDF

# Start the bundled server first with: python -m basic_robotics.plotting.vis_3js_server
client = DrawClient(host='127.0.0.1', port=5000)
arm = loadArmFromURDF('tests/test_helpers/irb_2400.urdf')

# Creating the plot registers and draws it against the running server
arm_plot = ArmPlot('arm1', arm, client)

# Move the arm and push the new configuration to the browser
arm.FK([0, 0.3, -0.3, 0, 0, 0])
arm_plot.update(send=True)
```

## Collisions
Provides support for detecting collisions between Robot objects and user supplied obstacles, or other robots, using [python-fcl](https://github.com/BerkeleyAutomation/python-fcl). Supports both creation of basic shapes, and loading in robot geometry (from URDFs for example). This package requires Octomap and python-fcl, which are only available on Linux.

### Usage

```python
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.collisions import ColliderManager, ColliderObject, ColliderArm, createMesh

arm1 = loadArmFromURDF('some_arm.urdf')
arm2 = loadArmFromURDF('another_arm.urdf')

manager = ColliderManager()

arm1_collider = ColliderArm(arm1, 'arm1')
arm2_collider = ColliderArm(arm2, 'arm2')

arm1_collider.bindManager(manager)
arm2_collider.bindManager(manager)

print(manager.checkCollisions())

random_mesh_series = ColliderObject()
random_mesh_series.bindManager(manager)
random_mesh_series.addMesh('mesh_1', createMesh('first_mesh.stl', tm()))
random_mesh_series.addMesh('mesh_2', createMesh('second_mesh.stl', tm()))

print(manager.checkCollisions())
```
Standalone obstacle sets (`ColliderObstacles`) are supported the same way as `ColliderObject` above. `ColliderSP` exists as a class but is currently a stub (it does not populate geometry from an `SP`'s plates/actuators) - Stewart Platform collision checking is not yet functional.

## Metrology
Metrology provides virtual camera and vision simulation tools for estimating object poses from simulated viewed scenes. It models a `Scene` populated with simulated pinhole `Camera` objects and trackable `SceneObj`/`Observed` points, and provides triangulation-style routines for reconstructing 3D point and object positions from multiple simulated camera views.

### Usage

```python
from basic_robotics.general import tm, fsr
from basic_robotics.metrology import Scene, Camera

scene = Scene()

# A camera's local Z axis is its viewing direction, so point each camera at the
# origin with fsr.lookAt rather than just placing it (an un-aimed camera sees nothing).
cam1_pose = fsr.lookAt(tm([2, 0, 1, 0, 0, 0]), tm())
cam2_pose = fsr.lookAt(tm([0, 2, 1, 0, 0, 0]), tm())

# Add a couple of simulated cameras (focal x/y, principal point x/y, sensor width/height, noise sigma, pose)
scene.addCam(Camera(800, 800, 320, 240, 640, 480, 0.0005, cam1_pose, id=1))
scene.addCam(Camera(800, 800, 320, 240, 640, 480, 0.0005, cam2_pose, id=2))

# Add an object made up of known points to track
scene.newSceneObj([tm(), tm([0.1, 0, 0, 0, 0, 0])], name='target')

# Reconstruct object/point positions from the simulated camera observations
observed_points = scene.GetObjPositionsFromPoints()
```

## Filtering
Filtering provides generic state estimation tools: a Low Pass Filter, a linear Kalman Filter, an Extended Kalman Filter (EKF), and a Particle Filter. Each tracks state internally as a flat numpy vector for the underlying linear algebra, then hands the result back in whichever representation (a `tm`, a numpy array, or a raw float/double) it was created with - so the same filter class works whether you're smoothing a scalar sensor reading, a numpy state vector, or a full 6DOF `tm` pose.

### Features
- `LowPassFilter`: a first-order exponential filter. Works directly on any type supporting `+` and scalar `*` (`tm`, `np.ndarray`, `float`), with no conversion overhead
- `KalmanFilter`: a standard linear Kalman Filter, with process/measurement models (`F`, `B`, `H`) and noise covariances (`Q`, `R`) given as plain numpy matrices, or as a scalar shorthand for an isotropic `scalar * identity(n)` matrix
- `ExtendedKalmanFilter`: an EKF for nonlinear process/measurement models, where the models themselves are ordinary functions operating on states in their native representation (so a `tm`-based motion model can freely use `tm`'s overloaded operators). Jacobians are estimated numerically by default, or may be supplied analytically
- `ParticleFilter`: a Sequential Importance Resampling (SIR) particle filter with systematic resampling, for non-Gaussian or strongly nonlinear estimation problems

### Usage

#### Low Pass Filter
```python
from basic_robotics.filtering import LowPassFilter

lpf = LowPassFilter(alpha=0.2, initial_state=0.0)
for noisy_reading in [1.1, 0.9, 1.4, 0.8, 1.0]:
    smoothed = lpf.update(noisy_reading)
```

#### Kalman Filter
```python
import numpy as np
from basic_robotics.filtering import KalmanFilter

# Track a scalar sensor reading assumed constant, but noisy
kf = KalmanFilter(initial_state=0.0, initial_covariance=1.0, process_noise=1e-4)
for z in [4.8, 5.3, 4.9, 5.1, 5.0]:
    kf.predict(F=1.0)
    estimate = kf.update(z, H=1.0, R=0.25)
```

#### Extended Kalman Filter (with a `tm` state)
```python
from basic_robotics.general import tm
from basic_robotics.filtering import ExtendedKalmanFilter

def motion_model(state, u=None):
    # Nonlinear motion expressed directly with tm's overloaded operators
    return state @ tm([0.1, 0, 0, 0, 0, 0.05])

def measurement_model(state):
    return state  # Assume direct (noisy) pose observations

ekf = ExtendedKalmanFilter(tm(), initial_covariance=0.05, process_noise=1e-5)
for noisy_pose in noisy_pose_observations:  # a list of tm poses
    ekf.predict(motion_model)
    pose_estimate = ekf.update(noisy_pose, measurement_model, R=0.01)
```

#### Particle Filter
```python
from basic_robotics.filtering import ParticleFilter

pf = ParticleFilter(initial_state=0.0, initial_covariance=1.0, num_particles=300)
for z in [4.8, 5.3, 4.9, 5.1, 5.0]:
    pf.predict(lambda x, u=None: x, process_noise=1e-3)
    estimate = pf.update(z, lambda x: x, measurement_noise=0.25)
```

## Workspace
Workspace analyzes how much of a robot's surrounding space it can actually reach, and how well it can orient its end effector once there. It was originally developed as a standalone companion package (`workspace_analysis_tools`) and has since been folded directly into basic_robotics. Any robot can be analyzed by wrapping it in a `RobotLink` adapter that exposes FK/IK/end-effector access; `basic_robotics.kinematics.Arm` is supported directly.

### Features
- Brute-force and alpha-shape based reachability analysis
- Unit-sphere and Jacobian-based manipulability scoring
- Manipulability analysis over the surface of an object, with optional collision checking against obstacles
- Manipulability analysis along a trajectory
- A matplotlib-based viewer for previously saved results
- A scriptable command line front end (`WorkspaceCommandLine`) for running analyses without writing Python

### Usage

```python
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.workspace import WorkspaceAnalyzer, RobotLink

arm = loadArmFromURDF('tests/test_helpers/irb_2400.urdf')

# Wrap the arm so the analyzer can drive it generically
link = RobotLink(arm)
link.bind_fk(lambda theta: (arm.FK(theta), True))
link.bind_ik(lambda goal: arm.IK(goal, protect=True))
link.bind_ee(arm.getEEPos)
link.bind_jt(arm.getJointTransforms)
link.joint_mins = arm.joint_mins
link.joint_maxs = arm.joint_maxs

analyzer = WorkspaceAnalyzer(link)

# Reachable point cloud using the fast, alpha-shape based method
pose_cloud = analyzer.analyze_total_workspace_functional(num_spread=8)

# Manipulability scoring over a set of target poses. manip_resolution and the
# number of poses scored drive runtime quadratically-ish - keep both small for
# a quick look, and raise them (with parallel=True) for a real analysis.
results = analyzer.analyze_task_space_manipulability(
        pose_cloud[:20], manip_resolution=6)
```

Saved results can be explored later with the viewer:
```python
from basic_robotics.workspace import view_workspace

view_workspace('results.dat', draw_alpha_shape=True, draw_slices=True)
```

Or driven entirely from the command line:
```bash
python -m basic_robotics.workspace.command_line
Command> loadRobot -fromURDF some_arm.urdf
Command> alphaMethodTotalWorkspace -numIterations=25 -o=workspace.dat -plot
```
Use `help` inside the command line prompt (or `help <command>`) for the full list of commands and flags.

## Utilities
Utilities contains display and logging tools generally useful for working with other components in this package
### Features
- matlab like display function 'disp' which is a drop in replacement for python print()
- JSON file logging tool
- Print matrices with appropriate labels
- ProgressBar display
### Usage
```python
import time
from basic_robotics.general import tm
from basic_robotics.utilities.disp import disp, progressBar
from basic_robotics.utilities.FaserLog import FaserLog

# disp() is a drop-in replacement for print() that understands tm, lists, and matrices
disp(tm([1, 2, 3, 0, 0, 0]), 'my_transform')

# progressBar() renders/updates a terminal progress bar in place
for i in range(10):
    progressBar(i, 9, prefix='Working')
    time.sleep(0.05)

# FaserLog writes timestamped entries (and matrices) to a log file on disk
log = FaserLog('my_run')
log.writeToLog('Started processing')
log.writeMatrixToLog(tm(), 'starting_pose')
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
