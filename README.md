# Basic Robotics

Basic Robotics is divided into several different component libraries:
 - basic_robotics.general - Basic robotics math, transforms, helper functions
 - basic_robotics.interfaces - Hardware communications skeletons for physical robots
 - basic_robotics.kinematics - Serial and Parallel robot kinematics classes
 - basic_robotics.modern_robotics_numba - Numba enabled version of NxRLab's Modern Robotics
 - basic_robotics.path_planning - RRT* path planning variations and spatial indexing
 - basic_robotics.plotting - Simple Plotting functions for visualizations of robots
 - basic_robotics.robot_collisions - Fast Collision Manager for robotics and obstacles
 - basic_robotics.utilities - Terminal Displays and Logging

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
Detailed Usage TODO

## Interfaces
FASER Interfaces is a simple toolkit for communicating over serial and udp through a common interface, and used generally to communicate with robots or sensors

### Features
- Generalized "Comms" implementation
- Communications supervisor
- Standard interface for Serial and UDP


### Usage

Detailed Usage TODO

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

Detailed Usage TODO

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
arm.setDynamicsProperties(
    basic_arm_link_mass_transforms,
    Tspace,
    basic_arm_inertia_list,
    basic_arm_link_box_dims)
arm.joint_mins = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])* -2
arm.joint_maxs= np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) * 2

#Draw the arm at a couple positions
#DrawArm(arm, ax, jdia = .3)
goal = arm.FK(np.array([np.pi/2, np.pi/4, -np.pi/4+.1, 0, 0, 0]))
DrawArm(arm, ax, jdia = .3)
plt.show()
```
## Modern Robotics - Numba
TODO Explanation
This repository contains the code library accompanying [_Modern Robotics:
Mechanics, Planning, and Control_](http://modernrobotics.org) (Kevin Lynch
and Frank Park, Cambridge University Press 2017). The
[user manual](/doc/MRlib.pdf) is in the doc directory.

The functions are available in:

* Python
* MATLAB
* Mathematica

Each function has a commented section above it explaining the inputs required for its use as well as an example of how it can be used and what the output will be. This repository also contains a pdf document that provides an overview of the available functions using MATLAB syntax. Functions are organized according to the chapter in which they are introduced in the book. Basic functions, such as functions to calculate the magnitude of a vector, normalize a vector, test if the value is near zero, and perform matrix operations such as multiplication and inverses, are not documented here.

The primary purpose of the provided software is to be easy to read and educational, reinforcing the concepts in the book. The code is optimized neither for efficiency nor robustness.

## Path Planning
FASER Path planning is a toolbox for using RRT* to plan paths quickly through adverse terrain in a generalized sense (compatible with a wide variety of robotic tools)
### Features
- RRT* generation for various configurations
- Fake terrain generation
- Collision detection and obstacle avoidance
- Bindable functions for advanced tuning
- Dual Path RRT* for quicker solution finding
### Usage

#### Simple Path Generation
```python
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
Detailed Usage TODO

## Plotting
FASER Plotting is a simple toolbox which extends matplotlib to draw simple shapes and show how a robot fits together.
### Features
- Animate videos using matplotlib frames
- Plot various primary shapes
- Plot FASER Robots
- Plot transforms and wrenches

### Usage

Detailed Usage TODO

## Robot Collisions
Provides support for detecting collisions between Robot objects and user supplied obstacles, or other robots. Supports both creation of basic shapes, and loading in robot geometry (from URDFs for example)

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
## Utilities
Utilities contains display and logging tools generally useful for working with other components in this package
### Features
- matlab like display function 'disp' which is a drop in replacement for python print()
- JSON file logging tool
- Print matrices with appropriate labels
- ProgressBar display
### Usage
Detailed Usage TODO


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
