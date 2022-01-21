# FASER Robotics Kinematics

FASER Robotics Kinematics is a toolbox for kinematics, statics, and dynamics of Stewart Platforms and Serial Manipulators, largely based on [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics). It is part of a set of related robotics repositories hosted here.

## Installation

Simply clone the repository and install dependencies

Python Dependencies:
```bash
pip install numpy
pip install numpy-stl
pip install scipy
pip install modern_robotics
pip install numba
pip install descartes

```
Other Dependencies:
Install on the same folder level as the kinematics repository
```bash
git clone https://github.com/64-B1T/faser_math.git
git clone https://github.com/64-B1T/faser_plotting.git
git clone https://github.com/64-B1T/modern_robotics_numba.git
git clone https://github.com/64-B1T/faser_utils.git
```

Or, if you are working on a repository featuring these, you can add them as submodules instead.

## Features

### Stewart Platforms:
- Forward and Inverse Kinematics
- Static Analysis and Force Calculations
- Custom Configurations
- Error detection and correction

### Serial Manipulators:
- Forward and Inverse Kinematics
- Static Analysis and Force Calculations
- Custom Configurations
- Error detection and correction
- Dynamics Analysis
- Visual Servoing and Path Planning

## Usage

Detailed Usage TBD

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.



## License
[MIT](https://choosealicense.com/licenses/mit/)
