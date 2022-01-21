# FASER Math

FASER Interfaces is a general toolkit for functions used elswehere in other related repositories, and feautres tm: a transformation library, FASER: a catchall repository for useful functions, and Faser High Performance, an extension of [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)'s robotics toolkit.

Simply clone the repository

Python Dependencies:
```bash
pip install numpy
pip install scipy
pip install modern_robotics
pip install numba

```
Other Dependencies:
Install on the same folder level as the kinematics repository
```bash
git clone https://github.com/64-B1T/modern_robotics_numba.git
```
Or, if you are working on a repository featuring these, you can add them as submodules instead.

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

## Usage

Detailed Usage TBD

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
