# FASER Path Planning

FASER Path planning is a toolbox for using RRT* to plan paths quickly through adverse terrain in a generalized sense (compatible with a wide variety of robotic tools)

## Installation

Simply clone the repository and install dependencies

Python Dependencies:
```bash
pip install rtree
pip install numpy
pip install scipy
pip install modern_robotics
```
Other Dependencies:
Install on the same folder level as the kinematics repository
```bash
git clone https://github.com/64-B1T/faser_math.git
git clone https://github.com/64-B1T/modern_robotics_numba.git
git clone https://github.com/64-B1T/faser_utils.git
```
Or, if you are working on a repository featuring these, you can add them as submodules instead.

## Features
- RRT* generation for various configurations
- Fake terrain generation
- Collision detection and obstacle avoidance
- Bindable functions for advanced tuning
- Dual Path RRT* for quicker solution finding
## Usage

Detailed Usage TBD

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
