"""
Basic Robotics: a toolbox for kinematics, dynamics, path planning, and visualization.

Aggregates the general math, kinematics, collision, metrology, path planning,
plotting, interfaces, and utilities subpackages under a single namespace.
"""

# __init__.py

# Version
__version__ = "0.3.05"
from . import interfaces
from . import modern_robotics_numba
from . import general
from . import plotting
from . import kinematics
from . import utilities
from . import collisions
from . import metrology
