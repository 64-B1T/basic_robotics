"""
Basic Robotics: a toolbox for kinematics, dynamics, path planning, and visualization.

Aggregates the general math, kinematics, collision, metrology, path planning,
plotting, interfaces, workspace analysis, filtering, and utilities subpackages
under a single namespace.

``collisions`` and ``workspace`` depend on extra packages (trimesh, python-fcl,
alphashape, ...) that are not installed by default. Install them with
``pip install basic_robotics[collisions]``, ``pip install basic_robotics[workspace]``,
or ``pip install basic_robotics[all]`` to enable those subpackages.
"""

# __init__.py

import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("basic_robotics")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from . import interfaces
from . import modern_robotics_numba
from . import general
from . import plotting
from . import kinematics
from . import utilities
from . import metrology
from . import filtering

try:
    from . import collisions
except ImportError as e:  # pragma: no cover
    warnings.warn(
        f"basic_robotics.collisions is unavailable ({e}). "
        "Install it with `pip install basic_robotics[collisions]`.",
        stacklevel=2,
    )

try:
    from . import workspace
except ImportError as e:  # pragma: no cover
    warnings.warn(
        f"basic_robotics.workspace is unavailable ({e}). "
        "Install it with `pip install basic_robotics[workspace]`.",
        stacklevel=2,
    )
