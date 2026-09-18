basic_robotics
===============

Basic Robotics is a general purpose robotics package featuring forward and
inverse kinematics for serial and parallel robots, static analysis,
transformations, path planning, and more.

It is divided into several component libraries:

- ``basic_robotics.general`` - Basic robotics math, transforms, helper functions
- ``basic_robotics.interfaces`` - Hardware communications skeletons for physical robots
- ``basic_robotics.kinematics`` - Serial and Parallel robot kinematics classes
- ``basic_robotics.modern_robotics_numba`` - Numba enabled version of NxRLab's Modern Robotics
- ``basic_robotics.path_planning`` - RRT* path planning variations and spatial indexing
- ``basic_robotics.plotting`` - Simple Plotting functions for visualizations of robots
- ``basic_robotics.collisions`` - Fast Collision Manager for robotics and obstacles
- ``basic_robotics.metrology`` - Camera/vision helpers for measurement and localization
- ``basic_robotics.utilities`` - Terminal Displays and Logging
- ``basic_robotics.workspace`` - Reachability and manipulability analysis, visualization, and a scriptable command line front end

Installation
------------

.. code-block:: bash

   pip install basic_robotics

See the :doc:`readme` for a full tour of each component library, and
:doc:`examples` for runnable scripts covering transforms, serial arms,
Stewart platforms, path planning, and more.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   examples

API Reference
-------------

.. toctree::
   :maxdepth: 4
   :caption: API Reference:

   api/modules

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
