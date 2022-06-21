"""
Use Python FCL to determine collisions between objects in a robotics system.

THIS PACKAGE REQUIRES OCTOMAP AND PYTHON-FCL WHICH ARE AVAILABLE ON LINUX ONLY.
"""
from .collision_manager import createBox, createCylinder, createSphere, createMesh
from .collision_manager import ColliderManager
from .collision_manager import ColliderObject
from .collision_manager import ColliderArm
from .collision_manager import ColliderSP
from .collision_manager import ColliderObstacles
