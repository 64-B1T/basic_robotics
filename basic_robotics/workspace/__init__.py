"""
Workspace reachability and manipulability analysis tools.

Analyzes how much of a robot's surrounding space it can actually reach and how well
it can orient its end effector once there. Includes brute-force and alpha-shape based
reachability analysis, unit-sphere and Jacobian-based manipulability scoring, optional
collision-aware analysis against obstacles or object surfaces, a matplotlib-based
viewer for saved results, and a scriptable command line front end.

This module was originally developed as a standalone companion package
(``workspace_analysis_tools``) and has been folded directly into basic_robotics.
"""
from .alpha_shape import AlphaShape
from .robot_link import RobotLink
from .analyzer import WorkspaceAnalyzer, optimize_robot_for_goals
from .viewer import WorkspaceViewer, view_workspace
from .command_line import WorkspaceCommandLine, CommandExecutor
