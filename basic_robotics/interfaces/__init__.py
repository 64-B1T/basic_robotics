"""
Submoddule for handling communications between Basic-Robotics and the outside world.

Currently capable of supporting Serial, UDP, ROS1 and ROS2 messaging.
"""
from . import communications as communications
from .ros_bridge import makeROSBridge, getMsgHandle, getRosHandle
