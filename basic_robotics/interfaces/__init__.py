"""
Submoddule for handling communications between Basic-Robotics and the outside world.

Currently capable of supporting Serial, UDP, ROS1 and ROS2 messaging.
"""
from .comms_object import CommsObject
from .comms_core import Comms
from .ros_bridge import makeROSBridge, getMsgHandle, getRosHandle
from .opc_bridge import OPCUA_Client
