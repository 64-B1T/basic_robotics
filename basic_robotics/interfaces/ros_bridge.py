"""
Implement communications from Basic-Robotics to ROS 1 and ROS 2.

This is a very basic implementation of communications, and may be slower than more tightly integrated methods.
This is as planned, as the intent is simply to provide a basic interface that can be used without much effort. 
It is expected that those who would desire a more tight integration (such as multithreaded pub/sub) would know
how to enable such features on their own, and would have no need of this bridging module on its own.
"""

from typing import Any
try:
    import rospy
    import std_msgs.msg as r1msg
except ImportError:
    pass
try: 
    import rclpy
    from rclpy.node import Node
    import std_msgs.msg as r2msg
except ImportError:
    pass

class ROSPub:
    """Create a new ROS Publisher Instance."""

    def __init__(self, name : str, node_type : Any) -> 'ROSPub':
        """
        Create a new ROS Publisher Instance.

        Args:
            name (str): Name of the publisher
            node_type (Any): Datatype of the publisher

        Returns:
            ROSPub: New publisher instance.
        """    
        self.node_type = node_type
        self.name = name

    def send(self, message : Any) -> None:
        """
        Send a message.

        Args:
            message (Any): Message data to send.
        """ 
        self.publisher.publish(message)

    def __eq__(self, o : Any) -> bool:
        """
        Test for equality between ROSPub instances.

        Args:
            o (Any): Other object

        Returns:
            bool: Equality boolean. True is equal.
        """ 
        if o == None:
            return False
        if self.name == o:
            return True
        if isinstance(o, ROSPub):
            if self.name == o.name:
                return True
        return False

class ROS1Pub(ROSPub):
    """Create a new ROS1 Publisher Instance."""

    def __init__(self, name, node_type):
        """
        Create a new ROS1 Publisher Instance.

        Args:
            name (str): Name of the publisher
            node_type (Any): Datatype of the publisher

        Returns:
            ROS1Pub: New publisher instance.
        """    
        super().__init__(name, type)
        self.publisher = rospy.Publisher(name, node_type)
    
    def send(self, message):
        """
        Send a message.

        Args:
            message (Any): Message data to send.
        """ 
        self.publisher.publish(message)

class ROS2Pub(ROSPub):
    """Create a new ROS2 Publisher Instance."""

    def __init__(self, name : str, node_type : Any, host_node : 'ROS2Bridge') -> 'ROS2Pub':
        """
        Create a new ROS2 Publisher Instance.

        Args:
            name (str): Name of the publisher
            node_type (Any): Datatype of the publisher
            host_node (ROS2Bridge): Reference to the ROS2 Bridge Node host.

        Returns:
            ROS2Pub: New publisher instance.
        """        
        super().__init__(name, node_type)
        self.host_node = host_node
        self.publisher = host_node.create_publisher(node_type, name, host_node.r)

    def send(self, message : Any) -> None:
        """
        Send a message.

        Args:
            message (Any): Message data to send.
        """        
        new_message = self.node_type()
        new_message.data = message
        super().publish(new_message)
    
class ROSSub:
    """Create a new ROS Subscriber Handle."""

    def __init__(self, name, node_type, host_node, call_back=None):
        """
        Create a new ROS Subscriber Handle.

        Args:
            name (str): Name of the Subscription Node
            node_type (Any): Message type of the subscription node
            call_back (Any, optional): Callback function for subscriber. Defaults to None.

        Returns:
            ROSSub: New ROS2 Subscriber Handle.
        """
        self.name = name
        self.node_type = node_type
        self.host_node = host_node
        self.call_back_handle = call_back
        if self.call_back_handle is not None:
            self.setCallBack(self.call_back_handle)
        self.mostRecent = None

    def setCallBack(self, call_back : Any) -> None:
        """
        Set callback function to external handle.

        Args:
            call_back (Any): External function handle.
        """        
        self.callBack = call_back

    def callBack(self, data : Any) -> None:
        """
        Execute default callback on most recent data.

        Args:
            data (Any): data from Ros subscriber.
        """        
        self.mostRecent = data.data

    def getUpdate(self) -> Any:
        """
        Get the most recent data.

        Returns:
            Any: Most recent data.
        """        
        return self.mostRecent

    def __eq__(self, o : Any) -> bool:
        """
        Test for equality between ROSSub instances.

        Args:
            o (Any): Other object

        Returns:
            bool: Equality boolean. True is equal.
        """        
        if o == None:
            return False
        if self.name == o:
            return True
        if isinstance(o, ROSSub):
            if self.name == o.name:
                return True
        return False

class ROS1Sub(ROSSub):
    """Create a new ROS1 Subscriber Handle."""

    def __init__(self, name : str, node_type : Any, call_back : Any = None) -> 'ROS1Sub':
        """
        Create a new ROS1 Subscriber Handle.

        Args:
            name (str): Name of the Subscription Node
            node_type (Any): Message type of the subscription node
            call_back (Any, optional): Callback function for subscriber. Defaults to None.

        Returns:
            ROS1Sub: New ROS2 Subscriber Handle.
        """
        super().__init__(name, node_type, call_back)
        self.subscriber = rospy.Subscriber(name, type, self.callBack)

class ROS2Sub(ROSSub):
    """Create a new ROS2 Subscriber Handle."""

    def __init__(self, name : str, node_type : Any, 
            host_node : 'ROS2Bridge', call_back : Any = None) -> 'ROS2Sub':
        """
        Create a new ROS2 Subscriber Handle.

        Args:
            name (str): Name of the Subscription Node
            node_type (Any): Message type of the subscription node
            host_node (ROS2Bridge): Instance of ROS2Bridge inheriting (NODE) properties.
            call_back (Any, optional): Callback function for subscriber. Defaults to None.

        Returns:
            ROS2Sub: New ROS2 Subscriber Handle.
        """        
        super().__init__(name, node_type, call_back)
        self.subscriber = host_node.create_subscription(
                node_type, name, self.callBack, host_node.r)

class Uplink:
    """Create a new Uplink from a function to a ROS topic."""

    def __init__(self, pub : ROSPub, call : Any) -> 'Uplink':
        """
        Create a new Uplink from a function to a ROS topic.

        Args:
            pub (ROSPub): Ros topic subscriber
            call (Any): method call to populate

        Returns:
            Uplink: New Uplink
        """    
        self.pub = pub
        self.call = call

    def update(self) -> None:
        """Update the uplink's internal publisher."""
        msg = self.call()
        self.pub.send(msg)

class Downlink:
    """Create a downlink from a ROS topic to a function."""   

    def __init__(self, sub : ROSSub, call : Any) -> 'Downlink':
        """
        Create a new downlink from a ROS topic to a function.

        Args:
            sub (ROSSub): Ros topic subscriber
            call (Any): method call to populate

        Returns:
            Downlink: New Downlink
        """        
        self.sub = sub
        self.call = call
        self.sub.callBack = call

    def update(self) -> Any:
        """
        Update the downlink's internal subscriber.

        Returns:
            Any: Most recent data.
        """        
        msg = self.sub.mostRecent
        self.call(msg)
        return msg

class ROSBridge:
    """Create a bridge between Basic Robotics and ROS."""

    def __init__(self, name = 'faserNode', rate = 10):
        """
        Create a new ROSBridge instance.

        Args:
            name (str, optional): Name of this Bridge Node. Defaults to 'faserNode'.
            rate (int, optional): Updates per second. Defaults to 10.
        """    
        super().__init__(name)
        self.r = rate
        self.pub_list = []
        self.sub_list = []
        self.updateables = []

    def newPub(self, name, node_type):
        """
        Create a new Publisher handle.

        Args:
            name (str): Name of the publisher.
            node_type (Any): Message type of the publisher.

        Returns:
            ROSPub: New Publisher node.
        """     
        pass

    def newSub(self, name, node_type, func):
        """
        Create a new Subscriber handle.

        Args:
            name (str): Name of the subscriber
            node_type (Any): Message type of the subscriber
            func (Any): Function handler for subscriber to bind to

        Returns:
            ROSSub : New Subscriber Node
        """     
        pass

    def bindUplink(self, name : str, node_type : Any, func : Any) -> None:
        """
        Bind a new uplink to the bridge, pulling data from a function and publishing over ROS.

        Args:
            name (str): Name of the upload handle.
            node_type (Any): Datatype for the upload handle.
            func (Any): Function handle to hold on to.
        """        
        new_uplink = None
        if name in self.pub_list:
            new_uplink = Uplink(self.pub_list[self.pub_list.index(name)], func)
        else:
            tpub = self.newPub(name, node_type, self)
            self.pub_list.append(tpub)
            new_uplink = Uplink(tpub, func)
        self.updateables.append(new_uplink)

    def bindDownlink(self, name : str, node_type : Any, func : Any) -> None:
        """
        Bind a new downlink to the bridge, pulling data from ROS and calling a function.

        Args:
            name (str): Name of download handle.
            node_type (Any): Datatype for the download handle.
            func (Any): Function to hold on to.
        """        
        new_downlink = None
        if name in self.sub_list:
            new_downlink = Downlink(self.sub_list[self.sub_list.index(name)], func)
        else:
            tsub = self.newSub(name, node_type)
            self.sub_list.append(tsub)
            new_downlink = Downlink(tsub, func)
        self.updateables.append(new_downlink)

    def spin(self):
        """Start a continous update cycle.""" 
        pass
try:
    class ROS2Bridge(Node, ROSBridge):
        """Create a bridge between Basic Robotics and ROS 2."""

        def __init__(self, name : str = 'faserNode', rate : int = 10):
            """
            Create a new ROS2 Bridge instance.

            Args:
                name (str, optional): Name of this Bridge Node. Defaults to 'faserNode'.
                rate (int, optional): Updates per second. Defaults to 10.
            """        
            super().__init__(name, rate)

        def newPub(self, name : str, node_type : Any) -> ROS2Pub:
            """
            Create a new Publisher handle.

            Args:
                name (str): Name of the publisher.
                node_type (Any): Message type of the publisher.

            Returns:
                ROS2Pub: New Publisher node.
            """ 
            return ROS2Pub(name, node_type, self)

        def newSub(self, name : str, node_type : Any, func : Any) -> ROS2Sub:
            """
            Create a new Subscriber handle.

            Args:
                name (str): Name of the subscriber
                node_type (Any): Message type of the subscriber
                func (Any): Function handler for subscriber to bind to

            Returns:
                ROS2Sub : New Subscriber Node
            """     
            return ROS2Sub(name, node_type, func)

        def spin(self):
            """Start a continous update cycle."""   
            rclpy.spin(self)
            rclpy.shutdown()
except: 
    pass

class ROS1Bridge(ROSBridge):
    """Create a bridge between Basic Robotics and ROS 1."""

    def __init__(self, name : str = 'faserNode', rate : int = 10) -> ROSBridge:
        """
        Create a new ROS1 Bridge instance.

        Args:
            name (str, optional): Name of this Bridge Node. Defaults to 'faserNode'.
            rate (int, optional): Updates per second. Defaults to 10.
        """        
        super().__init__(name, rate)

    def newPub(self, name : str, node_type : Any) -> ROS1Pub:
        """
        Create a new Publisher handle.

        Args:
            name (str): Name of the publisher.
            node_type (Any): Message type of the publisher.

        Returns:
            ROS1Pub: New Publisher node.
        """        
        return ROS1Pub(name, node_type)
    
    def newSub(self, name : str, node_type : Any, func : Any) -> ROS1Sub:
        """
        Create a new Subscriber handle.

        Args:
            name (str): Name of the subscriber
            node_type (Any): Message type of the subscriber
            func (Any): Function handler for subscriber to bind to

        Returns:
            ROS1Sub : New Subscriber Node
        """        
        return ROS1Sub(name, node_type, func)

    def spin(self) -> None:
        """Start a continous update cycle."""        
        while True: 
            for item in self.updateables:
                item.update()
            self.r.sleep()
    

def makeROSBridge(node_name : str, 
        node_rate : int = 10, ros_ver :int = 1, exargs : Any = None) -> ROSBridge:
    """
    Make a new ROS Bridge.

    Args:
        node_name (string): Name of this ROS node.
        node_rate (int, optional): updates per second. Defaults to 10.
        ros_ver (int, optional): ROS Bridge version to build. Defaults to 1.
        exargs (Any, optional): Extra arguments for ROS2. Defaults to None.

    Returns:
        ROSbridge: New ROS Bridging Function
    """    
    if ros_ver == 1:
        return ROS1Bridge(node_name, node_rate)
    else:
        rclpy.init(args=exargs)
        return ROS2Bridge(node_name, node_rate)

def getMsgHandle(ros_ver : int = 1):
    """
    Get a handle to this ROS version's form of std_msg.msg.

    Args:
        ros_ver (int, optional): ROS Version. Defaults to 1.

    Returns:
        std_msg.msg: ROS message types
    """    
    if ros_ver == 1:
        return r1msg
    else:
        return r2msg

def getRosHandle(ros_ver :int = 1):
    """
    Return a handle to the ROS python installation.

    Args:
        ros_ver (int, optional): ROS Version to return. Defaults to 1.

    Returns:
        Ros Handle : Handle to a ROS installation.
    """    
    if ros_ver == 1:
        return rospy
    else: 
        return rclpy