import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock

from basic_robotics.interfaces import ros_bridge


class _FakeBridge(ros_bridge.ROSBridge):
    """Minimal ROSBridge subclass used to exercise bindUplink/bindDownlink
    without needing rospy or rclpy at all."""

    def newPub(self, name, node_type):
        pub = ros_bridge.ROSPub(name, node_type)
        pub.publisher = MagicMock(name='publisher')
        return pub

    def newSub(self, name, node_type, func):
        return ros_bridge.ROSSub(name, node_type, self, func)


class test_interfaces_ros_base_classes(unittest.TestCase):
    """These cover logic that doesn't depend on rospy/rclpy being installed."""

    def test_rospub_equality(self):
        pub = ros_bridge.ROSPub('topic', int)
        self.assertFalse(pub == None)
        self.assertTrue(pub == 'topic')
        self.assertFalse(pub == 'other')
        other = ros_bridge.ROSPub('topic', float)
        self.assertTrue(pub == other)
        mismatched = ros_bridge.ROSPub('other', float)
        self.assertFalse(pub == mismatched)

    def test_rospub_send(self):
        pub = ros_bridge.ROSPub('topic', int)
        pub.publisher = MagicMock()
        pub.send(5)
        pub.publisher.publish.assert_called_once_with(5)

    def test_rossub_default_callback_and_update(self):
        sub = ros_bridge.ROSSub('topic', int, host_node=None)
        self.assertIsNone(sub.getUpdate())
        fake_msg = MagicMock(data=42)
        sub.callBack(fake_msg)
        self.assertEqual(sub.getUpdate(), 42)

    def test_rossub_custom_callback(self):
        seen = []
        sub = ros_bridge.ROSSub('topic', int, host_node=None, call_back=lambda d: seen.append(d))
        sub.callBack('payload')
        self.assertEqual(seen, ['payload'])

    def test_rossub_equality(self):
        sub = ros_bridge.ROSSub('topic', int, host_node=None)
        self.assertFalse(sub == None)
        self.assertTrue(sub == 'topic')
        self.assertFalse(sub == 'other')
        other = ros_bridge.ROSSub('topic', float, host_node=None)
        self.assertTrue(sub == other)

    def test_uplink_update(self):
        pub = MagicMock()
        uplink = ros_bridge.Uplink(pub, lambda: 'the message')
        uplink.update()
        pub.send.assert_called_once_with('the message')

    def test_downlink_update(self):
        sub = ros_bridge.ROSSub('topic', int, host_node=None)
        sub.mostRecent = 7
        seen = []
        downlink = ros_bridge.Downlink(sub, lambda msg: seen.append(msg))
        # Downlink's constructor rebinds the sub's callback to `call`
        self.assertIs(sub.callBack, downlink.call)
        result = downlink.update()
        self.assertEqual(result, 7)
        self.assertEqual(seen, [7])

    def test_rosbridge_construction_does_not_crash(self):
        # regression test: ROSBridge.__init__ used to call super().__init__(name),
        # which raises TypeError against plain object.__init__
        bridge = ros_bridge.ROSBridge('n', 5)
        self.assertEqual(bridge.r, 5)
        self.assertEqual(bridge.pub_list, [])
        self.assertEqual(bridge.sub_list, [])
        self.assertEqual(bridge.updateables, [])
        self.assertIsNone(bridge.newPub('t', int))
        self.assertIsNone(bridge.newSub('t', int, lambda d: d))
        self.assertIsNone(bridge.spin())

    def test_bindUplink_new_and_reuse(self):
        bridge = _FakeBridge('n', 10)
        bridge.bindUplink('topic', int, lambda: 42)

        self.assertEqual(len(bridge.pub_list), 1)
        self.assertEqual(len(bridge.updateables), 1)
        bridge.updateables[0].update()
        bridge.pub_list[0].publisher.publish.assert_called_with(42)

        # binding the same topic name again should reuse the existing publisher
        bridge.bindUplink('topic', int, lambda: 99)
        self.assertEqual(len(bridge.pub_list), 1)
        self.assertEqual(len(bridge.updateables), 2)
        bridge.updateables[1].update()
        bridge.pub_list[0].publisher.publish.assert_called_with(99)

    def test_bindDownlink_new_and_reuse(self):
        bridge = _FakeBridge('n', 10)
        results = []
        bridge.bindDownlink('topic', int, lambda msg: results.append(msg))

        self.assertEqual(len(bridge.sub_list), 1)
        bridge.sub_list[0].mostRecent = 7
        bridge.updateables[0].update()
        self.assertEqual(results, [7])

        bridge.bindDownlink('topic', int, lambda msg: results.append(msg * 2))
        self.assertEqual(len(bridge.sub_list), 1)
        bridge.sub_list[0].mostRecent = 3
        bridge.updateables[1].update()
        self.assertEqual(results, [7, 6])

    def test_ros1bridge_construction_and_spin(self):
        bridge = ros_bridge.ROS1Bridge('n', 10)
        seen = []
        bridge.updateables = [ros_bridge.Uplink(MagicMock(), lambda: seen.append(1))]

        class StopSpin(Exception):
            pass

        class FakeRate:
            def sleep(self):
                raise StopSpin()

        bridge.r = FakeRate()
        with self.assertRaises(StopSpin):
            bridge.spin()
        self.assertEqual(seen, [1])

    def test_makeROSBridge_not_ready(self):
        self.assertFalse(ros_bridge.READY)
        self.assertIsNone(ros_bridge.makeROSBridge('n'))


def _install_fake_rospy():
    fake_rospy = types.ModuleType('rospy')
    fake_rospy.Publisher = MagicMock(name='Publisher')
    fake_rospy.Subscriber = MagicMock(name='Subscriber')
    fake_std_msgs = types.ModuleType('std_msgs')
    fake_std_msgs_msg = types.ModuleType('std_msgs.msg')
    fake_std_msgs.msg = fake_std_msgs_msg
    sys.modules['rospy'] = fake_rospy
    sys.modules['std_msgs'] = fake_std_msgs
    sys.modules['std_msgs.msg'] = fake_std_msgs_msg
    importlib.reload(ros_bridge)
    return fake_rospy, fake_std_msgs_msg


def _remove_fake_rospy():
    del sys.modules['rospy']
    del sys.modules['std_msgs']
    del sys.modules['std_msgs.msg']
    importlib.reload(ros_bridge)


class test_interfaces_ros1_with_fake_rospy(unittest.TestCase):

    def setUp(self):
        self.fake_rospy, self.fake_r1msg = _install_fake_rospy()

    def tearDown(self):
        _remove_fake_rospy()

    def test_ready_flag_true(self):
        self.assertTrue(ros_bridge.READY)

    def test_ros1pub_uses_node_type_not_builtin_type(self):
        # regression test: ROS1Pub used to forward the builtin `type` instead
        # of the node_type argument to ROSPub.__init__ and rospy.Publisher
        pub = ros_bridge.ROS1Pub('topic', 'std_msgs/String')
        self.assertEqual(pub.node_type, 'std_msgs/String')
        self.fake_rospy.Publisher.assert_called_once_with('topic', 'std_msgs/String')

        pub.publisher = MagicMock()
        pub.send('hello')
        pub.publisher.publish.assert_called_once_with('hello')

    def test_ros1sub_uses_node_type_not_builtin_type(self):
        # regression test: same class of bug as ROS1Pub, in rospy.Subscriber
        seen = []

        def cb(data):
            seen.append(data)

        sub = ros_bridge.ROS1Sub('topic', 'std_msgs/String', cb)
        # the custom callback must actually be installed as self.callBack
        # (this used to silently stay as the ROSSub default due to a
        # positional-argument mismatch when forwarding to ROSSub.__init__)
        self.fake_rospy.Subscriber.assert_called_once_with('topic', 'std_msgs/String', sub.callBack)
        sub.callBack('data')
        self.assertEqual(seen, ['data'])

    def test_ros1bridge_newPub_newSub(self):
        bridge = ros_bridge.ROS1Bridge('n', 10)
        pub = bridge.newPub('topic', 'std_msgs/String')
        self.assertIsInstance(pub, ros_bridge.ROS1Pub)

        sub = bridge.newSub('topic2', 'std_msgs/String', lambda d: d)
        self.assertIsInstance(sub, ros_bridge.ROS1Sub)

    def test_bindUplink_bindDownlink_with_real_ros1_classes(self):
        bridge = ros_bridge.ROS1Bridge('n', 10)
        bridge.bindUplink('topic', 'std_msgs/String', lambda: 'hi')
        self.assertEqual(len(bridge.pub_list), 1)
        bridge.pub_list[0].publisher = MagicMock()
        bridge.updateables[0].update()
        bridge.pub_list[0].publisher.publish.assert_called_once_with('hi')

        results = []
        bridge.bindDownlink('topic2', 'std_msgs/String', lambda msg: results.append(msg))
        self.assertEqual(len(bridge.sub_list), 1)
        bridge.sub_list[0].mostRecent = 'downlink data'
        bridge.updateables[1].update()
        self.assertEqual(results, ['downlink data'])

    def test_getRosHandle_getMsgHandle_ready(self):
        self.assertIs(ros_bridge.getRosHandle(1), self.fake_rospy)
        self.assertIs(ros_bridge.getMsgHandle(1), self.fake_r1msg)

    def test_makeROSBridge_ros1(self):
        bridge = ros_bridge.makeROSBridge('n', 10, ros_ver=1)
        self.assertIsInstance(bridge, ros_bridge.ROS1Bridge)


class FakeRclpyNode:
    """Stand-in for rclpy.node.Node that cooperates with plain object.__init__."""

    def __init__(self, node_name, *args, **kwargs):
        self.node_name = node_name

    def create_publisher(self, node_type, name, qos):
        return MagicMock(name='rclpy_publisher')

    def create_subscription(self, node_type, name, callback, qos):
        return MagicMock(name='rclpy_subscription')


def _install_fake_rclpy():
    fake_rclpy = types.ModuleType('rclpy')
    fake_rclpy.init = MagicMock(name='init')
    fake_rclpy.spin = MagicMock(name='spin')
    fake_rclpy.shutdown = MagicMock(name='shutdown')
    fake_rclpy_node_mod = types.ModuleType('rclpy.node')
    fake_rclpy_node_mod.Node = FakeRclpyNode
    fake_rclpy.node = fake_rclpy_node_mod
    fake_std_msgs = types.ModuleType('std_msgs')
    fake_std_msgs_msg = types.ModuleType('std_msgs.msg')
    fake_std_msgs.msg = fake_std_msgs_msg
    sys.modules['rclpy'] = fake_rclpy
    sys.modules['rclpy.node'] = fake_rclpy_node_mod
    sys.modules['std_msgs'] = fake_std_msgs
    sys.modules['std_msgs.msg'] = fake_std_msgs_msg
    importlib.reload(ros_bridge)
    return fake_rclpy, fake_std_msgs_msg


def _remove_fake_rclpy():
    del sys.modules['rclpy']
    del sys.modules['rclpy.node']
    del sys.modules['std_msgs']
    del sys.modules['std_msgs.msg']
    importlib.reload(ros_bridge)


class test_interfaces_ros2_with_fake_rclpy(unittest.TestCase):

    def setUp(self):
        self.fake_rclpy, self.fake_r2msg = _install_fake_rclpy()

    def tearDown(self):
        _remove_fake_rclpy()

    def test_ready_flag_true_and_ros2bridge_defined(self):
        self.assertTrue(ros_bridge.READY)
        self.assertTrue(hasattr(ros_bridge, 'ROS2Bridge'))

    def test_ros2pub_send_uses_publisher_not_super(self):
        # regression test: ROS2Pub.send used to call super().publish(...), but
        # ROSPub has no publish() method -- it must go through self.publisher
        host_node = FakeRclpyNode('n')
        host_node.r = 5
        message_type = MagicMock(name='MsgType')
        pub = ros_bridge.ROS2Pub('topic', message_type, host_node)

        pub.send('payload')

        new_message = message_type.return_value
        pub.publisher.publish.assert_called_once_with(new_message)
        self.assertEqual(new_message.data, 'payload')

    def test_ros2sub_binds_real_host_node_and_callback(self):
        # regression test: ROS2Sub used to pass call_back positionally where
        # ROSSub expected host_node, so the custom callback never got installed
        host_node = FakeRclpyNode('n')
        host_node.r = 5
        seen = []

        sub = ros_bridge.ROS2Sub('topic', MagicMock(), host_node, lambda d: seen.append(d))

        self.assertIs(sub.host_node, host_node)
        sub.callBack('data')
        self.assertEqual(seen, ['data'])

    def test_ros2bridge_newPub_newSub(self):
        bridge = ros_bridge.ROS2Bridge('n', 10)
        bridge.r = 5

        pub = bridge.newPub('topic', MagicMock())
        self.assertIsInstance(pub, ros_bridge.ROS2Pub)

        sub = bridge.newSub('topic2', MagicMock(), lambda d: d)
        self.assertIsInstance(sub, ros_bridge.ROS2Sub)
        self.assertIs(sub.host_node, bridge)

    def test_ros2bridge_spin(self):
        bridge = ros_bridge.ROS2Bridge('n', 10)
        bridge.spin()
        self.fake_rclpy.spin.assert_called_once_with(bridge)
        self.fake_rclpy.shutdown.assert_called_once()

    def test_getRosHandle_getMsgHandle_ros2(self):
        self.assertIs(ros_bridge.getRosHandle(2), self.fake_rclpy)
        self.assertIs(ros_bridge.getMsgHandle(2), self.fake_r2msg)

    def test_makeROSBridge_ros2(self):
        bridge = ros_bridge.makeROSBridge('n', 10, ros_ver=2, exargs=['--foo'])
        self.fake_rclpy.init.assert_called_once_with(args=['--foo'])
        self.assertIsInstance(bridge, ros_bridge.ROS2Bridge)


if __name__ == '__main__':
    unittest.main()
