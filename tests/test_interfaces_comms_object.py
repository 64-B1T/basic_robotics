import unittest
from unittest.mock import MagicMock

from basic_robotics.interfaces.comms_object import CommsObject


class test_interfaces_comms_object(unittest.TestCase):

    def test_comms_object_defaults(self):
        obj = CommsObject()
        self.assertEqual(obj.name, 'CommObj')
        self.assertEqual(obj.type, 'UDP')
        self.assertIsNone(obj.comm_handle)
        self.assertFalse(obj.open)
        self.assertTrue(obj.getRxSuccess())
        self.assertTrue(obj.getTxSuccess())
        self.assertIsNone(obj.last_rx_data)

    def test_comms_object_name(self):
        obj = CommsObject('MyName', 'Serial')
        self.assertEqual(obj.getName(), 'MyName')
        self.assertEqual(obj.type, 'Serial')
        obj.setName('NewName')
        self.assertEqual(obj.getName(), 'NewName')

    def test_comms_object_rx_tx_success_flags(self):
        obj = CommsObject()
        obj.last_rx_success = False
        self.assertFalse(obj.getRxSuccess())
        obj.last_tx_success = False
        self.assertFalse(obj.getTxSuccess())

    def test_comms_object_closeCom_no_handle(self):
        obj = CommsObject()
        # comm_handle is None, so closing should fail regardless of open flag
        obj.open = True
        self.assertFalse(obj.closeCom())

    def test_comms_object_closeCom_not_open(self):
        obj = CommsObject()
        obj.comm_handle = MagicMock()
        obj.open = False
        self.assertFalse(obj.closeCom())
        obj.comm_handle.close.assert_not_called()

    def test_comms_object_closeCom_success(self):
        obj = CommsObject()
        handle = MagicMock()
        obj.comm_handle = handle
        obj.open = True
        self.assertTrue(obj.closeCom())
        handle.close.assert_called_once()
        self.assertFalse(obj.open)


if __name__ == '__main__':
    unittest.main()
