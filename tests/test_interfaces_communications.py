import unittest
import numpy as np
from basic_robotics.interfaces.communications import Communications

class test_interfaces_communications(unittest.TestCase):

    def test_interfaces_communications_UDPCommunication(self):
        com = Communications()
        this_ip = '127.0.0.1'
        com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)
        com.newComPort('test_B', 'UDP', this_ip, 9001, 9000)

        ref_msg = 'THIS IS A TEST'
        com.sendMessage('test_A', ref_msg)
        rx_msg, success = com.recvMessage('test_B')
        self.assertFalse(success)

        com.openCom('test_A')
        com.openCom('test_B')

        com.sendMessage('test_A', ref_msg)
        rx_msg, success = com.recvMessage('test_B')
        self.assertTrue(success)
        self.assertEqual(ref_msg, rx_msg)

        com.closeCom('test_A')
        com.closeCom('test_B')


    def test_interfaces_communications_UDPGetSet(self):
        com = Communications()
        this_ip = '127.0.0.1'
        com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)

        test_com = com.getCom('test_A')

        self.assertEqual(test_com.getIP(), this_ip)
        self.assertEqual(test_com.getName(), 'test_A')
        self.assertEqual(test_com.getTxPort(), 9001)
        self.assertEqual(test_com.getRxPort(), 9000)
        test_com.setBufferLen(1000)

        test_com.setIP('192.168.1.6')
        test_com.setRxPort(8001)
        test_com.setTxPort(8002)
        test_com.setName('newName')

        self.assertEqual(test_com.getIP(), '192.168.1.6')
        self.assertEqual(test_com.getName(), 'newName')
        self.assertEqual(test_com.getTxPort(), 8002)
        self.assertEqual(test_com.getRxPort(), 8001)
        self.assertEqual(test_com.getBufferLen(), 1000)

    def test_interfaces_communications_getCom(self):
        com = Communications()
        this_ip = '127.0.0.1'
        com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)

        test_com = com.getCom('test_A')
        self.assertIsNotNone(test_com)
        test_com_2 = com.getCom('None')
        self.assertIsNone(test_com_2)

