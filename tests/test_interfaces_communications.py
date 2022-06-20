import unittest
import numpy as np
from basic_robotics.interfaces.comms_core import Comms

class test_interfaces_communications(unittest.TestCase):

    def setUp(self):
        self.com = Comms()
    
    def tearDown(self):
        self.com.closeAll()

    def test_interfaces_communications_UDPCommunication(self):
        this_ip = '127.0.0.1'
        self.com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)
        self.com.newComPort('test_B', 'UDP', this_ip, 9001, 9000)

        ref_msg = 'THIS IS A TEST'
        self.com.sendData('test_A', ref_msg)
        rx_msg = self.com.getData('test_B')
        success = self.com.getCom('test_B').getRxSuccess()
        self.assertFalse(success)

        self.com.openCom('test_A')
        self.com.openCom('test_B')

        self.com.sendData('test_A', ref_msg)
        rx_msg = self.com.getData('test_B')
        success = self.com.getCom('test_B').getRxSuccess()
        self.assertTrue(success)
        self.assertEqual(ref_msg, rx_msg)

        self.com.closeCom('test_A')
        self.com.closeCom('test_B')


    def test_interfaces_communications_UDPGetSet(self):
        this_ip = '127.0.0.1'
        self.com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)

        test_com = self.com.getCom('test_A')

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
        this_ip = '127.0.0.1'
        self.com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)

        test_com = self.com.getCom('test_A')
        self.assertIsNotNone(test_com)
        test_com_2 = self.com.getCom('None')
        self.assertIsNone(test_com_2)

    def test_interfaces_communications_forwarding_rules(self):
        this_ip = '127.0.0.1'
        ref_msg = 'THIS IS A TEST'
        self.com.newComPort('test_A', 'UDP', this_ip, 9000, 9001)
        self.com.newComPort('test_B', 'UDP', this_ip, 9001, 9002)
        self.com.newComPort('test_C', 'UDP', this_ip, 9002, 9003)
        self.com.newComPort('test_D', 'UDP', this_ip, 9003, 9004)

        self.com.openAll()

        self.com.sendData('test_A', ref_msg)
        rx_msg = self.com.getData('test_B')
        self.assertEqual(ref_msg, rx_msg)

        self.com.sendData('test_B', ref_msg)
        rx_msg = self.com.getData('test_C')
        self.assertEqual(ref_msg, rx_msg)


        self.assertTrue(self.com.setForwardData('test_B', 'test_C'))
        self.assertFalse(self.com.setForwardData('test_N', 'test_C'))
        self.assertFalse(self.com.setForwardData('test_B', 'test_N'))
        self.assertFalse(self.com.setForwardData('test_N', 'test_N'))
        self.assertFalse(self.com.setForwardData('test_B', 'test_C'))
        self.assertTrue(self.com.setForwardData('test_B', 'test_D'))
        self.assertTrue(self.com.deleteForwardingRule('test_B', 'test_D'))
        self.assertFalse(self.com.deleteForwardingRule('test_B', 'test_D'))
        self.assertFalse(self.com.deleteForwardingRule('test_B', 'test_N'))
        self.assertFalse(self.com.deleteForwardingRule('test_N', 'test_B'))

        self.com.sendData('test_A', ref_msg)
        rx_msg = self.com.getData('test_B')

        success = self.com.getCom('test_B').getRxSuccess()
        self.assertTrue(success)

        rx_msg_2 = self.com.getData('test_D')

        success = self.com.getCom('test_D').getRxSuccess()
        self.assertTrue(success)
        
        
        
        self.assertTrue(success)
        self.assertEqual(ref_msg, rx_msg_2)

    def test_interfaces_communications_setDataSink(self):
        this_ip = '127.0.0.1'
        ref_msg = 'THIS IS A TEST'
        self.com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)
        self.com.newComPort('test_B', 'UDP', this_ip, 9001, 9002)

        self.com.openAll()

        def didGetData(rx_msg):
            self.assertEqual(ref_msg, rx_msg)

        def didGetData2(rx_msg):
            self.assertEqual(ref_msg, rx_msg)
        self.assertFalse(self.com.setDataSink('test_NotReal', didGetData))
        self.assertFalse(self.com.setDataSink('test_B', None))
        self.assertTrue(self.com.setDataSink('test_B', didGetData))
        self.assertFalse(self.com.setDataSink('test_B', didGetData))
        self.assertTrue(self.com.setDataSink('test_B', didGetData2))
        self.com.sendData('test_A', ref_msg)
        rx_msg = self.com.getData('test_B')
        self.assertEqual(ref_msg, rx_msg)

    def test_interfaces_communications_setDataSource(self):
        this_ip = '127.0.0.1'
        ref_msg = 'THIS IS A TEST'
        self.com.newComPort('test_A', 'UDP', ip = this_ip, rx_port = 9000, tx_port = 9001)
        self.com.newComPort('test_B', 'UDP', this_ip, 9001, 9002)

        self.com.openAll()

        def dataSource():
            return ref_msg
        
        def dataSource2():
            return ref_msg

        self.assertFalse(self.com.setDataSource('test_NotReal', dataSource))
        self.assertFalse(self.com.setDataSource('test_B', None))
        self.assertTrue(self.com.setDataSource('test_B', dataSource))
        self.assertFalse(self.com.setDataSource('test_B', dataSource))
        self.assertTrue(self.com.setDataSource('test_B', dataSource2))

        self.com.setDataSource('test_A', dataSource)
        self.com.spin(1)
        rx_msg = self.com.getData('test_B')
        self.assertEqual(ref_msg, rx_msg)

    def test_interfaces_communications_comNone(self):
        self.assertFalse(self.com.openCom('nOpe'))
        self.assertFalse(self.com.closeCom('nope'))
        self.assertFalse(self.com.sendData('Nope', 0))
        self.assertIsNone(self.com.getData('nope'))

    def test_interfaces_communications_spin(self):
        this_ip = '127.0.0.1'
        ref_msg = 'THIS IS A TEST'
        self.com.newComPort('test_A', 'UDP', this_ip, 9000, 9001)
        self.com.newComPort('test_B', 'UDP', this_ip, 9001, 9002)
        self.com.newComPort('test_C', 'UDP', this_ip, 9002, 9003)
        self.com.newComPort('test_D', 'UDP', this_ip, 9003, 9004)

        def dataSource():
            return ref_msg

        def didGetData(rx_msg):
            self.assertEqual(ref_msg, rx_msg)

        self.com.openAll()

        self.assertTrue(self.com.setDataSource('test_A', dataSource))
        self.assertTrue(self.com.setForwardData('test_B', 'test_C'))
        self.com.setDataSink('test_D', didGetData)
        

        self.com.spin(1)

        


