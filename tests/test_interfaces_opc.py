import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock

from basic_robotics.interfaces import opc_bridge


class test_interfaces_opc_not_ready(unittest.TestCase):
    """python-opcua is not installed in this environment, so these tests
    exercise the module's real not-ready state (READY == False)."""

    def test_opc_not_ready_flag(self):
        self.assertFalse(opc_bridge.READY)

    def test_opc_client_construction_short_circuits(self):
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        # __init__ returns before calling super().__init__(), so none of the
        # Comms/CommsObject state ever gets set on the instance
        self.assertFalse(hasattr(client, 'endpoints'))
        self.assertFalse(hasattr(client, 'client_handle'))


def _install_fake_opcua():
    fake_module = types.ModuleType('opcua')
    fake_module.Client = MagicMock(name='Client')
    fake_module.Node = MagicMock(name='Node')
    fake_ua = types.SimpleNamespace(
            AttributeIds=types.SimpleNamespace(Value='Value'),
            DataValue=MagicMock(name='DataValue', side_effect=lambda v: ('DataValue', v)))
    fake_module.ua = fake_ua
    sys.modules['opcua'] = fake_module
    importlib.reload(opc_bridge)
    return fake_module


def _remove_fake_opcua():
    del sys.modules['opcua']
    importlib.reload(opc_bridge)


class test_interfaces_opc_with_fake_opcua(unittest.TestCase):

    def setUp(self):
        self.fake_opcua = _install_fake_opcua()

    def tearDown(self):
        _remove_fake_opcua()

    def test_opc_ready_flag_true(self):
        self.assertTrue(opc_bridge.READY)

    def test_opc_client_init_creates_client(self):
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        self.fake_opcua.Client.assert_called_with('opc.tcp://localhost:4840')
        self.assertFalse(client.open)
        self.assertEqual(client.endpoints, {})

    def test_opc_client_openCom_closeCom_default(self):
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        self.assertTrue(client.openCom())
        client.client_handle.connect.assert_called_once()
        self.assertTrue(client.open)

        self.assertTrue(client.closeCom())
        client.client_handle.disconnect.assert_called_once()
        self.assertFalse(client.open)

    def test_opc_client_newComPort_forwards_type(self):
        # regression test: newComPort must forward `type` to Comms.newComPort,
        # otherwise Comms.newComPort raises TypeError for a missing `type` arg
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        client.newComPort('portA', 'Serial', port='COM1', baud=9600)

        endpoint = client.endpoints['portA']
        self.assertEqual(type(endpoint).__name__, 'SerialObject')
        self.assertEqual(endpoint.getPort(), 'COM1')
        self.assertEqual(endpoint.getBaud(), 9600)

    def test_opc_client_newComPort_opc_endpoint(self):
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        client.newComPort('ep1', 'OPCEndpoint', ['Objects', 'MyVar'])

        endpoint = client.endpoints['ep1']
        self.assertIsInstance(endpoint, opc_bridge.OPCUA_Endpoint)
        self.assertEqual(endpoint.opc_path, ['Objects', 'MyVar'])
        self.assertIs(endpoint.client_handle, client.client_handle)

    def test_opc_client_openCom_closeCom_named_port(self):
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        client.endpoints['portA'] = MagicMock()

        result_open = client.openCom('portA')
        client.endpoints['portA'].openCom.assert_called_once()
        self.assertIs(result_open, client.endpoints['portA'].openCom.return_value)

        result_close = client.closeCom('portA')
        client.endpoints['portA'].closeCom.assert_called_once()
        self.assertIs(result_close, client.endpoints['portA'].closeCom.return_value)

    def test_opc_client_setSecurity_while_closed(self):
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        client.setSecurity('Basic256', 'cert.pem', 'key.pem')

        client.client_handle.set_security_string.assert_called_once_with(
                'Basic256,SignAndEncrypt,cert.pem,key.pem')
        client.client_handle.connect.assert_not_called()
        client.client_handle.disconnect.assert_not_called()

    def test_opc_client_setSecurity_while_open_reopens(self):
        client = opc_bridge.OPCUA_Client('opc.tcp://localhost:4840')
        client.openCom()
        client.client_handle.reset_mock()

        client.setSecurity('Basic256', 'cert.pem', 'key.pem', mode=opc_bridge.SIGN)

        client.client_handle.disconnect.assert_called_once()
        client.client_handle.set_security_string.assert_called_once_with(
                'Basic256,Sign,cert.pem,key.pem')
        client.client_handle.connect.assert_called_once()
        self.assertTrue(client.open)

    def test_opc_endpoint_getData(self):
        client_handle = MagicMock()
        endpoint = opc_bridge.OPCUA_Endpoint('ep', ['Objects', 'MyVar'], client_handle)

        chain = client_handle.get_root_node.return_value.get_child.return_value
        chain.get_value.return_value = 42

        result = endpoint.getData()

        client_handle.get_root_node.return_value.get_child.assert_called_with(['Objects', 'MyVar'])
        self.assertEqual(result, 42)
        self.assertTrue(endpoint.last_tx_success)

    def test_opc_endpoint_sendData(self):
        client_handle = MagicMock()
        endpoint = opc_bridge.OPCUA_Endpoint('ep', ['Objects', 'MyVar'], client_handle)

        result = endpoint.sendData(3.14)

        chain = client_handle.get_root_node.return_value.get_child.return_value
        chain.set_attribute.assert_called_once_with('Value', ('DataValue', 3.14))
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
