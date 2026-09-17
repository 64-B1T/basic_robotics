import unittest
from unittest.mock import MagicMock, patch

from basic_robotics.interfaces.serial_bridge import SerialObject


class FakeUndecodableBytes:
    """Stand-in for a bytes object whose decode always fails, with a controllable length."""

    def __init__(self, length):
        self._length = length

    def decode(self, encoding):
        raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'bad byte')

    def __len__(self):
        return self._length


class test_interfaces_serial(unittest.TestCase):

    def test_serial_openCom_creates_handle(self):
        with patch('basic_robotics.interfaces.serial_bridge.serial.Serial') as mock_serial:
            mock_handle = MagicMock()
            mock_serial.return_value = mock_handle
            obj = SerialObject('test', port='COM3', baud=115200)

            self.assertTrue(obj.openCom())
            mock_serial.assert_called_once_with('COM3', 115200)
            self.assertTrue(obj.open)
            self.assertIs(obj.comm_handle, mock_handle)

            # already open, second call should fail
            self.assertFalse(obj.openCom())
            mock_serial.assert_called_once()

    def test_serial_sendData_not_open(self):
        obj = SerialObject('test')
        obj.sendData('hello')
        self.assertFalse(obj.getTxSuccess())

    def test_serial_sendData_success(self):
        obj = SerialObject('test')
        obj.open = True
        obj.comm_handle = MagicMock()
        obj.comm_handle.write.return_value = 5

        result = obj.sendData('hello')

        self.assertTrue(result)
        self.assertTrue(obj.getTxSuccess())
        obj.comm_handle.write.assert_called_once_with('hello')

    def test_serial_sendData_encode_fallback(self):
        obj = SerialObject('test')
        obj.open = True
        obj.comm_handle = MagicMock()
        obj.comm_handle.write.side_effect = [TypeError('needs bytes'), 5]

        result = obj.sendData('hello')

        self.assertTrue(result)
        self.assertEqual(obj.comm_handle.write.call_count, 2)
        obj.comm_handle.write.assert_called_with('hello'.encode('utf-8'))

    def test_serial_sendData_zero_bytes_written(self):
        obj = SerialObject('test')
        obj.open = True
        obj.comm_handle = MagicMock()
        obj.comm_handle.write.return_value = 0

        result = obj.sendData('hello')

        self.assertFalse(result)
        self.assertFalse(obj.getTxSuccess())

    def test_serial_getData_not_open(self):
        # comm_handle is still None since openCom() was never called; the
        # not-open guard must short-circuit before touching comm_handle
        obj = SerialObject('test')
        result = obj.getData(sleeptime=0)
        self.assertEqual(result, "")
        self.assertFalse(obj.getRxSuccess())

    def test_serial_getData_success(self):
        obj = SerialObject('test')
        obj.open = True
        obj.comm_handle = MagicMock()
        obj.comm_handle.in_waiting = 5
        obj.comm_handle.read.return_value = b'hello'

        result = obj.getData(sleeptime=0)

        self.assertEqual(result, 'hello')
        self.assertTrue(obj.getRxSuccess())
        self.assertEqual(obj.last_rx_data, 'hello')

    def test_serial_getData_decode_failure_empty(self):
        obj = SerialObject('test')
        obj.open = True
        obj.comm_handle = MagicMock()
        obj.comm_handle.in_waiting = 0
        obj.comm_handle.read.return_value = FakeUndecodableBytes(0)

        result = obj.getData(sleeptime=0)

        self.assertEqual(result, "")
        self.assertFalse(obj.getRxSuccess())

    def test_serial_getData_decode_failure_nonempty(self):
        obj = SerialObject('test')
        obj.open = True
        obj.comm_handle = MagicMock()
        obj.comm_handle.in_waiting = 3
        raw = FakeUndecodableBytes(3)
        obj.comm_handle.read.return_value = raw

        result = obj.getData(sleeptime=0)

        # decode failed but data was non-empty, so the raw payload passes through
        self.assertIs(result, raw)
        self.assertTrue(obj.getRxSuccess())
        self.assertIs(obj.last_rx_data, raw)

    def test_serial_port_and_baud_accessors(self):
        obj = SerialObject('test', port='COM5', baud=57600)
        self.assertEqual(obj.getPort(), 'COM5')
        self.assertEqual(obj.getBaud(), 57600)

        obj.setPort('COM9')
        obj.setBaud(9600)
        self.assertEqual(obj.getPort(), 'COM9')
        self.assertEqual(obj.getBaud(), 9600)


if __name__ == '__main__':
    unittest.main()
