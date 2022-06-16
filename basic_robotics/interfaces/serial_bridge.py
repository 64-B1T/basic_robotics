"""Handle Serial Communication as a CommsObject."""
import time

import serial

from .comms_object import CommsObject


class SerialObject(CommsObject):
    """Create a new CommsObject that handles Serial Communication."""

    def __init__(self, name : str, port : str = "COM1", baud : int = 9600) -> 'SerialObject':
        """
        Initialize a new Serial CommsObject.

        Args:
            name (str): name of the object
            port (str, optional): port to select. Defaults to "COM1".
            baud (int, optional): Baud rate. Defaults to 9600.

        Returns:
            SerialObject: New Serial CommsObject
        """        
        super().__init__(name, "Serial")
        self.port = port
        self.baud = baud
        self.comm_handle = None

    def openCom(self) -> bool:
        """
        Open a Communications Channel.

        Returns:
            bool: Success of Opening the Channel
        """
        if not self.open:
            self.open = True
            self.comm_handle = serial.Serial(self.port, self.baud)
            return True 
        return False

    def sendData(self, message : str) -> int:
        """
        Send a message.

        Args:
            message: message to be sent
        Returns:
            int: bytes written. 0 is failure.
        """
        if not self.open:
            self.last_tx_success = False
            return
        bwr = 0
        try:
            bwr = self.comm_handle.write(message)
        except:
            bwr = self.comm_handle.write(message.encode('utf-8'))
        if bwr > 0:
            self.last_tx_success = True
        else:
            self.last_tx_success = False
        return self.last_tx_success

    def getData(self, sleeptime : float = .2) -> tuple[str, float]:
        """
        Receive a Message.

        Args:
            sleeptime (float, optional): amount of time to sleep before checking for message
        Returns:
            Any: Message Data
            bool: Message Receive Success
        """ 
        time.sleep(sleeptime)
        msg = self.comm_handle.read(self.comm_handle.in_waiting)
        if not self.open:
            self.last_rx_success = False
            return ""
        try:
            msg = msg.decode('utf-8')
        except:
            if len(msg) == 0:
                self.last_rx_success = False
                return ""
        self.last_rx_success = True 
        self.last_rx_data = msg
        return msg

    def setPort(self, port : str) -> None:
        """
        Set the serial port.

        Args:
            port (str): port to be set
        """
        self.port = port

    def setBaud(self, baud : int) -> None:
        """
        Set the baud rate.

        Args:
            baud (int): baud rate to be set
        """
        self.baud = baud

    def getPort(self) -> str:
        """
        Return the set port.

        Returns:
            str: port
        """
        return self.port

    def getBaud(self) -> int:
        """
        Return the baud rate.

        Returns:
            int: baud
        """
        return self.baud
