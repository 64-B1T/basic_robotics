"""
Implement communications from Basic-Robotics to UDP, TCP, and Serial connections.

This is a basic implementation of these communications systems, and byte packing and more advanced features are not included.
Also not included are any sort of multithreaded capabilities. Users who desire to use such capabilities would be 
best off performing their own implementations or extending these classes.
"""
from ast import Str
import socket
import time
from typing import Any

import serial


class CommsObject:
    """Base Class for a variety of communications types with standard interfaces."""

    def __init__(self, name : str = "CommObj", type : str = "UDP") -> 'CommsObject':
        """
        Create a new CommsObject.

        Args:
            name (str, optional): Name of this Comm Object. Defaults to "CommObj".
            type (str, optional): Type of this Comm Object. Defaults to "UDP".

        Returns:
            CommsObject: _description_
        """        
        self.name = name
        self.type = type
        self.forwardList = []
        self.comm_handle = None
        self.open = False

    def sendMessage(self, message : Any) -> bool:   # pragma: no cover
        """
        Send a message.

        Args:
            message: message to be sent
        Returns:
            bool: message send success
        """
        return False

    def recvMessage(self) -> tuple[Any, bool]:  # pragma: no cover
        """
        Receive a Message.

        Returns:
            Any: Message Data
            bool: Message Receive Success
        """        
        return None, False

    def setName(self, name : str) -> None:
        """
        Set the name of the object.
        
        Args:
            name (str): Name of the Comms Object
        """
        self.name = name

    def getName(self) -> str:
        """
        Return the name of the object.

        Returns:
            str: Name of the object
        """
        return self.name

    def openCom(self) -> None:   # pragma: no cover
        """Open the communications channel."""
        self.open = True
        pass

    def closeCom(self) -> None:
        """Close the communications channel."""
        if self.comm_handle is not None:
            self.comm_handle.close()
            self.open = False

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

    def openCom(self) -> None:
        """Open the communications line."""
        self.open = True
        self.comm_handle = serial.Serial(self.port, self.baud)

    def sendMessage(self, message : str) -> int:
        """
        Send a message.

        Args:
            message: message to be sent
        Returns:
            int: bytes written. 0 is failure.
        """
        if not self.open:
            return
        bwr = 0
        try:
            bwr = self.comm_handle.write(message)
        except:
            bwr = self.comm_handle.write(message.encode('utf-8'))
        return bwr > 0

    def recvMessage(self, sleeptime : float = .2) -> tuple[Str, float]:
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
            return "", False
        try:
            msg = msg.decode('utf-8')
        except:
            if len(msg) == 0:
                return "", False
        return msg, True

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

class UDPObject(CommsObject):
    """Create a new CommsObject that handles UDP Communication."""

    def __init__(self, name, ip = "192.168.1.1", rx_port = 8000, tx_port = 9000):
        """
        Create a new UDP CommsObject.

        Args:
            name (str): name of the object
            ip (str, optional): port to select. Defaults to "192.168.1.1".
            rx_port (int, optional): Network Receive. Defaults to 8000.
            tx_port (int, optional): Network Transmit. Defaults to 9000.
        Returns:
            CommsObject instance
        """
        super().__init__(name, "UDP")
        self.ip = ip
        self.rx_port = rx_port
        self.tx_port = tx_port
        self.bufferLen = 1024
        self.lastAddr = None

    def sendMessage(self, message : str, port : int = None) -> None:
        """
        Send a message.

        Args:
            message (Str): data to be sent along the connection
            port (int, Optional) : port to transmit. Defaults to None.
        """
        if not self.open:
            return
        if port is None:
            port = self.tx_port
        self.comm_handle.sendto(message.encode('utf-8'), (self.ip, port))

    def recvMessage(self) -> tuple[Any, bool]:
        """
        Receive a message.

        Returns:
            msg: data retrieved, if any
            success: boolean for whether or not data was retrieved
        """
        if not self.open:
            return None, False
        success = True
        data, addr = self.comm_handle.recvfrom(self.bufferLen)
        if data == None:
            success = False
        else:
            self.lastAddr = addr
        return data.decode('utf-8'), success

    def setIP(self, ip : str) -> None:
        """
        Set the IP address of the comms handle.

        Args:
            ip: String - ip to bind to
        """
        self.ip = ip

    def setRxPort(self, port : int) -> None:
        """
        Set the Rx port of the comms handle.

        Args:
            port: Int - port to bind to
        """
        self.rx_port = port

    def setTxPort(self, port : int) -> None:
        """
        Set the Tx port of the comms handle.

        Args:
            port: Int - port to bind to
        """
        self.tx_port = port

    def openCom(self) -> None:
        """Open the communications line."""
        self.open = True
        self.comm_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.comm_handle.bind((self.ip, self.rx_port))

    def setBufferLen(self, bufferLen : int) -> None:
        """
        Set the buffer length.

        Args:
            bufferLen (int) : length of the buffer in bytes
        """
        self.bufferLen = bufferLen

    def getIP(self) -> str:
        """
        Return the bound IP Address.

        Returns:
            String: ip
        """
        return self.ip

    def getRxPort(self) -> int:
        """
        Return the bound rx port number.

        Returns:
            Int: port
        """
        return self.rx_port
    
    def getTxPort(self) -> int:
        """
        Return the bound tx port number.

        Returns:
            Int: port
        """
        return self.tx_port

    def getBufferLen(self) -> int:
        """
        Return the buffer length.

        Returns:
            Int: Buffer length
        """
        return self.bufferLen


class Communications:
    """Communications wrapper class for multiple communications objects."""

    def __init__(self) -> 'Communications':
        """
        Create an empty communications object.

        Returns:
            Communications: new communications object
        """
        self.commsObjects = []
        self.updateObjects = []

    def newComPort(self, name : str, type : str, *args, **kwargs) -> None:
        """
        Create a new named comm port and adds it to the register.

        Args:
            name: String - name to identify new port
            type: String - type of port "UDP" or "Serial"
            args: Optional - List of arguments specific to comm port
        """
        newObj = None
        if type == "UDP":
            newObj = UDPObject(name, *args, **kwargs)
        elif type == "Serial":
            newObj = SerialObject(name, *args, **kwargs)
        self.commsObjects.append(newObj)

    def getCom(self, name : str) -> CommsObject:
        """
        Return a reference to a specified CommsObject.

        Args:
            name (str): Desired Comms Object

        Returns:
            CommsObject: Desired Comms Object
        """
        for i in range(len(self.commsObjects)):
            if self.commsObjects[i].name == name:
                return self.commsObjects[i]
        return None

    def openCom(self, name : str) -> None:
        """
        Open a specified Communications line.

        Args:
            name (str): CommsObject name
        """
        comms_obj = self.getCom(name)
        if comms_obj is not None:
            comms_obj.openCom()

    def closeCom(self, name : str) -> None:
        """
        Close a specified Communications line.

        Args:
            name (str): CommsObject name.
        """        
        comms_obj = self.getCom(name)
        if comms_obj is not None:
            comms_obj.closeCom()

    def sendMessage(self, name, message):
        """
        Send a message from a comm port with a specific name.

        Args:
            name: String - unique name of comm port
            message: Data to send
        """
        for i in range(len(self.commsObjects)):
            if self.commsObjects[i].name == name:
                self.commsObjects[i].sendMessage(message)
                return

    def recvMessage(self, name):
        """
        Receives a message from a comm port with a specific name.

        Args:
            name: String - unique name of comm port
        Returns:
            data: Retrieved data
            success: Whether it was able to retrieve anything at all
        """
        for i in range(len(self.commsObjects)):
            if self.commsObjects[i].name == name:
                return self.commsObjects[i].recvMessage()
