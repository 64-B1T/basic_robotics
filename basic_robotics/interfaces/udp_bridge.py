"""Handle UDP Communications as a CommsObject."""
import socket
from .comms_object import CommsObject

class UDPObject(CommsObject):
    """Create a new CommsObject that handles UDP Communication."""

    def __init__(self, name, ip : str = "192.168.1.1", rx_port : int = 8000, 
        tx_port : int = 9000, timeout : float = 0.1) -> 'UDPObject':
        """
        Create a new UDP CommsObject.

        Args:
            name (str): name of the object
            ip (str, optional): port to select. Defaults to "192.168.1.1".
            rx_port (int, optional): Network Receive. Defaults to 8000.
            tx_port (int, optional): Network Transmit. Defaults to 9000.
            timeout (float, optional): Network timeout. Defaults to 0.1
        Returns:
            CommsObject instance
        """
        super().__init__(name, "UDP")
        self.ip = ip
        self.rx_port = rx_port
        self.tx_port = tx_port
        self.bufferLen = 1024
        self.last_source_address = None
        self.timeout = timeout

    def sendData(self, message : str, port : int = None) -> None:
        """
        Send a message.

        Args:
            message (Str): data to be sent along the connection
            port (int, Optional) : port to transmit. Defaults to None.
        """
        if not self.open:
            self.last_tx_success = False
            return
        if port is None:
            port = self.tx_port
        self.comm_handle.sendto(message.encode('utf-8'), (self.ip, port))
        self.last_tx_success = True
        return self.last_tx_success

    def getData(self) -> tuple[any, bool]:
        """
        Receive a message.

        Returns:
            msg: data retrieved, if any
            success: boolean for whether or not data was retrieved
        """
        if not self.open:
            self.last_rx_success = False
            return None
        try:
            data, addr = self.comm_handle.recvfrom(self.bufferLen)
        except TimeoutError:
            data = None
        if data == None:
            self.last_rx_success = None
            return None
        else:
            self.last_source_address = addr
            self.last_rx_success = True
            self.last_rx_data = data.decode('utf-8')
            return self.last_rx_data

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

    def openCom(self) -> bool:
        """
        Open a Communications Channel.

        Returns:
            bool: Success of Opening the Channel
        """
        if not self.open:
            self.open = True
            self.comm_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.comm_handle.settimeout(self.timeout)
            self.comm_handle.bind((self.ip, self.rx_port))
            return True 
        return False

    def closeCom(self) -> bool:
        """
        Close a Communications Channel.

        Returns:
            bool: Success of Closing the Channel
        """
        if self.comm_handle is not None and self.open == True:
            self.comm_handle.shutdown(socket.SHUT_RDWR)
            self.comm_handle.close()
            self.open = False
            return True 
        return False

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