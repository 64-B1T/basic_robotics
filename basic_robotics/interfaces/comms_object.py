"""Base Class for Communications Bridges in Basic-Robotics."""

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
        self.comm_handle = None
        self.open = False
        self.last_rx_success = True
        self.last_tx_success = True
        self.last_rx_data = None

    def getRxSuccess(self) -> bool:
        """
        Get success of last data receiving attempt.

        Returns:
            bool: data read success
        """        
        return self.last_rx_success
    
    def getTxSuccess(self) -> bool:
        """
        Get success of last data writing attempt.

        Returns:
            bool: data write success
        """        
        return self.last_tx_success

    def sendData(self, data) -> bool:   # pragma: no cover
        """
        Send data.

        Args:
            data: data to be sent
        Returns:
            bool: message send success
        """
        return False

    def getData(self):  # pragma: no cover
        """
        Receive data.

        Returns:
            Any: Message Data
        """        
        return None

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

    def openCom(self) -> bool:   # pragma: no cover
        """
        Open a Communications Channel.

        Returns:
            bool: Success of Opening the Channel
        """        
        self.open = True
        return True

    def closeCom(self) -> None:
        """
        Close a Communications Channel.

        Returns:
            bool: Success of Closing the Channel
        """
        if self.comm_handle is not None and self.open:
            self.comm_handle.close()
            self.open = False
            return True
        return False