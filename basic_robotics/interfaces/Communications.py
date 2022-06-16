"""
Implement communications from Basic-Robotics to UDP, TCP, and Serial connections.

This is a basic implementation of these communications systems, and byte packing and more advanced features are not included.
Also not included are any sort of multithreaded capabilities. Users who desire to use such capabilities would be 
best off performing their own implementations or extending these classes.
"""
from .comms_object import CommsObject
from .serial_bridge import SerialObject
from .udp_bridge import UDPObject
from ast import Str
from typing import Any

class Communications:
    """Communications wrapper class for multiple communications objects."""

    def __init__(self) -> 'Communications':
        """
        Create an empty communications object.

        Returns:
            Communications: new communications object
        """
        self.endpoints = {}
        self.forwarding = {}
        self.output_functions = {}
        self.input_functions = {}

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
        self.endpoints[name] = newObj

    def getCom(self, name : str) -> CommsObject:
        """
        Return a reference to a specified CommsObject.

        Args:
            name (str): Desired Comms Object

        Returns:
            CommsObject: Desired Comms Object
        """
        if name in self.endpoints:
            return self.endpoints[name]
        return None

    def openAll(self) -> None: 
        """Open All Com Ports."""
        for com_name in self.endpoints:
            self.getCom(com_name).openCom()
    
    def closeAll(self) -> None:
        """Close ALl Com Ports."""
        for com_name in self.endpoints:
            self.getCom(com_name).closeCom()

    def openCom(self, name : str) -> None:
        """
        Open a specified Communications line.

        Args:
            name (str): CommsObject name
        Returns:
            bool: Success of Opening the Channel
        """
        comms_obj = self.getCom(name)
        if comms_obj is not None:
            return comms_obj.openCom()
        return False 

    def closeCom(self, name : str) -> bool:
        """
        Close a specified Communications line.

        Args:
            name (str): CommsObject name.
        Returns:
            bool: Success of Closing the Channel
        """       
        comms_obj = self.getCom(name)
        if comms_obj is not None:
            return comms_obj.closeCom()
        return False

    def setDataSource(self, output_name : str, input_handle: Any) -> bool:
        """
        Set a function that produces a compatible data output as an automatic message source.

        Do not bind a function to both a data input and a data output.
        Args:
            output_name (str): data sender to use.
            input_handle (Any): input handle to use. Must produce one output.

        Returns:
            bool: Success on binding function.
        """        
        output_com = self.getCom(output_name)
        if output_com is None or input_handle is None: 
            return False
        if output_name in self.input_functions:
            if input_handle not in self.input_functions[output_name]:
                self.input_functions[output_name].append(input_handle)
                return True
            return False
        else:
            self.input_functions[output_name] = [input_handle]
        return True

    def setDataSink(self, input_name : str, output_handle : Any) -> bool:
        """
        Set an output function for received data to automatically populate.
        
        Do not bind a function to both a data input and a data output. 
        Args:
            input_name (str): Name of input data handle.
            output_handle (Any): function to be called. Must take only one argument.

        Returns:
            bool: Success with binding data output
        """        
        input_com = self.getCom(input_name)
        if input_com is None or output_handle is None: 
            return False
        if input_name in self.output_functions:
            if output_handle not in self.output_functions[input_name]:
                self.output_functions[input_name].append(output_handle)
                return True
            return False
        else:
            self.output_functions[input_name] = [output_handle]
        return True

    def deleteForwardingRule(self, input_name : str, output_name : str) -> bool:
        """
        Delete a previously established forwarding rule.

        Args:
            input_name (str): receiving com object
            output_name (str): destination com object

        Returns:
            bool: success in removing a forwarding rule.
        """
        output_com = self.getCom(output_name)
        if output_com is None: 
            return False
        if input_name in self.forwarding and output_com in self.forwarding[input_name]:
            self.forwarding[input_name].remove(output_com)
            return True
        return False

    def setForwardData(self, input_name : str, output_name : str) -> bool:
        """
        Set a forwarding rule, from one com object to another.
    
        If data is pulled from one com_port that is linked to another, 
        then the data will be automatically sent to the next com port.

        Args:
            input_name (str): receiving com object
            output_name (str): destination com object

        Returns:
            bool: Success in establishing forwarding rule.
        """        
        input_com = self.getCom(input_name)
        output_com = self.getCom(output_name)
        if input_com is None or output_com is None: 
            return False
        if input_name in self.forwarding:
            if output_com not in self.forwarding[input_name]:
                self.forwarding[input_name].append(output_com)
                return True
            return False
        else:
            self.forwarding[input_name] = [output_com]
        return True

    def _single_spin(self):
        """Execute a single spin."""
        for name in self.endpoints:
            this_com = self.getCom(name)
            if name in self.input_functions:
                for function in self.input_functions[name]:
                    this_com.sendData(function())
            if name in self.output_functions or name in self.forwarding:
                self.getData(name)

    def spin(self, spin_iterations = -1):
        """
        Spin the communications instance, executing all forwarding rules, input and output functions.

        if spin_iterations < 0, will spin infinitely.
        Args:
            spin_iterations (int, optional): spin iterations. Defaults to -1.
        """        
        if spin_iterations < 0:
            while True:
                self._single_spin()
        for i in range(spin_iterations):
            self._single_spin()

    def sendData(self, name, data):
        """
        Send data from a comm port with a specific name.

        Args:
            name: String - unique name of comm port
            message: Data to send
        """
        comms_obj = self.getCom(name)
        if comms_obj is not None:
            return comms_obj.sendData(data)

    def getData(self, name):
        """
        Receives data from a comm port with a specific name.

        Args:
            name: String - unique name of comm port
        Returns:
            data: Retrieved data
            success: Whether it was able to retrieve anything at all
        """
        comms_obj = self.getCom(name)
        if comms_obj is not None:
            rx_data = comms_obj.getData()
            if name in self.forwarding:
                for destination in self.forwarding[name]:
                    destination.sendData(rx_data)
            if name in self.output_functions:
                for destination_function in self.output_functions[name]:
                    destination_function(rx_data)
            return rx_data
        return None
