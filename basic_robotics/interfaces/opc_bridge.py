"""Handle communication with OPC Unified Automation Protocol."""

READY = True 
try:
    from opcua import Client, Node, ua
except ImportError:
    READY = False
from .comms_object import CommsObject
from .comms_core import Comms

SIGN = "Sign"
SIGN_AND_ENCRYPT = "SignAndEncrypt"

class OPCUA_Endpoint(CommsObject):
    """Create a new OPCUA Endpoint."""

    def __init__(self, name : str, opc_path : list[str], client_ref : 'Client'):
        """
        Initialize a new OPCUA Endpoint.

        Args:
            name (str): endpoint name
            opc_path (list[str]): endpoint path from root node
            client_ref (Client): reference to OPCUA Client
        """        
        super().__init__(name, 'OPCEndpoint')
        self.opc_path = opc_path
        self.client_handle = client_ref
    
    def getData(self) -> any:
        """
        Receive data.

        Returns:
            Any: Message Data
        """ 
        data = self.client_handle.get_root_node().get_child(self.opc_path).get_value()
        self.last_tx_success = True
        return data


    def sendData(self, data : any) -> bool:
        """
        Send data.

        Args:
            data: data to be sent
        Returns:
            bool: message send success
        """
        self.client_handle.get_root_node().get_child(
                self.opc_path).set_attribute(
                    ua.AttributeIds.Value, ua.DataValue(data))
        self.last_tx_success = True 
        return self.last_tx_success

class OPCUA_Client(Comms):
    """Wrap a OPCUA Client object to be compatible with Basic-Robotics Communications."""

    def __init__(self, end_point : str) -> 'OPCUA_Client':
        """
        Create a new OPCUA_Client.

        Args:
            end_point (str): OPC Server URL

        Returns:
            OPCUA_Client: New OPCUA_Client
        """        
        if not READY:
            print("python-opcua is not installed," +
                 "OPCUA_Client is not available. Please install python-opcua to proceed.")
            return
        super().__init__()
        self.client_handle = Client(end_point)
        self.open = False

    def setSecurity(self, security_policy : str, cert_file : str, private_key_file : str, 
            server_certificate_file : str = None, mode : str = SIGN_AND_ENCRYPT) -> None:
        """
        Set Security Settings for OPC Client.

        if the client is already open, this function will close the client,
        apply settings, and then reopen the client connection.
        Args:
            security_policy (str): security policy path
            cert_file (str): certificate path
            private_key_file (str): private key path
            server_certificate_file (str, optional): server certificate path. Defaults to None.
            mode (str, optional): policy mode. Sign or SignAndEncrypt. Defaults to SignAndEncrypt.
        """
        was_open = False
        if self.open:
            was_open = True
            self.closeCom()
        param_list = [y for y in [security_policy, mode, cert_file, 
            private_key_file, server_certificate_file] if y is not None]
        self.client_handle.set_security_string(','.join(param_list))
        if was_open:
            self.openCom()

    def newComPort(self, name : str, type : str, *args, **kwargs) -> None:
        """
        Create a new named comm port and adds it to the register.

        Args:
            name: String - name to identify new port
            type: String - type of port "UDP" or "Serial" or "OPCEndpoint"
            args: Optional - List of arguments specific to comm port
        """  
        if type == "OPCEndpoint":
            return self.addEndpoint(name, *args, **kwargs)
        super().newComPort(name, *args, **kwargs)

    def addEndpoint(self, reference_name : str, opc_path : list[str]):
        """
        Add an OPC Endpoint.

        Args:
            reference_name (str): easy reference name for the opc endpoint.
            opc_path ([str]]): opc path to the endpoint from the root node.
        """
        self.endpoints[reference_name] = OPCUA_Endpoint(reference_name, opc_path, self.client_handle)

    def openCom(self, name : str = "OPC") -> bool:
        """
        Open either a specific Com port, or the OPC Client itself.

        Args:
            name (str, optional): com port to open. Defaults to "OPC".
        Returns:
            bool: Success of Closing the Channel
        """         
        if name == "OPC":
            self.client_handle.connect()
            self.open = True
        else:
            return super().openCom(name)

    def closeCom(self, name : str = "OPC") -> bool:
        """
        Close either a specific Com port, or the OPC Client itself.

        Args:
            name (str, optional): Com port to close. Defaults to "OPC".
        Returns:
            bool: Success of Closing the Channel
        """         
        if name == "OPC":
            self.client_handle.disconnect()
            self.open = False
        else:
            return super().closeCom(name)

    
