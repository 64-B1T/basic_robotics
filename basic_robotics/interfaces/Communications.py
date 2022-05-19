import socket
import serial
import time

class CommsObject:
    """
    Base Class for a variety of communications types with standard interfaces
    """
    def __init__(self, name = "", type = ""):
        """
        Initializes the class
        Args:
            name: String- name of the object
            type: String- type of the object
        Returns:
            comms object
        """
        self.name = name
        self.type = type
        self.forwardList = []

    def sendMessage(self):
        """
        Sends a message
        """
        return True

    def recvMessage(self, sleeptime=.2):
        """
        Receives a message
        Args:
            sleeptime: optional sleeptime float, if waiting for a response
        """
        return None, False

    def setName(self, name):
        """
        Sets the name of the object
        """
        self.name = name

    def getName(self):
        """
        Returns the name of the object
        """
        return self.name

    def openCom(self):
        """
        Opens the communications channel
        """
        return 0

    def closeCom(self):
        """
        Closes the communications channel
        """
        return 0

class SerialObject(CommsObject):
    """
    Handler for Serial Communication
    """
    def __init__(self, name, port = "COM1", baud = 9600):
        """
        Initializes a serial connection
        Args:
            name: String - name of the object
            port: Optional String, Defaults to COM1, port to select
            baud: Optional int, Defaults to 9600, baud rate to select
        Returns:
            CommsObject
        """
        super().__init__(name, "Serial")
        self.port = port
        self.baud = baud
        self.ser = None

    def openCom(self):
        """
        Opens the communications line
        """
        self.ser = serial.Serial(self.port, self.baud)

    def closeCom(self):
        """
        Closes the communications line
        """
        self.ser.close()

    def sendMessage(self, message):
        """
        Sends a message
        Args:
            message: data to be sent along the connection
        Returns:
            bytes written
        """
        bwr = 0
        try:
            bwr = self.ser.write(message)
        except:
            bwr = self.ser.write(message.encode('utf-8'))
        return bwr > 0

    def recvMessage(self, sleeptime = .2):
        """
        Receives a message
        Args:
            sleeptime: Optional Float - amount of time to sleep before checking for message
        Returns:
            msg: data retrieved, if any
            success: boolean for whether or not data was retrieved
        """
        time.sleep(.2)
        msg = self.ser.read(self.ser.in_waiting)
        try:
            msg = msg.decode('utf-8')
        except:
            if len(msg) == 0:
                return "", False
        return msg, True


    def setPort(self, port):
        """
        Sets the serial port
        Args:
            port: String - port to be set
        """
        self.port = port

    def setBaud(self, baud):
        """
        Sets the baud rate
        Args:
            baud: Int - Baud rate to be set
        """
        self.baud = baud

    def getPort(self):
        """
        Returns the set port
        Returns:
            String: port
        """
        return self.port

    def getBaud(self):
        """
        Returns the baud rate
        Returns:
            Int: baud
        """
        return self.baud

class UDPObject(CommsObject):

    def __init__(self, name, IP = "192.168.1.1", Port = 8000):
        super().__init__(name, "UDP")
        """
        Creates a UDP Communications instance
        Args:
            name: String - Name to give instance
            IP: String - IP address to bind to
            Port: Int - Port to bind to
        Returns:
            CommsObject instance
        """
        self.IP = IP
        self.Port = Port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bufferLen = 1024
        self.lastAddr = None

    def sendMessage(self, message):
        """
        Sends a message
        Args:
            message: data to be sent along the connection
        """
        print("Send Message: " + message)
        self.sock.sendto(message.encode('utf-8'), (self.IP, self.Port))

    def recvMessage(self):
        """
        Receives a message
        Returns:
            msg: data retrieved, if any
            success: boolean for whether or not data was retrieved
        """
        success = True
        data, addr = self.sock.recvfrom(self.bufferLen)
        if data == None:
            success = False
        else:
            self.lastAddr = addr
        return data, success

    def setIP(self, IP):
        """
        Sets the IP address of the comms handle
        Args:
            IP: String - IP to bind to
        """
        self.IP = IP

    def setPort(self, Port):
        """
        Sets the Port of the comms handle
        Args:
            Port: Int - Port to bind to
        """
        self.Port = Port

    def openCom(self):
        """
        Opens a communications object and binds
        """
        self.sock.bind((self.IP, self.Port))

    def closeCom(self):
        """
        Closes a communications object
        """
        self.sock.close()

    def setBufferLen(self, bufferLen):
        """
        Sets the buffer length
        """
        self.bufferLen = bufferLen

    def getIP(self):
        """
        Returns the bound IP Address
        Returns:
            String: IP
        """
        return self.IP

    def getPort(self):
        """
        Returns the bound port number
        Returns:
            Int: Port
        """
        return self.Port

    def getBufferLen(self):
        """
        Returns the buffer length
        Returns:
            Int: Buffer length
        """
        return self.bufferLen


class Communications:
    """
    Communications wrapper class for multiple communications objects
    """
    def __init__(self):
        """
        Initializes an empty communications object
        """
        self.commsObjects = []
        self.updateObjects = []

    def newComPort(self, name, type, args = []):
        """
        Creates a new named comm port and adds it to the register
        Args:
            name: String - name to identify new port
            type: String - type of port "UDP" or "Serial"
            args: Optional - List of arguments specific to comm port
        """
        newObj = None
        if type == "UDP":
            if len(args) == 2:
                newObj = UDPObject(name, args[0], args[1])
            else:
                newObj = UDPObject(name)
        elif type == "Serial":
            if (len(args) == 2):
                newObj = SerialObject(name, args[0], args[1])
            else:
                newObj = SerialObject(name)
        self.commsObjects.append(newObj)

    def openComm(self, name):
        for i in range(len(self.commsObjects)):
            if self.commsObjects[i].name == name:
                self.commsObjects[i].openCom()
                return

    def closeComm(self, name):
        for i in range(len(self.commsObjects)):
            if self.commsObjects[i].name == name:
                self.commsObjects[i].closeCom()
                return

    def sendMessage(self, name, message):
        """
        Send a message from a comm port with a specific name
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
        Receives a message from a comm port with a specific name
        Args:
            name: String - unique name of comm port
        Returns:
            data: Retrieved data
            success: Whether it was able to retrieve anything at all
        """
        for i in range(len(self.commsObjects)):
            if self.commsObjects[i].name == name:
                return self.commsObjects[i].recvMessage()
