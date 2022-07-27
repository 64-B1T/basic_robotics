"""Client functions for using the 3JS visualizer."""
import requests
import json
import time
from basic_robotics.general import fsr, tm
from basic_robotics.utilities.disp import disp
import numpy as np
import os

from sqlalchemy import true 

package_directory = os.path.dirname(os.path.abspath(__file__))


def newFloor(filename : str = 'ChFloor2.glb') -> dict:
    """
    Create a new floor.

    Args:
        filename (str, optional): Floor file. Defaults to 'ChFloor2.glb'.

    Returns:
        dict: dict representing the floor object.
    """    
    if filename == 'ChFloor2.glb':
        filename = package_directory + '\\' + filename
    return {"Key" : "Floor", "File" : filename, "Category" : "Model"}

def newMaterial(color : hex = 0x3238a8, transparent : bool = True, opacity : float = .5) -> dict:
    """
    Create a new material for an object primitive.

    Args:
        color (hexadecimal, optional): Color of the Object. Defaults to 0xff8800.
        transparent (bool, optional): transparency boolean. Defaults to True.
        opacity (float, optional): opacity between 0 and 1. Defaults to .5.

    Returns:
        dict: dict representing the new material.
    """    
    return {"Color" : color, "transparent" : transparent, "opacity" : opacity}

def newPrimitive(name : str = "CubeLet", file : str = "Cube3.glb", Category : str = "Model", 
        scale = [1.0, 1.0, 1.0], color : dict = None) -> dict:
    """
    Create a new primative for the visualizer.

    Args:
        name (str, optional): name of the object. Defaults to "CubeLet"
        file (str, optional): filename of the primitive. Defaults to "Cube3.glb".
        Category (str, optional): type of object. Defaults to "Model".
        scale (list[float], optional): scale of object dimensions XYZ. Defaults to [1.0, 1.0, 1.0].
        color (dict, optional): color dict of the object. Defaults to None.

    Returns:
        dict: New dict representing primitive.
    """    
    if color == None:
        color = newMaterial()
    return {"Key" : name, "File" : file, "Category" : Category, "Scale" : scale, "Material" : color}

def determineAxis(joint_location : tm, axis : 'np.ndarray[float]') -> tm:
    """
    Determine axis of rotation in global space of a joint.

    Args:
        joint_location (tm): joint location in global space
        axis (np.ndarray[float]): Axis of the joint pulled from the Arm model.

    Returns:
        tm: tm indicating rotation of joint.
    """    
    joint_rotation = tm([joint_location[3], joint_location[4], joint_location[5]])
    axis_unit = tm([axis[0], axis[1], axis[2], 0, 0, 0])
    axis_new = (joint_rotation @ axis_unit)[0:3]
    return axis_new.flatten()

class DrawClient:
    """Supervisor clieent for accessing 3JS visualizer."""

    def __init__(self, host  : str = '127.0.0.1', port : int = 5000) -> 'DrawClient':
        """
        Create a new DrawClient for the 3JS visualizer.

        Args:
            host (str, optional): Host IP of the visualizer. Defaults to '127.0.0.1'.
            port (int, optional): Port of the visualizer. Defaults to 5000.

        Returns:
            DrawClient: new Draw Client.
        """        
        self.host = host 
        self.port = port
        self.makeHostURL()
        self.json_header = {"Content-Type": "application/json"}
        self.ses = requests.Session()
        self.unnamed_registry = []
        self.unnamed_counter = 0

    def newName(self) -> str:
        """
        Allocate a new name for an object to use, if default name was not assigned.

        Returns:
            str : new name to use
        """
        self.unnamed_registry.append("Object_" + str(self.unnamed_counter))
        return self.unnamed_registry[-1]

    def deleteName(self, name : str) -> bool:
        """
        Delete a particular object from the default/unammed registry.

        Args:
            str (name): Object to remove.

        Returns:
            bool: Success of removal
        """        
        if name in self.unnamed_registry:
            self.unnamed_registry.remove(name)
            return self.delete({"Keys" : [name]})
        else:
            return False

    def deleteAllUnnamed(self) -> bool:
        """
        Delete all objects that were named by default.

        Returns:
            bool: Success of Default.
        """        
        success = self.delte({"Keys" : self.unnamed_registry})
        self.unnamed_registry = [] 
        self.unnamed_counter = 0
        return success

    def setPort(self, new_port : int) -> None:
        """
        Set a new host port.

        Args:
            new_port (int): new host port.
        """        
        self.port = new_port
        self.makeHostURL()

    def setHost(self, new_host : str) -> None:
        """
        Set a new host IP.

        Args:
            new_host (str): New Host IP.
        """        
        self.host = new_host
        self.makeHostURL()

    def makeHostURL(self) -> None:
        """Make the host url from the host IP and port saved."""
        self.url = 'http://' + self.host + ':' + str(self.port) + '/api/json'


    def detHost(self, host_alt : str = None) -> str:
        """
        Switch hosts between default and an alternate (if supplied).

        Args:
            host_alt (str, optional): Alternate host URL. Defaults to 'None'

        Returns:
            str: Host URL
        """        
        host_url = self.url
        if host_alt is not None:
            host_url = host_alt
        return host_url

    def tryDictMatch(self, tdict : dict, val : str, init : any) -> any:
        """
        Try and retrieve appropriate value from a dict.

        Args:
            tdict (dict): dict of interest.
            val (str): value key of interest.
            init (any): default value.

        Returns:
            any: return type of accessed dict or default.
        """
        t = init
        if val in tdict.keys():
            t = tdict[val]
        return t

    def makeFloor(self, floor_file = 'jsons/SceneFloor.json') -> None:
        """
        Make a floor.

        Args:
            floor_file (str, optional): optional floor file. Defaults to 'jsons/SceneFloor.json'.
        """
        if floor_file == 'jsons/SceneFloor.json':
            floor_file = package_directory + '\\' + floor_file
        self.sendFile(floor_file)

    def prepAggregated(self, dict_list) -> dict:
        """
        Turn a list of dicts into a single dict with named elements.

        Args:
            dictarray (list[dict]): Input list of dicts.

        Returns:
            dict: output dict.
        """        
        return {"Keys" : {k["Key"] : k for k in dict_list if k is not None}}

    def sendAggregated(self, dict_list, hostalt : str = None) -> bool:
        """
        Send an aggregated list of dicts to the visualizer.

        Args:
            dict_list (list[dict]): Input list of dicts.
            hostalt (str, optional): Alternate Host. Defaults to None.

        Returns:
            bool: _description_
        """        
        return self.send(self.prepAggregated(dict_list), "PUT", hostalt, addtime = True)

    def send(self, dat : dict, msgtype : str, hostalt : str = None, addtime : bool = False) -> bool:
        """
        Send a dict to the visualization server.

        Args:
            dat (dict): Dict to send
            msgtype (str): Type of message (PUT or POST)
            hostalt (str, optional): alternate host. Defaults to None.
            addtime (bool, optional): add network time to package. Defaults to False.

        Returns:
            bool: Success on placing message
        """        
        url = self.detHost(hostalt)
        if addtime:
            if "Key" in dat.keys():
                dat["UnixTime"] = time.time()
            elif "Keys" in dat.keys():
                for keyval in dat["Keys"].keys():
                    dat["Keys"][keyval]["UnixTime"] = time.time()

        if msgtype == "POST":
            nreq = self.ses.post(url = url, headers = self.json_header, json=dat)
        else:
            #print(msgtype)
            #print(dat)
            nreq = self.ses.put(url = url, headers = self.json_header, json=dat)

        if nreq.status_code == 200:
            return True
        else:
            print('failed')
            return False

    def sendFile(self, fname : str, host_alt : str = None, 
            add_time : bool = True, key_name : str = None) -> json:
        """
        Send a file to the server.

        Args:
            fname (str): File name to send.
            host_alt (str, optional): alternate host to send to. Defaults to None.
            add_time (bool, optional): add network time to package. Defaults to True.
            key_name (str, optional): key file name. Defaults to None.

        Returns:
            json: response json
        """        
        url = self.detHost(host_alt)
        with open(fname, 'r') as import_json:
            loaded = json.load(import_json)

        if "Key" not in loaded:
            loaded = {"Keys" : loaded}
            if add_time:
                for key in loaded["Keys"]:
                    loaded["Keys"][key]['UnixTime'] = time.time()
        else:
            if add_time:
                loaded['UnixTime'] = time.time()
        if key_name is not None and "Key" in loaded:
            loaded["Key"] = key_name

        response = self.ses.put(
            url, json=loaded, headers=self.json_header)
        res = response.json
        return res

    def prepTM(self, tmobj : tm, params : dict) -> dict:
        """
        Add a Tm to an outbound dict.

        Args:
            tmobj (tm): transform to add in.
            params (dict): dict to send out.

        Returns:
            dict: dict to send out.
        """        
        tvec = tmobj.gTM().T.ravel().tolist()
        params["Matrix"] = tvec
        return params

    def sendTM(self, tmobj : tm, params : dict) -> bool:
        """
        Send a transformation matrix complete dict to the visualizer.

        Args:
            tmobj (tm): transformation
            params (dict): dict needing new position

        Returns:
            bool: success of placement.
        """        
        params = self.prepTM(tmobj, params)
        return self.send(params, "PUT", addtime = True)

    def sendAxes(self, loc : tm, name : str = "Frame", scale : float = 1.0) -> bool:
        """
        Send Axes to the client.

        Args:
            loc (tm): Origin of the axes
            name (str, optional): name of the axes object. Defaults to "Frame".
            scale (float, optional): scale (default is 1m axes). Defaults to 1.0.

        Returns:
            bool: success of placement.
        """        
        params = {
            name : {
                "Frame":1, 
                "Scale":scale, 
                "Matrix" : loc.gTM().T.reshape((16,1)).tolist()
            }
        }
        return self.send(params, "PUT", addtime=True)

    def sendLine(self, line_tms, host_alt : str = None, 
            key : str = None, type : str = "Arrow", 
            color = [0., 1., 0.], linewidth : float = .0025) -> str:
        """Send a line between two transforms to the visualization server.

        Args:
            line_tms (list[tm, tm]): list containing two tms, that is the line to be drawn.
            host_alt (str, optional): Alternate Host. Defaults to None.
            key (str, optional): name of the line. Defaults to None.
            type (str, optional): type of line ("Line" or "Arrow"). Defaults to "Arrow".
            color (list[float], optional): color of line. Defaults to [0., 1., 0.].
            linewidth (float, optional): line width. Defaults to .0025.

        Returns:
            str: name of placed line.
        """            
        url = self.detHost(host_alt)
        #unit, distance = fsr.getUnitVec(tm1, tm2, return_dist=True)
        if key is None:
            key = self.newName()
        if type == "Arrow":
            distance = fsr.distance(line_tms[0], line_tms[1])
            unit = fsr.getUnitVec(line_tms[0], line_tms[1])
            lineparams = {"Key" : key, "LineParameters" : {
                    "ArrowBase": line_tms[0][0:3].flatten().tolist(),
                    "ArrowDirection": unit[0:3].flatten().tolist(),
                    "ArrowLength": distance,
                    "Arrow" : 1, "Color" : color,
                    "lineWidth" : linewidth},
                    "Segments":[[]]}
        else:
            lineparams = {"Key" : key, "LineParameters" : {"Color" : color, "lineWidth" : linewidth}}
            segments = []
            for ttm in line_tms:
                segments.append(ttm[0:3].flatten().tolist())
            lineparams["Segments"] = segments
        self.send(lineparams, "PUT", url, True)
        return key

    def get(self, params : dict = {}, hostalt : str = None) -> dict:
        """
        Get a dict currently hosted on the visualization server.

        Args:
            params (dict, optional): parameters to match. Defaults to {}.
            hostalt (str, optional): alternate host. Defaults to None.

        Returns:
            dict: dict matching desired request.
        """        
        hosturl = self.detHost(hostalt)

        kind = self.tryDictMatch(params, "Kind", "")

        if kind == "" and len(params) == 0:
            kind = "Latest"

        thekeys    = self.tryDictMatch(params, "Keys", {})
        category   = self.tryDictMatch(params, "Category", "")
        index      = self.tryDictMatch(params, "Index", -1)
        getrange   = self.tryDictMatch(params, "GetRange", "")
        valuerange = self.tryDictMatch(params, "ValueRange", {})

        q = "?"
        if len(category) > 0:
            q = q + "Category-" + category + "&"

        if kind == "Complete":
            q = q + "Complete=1&"

        elif kind == "Latest":
            q = q + "Latest=1&"

        elif len(thekeys) > 0:
            q = q + "Key"
            for keyval in thekeys:
                q = q + keyval + ","
            q = q[:-1]
            q = q + "&"

            if index != -1:
                q = q + "Index=" + str(index) + "&"
            elif len(getrange) > 0:
                q = q + "GetRange=" + str(getrange) + "&ValueRange="
                for val in valuerange:
                    q = q + str(val) + ","
                q = q[:-1]
                q = q + "&"
        q.replace(" ", "%20")
        #print(hosturl + "/api/json" + q)
        nreq = self.ses.get(url = hosturl + q, headers = self.json_header)
        data = nreq.json()

        if nreq.status_code == 200:
            return data, True
        else:
            return {}, False

    def delete(self, params : dict = {}, hostalt : str = None) -> bool:
        """
        Delete something from the server.

        Args:
            params (dict, optional): params to match. Defaults to {} (DeleteAll).
            hostalt (str, optional): Alternate host. Defaults to None.

        Returns:
            bool: success of deletion
        """        
        hosturl = self.detHost(hostalt)
        #print(hosturl)
        thekeys  = self.tryDictMatch(params, "Keys", {})
        category = self.tryDictMatch(params, "Category", "")

        if len(thekeys) == 0 and len(category) == 0:
             delall = {'DeleteAll' : 1}
             #print(json.dumps(delall))
             nreq = self.ses.put(hosturl, json.dumps(delall), headers = self.json_header)
        elif len(thekeys) == 1:
             delkey = {"DeleteKey" : thekeys[0]}
             nreq = self.ses.put(hosturl, json.dumps(delkey), headers = self.json_header)
        elif len(thekeys) > 1:
             delkey = {"DeleteKey" : thekeys}
             nreq = self.ses.put(hosturl, json.dumps(delkey), headers = self.json_header)
        elif len(category) > 1:
             delcat = {"DeleteCategory" : category}
             nreq = self.ses.put(hosturl, json.dumps(delcat), headers = self.json_header)

        if nreq.status_code == 200:
            return True
        else:
            return False

class VisPlot:
    """A Simple Holding Class for Drawing on the 3JS Server."""

    def __init__(self, name : str, client : 'DrawClient' = None) -> 'VisPlot':
        """
        Create a new Visplot object.

        Args:
            name (str): Name of the object
            client (DrawClient, optional): Reference to active DrawClient. Defaults to None.

        Returns:
            VisPlot: New Visplot object.
        """        
        if client is None:
            client = DrawClient()
        if name is None:
            name = client.newName()
        self.keys = []
        self.name = name
        self.c = client

    def delete(self):
        """Delete this Object from the Visualizer."""
        self.c.delete({"Keys" : self.keys})
        self.c.delete({"Category" : self.name})

    def initialize(self):
        """Initialize this Vis Plot Instance."""
        pass

    def update(self, send = False):
        """Update visualized robot to match current configuration."""
        pass

class PrimitivePlot(VisPlot):
    """A Simple Holding Class for Drawing on the 3JS Server. Models a primitive shape."""

    def __init__(self, name : str, transform_origin : tm, 
            dimensions, client : 'DrawClient' = None, 
            c : str = 'grey', a : float = 0.1) -> 'PrimitivePlot':
        """
        Generate a new primitive plot instance.

        Args:
            name (str): Name of this new object.
            transform_origin (tm): Transformation of this object in global space.
            dimensions (list[float]): Dimensions of this object.
            client (DrawClient, optional): Reference to DrawClient object. Defaults to None.
            c (str, optional): Color of this object. Defaults to 'grey'.
            a (float, optional): Transparency of this object. Defaults to 0.1.

        Returns:
            PrimitivePlot: _description_
        """        
        super().__init__(name, client)
        self.transform_origin = transform_origin
        self.dimensions = dimensions
        self.color = c 
        self.transparency = a
        self.object = None
        self.initialize()

    def setTM(self, transform_origin : tm, send : bool = False):
        """
        Set a new Transformation for this object.

        Args:
            transform_origin (tm): Object transformation
            send (bool, optional): Whether or not to send object. Defaults to False.
        Returns:
            dict reference to this object.
        """        
        self.transform_origin = transform_origin
        self.initialize()
        return self.update(send)

    def update(self, send : bool = False):
        """
        Update the object in the visualizer.

        Args:
            send (bool, optional): Whether or not to send object. Defaults to False.
        Returns:
            dict reference to this object.
        """ 
        if send: 
            self.c.send(self.c.prepTM(self.transform_origin, self.object) , "PUT")
        return [self.object]
        
class CubePlot(PrimitivePlot):
    """A Simple Holding Class for Drawing on the 3JS Server. Models a cubic shape."""

    def __init__(self, transform_origin : tm, dimensions, 
            client : 'DrawClient' = None, c : str = 'grey', 
            a : float = 0.1, name : str = None) -> 'CubePlot':
        """
        Generate a new CubePlot Object.

        Args:
            transform_origin (tm): Transform of the object.
            dimensions (list[float]): Dimesnions of the object.
            client (DrawClient, optional): Reference to Drawclient. Defaults to None.
            c (str, optional): color of the object. Defaults to 'grey'.
            a (float, optional): transparency of the object. Defaults to 0.1.
            name (str, optional): Name of the object. Defaults to None.

        Returns:
            CubePlot: A new CubePlot
        """        
        super().__init__(name, transform_origin, dimensions, client, c, a)

    def initialize(self):
        """Initialize a new cube."""
        self.keys.append(self.name)
        self.object = newPrimitive(
            name = self.name,
            file = "internal/models/Cube3.glb",
            scale = self.dimensions,
            color = newMaterial(self.color, opacity = self.transparency)
        )
        self.update(True)

class AxesPlot(PrimitivePlot):
    """A Simple Holding Class for Drawing on the 3JS Server. Models an Axes object."""

    def __init__(self, transform_origin : tm, 
            client : 'DrawClient' = None, scale : float = 1.0, name : str = None):
        """
        Generate a new AxesPlot Object.

        Args:
            transform_origin (tm): Transform of the object.
            client (DrawClient, optional): Reference to Drawclient. Defaults to None.
            scale (float, optional): scale of the axes. Defaults to 1.0
            name (str, optional): Name of the object. Defaults to None.

        Returns:
            AxesPlot : new AxesPlot object
        """      
        super().__init__(name, transform_origin, scale, client, None, None)
    
    def initialize(self):
        """Initialize a new Axes object."""
        self.keys.append(self.name)
        self.object = {
            self.name : {
                "Frame":1, 
                "Scale": self.dimensions, 
                "Matrix" : self.transform_origin.gTM().T.reshape((16,1)).tolist()
            }
        }

class TubePlot(PrimitivePlot):
    """A Simple Holding Class for Drawing on the 3JS Server. Models a tube."""

    def __init__(self, transform_origin : tm, 
            height : float, radius : float, client : 'DrawClient', 
            color : str = 'blue', a : float = 0.1, name : str = None) -> 'TubePlot':
        """
        Generate a new TubePlot Object.

        Args:
            transform_origin (tm): Transform of the object.
            height (float): height of the cylinder.
            radius (float): radius of the cylinder.
            client (DrawClient, optional): Reference to Drawclient. Defaults to None.
            c (str, optional): color of the object. Defaults to 'grey'.
            a (float, optional): transparency of the object. Defaults to 0.1.
            name (str, optional): Name of the object. Defaults to None.

        Returns:
            TubePlot : new TubePlot object
        """           
        super().__init__(name, transform_origin, [height, radius], client, color, a)

    def initialize(self):
        """Initialize a new tube."""
        self.keys.append(self.name)
        self.object = newPrimitive(
            name = self.name,
            file = "internal/models/Cylinder.glb",
            scale = [self.dimensions[1], self.dimensions[1], self.dimensions[0]],
            color = newMaterial(self.color, opacity = self.transparency)
        )
        self.object = {"Key" : self.name,
                    "Primitive":"Cylinder", 
                    "Scale":[self.dimensions[1], self.dimensions[1], self.dimensions[0]], 
                    "Matrix" : self.transform_origin.gTM().T.reshape((16,1)).tolist(),
                    "Material" : newMaterial(self.color, opacity = self.transparency)
                    }
        self.update(True)



class RobotPlot(VisPlot):
    """General modelling class for a robot in the 3JS visualizer system."""

    def __init__(self, name : str, bot, client : 'DrawClient' = None) -> 'RobotPlot':
        """
        Initialize a new RobotPlot instance.

        Args:
            name (str): Name of this robot for reference.
            bot (Robot): Reference to the robot object desired to draw.
            client (DrawClient, optional): DrawClient instance. If none, make a new one. Defaults to None.

        Returns:
            RobotPlot: New RobotPlot Instance.
        """        
        super().__init__(name, client)
        self.bot = bot    

class ArmPlot(RobotPlot):
    """Implement Creation of Serial Arm in the Visualizer."""    

    def __init__(self, name : str, arm, client : 'DrawClient' = None):
        """
        Initialize a new ArmPlot instance.

        Args:
            name (str): Name of this robot for reference.
            arm (Arm): Reference to the robot object desired to draw.
            client (DrawClient, optional): DrawClient instance. If none, make a new one. Defaults to None.

        Returns:
            ArmPlot: New ArmPlot Instance.
        """  
        super().__init__(name, arm, client)
        self.links = []
        self.bot_data_tms = []
        self.joint_dia = .2
        self.link_end_ind = 0
        self.keys = []
        self.cyl_type = True
        self.initialize()
        

    def initialize(self):
        """Initialize this ArmPlot Instance."""
        Dims = np.copy(self.bot._link_dimensions).T
        links = []
        joints = []
        if self.bot._vis_props is not None:
            self.cyl_type = False
            for i in range(len(self.bot._vis_props)):
                if self.bot._vis_props[i].geo_type == 'mesh':
                    newp = newPrimitive(
                        name = self.bot.link_names[i],
                        file = self.bot._vis_props[i].file_name,
                        Category = self.name,
                        scale = [1.0, 1.0, 1.0]
                    )
                    links.append(newp)
                    newj = newPrimitive(
                        name = self.name + '_joint_' + str(i),
                        file = "internal/models/CylRed.glb",
                        Category = self.name,
                        scale = [0.01, 0.01, .001])
                    joints.append(newj)
                    self.bot_data_tms.append(None)
                    self.bot_data_tms.append(None)
        else:
            for i in range(len(Dims)):
                self.keys.append(self.name+str(i))
                newp = newPrimitive(
                    name = self.name + '_link_' + str(i),
                    file = "internal/models/CylGrey.glb",
                    Category = self.name,
                    scale = [Dims[i,0], Dims[i,1], Dims[i,2]])
                newj = newPrimitive(
                    name = self.name + '_joint_' + str(i),
                    file = "internal/models/CylRed.glb",
                    Category = self.name,
                    scale = [self.joint_dia, self.joint_dia, .1])
                links.append(newp)
                joints.append(newj)
                self.bot_data_tms.append(None)
                self.bot_data_tms.append(None)
        arm_data = links
        self.link_end_ind = len(links)
        arm_data.extend(joints)
        self.bot_data = arm_data

    def update(self, send = False):
        """Update visualized robot to match current configuration."""
        if not self.cyl_type:
            poses = self.bot.getJointTransforms()
            for i in range(len(poses) - 1):
                self.bot_data_tms[i] = self.c.prepTM(poses[i], self.bot_data[i])
        else:
            poses = self.bot.getJointTransforms()[1:]
            self.bot_data_tms[0] = self.c.prepTM(tm(), self.bot_data[0])
            for i in range(len(poses) -1):
                if i < len(poses) - 2:
                    Tp = fsr.tmInterpMidpoint(poses[i], poses[i+1])
                    T = fsr.lookAt(Tp, poses[i+1] + tm([0.0000001, 0, 0, 0, 0, 0]))
                    self.bot_data_tms[i+1] = self.c.prepTM(T, self.bot_data[i])
                elif not self.cyl_type:
                    self.bot_data_tms[i+1] = self.c.prepTM(poses[i], self.bot_data[i])
                if (self.bot.joint_axes[0, i] == 1):
                    joint_tm = poses[i] @ tm([0, 0, 0, 0, np.pi/2, 0])
                elif (self.bot.joint_axes[1, i] == 1):
                    joint_tm = poses[i] @ tm([0, 0, 0, np.pi/2, 0, 0])
                elif (self.bot.joint_axes[2, i] == 1):
                    joint_tm = poses[i] @ tm([0, 0, 0, 0, 0, np.pi])
                else:
                    joint_tm = poses[i].copy()
                    ax = determineAxis(joint_tm, self.bot.joint_axes[0:3,i]) * np.pi/2
                    joint_tm = poses[i] @ tm([0, 0, 0, ax[0], ax[1], ax[2]])
                self.bot_data_tms[i+self.link_end_ind] = self.c.prepTM(joint_tm, self.bot_data[i+self.link_end_ind])
        if send:
            self.c.sendAggregated(self.bot_data_tms)
        else:
            return self.bot_data_tms

class SPPlot(RobotPlot):
    """General modelling class for a stewart platform in the 3JS visualizer system."""

    def __init__(self, name: str, sp, client : 'DrawClient' = None):
        """
        Initialize a new SPPlot instance.

        Args:
            name (str): Name of this sp for reference.
            sp (SP): Reference to the robot object desired to draw.
            client (DrawClient, optional): DrawClient instance. If none, make a new one. Defaults to None.

        Returns:
            SPPlot: New SPPlot Instance.
        """  
        super().__init__(name, sp, client)
        self.legs = []
        self.links = [None, None]
        self.lwidth = .05
        self.keys = []
        self.leg_bot_mag = tm([0, 0, self.bot.leg_ext_min/2, 0, 0, 0])
        self.leg_top_mag = tm([0, 0, (self.bot.leg_ext_max - self.bot.leg_ext_min + .01)/2, 0, 0, 0])
        self.top_plate_mod = tm([0, 0, -self.bot.top_plate_thickness/2, 0, 0, 0])
        self.bottom_plate_mod = tm([0, 0, self.bot.bottom_plate_thickness/2, 0, 0, 0])
        self.out_list = [
            None, None, #Plates
            None, None, None, None, None, None, #Top Actuators
            None, None, None, None, None, None, #Bottom Actuators
        ]
        self.initialize()

    def legBot(self, i : int) -> tm:
        """
        Get bottom part of an actuator position and rotation in global space.

        Args:
            i (int): index of leg.

        Returns:
            tm: position of bottom part of actuator.
        """        
        bleg = tm([self.bot._bottom_joints_space[0,i], self.bot._bottom_joints_space[1,i],self.bot._bottom_joints_space[2,i],0,0,0])
        tleg = tm([self.bot._top_joints_space[0,i], self.bot._top_joints_space[1,i],self.bot._top_joints_space[2,i],0,0,0])
        #return fsr.adjustRotationToMidpoint(bleg, bleg, tleg, mode = 1) @ self.legmag
        return fsr.lookAt(bleg, tleg) @ self.leg_bot_mag

    def legTop(self, i : int) -> tm:
        """
        Get top part of an actuator position and rotation in global space.

        Args:
            i (int): index of leg.

        Returns:
            tm: position of top part of actuator.
        """    
        bleg = tm([self.bot._bottom_joints_space[0,i], self.bot._bottom_joints_space[1,i],self.bot._bottom_joints_space[2,i],0,0,0])
        tleg = tm([self.bot._top_joints_space[0,i], self.bot._top_joints_space[1,i],self.bot._top_joints_space[2,i],0,0,0])
        return fsr.lookAt(tleg, bleg) @ self.leg_top_mag


    def initialize(self):
        """Initialize this ArmPlot Instance."""
        bottom = newPrimitive(name=self.name + "B",
            file = "internal/models/CylGrey.glb",
            Category = self.name,
            scale = [self.bot._outer_bottom_radius*2, self.bot._outer_bottom_radius*2, max(.01, self.bot.bottom_plate_thickness)])
        top = newPrimitive(name=self.name + "T",
            file = "internal/models/CylGrey.glb",
            Category = self.name,
            scale = [self.bot._outer_top_radius*2, self.bot._outer_top_radius*2, max(.01, self.bot.top_plate_thickness)])
        self.links[0] = bottom
        self.links[1] = top
        #self.c.sendTM(self.bot.getBottomT() @ tm([0, 0, self.bot.bottom_plate_thickness/2, 0, 0, 0]), bottom)
        #self.c.sendTM(self.bot.getTopT() @ tm([0, 0, -self.bot.top_plate_thickness/2, 0, 0, 0]), top)

        for i in range(6):
            lt = newPrimitive(name = self.name + str(i) + 't',
                file = "internal/models/CylRed.glb",
                Category = self.name,
                scale = [self.lwidth/2, self.lwidth/2, (self.bot.leg_ext_max - self.bot.leg_ext_min + 0.01)])
            lb = newPrimitive(name = self.name + str(i) + 'b',
                file = "internal/models/CylGrey.glb",
                Category = self.name,
                scale = [self.lwidth, self.lwidth, self.bot.leg_ext_min])
            self.legs.append([lb, lt])

    def update(self, send = False):
        """Update visualized robot to match current configuration."""
        self.out_list[0] = self.c.prepTM(self.bot.getBottomT() @ self.bottom_plate_mod, self.links[0])
        self.out_list[1] = self.c.prepTM(self.bot.getTopT() @ self.top_plate_mod, self.links[1])
        for i in range(6):
            self.out_list[i+2] = self.c.prepTM(self.legTop(i), self.legs[i][1])
            self.out_list[i+8] = self.c.prepTM(self.legBot(i), self.legs[i][0])
        if send:
            self.c.sendAggregated(self.out_list)
        else:
            return self.out_list