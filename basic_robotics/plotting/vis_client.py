import requests
import json
import time
from basic_robotics.general import fsr, tm
from basic_robotics.utilities.disp import disp
import numpy as np

def new_floor(filename):
    return {"Key" : "Floor", "File" : filename, "Category" : "Model"}

def new_material(color = 0xff8800, transparent = True, opacity = .5):
    return {"color" : color, "transparent" : transparent, "opacity" : opacity}

def new_primitive(name = "CubeLet", file = "Cube3.glb", Category = "Model", Scale = [1.0, 1.0, 1.0], color = None):
    if color == None:
        color = new_material()
    return {"Key" : name, "File" : file, "Category" : Category, "Scale" : Scale, "Material" : color}

def determineAxis(joint_location, axis):
        joint_rotation = tm([joint_location[3], joint_location[4], joint_location[5]])
        axis_unit = tm([axis[0], axis[1], axis[2], 0, 0, 0])
        axis_new = (joint_rotation @ axis_unit)[0:3]
        #if sum(abs(axis)) > 0:
        #    axis_new = abs(axis_new)
        #else:
        #    axis_new = abs(axis_new) * -1
        return axis_new.flatten()

class ArmPlot:
    def __init__(self, name, arm, client=None, json_folder = "jsons/", model_folder = "models/"):
        if client == None:
            client = DrawClient()
        self.name = name
        self.c = client
        self.arm = arm
        self.links = []
        self.joints = []
        self.link_tms = []
        self.arm_data_tms = []
        self.json_folder = json_folder
        self.model_folder = model_folder
        self.joint_dia = .2
        self.link_end_ind = 0
        self.keys = []
        self.Initialize()

    def Initialize(self):
        poses = self.arm.getJointTransforms()
        Dims = np.copy(self.arm.link_dimensions).T
        dofs = self.arm.screw_list.shape[1]
        links = []
        joints = []
        if self.arm._vis_props is not None:
            for i in range(len(self.arm._vis_props)):
                if self.arm._vis_props[i][0] == 'msh':
                    newp = new_primitive(
                        name = self.arm.link_names[i],
                        file = self.arm._vis_props[i][2][0],
                        Category = self.name,
                        Scale = [1.0, 1.0, 1.0]
                    )
        for i in range(len(Dims)):
             self.keys.append(self.name+str(i))
             newp = new_primitive(
                name = self.name + '_link_' + str(i),
                file = "models/CylGrey.glb",
                Category = self.name,
                Scale = [Dims[i,0], Dims[i,1], Dims[i,2]])
             newj = new_primitive(
                name = self.name + '_joint_' + str(i),
                file = "models/CylRed.glb",
                Category = self.name,
                Scale = [self.joint_dia, self.joint_dia, .1])
             links.append(newp)
             joints.append(newj)
             self.arm_data_tms.append(None)
             self.arm_data_tms.append(None)
        arm_data = links
        self.link_end_ind = len(links)
        arm_data.extend(joints)
        self.arm_data = arm_data



    def Update(self, send = False):
        poses = self.arm.getJointTransforms()
        self.arm_data_tms[0] = self.c.PrepTM(tm(), self.arm_data[0])
        disp(poses)
        for i in range(len(poses)):
            if i < len(poses) - 1:
                Tp = fsr.tmInterpMidpoint(poses[i], poses[i+1])
                T = fsr.lookAt(Tp, poses[i+1] + tm([0.0000001, 0, 0, 0, 0, 0]))
                self.arm_data_tms[i+1] = self.c.PrepTM(T, self.arm_data[i])
            print(len(self.arm_data_tms), self.link_end_ind, i)
            if (self.arm.joint_axes[0, i] == 1):
                joint_tm = poses[i] @ tm([0, 0, 0, 0, np.pi/2, 0])
            elif (self.arm.joint_axes[1, i] == 1):
                joint_tm = poses[i] @ tm([0, 0, 0, np.pi/2, 0, 0])
            elif (self.arm.joint_axes[2, i] == 1):
                joint_tm = poses[i] @ tm([0, 0, 0, 0, 0, np.pi])
            else:
                joint_tm = poses[i].copy()
                ax = determineAxis(joint_tm, self.arm.joint_axes[0:3,i]) * np.pi/2
                joint_tm = poses[i] @ tm([0, 0, 0, ax[0], ax[1], ax[2]])
            disp(joint_tm)
            self.arm_data_tms[i+self.link_end_ind] = self.c.PrepTM(joint_tm, self.arm_data[i+self.link_end_ind])
        if send:
            self.c.SendAggregated(self.arm_data_tms)
        else:
            return self.arm_data_tms


    def Delete(self):
        self.c.Delete({"Keys" : self.keys})
        self.c.Delete({"Category" : self.name})

class SPPlot:
    def __init__(self, name, sp, client = None, json_folder = "jsons/", model_folder = "models/"):
        if client == None:
            client = DrawClient()
        self.name = name
        self.c = client
        self.sp = sp
        self.legs = []
        self.links = [None, None]
        self.lwidth = .05
        self.json_folder = json_folder
        self.model_folder = model_folder
        self.keys = []
        self.leg_bot_mag = tm([0, 0, self.sp.leg_ext_min/2, 0, 0, 0])
        self.leg_top_mag = tm([0, 0, (self.sp.leg_ext_max - self.sp.leg_ext_min + .01)/2, 0, 0, 0])
        self.top_plate_mod = tm([0, 0, -self.sp.top_plate_thickness/2, 0, 0, 0])
        self.bottom_plate_mod = tm([0, 0, self.sp.bottom_plate_thickness/2, 0, 0, 0])
        self.out_list = [
            None, None, #Plates
            None, None, None, None, None, None, #Top Actuators
            None, None, None, None, None, None, #Bottom Actuators
        ]
        self.Initialize()

    def legBot(self, i):
        bleg = tm([self.sp._bottom_joints_space[0,i], self.sp._bottom_joints_space[1,i],self.sp._bottom_joints_space[2,i],0,0,0])
        tleg = tm([self.sp._top_joints_space[0,i], self.sp._top_joints_space[1,i],self.sp._top_joints_space[2,i],0,0,0])
        #return fsr.adjustRotationToMidpoint(bleg, bleg, tleg, mode = 1) @ self.legmag
        return fsr.lookAt(bleg, tleg) @ self.leg_bot_mag

    def legTop(self, i):
        bleg = tm([self.sp._bottom_joints_space[0,i], self.sp._bottom_joints_space[1,i],self.sp._bottom_joints_space[2,i],0,0,0])
        tleg = tm([self.sp._top_joints_space[0,i], self.sp._top_joints_space[1,i],self.sp._top_joints_space[2,i],0,0,0])
        return fsr.lookAt(tleg, bleg) @ self.leg_top_mag


    def Initialize(self):
        bottom = new_primitive(name=self.name + "B",
            file = "models/CylGrey.glb",
            Category = self.name,
            Scale = [self.sp._outer_bottom_radius*2, self.sp._outer_bottom_radius*2, max(.01, self.sp.bottom_plate_thickness)])
        top = new_primitive(name=self.name + "T",
            file = "models/CylGrey.glb",
            Category = self.name,
            Scale = [self.sp._outer_top_radius*2, self.sp._outer_top_radius*2, max(.01, self.sp.top_plate_thickness)])
        self.links[0] = bottom
        self.links[1] = top
        #self.c.SendTM(self.sp.getBottomT() @ tm([0, 0, self.sp.bottom_plate_thickness/2, 0, 0, 0]), bottom)
        #self.c.SendTM(self.sp.getTopT() @ tm([0, 0, -self.sp.top_plate_thickness/2, 0, 0, 0]), top)

        for i in range(6):
            lt = new_primitive(name = self.name + str(i) + 't',
                file = "models/CylRed.glb",
                Category = self.name,
                Scale = [self.lwidth/2, self.lwidth/2, (self.sp.leg_ext_max - self.sp.leg_ext_min + 0.01)])
            lb = new_primitive(name = self.name + str(i) + 'b',
                file = "models/CylGrey.glb",
                Category = self.name,
                Scale = [self.lwidth, self.lwidth, self.sp.leg_ext_min])
            self.legs.append([lb, lt])

    def Update(self, send = False):
        self.out_list[0] = self.c.PrepTM(self.sp.getBottomT() @ self.bottom_plate_mod, self.links[0])
        self.out_list[1] = self.c.PrepTM(self.sp.getTopT() @ self.top_plate_mod, self.links[1])
        for i in range(6):
            self.out_list[i+2] = self.c.PrepTM(self.legTop(i), self.legs[i][1])
            self.out_list[i+8] = self.c.PrepTM(self.legBot(i), self.legs[i][0])
        if send:
            self.c.SendAggregated(self.out_list)
        else:
            return self.out_list

class DrawClient:

    def __init__(self, url = 'http://127.0.0.1:5000/api/json'):
        self.url = url
        self.json_header = {"Content-Type": "application/json"}
        self.ses = requests.Session()

    def detHost(self, host_alt):
        host_url = self.url
        if host_alt != None:
            host_url = host_alt
        return host_url

    def tryDictMatch(self, tdict, val, init):
        t = init
        if val in tdict.keys():
            t = tdict[val]
        return t


    def PrepAggregated(self, dictarray):
        return {"Keys" : {k["Key"] : k for k in dictarray}}

    def SendAggregated(self, dictarray, hostalt = None):
        hosturl = self.detHost(hostalt)
        return self.Send(self.PrepAggregated(dictarray), "PUT", addtime = True)

    def Send(self, dat, msgtype, hostalt = None, addtime = False):
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

    def sendFile(self, fname, host_alt = None, add_time = True, key_name = None):
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

    def PrepTM(self, tmobj, params):
        tvec = tmobj.gTM().T.ravel().tolist()
        params["Matrix"] = tvec
        return params

    def SendTM(self, tmobj, params):
        params = self.PrepTM(tmobj, params)
        return self.Send(params, "PUT", addtime = True)

    def SendAxes(self, loc, name = "Frame", scale=1.0):
        params = {name : {"Frame":1, "Scale":scale, "Matrix" : loc.gTM().T.reshape((16,1)).tolist()}}
        return self.Send(params, "PUT", addtime=True)

    def SendLine(self, line_tms, host_alt = None, key = None, type = "Arrow", color = [0., 1., 0.], linewidth = .0025):
        url = self.detHost(host_alt)
        #unit, distance = fsr.getUnitVec(tm1, tm2, return_dist=True)
        if key is None:
            key = type
        if type == "Arrow":
            distance = fsr.distance(line_tms[0], line_tms[1])
            unit = fsr.getUnitVec(line_tms[0], line_tms[1])
            lineparams = {"Key" : key, "LineParameters" : {
                    "ArrowBase": line_tms[0][0:3].flatten().tolist(),
                    "ArrowDirection": unit[0:3].flatten().tolist(),
                    "ArrowLength": distance,
                    "Arrow" : 1, "color" : color,
                    "lineWidth" : linewidth},
                    "Segments":[[]]}
        else:
            lineparams = {"Key" : key, "LineParameters" : {"color" : color, "lineWidth" : linewidth}}
            segments = []
            for ttm in line_tms:
                segments.append(ttm[0:3].flatten().tolist())
            lineparams["Segments"] = segments
        self.Send(lineparams, "PUT", url, True)

    def Get(self, params = {}, hostalt = None):
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

    def Delete(self, params = {}, hostalt = None):
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
