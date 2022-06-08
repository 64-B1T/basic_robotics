import numpy as np
import scipy.linalg as ling
import scipy.stats as st
import scipy as sci
import random

from ..general import tm, fsr

class Scene: #The Whole Scene, Cameras and Objects
    def __init__(self):
        self.objList = []
        self.camList = []
        self.observed = []
        self.grid = None

    def newSceneObj(self, objList, tol = .01, name = "OBJ"):
        self.objList.append(SceneObj(objList, tol, name))

    def addSceneObj(self, obj):
        self.objList.append(obj)

    def addCam(self, obj):
        self.camList.append(obj)

    def CalculateGrid(self):
        self.grid = np.zeros((len(self.observed), len(self.observed)))
        for i in range(len(self.observed)):
            for j in range(i, len(self.observed)): #Lists are mirrored, so it's ok
                if (i == j):
                    continue
                d = fsr.Distance(self.observed[i].cPos, self.observed[j].cPos)
                self.grid[i,j] = d
                self.grid[j,i] = d
        return self.grid

    def GetObjPositionsFromPoints(self):
        rawList = []
        #Identify All Objects
        for camera in self.camList:
            for ls in self.objList:
                obsv = camera.getScene(ls.objs)
                for obs in obsv:
                    rawList.append(Observed(obs[0], obs[1], camera))
        if len(rawList) == 0:
            return None

        refinedList = []
        refinedList.append(rawList[0])
        k = 1
        while k < len(rawList):
            p = False
            for j in range(len(refinedList)):
                if(refinedList[j].eq(rawList[k])):
                    refinedList[j].sync(rawList[k])
                    p = True
            if not p:
                refinedList.append(rawList[k])
            k+=1
        retList = []
        for o in refinedList:
            if(o.inView < 2):
                continue
            o.CalcAvgGuess()
            retList.append(o)
        self.observed = retList
        return retList

class Observed: #A Point That is Viewed
    def __init__(self, pixpos, q, camera, tol = .0003):
        self.pixpos = pixpos
        self.camera = camera
        self.q = q
        self.tol = tol
        self.gl = self.collateVector()
        self.inView = 1
        self.camerasViewing = [camera]
        self.gls = [self.gl]
        self.draftPoses = []
        self.cPos = None

    def CalcAvgGuess(self):
        numPos = len(self.draftPoses)
        sumPos = tm()
        for i in range(numPos):
            sumPos = sumPos + self.draftPoses[i]
        sumPos = sumPos/numPos
        self.cPos = sumPos
        return sumPos

    def AvgWDist(self, x):
        ts = tm()
        tls = []
        for i in range(len(x)):
            tp = self.camerasViewing[i].CamT @ (self.gls[i] * x[i])
            ts = ts + tp
            tls.append(tp)
        ts = ts / len(x)
        davg = 0
        for i in range(len(x)):
            davg+=fsr.Distance(tp, tls[i])
        davg = davg/len(x)
        return davg, ts

    def CalcAvgCoalated(self):
        res = lambda x : self.AvgWDist(x)[0]
        x0 = np.zeros((len(self.gls),))
        xs = sci.optimize.minimize(res, x0, method = "SLSQP")
        xs = xs.x
        return self.AvgWDist(xs)[1]


    def collateVector(self):
        p1 = self.camera.CamT
        ps = self.camera.getLocalPos(self.pixpos)
        p2 = p1 @ tm([ps[0], ps[1], 1, 0, 0, 0])
        gla = (fsr.GlobalToLocal(p1, p2)[0:3])/fsr.Distance(p1, p2)
        gl = tm([gla[0], gla[1], gla[2], 0, 0, 0])
        return gl

    def eq(self, other):
        if(self.camera == other.camera):
            return False
        res = lambda x : fsr.Distance(self.camera.CamT @ (self.gl * x[0]), other.camera.CamT @ (other.gl * x[1]))
        x0 = np.zeros((2,))
        bnds = ((0, None), (0, None))
        xs = sci.optimize.minimize(res, x0, method = "SLSQP", bounds = bnds)
        xs = xs.x
        dist = res(xs)
        print(dist)
        if (dist < self.tol):
            if(self.inView == 1):
                t = self.camera.CamT @ (self.gl * xs[0])
                self.draftPoses.append(tm([t[0], t[1], t[2], 0, 0, 0]))
            t = other.camera.CamT @ (other.gl * xs[1])
            self.draftPoses.append(tm([t[0], t[1], t[2], 0, 0, 0]))
            return True
        return False

    def sync(self, other):
        self.inView+=1
        self.camerasViewing.append(other.camera)
        self.gls.append(other.gl)


class SceneObj: #Object within a scene that needs to be reconstructed
    def __init__(self, objList, tol = .01, name = "OBJ"):
        if isinstance(objList, list):
            self.sz = len(objList)
            self.objs = objList
            self.lead = objList[0]
        else:
            print("Generating Additional Points")
            self.sz = 4
            self.objs = [objList]
            self.objs.append(objList @ tm([random.uniform(0,2), 0, 0, 0, 0, 0]))
            self.objs.append(objList @ tm([0, random.uniform(0,2), 0, 0, 0, 0]))
            self.objs.append(objList @ tm([0, 0, random.uniform(0,2), 0, 0, 0]))
            self.lead = objList
        self.rels = []
        self.dists = []
        self.getRels()
        self.tol = tol
        self.name = name
        self.min = 0

    def getRels(self):
        for i in range(self.sz-1):
            self.rels.append(fsr.GlobalToLocal(self.lead, self.objs[i+1]))
            self.dists.append(fsr.Distance(self.lead, self.objs[i+1]))

    def adjRot(self, x, y):
        temp = y
        temp[3] = x[0]
        temp[4] = x[1]
        temp[5] = x[2]
        sum = 0
        for i in range(self.sz - 1):
            d = fsr.Distance(fsr.LocalToGlobal(temp, self.rels[i]), self.objs[i])
            sum+=d
        return sum

    def testAll(self, x, ob):
        ylist = []
        for i in range(len(ob)):
            ylist.append(adjRot(x, ob[i]))
        self.min = ylist.index(min(ylist))
        return min(ylist)

    def getPos(self, scene):
        found = 0
        foundL = []
        for i in range(len(scene.observed)-1):
            obs = scene.observed[i]
            for j in range(i + 1, len(scene.observed)):
                obs2 = scene.observed[j]
                if (obs == obs2):
                    continue
                if (fsr.Distance(obs.cPos, obs2.cPos) < self.tol):
                    found+=1
                    found.append(obs.cPos)
                if found == self.sz:
                    break
            if found == self.sz:
                break
        x0 = np.zeros((found,))
        res = lambda x : self.testAll(x, foundL)
        xs = sci.optimize.minimize(res, x0, method = "SLSQP")
        self.cPos = foundL[self.min]
        return self.cPos


class Camera:


    def __init__(self, aptx, apty, pixX, pixY, maxX, maxY, sigma, camT, id = 0):
        self.id = id
        self.focx = aptx
        self.focy = apty
        self.pixX = pixX
        self.pixY = pixY
        self.sigma = sigma^2
        self.maxX = maxX
        self.maxY = maxY
        self.CamT = camT.copy()
        self.fs = self.getFrameSize()

    def getLocalPos(self, pix):
        sz = self.getFrameSize()
        return [-sz[0] + pix[0]/self.focx, -sz[1] + pix[1]/self.focy]

    def getFrameSize(self, sz = 1):
        return [self.maxX/self.focx*sz/2, self.maxY/self.focy*sz/2]

    def getScene(self, listPoints):
        observed = []
        print("Scanning " + str(len(listPoints)) + " points")
        for i in range(len(listPoints)):
            a, b, c = self.getPhoto(listPoints[i])
            if(c):
                observed.append([a, b, listPoints[i]])
        print("Found: " + str(len(observed)))
        return observed

    def getPhoto(self, mat):
        success = True
        x = mat[0:3].reshape((3,1))
        p = np.linalg.lstsq(self.CamT.TM, np.vstack((x, 1.0)), rcond=-1)[0]
        #p = ling.inv(self.CamT) @ np.vstack((x, 1.0))
        pscale = p[0:3] / p[2]

        imgT = np.array([[self.focx, 0, self.pixX],[0, self.focy, self.pixY],[0.0, 0.0,0.0]]) @ pscale

        #Pixel Location
        img = imgT[0:2]

        Q = np.diag(np.array([self.sigma, self.sigma]))
        if (img[0] > self.maxX or img[1] > self.maxY or img[0] < 0 or img[1] < 0):
            img[0] = 10000;
            img[1] = 10000;
            Q = 99999 * Q;
            success = False

        return img, Q, success

    def dhdx(self, mat):
        r = lambda x : (self.getPhoto(mat[0:3].reshape((3,1)))[0].conj().T)
        rate = self.NumJac(r, mat[0:3].reshape((3,1)), .005)

        return rate

    def NumJac(self, f, x0, h):
        x0p = np.copy(x0)
        x0p[0] = x0p[0] + h
        x0m = np.copy(x0)
        x0m[0] = x0m[0] - h
        dfdx = (f(x0p)-f(x0m))/(2*h)

        for i in range(1,x0.size):
            x0p =  np.copy(x0)
            x0p[i] = x0p[i] + h
            x0m =  np.copy(x0)
            x0m[i] = x0m[i] - h
            #Conversion paused here. continue evalutation
            dfdx=np.concatenate((dfdx,(f(x0p)-f(x0m))/(2*h)), axis = 0)
        dfdx=dfdx.conj().T
        f(x0)

        # Call the function with the initial input to reset state, if
        # applicable.
        #f(x0)

        return dfdx

    def getProbability(self, mat, mean):
        img, Q, suc = self.getPhoto(mean)
        prob = st.multivariate_normal.pdf(mat, img.reshape((2)), Q)
        if Q[0,0] > self.sigma:
            prob = 0
        return prob

    def attachment(self, target, mat):
        targ = self.checkCoord(target)
        self.moveCamera(mat @ TAAtoTM([0, 0, 0, 0, np.pi/2, 0]))
        img, Q, suc = self.getPhoto(targ)
        img = img - np.array([1024, 1024])
        return img

    def updateFocal(self, x, y):
        self.focx = x
        self.focy = y

    def updateResolution(self, x, y):
        self.pixX = x
        sef.pixY = y

    def moveCamera(self, newT):
        self.CamT = newT.copy()

    def __eq__(self, o):
        if(self.id == o.id):
            return True
