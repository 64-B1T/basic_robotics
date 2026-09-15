"""
Virtual camera and vision simulation tools for estimating object poses from viewed scenes.

Models a scene populated by simulated pinhole Cameras and trackable SceneObj/Observed
points, and provides triangulation-style routines for reconstructing 3D point and
object positions from multiple simulated camera views.
"""

import numpy as np
import scipy.linalg as ling
import scipy.stats as st
import scipy as sci
import random

from ..general import tm, fsr

class Scene: #The Whole Scene, Cameras and Objects
    """
    Container for all Cameras and SceneObjs in a simulated vision scene.

    Coordinates gathering observations from every camera and reconstructing 3D
    point positions from those observations.
    """

    def __init__(self):
        """
        Create a new, empty Scene with no objects, cameras, or observations.
        """
        self.objList = []
        self.camList = []
        self.observed = []
        self.grid = None

    def newSceneObj(self, objList, tol = .01, name = "OBJ"):
        """
        Create a new SceneObj from a list (or single transform) of points and add it to the scene.

        Args:
            objList: List of tm points defining the object, or a single tm point.
            tol (float, optional): Distance tolerance used when matching this object. Defaults to .01.
            name (str, optional): Name of the object. Defaults to "OBJ".
        """
        self.objList.append(SceneObj(objList, tol, name))

    def addSceneObj(self, obj):
        """
        Add an existing SceneObj to the scene.

        Args:
            obj (SceneObj): Object to add.
        """
        self.objList.append(obj)

    def addCam(self, obj):
        """
        Add an existing Camera to the scene.

        Args:
            obj (Camera): Camera to add.
        """
        self.camList.append(obj)

    def CalculateGrid(self):
        """
        Compute the pairwise distance matrix between all currently observed points.

        Returns:
            ndarray: Symmetric matrix of distances between every pair of entries in
            self.observed.
        """
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
        """
        Gather observations from every camera and reconstruct 3D positions of the seen points.

        Each camera captures every SceneObj's points; matching observations of the
        same point across different cameras are merged, and points seen by fewer
        than two cameras are discarded since their position cannot be triangulated.

        Returns:
            list[Observed]: Reconstructed points, each with an averaged 3D position,
            or None if no points were observed by any camera.
        """
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
    """
    A single 3D point as observed by one or more Cameras, used to triangulate its world position.
    """

    def __init__(self, pixpos, q, camera, tol = .0003):
        """
        Create a new Observed point from a single camera's observation.

        Args:
            pixpos: Pixel position of the observed point in the camera image.
            q: Measurement covariance associated with this observation.
            camera (Camera): Camera that produced this observation.
            tol (float, optional): Distance tolerance used when matching this
                observation to others. Defaults to .0003.
        """
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
        """
        Average the draft position guesses collected from matching observations.

        Returns:
            tm: Averaged position estimate, also stored on self.cPos.
        """
        numPos = len(self.draftPoses)
        sumPos = tm()
        for i in range(numPos):
            sumPos = sumPos + self.draftPoses[i]
        sumPos = sumPos/numPos
        self.cPos = sumPos
        return sumPos

    def AvgWDist(self, x):
        """
        Compute an averaged 3D position estimate and its residual error for a set of ray scales.

        For each viewing camera, projects along that camera's collated direction
        vector by the corresponding scale in x, averages the resulting points, and
        measures how far each individual projected point is from that average.

        Args:
            x: Per-camera scale factors along each camera's viewing ray.

        Returns:
            tuple: (davg, ts) where davg (float) is the average distance of each
            per-camera projected point from the averaged position, and ts (tm) is
            the averaged position estimate.
        """
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
        """
        Solve for the ray scales that minimize triangulation residual error and return the position.

        Returns:
            tm: Averaged position estimate corresponding to the optimized ray scales.
        """
        res = lambda x : self.AvgWDist(x)[0]
        x0 = np.zeros((len(self.gls),))
        xs = sci.optimize.minimize(res, x0, method = "SLSQP")
        xs = xs.x
        return self.AvgWDist(xs)[1]


    def collateVector(self):
        """
        Compute the unit direction vector from the camera through the observed pixel.

        Args:
            None

        Returns:
            tm: Unit vector (as a tm with zero rotation) pointing from the camera's
            position toward the observed point, expressed in the camera's local frame.
        """
        p1 = self.camera.CamT
        ps = self.camera.getLocalPos(self.pixpos)
        p2 = p1 @ tm([ps[0], ps[1], 1, 0, 0, 0])
        gla = np.asarray(fsr.GlobalToLocal(p1, p2)[0:3]).flatten()/fsr.Distance(p1, p2)
        gl = tm([gla[0], gla[1], gla[2], 0, 0, 0])
        return gl

    def eq(self, other):
        """
        Determine whether this observation and another (from a different camera) view the same point.

        Finds the pair of ray scales that minimizes the distance between the two
        cameras' projected rays; if that minimum distance is within tolerance, the
        candidate 3D position(s) are recorded as draft poses.

        Args:
            other (Observed): Observation from another camera to compare against.

        Returns:
            bool: True if the two observations are judged to be the same point.
        """
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
        """
        Merge another camera's matching observation of this same point into this one.

        Args:
            other (Observed): Matching observation from another camera.
        """
        self.inView+=1
        self.camerasViewing.append(other.camera)
        self.gls.append(other.gl)


class SceneObj: #Object within a scene that needs to be reconstructed
    """
    A rigid object in the scene, defined by a set of points with known relative geometry.

    Used to identify and reconstruct the object's pose from observed points by
    matching the known relative distances/transforms between its defining points.
    """

    def __init__(self, objList, tol = .01, name = "OBJ"):
        """
        Create a new SceneObj from a list of points, or generate one from a single point.

        If objList is a list of tm points, it is used directly with the first
        element treated as the lead point. Otherwise, objList is treated as a
        single lead point and three additional points are generated around it at
        random offsets along each axis.

        Args:
            objList: List of tm points defining the object, or a single tm point.
            tol (float, optional): Distance tolerance used when matching this object. Defaults to .01.
            name (str, optional): Name of the object. Defaults to "OBJ".
        """
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
        """
        Compute each point's relative transform and distance from the lead point.

        Populates self.rels and self.dists in place.
        """
        for i in range(self.sz-1):
            self.rels.append(fsr.GlobalToLocal(self.lead, self.objs[i+1]))
            self.dists.append(fsr.Distance(self.lead, self.objs[i+1]))

    def adjRot(self, x, y):
        """
        Score a candidate lead-point orientation against the object's known point geometry.

        Applies the orientation in x to pose y, then sums the distance error
        between where each defining point would be under that orientation and its
        actual observed position.

        Args:
            x: Candidate orientation (roll, pitch, yaw) to test.
            y (tm): Candidate lead-point pose to adjust with x.

        Returns:
            float: Sum of distance errors across all defining points.
        """
        temp = y
        temp[3] = x[0]
        temp[4] = x[1]
        temp[5] = x[2]
        sum = 0
        for i in range(self.sz - 1):
            d = fsr.Distance(fsr.LocalToGlobal(temp, self.rels[i]), self.objs[i+1])
            sum+=d
        return sum

    def testAll(self, x, ob):
        """
        Score a candidate orientation against every candidate lead pose and track the best match.

        Args:
            x: Candidate orientation to test against each pose in ob.
            ob: Candidate lead poses to score.

        Returns:
            float: The smallest (best) score found across all candidate poses; the
            index of that best match is stored in self.min.
        """
        ylist = []
        for i in range(len(ob)):
            ylist.append(self.adjRot(x, ob[i]))
        self.min = ylist.index(min(ylist))
        return min(ylist)

    def getPos(self, scene):
        """
        Attempt to locate this object's position among a scene's observed points.

        Searches for a set of observed points in the scene whose mutual distances
        match this object's expected geometry within tolerance, then selects the
        best-fitting candidate pose.

        Args:
            scene (Scene): Scene whose observed points are searched.

        Returns:
            tm: Best-fit position for this object, also stored on self.cPos.
        """
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
                    foundL.append(obs.cPos)
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
    """
    Simulated pinhole camera that projects 3D scene points into pixel observations.
    """


    def __init__(self, aptx, apty, pixX, pixY, maxX, maxY, sigma, camT, id = 0):
        """
        Create a new simulated Camera.

        Args:
            aptx (float): Focal length / aperture parameter along the x axis.
            apty (float): Focal length / aperture parameter along the y axis.
            pixX (float): Pixel x coordinate of the image center.
            pixY (float): Pixel y coordinate of the image center.
            maxX (float): Maximum resolvable x resolution/extent of the sensor.
            maxY (float): Maximum resolvable y resolution/extent of the sensor.
            sigma (float): Measurement noise standard deviation (squared internally).
            camT (tm): Pose of the camera in the world/global frame.
            id (int, optional): Identifier used to compare cameras. Defaults to 0.
        """
        self.id = id
        self.focx = aptx
        self.focy = apty
        self.pixX = pixX
        self.pixY = pixY
        self.sigma = sigma**2
        self.maxX = maxX
        self.maxY = maxY
        self.CamT = camT.copy()
        self.fs = self.getFrameSize()

    def getLocalPos(self, pix):
        """
        Convert a pixel coordinate into a local, focal-length-normalized offset.

        Args:
            pix: Pixel coordinate [x, y] to convert.

        Returns:
            list: [x, y] offset from the frame center in focal-length-normalized units.
        """
        sz = self.getFrameSize()
        pix = np.asarray(pix).flatten()
        return [-sz[0] + pix[0]/self.focx, -sz[1] + pix[1]/self.focy]

    def getFrameSize(self, sz = 1):
        """
        Compute the half-frame size of the sensor in normalized units.

        Args:
            sz (float, optional): Scale factor applied to the computed frame size. Defaults to 1.

        Returns:
            list: [x, y] half-frame size at the given scale.
        """
        return [self.maxX/self.focx*sz/2, self.maxY/self.focy*sz/2]

    def getScene(self, listPoints):
        """
        Project a list of 3D points into this camera and return those that land within the sensor.

        Args:
            listPoints: List of 3D points (as tm-compatible arrays) to project.

        Returns:
            list: Entries of [pixel_position, covariance, original_point] for each
            input point that falls within the camera's field of view.
        """
        observed = []
        print("Scanning " + str(len(listPoints)) + " points")
        for i in range(len(listPoints)):
            a, b, c = self.getPhoto(listPoints[i])
            if(c):
                observed.append([a, b, listPoints[i]])
        print("Found: " + str(len(observed)))
        return observed

    def getPhoto(self, mat):
        """
        Project a single 3D point into this camera's pixel space.

        Args:
            mat: 3D point (or tm) to project, using the first three elements as x, y, z.

        Returns:
            tuple: (img, Q, success) where img is the projected [x, y] pixel
            location, Q is the associated measurement covariance (inflated if the
            point falls outside the sensor bounds), and success (bool) indicates
            whether the point landed within the sensor's visible bounds.
        """
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
        """
        Compute the numerical Jacobian of pixel projection with respect to a 3D point.

        Args:
            mat: 3D point (or tm) to linearize the projection around.

        Returns:
            ndarray: Jacobian of the projected pixel coordinates with respect to
            the point's x, y, z position.
        """
        r = lambda x : (self.getPhoto(mat[0:3].reshape((3,1)))[0].conj().T)
        rate = self.NumJac(r, mat[0:3].reshape((3,1)), .005)

        return rate

    def NumJac(self, f, x0, h):
        """
        Compute the numerical (central-difference) Jacobian of a function at a point.

        Args:
            f: Function to differentiate.
            x0: Point at which to evaluate the Jacobian.
            h: Step size used for the central difference.

        Returns:
            ndarray: Numerical Jacobian of f evaluated at x0.
        """
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
        """
        Compute the likelihood of an observed pixel given an expected mean position.

        Args:
            mat: Observed pixel position.
            mean: 3D point whose projection gives the expected pixel position.

        Returns:
            float: Probability density of observing mat given the projection of
            mean, or 0 if the expected point's measurement covariance exceeds the
            camera's noise level (i.e. it falls outside the sensor bounds).
        """
        img, Q, suc = self.getPhoto(mean)
        prob = st.multivariate_normal.pdf(mat, img.reshape((2)), Q)
        if Q[0,0] > self.sigma:
            prob = 0
        return prob

    def attachment(self, target, mat):
        """
        Reposition the camera relative to a pose and measure the pixel offset to a target.

        Args:
            target: Target point to project after moving the camera.
            mat (tm): Pose the camera is moved to, offset by a fixed rotation.

        Returns:
            ndarray: Pixel offset of the projected target from the image center (1024, 1024).
        """
        targ = self.checkCoord(target)
        self.moveCamera(mat @ fsr.TAAtoTM([0, 0, 0, 0, np.pi/2, 0]))
        img, Q, suc = self.getPhoto(targ)
        img = img - np.array([1024, 1024])
        return img

    def updateFocal(self, x, y):
        """
        Update the camera's focal length parameters.

        Args:
            x: New focal length along the x axis.
            y: New focal length along the y axis.
        """
        self.focx = x
        self.focy = y

    def updateResolution(self, x, y):
        """
        Update the camera's pixel resolution parameters.

        Args:
            x: New x pixel resolution.
            y: New y pixel resolution.
        """
        self.pixX = x
        self.pixY = y

    def moveCamera(self, newT):
        """
        Move the camera to a new pose.

        Args:
            newT (tm): New pose to assign to the camera.
        """
        self.CamT = newT.copy()

    def __eq__(self, o):
        """
        Compare two cameras for equality by their id.

        Args:
            o (Camera): Other camera to compare against.

        Returns:
            bool: True if both cameras share the same id.
        """
        if(self.id == o.id):
            return True
