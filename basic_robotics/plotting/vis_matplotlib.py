"""Utility Plotting Functions for Basic-Robotics using matplotlib."""
import os
from stl import mesh

#Math Functions
import math
import numpy as np
import scipy.linalg as ling


#Drawing Functios
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from matplotlib import cm

#Other Modules
from ..general import fsr, tm



# Create an instance of a LightSource and use it to illuminate the surface.

def drawManipulability(jacobian_matrix : 'np.ndarray[float]', 
        transform : tm, len_factor : float, ax : Axes3D) -> None:   # pragma: no cover
    """
    Draw Manipulability Ellipsoid based on a Jacobian Matrix.

    Args:
        jacobian_matrix (np.ndarray[float]): jacobian matrix
        transform (tm): position to draw ellipsoid at
        len_factor (float): scaling
        ax (Axes3D): axes object to draw on
    """    
    p = transform[0:3, 3]
    R = transform[0:3, 2]
    Aw = jacobian_matrix[0:3,:] @ jacobian_matrix[0:3,:].conj().transpose()
    Av = jacobian_matrix[3:6,:] @ jacobian_matrix[3:6,:].conj().transpose()

    weigv, weigd = ling.eig(Aw)
    weig = math.sqrt(np.diag(weigd))
    [veigv, veigd] = ling.eig(Av)
    veig = math.sqrt(np.diag(veigd))

    weigvs = weigv.copy()
    veigvs = veigv.copy()

    for i in range(0, 3):
        weigvs[0:3, i] = R @ weigv[0:3, i]
        veigvs[0:3, i] = R @ veigv[0:3, i]

    for i in range(0, 3):
        pw = p + len_factor * weigvs[0:3, i] * weig[i]
        pv = p + len_factor * veigvs[0:3, i] * veig[i]

        ax.plot3D(p, pw)
        ax.plot3D(p, pv)

def drawSTL(transform : tm, file_name : str, ax : Axes3D, scale : float = 1.0):  # pragma: no cover
    """
    Draw an STL file at a set scale.

    Args:
        transform (tm): transform to draw at
        file_name (str): filename of the stl
        ax (Axes3D): Axis to draw on
        scale (float, optional): scaling factor to apply. Defaults to 1.0.
    """    
    #make sure to install nuumpy-stl and not stl


    t_mesh = mesh.Mesh.from_file(file_name)
    for i in range(len(t_mesh.x)):
        for j in range(3):
            t_mesp = fsr.TAAtoTM(np.array([t_mesh.x[i, j], t_mesh.y[i, j], t_mesh.z[i, j], 0, 0, 0]))
            #disp(t_mesh.x[i, j])
            t_new = transform @ t_mesp
            mesp_aa = fsr.TMtoTAA(t_new)
            t_mesh.x[i, j] = mesp_aa[0] * scale
            t_mesh.y[i, j] = mesp_aa[1] * scale
            t_mesh.z[i, j] = mesp_aa[2] * scale


    X = t_mesh.x
    Y = t_mesh.y
    Z = t_mesh.z

    light = LightSource(90, 45)
    illuminated_surface = light.shade(Z, cmap=cm.coolwarm)

    ax.plot_surface(t_mesh.x, t_mesh.y, t_mesh.z, rstride=1, cstride=1, linewidth=0, antialiased=False,
                    facecolors=illuminated_surface)
    #ax.add_collection3d(mplot3d.art3d.Poly3DCollection(t_mesh.vectors))
    #ax.auto_scale_xyz(scale, scale, scale)

def getSTLProps(fname : str):  # pragma: no cover
    """
    Get Mass properties of a Mesh.

    Args:
        fname (str): file name

    Returns:
        Mass Props: Mesh Mass Properties
    """    
    #Return center of Mass, inertia, etc
    new_mesh = mesh.Mesh.from_file(fname)

    return new_mesh.get_mass_properties()

def drawArm(arm, ax : Axes3D, jheight : float = .1, jdia : float = .3, 
        axes_lens : float = 1.0, c : str = 'grey', forces : 'np.ndarray[float]' = np.zeros((1))):  # pragma: no cover
    """
    Draw A Serial Arm.

    Args:
        arm (Arm): Serial Arm To Draw
        ax (Axes3D):  Axes object to plot to.
        jrad (float, optional): joint circle height. Defaults to .1.
        jdia (float, optional): joint circle diameter. Defaults to .3.
        axes_lens (float, optional): joint axis marker lengths. Defaults to 1.0.
        c (str, optional): _description_. Defaults to 'grey'.
        forces (np.ndarray[float], optional): arm joint torques. Defaults to np.zeros((1)).
    """    
    startind = 0
    while (sum(arm.screw_list[3:6, startind]) == 1):
        startind = startind + 1
    poses = arm.getJointTransforms()
    p = np.zeros((3, len(poses[startind:])))
    for i in range(startind, len(poses[startind:])):
        if poses[i] == None:
            continue
        p[0, i] = (poses[i].TAA[0])
        p[1, i] = (poses[i].TAA[1])
        p[2, i] = (poses[i].TAA[2])
    ax.scatter3D(p[0,:], p[1,:], p[2,:])
    ax.plot3D(p[0,:], p[1,:], p[2,:])
    Dims = np.copy(arm._link_dimensions).T
    dofs = len(poses)
    yrot = poses[0].spawnNew([0, 0, 0, 0, np.pi/2, 0])
    xrot = poses[0].spawnNew([0, 0, 0, np.pi/2, 0, 0])
    zrot = poses[0].spawnNew([0, 0, 0, 0, 0, np.pi])
    rdim = 0
    for i in range(startind, dofs - 1):
        zed = poses[i]
        drawAxes(zed, axes_lens, ax)

        if i < len(Dims):
            rdim = min(Dims[i, :])
        drawTube(fsr.lookAt(fsr.tmInterpMidpoint(poses[i], poses[i+1]), poses[i+1]), fsr.distance(poses[i], poses[i+1]), rdim, ax, 'red', res=6)
        #QuadPlot(poses[i], poses[i+1], Dims[i+1, 0:3], ax, c = c)

        if i > 0:
            if len(forces) != 1:
                label = '%.1fNm' % (forces[i])
                ax.text(poses[i][0], poses[i][1], poses[i][2], label)
            if (arm.joint_axes[0, i - 1] == 1):
                if len(forces) != 1:
                    drawTube(zed @ yrot, jheight, forces[i]/300, ax)
                else:
                    drawTube(zed @ yrot, jheight, jdia, ax)
            elif (arm.joint_axes[1, i - 1] == 1):
                if len(forces) != 1:
                    drawTube(zed @ xrot, jheight, forces[i]/300, ax)
                else:
                    drawTube(zed @ xrot, jheight, jdia, ax)
            else:
                if len(forces) != 1:
                    drawTube(zed @ zrot, jheight, forces[i]/300, ax)
                else:
                    drawTube(zed @ zrot, jheight, jdia, ax)

    zed = poses[0].gTAA()
    if startind ==0:
        drawRectangle(arm._base_pos_global @
            fsr.TAAtoTM(np.array([0, 0, Dims[len(Dims)-1, 2]/2, 0, 0, 0])),
            Dims[len(Dims)-1, 0:3], ax, c = c)
    for i in range(len(arm.cameras)):
        drawCamera(arm.cameras[i][0], 1, ax)

def drawLine(tf1 : tm, tf2 : tm, ax : Axes3D, col : str = 'blue'):  # pragma: no cover
    """
    Draw a line Between Two Points.

    Args:
        tf1 (tm): Point 1
        tf2 (tm): Point 2
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'blue'.
    """    
    ax.plot3D([tf1[0], tf2[0]], [tf1[1], tf2[1]], [tf1[2], tf2[2]], col)

def drawSP(sp, ax : Axes3D, col : str = 'green', forces : bool = False):  # pragma: no cover
    """
    Draw A Stewart Platform.

    Args:
        sp (SP): SP model to draw.
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'green'.
        forces (bool, optional): whether or not to indicate leg forces. Defaults to True.
    """    
    for i in range(6):

        ax.plot3D([sp.getBottomJoints()[0, i], sp.getBottomJoints()[0,(i+1)%6]],
            [sp.getBottomJoints()[1, i], sp.getBottomJoints()[1,(i+1)%6]],
            [sp.getBottomJoints()[2, i], sp.getBottomJoints()[2,(i+1)%6]], 'blue')
        ax.plot3D([sp.getTopJoints()[0, i], sp.getTopJoints()[0,(i+1)%6]],
            [sp.getTopJoints()[1, i], sp.getTopJoints()[1,(i+1)%6]],
            [sp.getTopJoints()[2, i], sp.getTopJoints()[2,(i+1)%6]], 'blue')
        if i == 0:
            ax.plot3D([sp.getBottomJoints()[0, i], sp.getTopJoints()[0, i]],
                [sp.getBottomJoints()[1, i], sp.getTopJoints()[1, i]],
                [sp.getBottomJoints()[2, i], sp.getTopJoints()[2, i]], 'darkred')
        elif i == 1:
            ax.plot3D([sp.getBottomJoints()[0, i], sp.getTopJoints()[0, i]],
                [sp.getBottomJoints()[1, i], sp.getTopJoints()[1, i]],
                [sp.getBottomJoints()[2, i], sp.getTopJoints()[2, i]], 'salmon')
        else:
            ax.plot3D([sp.getBottomJoints()[0, i], sp.getTopJoints()[0, i]],
                [sp.getBottomJoints()[1, i], sp.getTopJoints()[1, i]],
                [sp.getBottomJoints()[2, i], sp.getTopJoints()[2, i]], col)
        if(sp.bottom_plate_thickness != 0):
            aa = sp._nominal_plate_transform.spawnNew([
                sp.getBottomJoints()[0, i],
                sp.getBottomJoints()[1, i],
                sp.getBottomJoints()[2, i],
                sp.getBottomT()[3],
                sp.getBottomT()[4],
                sp.getBottomT()[5]]) @ tm([0, 0, -1 * sp.bottom_plate_thickness, 0, 0, 0])
            ab = sp._nominal_plate_transform.spawnNew([
                sp.getBottomJoints()[0,(i+1)%6],
                sp.getBottomJoints()[1,(i+1)%6],
                sp.getBottomJoints()[2,(i+1)%6],
                sp.getBottomT()[3],
                sp.getBottomT()[4],
                sp.getBottomT()[5]]) @ tm([0, 0, -1 * sp.bottom_plate_thickness, 0, 0, 0])
            ba = sp._nominal_plate_transform.spawnNew([
                sp.getTopJoints()[0, i],
                sp.getTopJoints()[1, i],
                sp.getTopJoints()[2, i],
                sp.getTopT()[3],
                sp.getTopT()[4],
                sp.getTopT()[5]]) @ tm([0, 0, sp.top_plate_thickness, 0, 0, 0])
            bb = sp._nominal_plate_transform.spawnNew([
                sp.getTopJoints()[0,(i+1)%6],
                sp.getTopJoints()[1,(i+1)%6],
                sp.getTopJoints()[2,(i+1)%6],
                sp.getTopT()[3],
                sp.getTopT()[4],
                sp.getTopT()[5]]) @ tm([0, 0, sp.top_plate_thickness, 0, 0, 0])
            ax.plot3D([aa[0], ab[0]],[aa[1], ab[1]],[aa[2], ab[2]], 'blue')
            ax.plot3D([ba[0], bb[0]],[ba[1], bb[1]],[ba[2], bb[2]], 'blue')
            ax.plot3D([sp.getBottomJoints()[0, i], aa[0]],
                [sp.getBottomJoints()[1, i], aa[1]],
                [sp.getBottomJoints()[2, i], aa[2]], 'blue')
            ax.plot3D([sp.getTopJoints()[0, i], ba[0]],
                [sp.getTopJoints()[1, i], ba[1]],
                [sp.getTopJoints()[2, i], ba[2]], 'blue')

    if forces and sp._last_tau.size > 1:
        for i in range(6):
            label = '%.1fN' % (sp.getActuatorForces()[i])
            if i % 2 == 0:
                pos = sp.getActuatorLoc(i, 'b')
            else:
                pos = sp.getActuatorLoc(i, 't')
            ax.text(pos[0], pos[1], pos[2], label)

def drawInterPlate(sp1, sp2, ax : Axes3D, col : str = 'green'):  # pragma: no cover
    """
    Draw interplate medium between two stacked stewart platforms.

    Congratulations! You found an Easter Egg. ASSEMBLERS aren't included in Basic-Robotics yet. 

    Args:
        sp1 (SP): SP 1
        sp2 (SP): SP 2
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'green'.
    """    
    for i in range(6):
        aa = sp1.nominal_plate_transform.spawnNew([
            sp1.getTopJoints()[0, i],
            sp1.getTopJoints()[1, i],
            sp1.getTopJoints()[2, i],
            sp1.getTopT()[3],
            sp1.getTopT()[4],
            sp1.getTopT()[5]]) @ (sp1.nominal_plate_transform)
        ab = sp1.nominal_plate_transform.spawnNew([
            sp1.getTopJoints()[0,(i+1)%6],
            sp1.getTopJoints()[1,(i+1)%6],
            sp1.getTopJoints()[2,(i+1)%6],
            sp1.getTopT()[3],
            sp1.getTopT()[4],
            sp1.getTopT()[5]]) @ (sp1.nominal_plate_transform)
        ba = sp2.nominal_plate_transform.spawnNew([
            sp2.getBottomJoints()[0, i],
            sp2.getBottomJoints()[1, i],
            sp2.getBottomJoints()[2, i],
            sp2.getBottomT()[3],
            sp2.getBottomT()[4],
            sp2.getBottomT()[5]]) @ (-1 * sp2.nominal_plate_transform)
        bb = sp2.nominal_plate_transform.spawnNew([
            sp2.getBottomJoints()[0,(i+1)%6],
            sp2.getBottomJoints()[1,(i+1)%6],
            sp2.getBottomJoints()[2,(i+1)%6],
            sp2.getBottomT()[3],
            sp2.getBottomT()[4],
            sp2.getBottomT()[5]]) @ (-1 * sp2.nominal_plate_transform)
        #ax.plot3D([aa[0], ab[0]],[aa[1], ab[1]],[aa[2], ab[2]], 'g')
        #ax.plot3D([ba[0], bb[0]],[ba[1], bb[1]],[ba[2], bb[2]], 'g')
        ax.plot3D(
            [sp2.getBottomJoints()[0, i], aa[0]],
            [sp2.getBottomJoints()[1, i], aa[1]],
            [sp2.getBottomJoints()[2, i], aa[2]], col)
        ax.plot3D(
            [sp1.getTopJoints()[0, i], ba[0]],
            [sp1.getTopJoints()[1, i], ba[1]],
            [sp1.getTopJoints()[2, i], ba[2]], col)

def drawAssembler(spl, ax : Axes3D, col : str = 'green', forces : bool = True):  # pragma: no cover
    """
    Draw an Assembler.

    Congratulations! You found an Easter Egg. ASSEMBLERS aren't included in Basic-Robotics yet. 

    Args:
        spl (List[SP]): List of SP objects
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color . Defaults to 'green'.
        forces (bool, optional): whether to indicate leg forces. Defaults to True.
    """    
    for i in range(spl.numsp):
        DrawSP(spl.splist[i], ax , col, forces)
        if i + 1 < spl.numsp:
            DrawInterPlate(spl.splist[i], spl.splist[i+1], ax, col)

def drawCamera(cam, size : float, ax : Axes3D):  # pragma: no cover
    """
    Draw a Camera.

    Args:
        cam (Camera): virtual camera
        size (float): camera screen size
        ax (Axes3D):  Axes object to plot to.
    """    
    drawAxes(cam.CamT, size/2, ax)
    ScreenLoc = cam.CamT @ fsr.TAAtoTM(np.array([0, 0, size, 0, 0, 0]))
    imgT = cam.getFrameSize(size)
    print(imgT)
    Scr = np.zeros((4, 3))
    t = ScreenLoc @ fsr.TAAtoTM(np.array([-imgT[0], imgT[1], 0, 0, 0, 0]))
    Scr[0, 0:3] = t[0:3].flatten()
    t = ScreenLoc @ fsr.TAAtoTM(np.array([imgT[0], imgT[1], 0, 0, 0, 0]))
    Scr[1, 0:3] = t[0:3].flatten()
    t = ScreenLoc @ fsr.TAAtoTM(np.array([-imgT[0], -imgT[1], 0, 0, 0, 0]))
    Scr[3, 0:3] = t[0:3].flatten()
    t = ScreenLoc @ fsr.TAAtoTM(np.array([imgT[0], -imgT[1], 0, 0, 0, 0]))
    Scr[2, 0:3] = t[0:3].flatten()
    for i in range(4):
        ax.plot3D((cam.CamT[0],Scr[i, 0]),
            (cam.CamT[1],Scr[i, 1]),
            (cam.CamT[2],Scr[i, 2]), 'green')
    ax.plot3D(np.hstack((Scr[0:4, 0], Scr[0, 0])),
        np.hstack((Scr[0:4, 1], Scr[0, 1])),
        np.hstack((Scr[0:4, 2], Scr[0, 2])), 'red')

def drawAxes(zed : tm, lv : float, ax : Axes3D, makelegend : str = None, zdir : tm = None):  # pragma: no cover
    """
    Draw a set of Axes in a frame.

    Red indicates X
    Blue indicates Y
    Green indicates Z

    Args:
        zed (tm): transform to draw
        lv (float): axes length scalar
        ax (Axes3D):  Axes object to plot to.
        makelegend (str, optional): text to display
        zdir (tm, optional): offset from zdir to draw text
    """    
    zx, zy, zz = zed.tripleUnit(lv)
    poses = zed.gTAA().flatten()
    if makelegend is not None:
        if zdir is not None:
            zed = zed @ zdir
        ax.text(zed[0], zed[1], zed[2], makelegend)
    ax.plot3D([poses[0], zx[0]], [poses[1], zx[1]], [poses[2], zx[2]], 'red')
    ax.plot3D([poses[0], zy[0]], [poses[1], zy[1]], [poses[2], zy[2]], 'blue')
    ax.plot3D([poses[0], zz[0]], [poses[1], zz[1]], [poses[2], zz[2]], 'green')

def drawTrussElement(transform : tm, truss_length : float, truss_radius : float, ax : Axes3D, c : str ='blue', 
        c2 : str = 'blue', hf : bool = False, delt : float = .5, RB : float = .1):  # pragma: no cover
    """
    Draw a Three Segment Truss.

    Args:
        transform (tm): Truss Center and Orientation transform
        truss_length (float): Length of Truss
        truss_radius (float): Truss Radius
        ax (Axes3D):  Axes object to plot to.
        c (str, optional): Truss Tower Color. Defaults to 'blue'.
        c2 (str, optional): Truss Crosses Color. Defaults to 'blue'.
        hf (bool, optional): Draw truss crosses. Defaults to False.
        delt (float, optional): Distance for truss crossees. Defaults to .5.
        RB (float, optional): Truss Crosses Radius. Defaults to .1.
    """        
    if hf == True:
        R1 = transform @ transform.spawnNew([truss_radius, 0, 0, 0, 0, 0])
        R2 = transform @ transform.spawnNew([0, 0, 0, 0, 0, 2*np.pi/3]) @ transform.spawnNew([truss_radius, 0, 0, 0, 0, 0])
        R3 = transform @ transform.spawnNew([0, 0, 0, 0, 0, 4*np.pi/3]) @ transform.spawnNew([truss_radius, 0, 0, 0, 0, 0])
        R1 = R1 @ transform.spawnNew([0, 0, -truss_length/2, 0, 0, 0])
        R2 = R2 @ transform.spawnNew([0, 0, -truss_length/2, 0, 0, 0])
        R3 = R3 @ transform.spawnNew([0, 0, -truss_length/2, 0, 0, 0])
        for i in range(int(truss_length/delt)):
            R1A = R1 @ transform.spawnNew([0, 0, delt, 0, 0, 0])
            R2A = R2 @ transform.spawnNew([0, 0, delt, 0, 0, 0])
            R3A = R3 @ transform.spawnNew([0, 0, delt, 0, 0, 0])
            if cycle ==1:
                drawTube(fsr.adjustRotationToMidpoint(
                    fsr.tmInterpMidpoint(R1, R2A), R1, R2A, mode=1),
                    fsr.distance(R1, R2A)-RB, RB/3, ax, c2, res = 3)
                drawTube(fsr.adjustRotationToMidpoint(
                    fsr.tmInterpMidpoint(R2, R3A), R2, R3A, mode=1),
                    fsr.distance(R2, R3A)-RB, RB/3, ax, c2, res = 3)
                drawTube(fsr.adjustRotationToMidpoint(
                    fsr.tmInterpMidpoint(R3, R1A), R3, R1A, mode=1),
                    fsr.distance(R3, R1A)-RB, RB/3, ax, c2, res = 3)
                R1 = R1A
                R2 = R2A
                R3 = R3A
            else:
                drawTube(fsr.adjustRotationToMidpoint(
                    fsr.tmInterpMidpoint(R1, R3A), R1, R3A, mode=1),
                    fsr.distance(R1, R3A)-RB, RB/3, ax, c2, res = 3)
                drawTube(fsr.adjustRotationToMidpoint(
                    fsr.tmInterpMidpoint(R2, R1A), R2, R1A, mode=1),
                    fsr.distance(R2, R1A)-RB, RB/3, ax, c2, res = 3)
                drawTube(fsr.adjustRotationToMidpoint(
                    fsr.tmInterpMidpoint(R3, R2A), R3, R2A, mode=1),
                    fsr.distance(R3, R2A)-RB, RB/3, ax, c2, res = 3)
                R1 = R1A
                R2 = R2A
                R3 = R3A
            cycle*=-1
        drawTube(fsr.adjustRotationToMidpoint(fsr.tmInterpMidpoint(R1, R2), R1, R2, mode=1),
            fsr.distance(R1, R2)-RB, RB/3, ax, 'r', res = 3)
        drawTube(fsr.adjustRotationToMidpoint(fsr.tmInterpMidpoint(R2, R3), R2, R3, mode=1),
            fsr.distance(R2, R3)-RB, RB/3, ax, 'r', res = 3)
        drawTube(fsr.adjustRotationToMidpoint(fsr.tmInterpMidpoint(R3, R1), R3, R1, mode=1),
            fsr.distance(R3, R1)-RB, RB/3, ax, 'r', res = 3)
    R1 = transform @ transform.spawnNew([truss_radius, 0, 0, 0, 0, 0])
    R2 = transform @ transform.spawnNew([0, 0, 0, 0, 0, 2*np.pi/3]) @ transform.spawnNew([truss_radius, 0, 0, 0, 0, 0])
    R3 = transform @ transform.spawnNew([0, 0, 0, 0, 0, 4*np.pi/3]) @ transform.spawnNew([truss_radius, 0, 0, 0, 0, 0])
    drawTube(R1, truss_length, RB, ax, c, res = 3)
    drawTube(R2, truss_length, RB, ax, c, res = 3)
    drawTube(R3, truss_length, RB, ax, c, res = 3)

def drawRectangle(transform : tm, 
        dims : '[float]', ax : Axes3D, c : str = 'grey', a : float = 0.1):  # pragma: no cover
    """
    Draw a Rectanglular Prism.

    Args:
        transform (tm): prism origin.
        dims (List[float]): rectangle dimensions
        ax (Axes3D):  Axes object to plot to.
        c (str, optional): color. Defaults to 'grey'.
        a (float, optional): transparency. Defaults to 0.1.
    """    
    dx = dims[0]
    dy = dims[1]
    dz = dims[2]
    corners = .5 * np.array([
        [-dx, -dy, -dz], #BBL
        [dx, -dy, -dz], #BBR
        [-dx, dy, -dz], #BFL
        [dx, dy, -dz], #BFR
        [-dx, -dy, dz], #TBL
        [dx, -dy, dz], #TBR
        [-dx, dy, dz], #TFL
        [dx, dy, dz]]).T #TFR
    Tc = np.zeros((3, 8))
    for i in range(0, 8):
        h = transform.gTM() @ np.array([corners[0, i], corners[1, i], corners[2, i], 1]).T
        Tc[0:3, i] = np.squeeze(h[0:3])
    verts = [Tc[:,(0, 1, 3, 2)].T, Tc[:,(4, 5, 7, 6)].T, Tc[:,(0, 1, 5, 4)].T,
    Tc[:,(2, 3, 7, 6)].T, Tc[:,(0, 2, 6, 4)].T, Tc[:,(1, 3, 7, 5)].T]
    ax.add_collection3d(Poly3DCollection(verts, facecolors = c, edgecolors=c, alpha = a))

def makeVideo(dir : str = "VideoTemp"):  # pragma: no cover
    """
    Make a video from a directory full of images.

    Args:
        dir (str, optional): Directory. Defaults to "VideoTemp".
    """    
    os.chdir(dir)
    os.system("ffmpeg -r 30 -i img%04d.png -vcodec mpeg4 -qscale 0 -y movie.mp4")
    #for file_name in glob.glob("*.png"):
    #    os.remove(file_name)

def drawRRT(listPoints, ax : Axes3D):  # pragma: no cover
    """
    Draw A RRT.

    Args:
        listPoints (RRT Nodes List): RRT Nodes List
        ax (Axes3D):  Axes object to plot to.
    """    
    for i in range(len(listPoints)):
        temp = listPoints[i].object.getPosition()
        ax.scatter3D(temp[0], temp[1], temp[2], s = 3)
    for i in range(len(listPoints)):
        temp1 = listPoints[i].object.getPosition()
        if listPoints[i].object.getParent() == None:
            drawRectangle(temp1, [.05, .05, .05], ax, 'b')
            continue
        temp2 = listPoints[i].object.getParent().getPosition()
        c = 'green'
        if  listPoints[i].object.type == 1 and  listPoints[i].object.getParent().type == 1:
            c = 'm'
            continue
        elif listPoints[i].object.type == 0 and listPoints[i].object.getParent().type == 0:
            c = 'blue'
        ax.plot3D([temp1[0], temp2[0]], [temp1[1], temp2[1]], [temp1[2], temp2[2]], c)

def drawRRTPath(listPoints, ax : Axes3D, col : str = 'red'):  # pragma: no cover
    """
    Draw a RRT path do destination.

    Args:
        listPoints (List of RRT Nodes): List of RRT Nodes
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'red'.
    """    
    drawRectangle(listPoints[0], [.1, .1, .1], ax, 'b')
    drawRectangle(listPoints[len(listPoints)-1], [.1, .1, .1], ax, 'r', )
    for i in range(len(listPoints) - 1):
        temp1 = listPoints[i]
        temp2 = listPoints[i+1]
        ax.plot3D([temp1[0], temp2[0]], [temp1[1], temp2[1]], [temp1[2], temp2[2]], col)

def drawObstructions(listObs, ax : Axes3D, col : str = 'red', a : float = .1):  # pragma: no cover
    """
    Draw RRT Obstructions.

    Args:
        listObs (List[Obstructions]): List of RRT Obstructions
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'red'.
        a (float, optional): transparency. Defaults to .1.
    """    
    for obs in listObs:
        center = obs[1].spawnNew([(obs[1][0] + obs[0][0])/2, (obs[1][1] + obs[0][1])/2, (obs[1][2] + obs[0][2])/2, 0, 0, 0])
        dims = [(obs[1][0] - obs[0][0]), (obs[1][1] - obs[0][1]), (obs[1][2] - obs[0][2])]
        drawRectangle(center, dims, ax, col, a)

def drawMesh(mesh : mesh.Mesh, ax : Axes3D):  # pragma: no cover
    """
    Draw a Mesh.

    Args:
        mesh (mesh.Mesh): Mesh To Draw
        ax (Axes3D):  Axes object to plot to.
    """    
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1],
            triangles=mesh.faces, Z=mesh.vertices[:, 2])

def drawWrench(wrench, weight : float, ax : Axes3D):  # pragma: no cover
    """
    Draw a Wrench.

    Args:
        wrench (Wrench): wrench to draw
        weight (float): length of wrench vector arrow
        ax (Axes3D):  Axes object to plot to.
    """    
    tr = wrench.frame_applied @ wrench.position_applied
    direction = wrench.getForce().flatten()/math.sqrt((wrench.getForce()[0]**2 + wrench.getForce()[1]**2 + wrench.getForce()[2]**2))
    trb = tr.spawnNew([tr[0], tr[1], tr[2], 0, 0, 0])
    other = tr.spawnNew([direction[0], direction[1], direction[2], 0, 0, 0])
    np = trb @ (.4*other)
    a = trb @ (.3*other)
    drawRectangle(trb, [.1, .1, .1], ax)
    a1 = a @ tr.spawnNew([.05, 0, 0, 0, 0, 0])
    a2 = a @ tr.spawnNew([-.05, 0, 0, 0, 0, 0])
    a3 = a @ tr.spawnNew([0, -.05,  0, 0, 0, 0])
    a4 = a @ tr.spawnNew([0, .05, 0, 0, 0, 0])
    label = '%.1fN' % weight
    ax.text(tr[0], tr[1], tr[2], label)

    ax.plot3D([tr[0], np[0]], [tr[1], np[1]], [tr[2], np[2]], 'b')
    ax.plot3D([np[0], a1[0]],[np[1], a1[1]],[np[2], a1[2]], 'r')
    ax.plot3D([np[0], a2[0]],[np[1], a2[1]],[np[2], a2[2]] ,'r')
    ax.plot3D([np[0], a3[0]],[np[1], a3[1]],[np[2], a3[2]] ,'r')
    ax.plot3D([np[0], a4[0]],[np[1], a4[1]],[np[2], a4[2]] ,'r')

def drawTube(T : tm, height : float, r : float, ax : Axes3D, c : str = 'blue', res : int = 12):  # pragma: no cover
    """
    Draw a Tube.

    Args:
        T (tm): Transform of the Tube
        height (float): Height of the tube
        r (float): Radius of the Tube
        ax (Axes3D):  Axes object to plot to.
        c (str, optional): color. Defaults to 'blue'.
        res (int, optional): resolution of the tube. Defaults to 12.
    """
    # find points along x and y axes
    points  = np.linspace(0, 2*np.pi, res+1)
    x = np.cos(points)*r
    y = np.sin(points)*r

    # find points along z axis
    rpoints = np.atleast_2d(np.linspace(0, 1, 1))
    z = np.ones((res+1))*height/2
    tres = np.zeros((6, len(x)))
    tres2 = np.zeros((6, len(x)))
    for i in range(len(x)):
        #disp(i)
        tres[0:6, i] = (T @ T.spawnNew([x[i], y[i], z[i], 0, 0, 0])).TAA.flatten()
        tres2[0:6, i] = (T @ T.spawnNew([x[i], y[i], -z[i], 0, 0, 0])).TAA.flatten()
        bx = np.array([tres[0, i], tres2[0, i]], dtype=object)
        by = np.array([tres[1, i], tres2[1, i]], dtype=object)
        bz = np.array([tres[2, i], tres2[2, i]], dtype=object)
        ax.plot3D(bx, by, bz, c)
    ax.plot3D(tres[0,:], tres[1,:], tres[2,:], c)
    ax.plot3D(tres2[0,:], tres2[1,:], tres2[2,:], c)

def DrawManipulability(jacobian_matrix : 'np.ndarray[float]', 
    transform : tm, len_factor : float, ax : Axes3D) -> None:   # pragma: no cover
    """
    Draw Manipulability Ellipsoid based on a Jacobian Matrix.

    Args:
        jacobian_matrix (np.ndarray[float]): jacobian matrix
        transform (tm): position to draw ellipsoid at
        len_factor (float): scaling
        ax (Axes3D): axes object to draw on
    """
    drawManipulability(jacobian_matrix, transform, len_factor, ax)  

def DrawSTL(transform : tm, file_name : str, ax : Axes3D, scale : float = 1.0):  # pragma: no cover
    """
    Draw an STL file at a set scale.

    Args:
        transform (tm): transform to draw at
        file_name (str): filename of the stl
        ax (Axes3D): Axis to draw on
        scale (float, optional): scaling factor to apply. Defaults to 1.0.
    """
    return drawSTL(transform, file_name, ax, scale)

def DrawArm(arm, ax : Axes3D, jheight : float = .1, jdia : float = .3, axes_lens : float = 1.0, c : str = 'grey', forces : 'np.ndarray[float]' = np.zeros((1))):  # pragma: no cover
    """
    Draw A Serial Arm.

    Args:
        arm (Arm): Serial Arm To Draw
        ax (Axes3D):  Axes object to plot to.
        jrad (float, optional): joint circle height. Defaults to .1.
        jdia (float, optional): joint circle diameter. Defaults to .3.
        axes_lens (float, optional): joint axis marker lengths. Defaults to 1.0.
        c (str, optional): _description_. Defaults to 'grey'.
        forces (np.ndarray[float], optional): arm joint torques. Defaults to np.zeros((1)).
    """  
    drawArm(arm, ax, jheight, jdia, axes_lens, c, forces)

def DrawLine(tf1 : tm, tf2 : tm, ax : Axes3D, col : str = 'blue'):  # pragma: no cover
    """
    Draw a line Between Two Points.

    Args:
        tf1 (tm): Point 1
        tf2 (tm): Point 2
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'blue'.
    """

def DrawSP(sp, ax : Axes3D, col : str = 'green', forces : bool = True):  # pragma: no cover
    """
    Draw A Stewart Platform.

    Args:
        sp (SP): SP model to draw.
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'green'.
        forces (bool, optional): whether or not to indicate leg forces. Defaults to True.
    """
    drawSP(sp, ax, col, forces)

def DrawInterPlate(sp1, sp2, ax : Axes3D, col : str = 'green'):  # pragma: no cover
    """
    Draw interplate medium between two stacked stewart platforms.

    Congratulations! You found an Easter Egg. ASSEMBLERS aren't included in Basic-Robotics yet. 

    Args:
        sp1 (SP): SP 1
        sp2 (SP): SP 2
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'green'.
    """
    drawInterPlate(sp1, sp2, ax, col)

def DrawAssembler(spl, ax : Axes3D, col : str = 'green', forces : bool = True):  # pragma: no cover
    """
    Draw an Assembler.

    Congratulations! You found an Easter Egg. ASSEMBLERS aren't included in Basic-Robotics yet. 

    Args:
        spl (List[SP]): List of SP objects
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color . Defaults to 'green'.
        forces (bool, optional): whether to indicate leg forces. Defaults to True.
    """ 
    drawAssembler(spl, ax, col, forces)

def DrawCamera(cam, size : float, ax : Axes3D):  # pragma: no cover
    """
    Draw a Camera.

    Args:
        cam (Camera): virtual camera
        size (float): camera screen size
        ax (Axes3D):  Axes object to plot to.
    """
    drawCamera(cam, size, ax)

def DrawAxes(zed : tm, lv : float, ax : Axes3D, makelegend : str = None, zdir : tm = None):  # pragma: no cover
    """
    Draw a set of Axes in a frame.

    Red indicates X
    Blue indicates Y
    Green indicates Z

    Args:
        zed (tm): transform to draw
        lv (float): axes length scalar
        ax (Axes3D):  Axes object to plot to.
        makelegend (str, optional): text to display
        zdir (tm, optional): offset from zdir to draw text
    """
    drawAxes(zed, lv, ax, makelegend, zdir)

def DrawTrussElement(transform : tm, truss_length : float, truss_radius : float, ax : Axes3D, c : str ='blue', 
        c2 : str = 'blue', hf : bool = False, delt : float = .5, RB : float = .1):  # pragma: no cover
    """
    Draw a Three Segment Truss.

    Args:
        transform (tm): Truss Center and Orientation transform
        truss_length (float): Length of Truss
        truss_radius (float): Truss Radius
        ax (Axes3D):  Axes object to plot to.
        c (str, optional): Truss Tower Color. Defaults to 'blue'.
        c2 (str, optional): Truss Crosses Color. Defaults to 'blue'.
        hf (bool, optional): Draw truss crosses. Defaults to False.
        delt (float, optional): Distance for truss crossees. Defaults to .5.
        RB (float, optional): Truss Crosses Radius. Defaults to .1.
    """
    drawTrussElement(transform, truss_length, truss_radius, ax, c, c2, hf, delt, RB)

def DrawRectangle(transform : tm, 
        dims : '[float]', ax : Axes3D, c : str = 'grey', a : float = 0.1):  # pragma: no cover
    """
    Draw a Rectanglular Prism.

    Args:
        transform (tm): prism origin.
        dims (List[float]): rectangle dimensions
        ax (Axes3D):  Axes object to plot to.
        c (str, optional): color. Defaults to 'grey'.
        a (float, optional): transparency. Defaults to 0.1.
    """
    drawRectangle(transform, dims, ax, c, a)

def DrawRRT(listPoints, ax : Axes3D):  # pragma: no cover
    """
    Draw A RRT.

    Args:
        listPoints (RRT Nodes List): RRT Nodes List
        ax (Axes3D):  Axes object to plot to.
    """    
    drawRRT(listPoints, ax)

def DrawRRTPath(listPoints, ax : Axes3D, col : str = 'red'):  # pragma: no cover
    """
    Draw a RRT path do destination.

    Args:
        listPoints (List of RRT Nodes): List of RRT Nodes
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'red'.
    """    
    drawRRTPath(listPoints, ax, col)

def DrawObstructions(listObs, ax : Axes3D, col : str = 'red', a : float = .1):  # pragma: no cover
    """
    Draw RRT Obstructions.

    Args:
        listObs (List[Obstructions]): List of RRT Obstructions
        ax (Axes3D):  Axes object to plot to.
        col (str, optional): color. Defaults to 'red'.
        a (float, optional): transparency. Defaults to .1.
    """    
    drawObstructions(listObs, ax, col, a)

def DrawWrench(wrench, weight : float, ax : Axes3D):  # pragma: no cover
    """
    Draw a Wrench.

    Args:
        wrench (Wrench): wrench to draw
        weight (float): length of wrench vector arrow
        ax (Axes3D):  Axes object to plot to.
    """
    drawWrench(wrench, weight, ax)

def DrawTube(T : tm, height : float, r : float, ax : Axes3D, c : str = 'blue', res : int = 12):  # pragma: no cover
    """
    Draw a Tube.

    Args:
        T (tm): Transform of the Tube
        height (float): Height of the tube
        r (float): Radius of the Tube
        ax (Axes3D):  Axes object to plot to.
        c (str, optional): color. Defaults to 'blue'.
        res (int, optional): resolution of the tube. Defaults to 12.
    """
    drawTube(T, height, r, ax, c, res)