from itertools import combinations

import numpy as np
import trimesh
from trimesh.collision import CollisionManager

from ..general import fsr, tm
from ..kinematics import Arm
from ..plotting.Draw import drawMesh


def createBox(position, dims):
    """Creates a trimesh representation of a box

    Args:
        position (tm): transformation of the box in space
        dims: Dimensions list of the box

    Returns:
        trimesh.Trimesh: new trimesh mesh
    """
    box = trimesh.creation.box(dims, position.gTM())
    return box

def createCylinder(position, radius, height):
    """
    Creates a new mesh representing a cylinder

    Args:
        position (tm): Transformation of the cylinder in space
        radius: radius of cylinder
        height: height of the cylinder

    Returns:
        trimesh.Trimesh: new trimesh mesh

    """
    cyl = trimesh.creation.cylinder(radius, height, transform = position.gTM())
    return cyl

def createSphere(position, radius):
    """
    Creates a new mesh representing a sphere

    Args:
        position (tm): Transformation of the sphere in space
        radius: radius of the sphere

    Returns:
        trimesh.Trimesh: new trimesh mesh

    """
    sphr = trimesh.creation.icosphere(radius = radius)
    sphr.apply_transform(position.gTM())
    return sphr

def createMesh(position, file_name, type='stl'):
    """
    Creates a new mesh from a mesh file

    Args:
        position (tm): position of the mesh in space
        file_name: File name of the STL file to load
        type: type of object, such as 'stl'

    Returns:
        trimesh.Trimesh: new trimesh mesh

    """
    new_mesh = trimesh.load(file_name, type = type)
    new_mesh.apply_transform(position.gTM())
    return new_mesh

class ColliderManager:
    """Provides an interface for multiple collision managers
        to afford greater control over a workspace

    Attributes:
        collision_objects: list of trimesh collision managers

    """
    def __init__(self):
        """Initializes a new collision manager"""
        self.collision_objects = []

    def update(self):
        """updates all objects in the collision manager"""
        for collision_object in self.collision_objects:
            collision_object.update()

    def bind(self, collision_object):
        """
        Bind a new ColliderObject into the collision manager

        Args:
            collision_object: New collision object to bind to the Collider Manager
        """
        if collision_object not in self.collision_objects:
            self.collision_objects.append(collision_object)
            collision_object.super_manager = self

    def checkCollisions(self):
        """
        Checks for collisions between any set of management groups

        Returns:
            type: Boolean for whether or not collisions are present
            names: tuple containing first colliding set names

        """
        combos = list(combinations(self.collision_objects, 2))
        for combo in combos:
            if combo[0].manager.in_collision_other(combo[1].manager):
                return True, (combo[0].name, combo[1].name)
        return False, None

class ColliderObject:
    """
    General class for a collider object

    Args:
        name: identifier for this collision group

    Attributes:
        super_manager: reference to the ColliderManager if there is one
        manager: trimesh manager of this particular group
        name: name of this instance
        meshes: dict containing name-mesh pairs

    """
    def __init__(self, name='newCollider'):
        """
        Initializes a new collider object

        Args:
            name: name of the new collider object
        """
        self.name = name
        self.super_manager = None
        self.manager = CollisionManager()
        self.meshes = {}

    def bindManager(self, super_manager):
        """
        Binds this instance to a collider manager

        Args:
            super_manager: collider manager to bind to
        """
        super_manager.bind(self)

    def update(self):
        """Handle for update, may not do anything for all subclasses"""
        pass

    def addMesh(self, name, object):
        self.manager.add_object(name, object)
        self.meshes[name] = object

    def checkExternalCollisions(self):
        """
        Checks for collisions external to the collision group

        Returns:
            bool: collisions
            [tuples]: names of colliding objects
        """
        colliding = False
        names = []
        if self.super_manager is not None:
            for other_manager in self.super_manager.collision_objects:
                if other_manager == self.manager:
                    continue
                else:
                    collide_bool, collide_names = self.manager.in_collision_other(
                            other_manager, return_names = True)
                    if collide_bool:
                        colliding = True
                    names.extend(list(collide_names))
        return colliding, names


    def checkInternalCollisions(self):
        """
        Check for collisions internal to the collision group

        Returns:
            bool: whether or not there are collisions

        """
        return self.manager.in_collision_internal()

    def checkAllCollisions(self):
        """
        Check for internal and external collisions to the collision group

        Returns:
            bool: Collisions in a global space

        """
        return self.checkInternalCollisions() and self.checkExternalCollisions()[0]

class ColliderArm(ColliderObject):
    """
    Subclass of ColliderObject which deals with collision groups resulting from serial arms

    Args:
        arm: serial arm model
        name: name of the serial arm model/collisiong roup

    Attributes:
        num_links: number of links in the serial arm (corresponding to collision objects)
        ignore_connected_links: Description of parameter `ignore_connected_links`.
        arm: serial arm model
        name: name of this instance

    """
    def __init__(self, arm, name = "arm"):
        """
        Initialize new ColliderArm instance

        Args:
            arm: serial arm model
            name: name of this instance
        """
        super().__init__()
        self.arm = arm
        self.name = name
        self.num_links = len(self.arm.link_names)
        self.old_transforms = []
        self.populateSerialArm()
        self.ignore_connected_links = True
        self.ignore_ee = False
        self.include_base = True

    def deleteEE(self):
        """
        Delete the end effector to more effectively remove it from collision checking
        """
        self.manager.remove_object(self.arm.link_names[self.num_links - 1])
        self.num_links = self.num_links - 1

    def populateSerialArm(self):
        """
        populateColliderObject with serial arm properties
        """
        props = self.arm._col_props
        if props is None:
            props = self.arm._vis_props
        for i in range(len(props)):
            link_name = self.arm.link_names[i]
            p = props[i]
            self.old_transforms.append(tm())
            if p[0] == 'box':
                new_obj = createBox(tm(), p.box_size)
            elif p[0] == 'cyl':
                new_obj = createCylinder(tm(), p.radius, p.length)
            elif p[0] == 'spr':
                new_obj = createSphere(tm(), p.radius)
            elif p[0] == 'msh':
                new_obj = createMesh(p.origin, p.file_name)
            self.addMesh(link_name, new_obj)
        self.update()

    def update(self):
        """
        update positions of collision meshes to match with the arm current state
        """
        joint_transforms = self.arm.getJointTransforms(self.include_base)
        for i in range(self.num_links):
            self.manager.set_transform(self.arm.link_names[i], joint_transforms[i].gTM())

    def checkInternalCollisions(self):
        """
        Check for collisions within the arm object, potentialy ignoring consecutive links

        Returns:
            bool: whether or not there's a countable internal collision
        """
        colliding, names = self.manager.in_collision_internal(return_names=True)
        if not colliding or not self.ignore_connected_links:
            return colliding
        # Here we assume *something* is in collision, we just need to know what exactly
        for name_tup in names:
            first = name_tup[0]
            second = name_tup[1]
            if self.ignore_ee:
                if first == self.arm.link_names[-1] or second == self.arm.link_names[-1]:
                    continue
            link_index = self.arm.link_names.index(first)
            if link_index > 0 and self.arm.link_names[link_index - 1] == second:
                continue
            elif link_index < self.num_links - 1 and self.arm.link_names[link_index + 1] == second:
                continue
            else:
                return True
        return False

    def drawArmMeshes(self, ax):
        """
        Draw Arm Meshes (and update backup, displayable meshes)
        Args:
            ax: matplotlib axis object to draw to
        """
        joint_transforms = self.arm.getJointTransforms(self.include_base)
        for i in range(self.num_links):
            #Due to a limitation in trimesh.mesh, it is not possible to directly set the transform
            #So to apply a global transform, one must undo the previous transform, and then apply
            #The next one in sequence. For whatever reason old.inv @ new didn't work, so the two
            #step process is what is being used for now.
            #This is only called during the draw function, so it is not resource intensive
            #while in normal use.
            self.meshes[self.arm.link_names[i]].apply_transform(self.old_transforms[i].inv().gTM())
            self.meshes[self.arm.link_names[i]].apply_transform(joint_transforms[i].gTM())
            self.old_transforms[i] = joint_transforms[i]
        for item in self.meshes:
            drawMesh(self.meshes[item], ax)

class ColliderSP(ColliderObject):
    """
    Handler class for Stewart Platform Collision Groups

    Attributes:
        super: Description of parameter `super`.

    """
    def __init__(self):
        """
        Initializes new SP Collider Object
        """
        super().__init__()

class ColliderObstacles(ColliderObject):
    """
    Generic handler in ColliderObject for various workspace obstacles

    Args:
        name: name of this collision group e.g. 'asteroids'

    Attributes:
        name: name of this instance

    """
    def __init__(self, name = 'obstacles'):
        """
        Initialize new ColliderObstacles object

        Args:
            name: name of this instance of ColliderObstacles
        """
        super().__init__()
        self.name = name

    def update_component(self, name, position):
        """
        Update the position of an obstacle in the group

        Args:
            name: name of the object to update
            position (tm): new position to setct.

        """
        self.manager.set_transform(name, position.gTM())
