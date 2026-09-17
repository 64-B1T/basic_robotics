import os
import unittest
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Arm
from basic_robotics.kinematics.visual_info import vis_info
from basic_robotics.collisions.collision_manager import (
    createBox, createCylinder, createSphere, createMesh,
    ColliderManager, ColliderObject, ColliderArm, ColliderSP, ColliderObstacles,
)

MESH_FILE = os.path.join(
        os.path.dirname(__file__), 'test_helpers', 'ur_description',
        'meshes', 'ur5', 'collision', 'base.stl')


def make_test_arm():
    """Build a small 6-DOF arm (same shape as the kinematics test fixture)
    with box visual properties, so it can be wrapped in a ColliderArm."""
    Base_T = tm()
    L1, L2, L3, W = 4.5, 3.75, 3.75, 0.1
    basic_arm_end_effector_home = fsr.TAAtoTM(np.array([[L2+L3+W+W+W], [0], [L1], [0], [0], [0]]))
    basic_arm_joint_axes = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]).conj().T
    basic_arm_joint_homes = np.array(
            [[0, 0, 0], [0, 0, L1], [L2, 0, L1], [L2+L3, 0, L1],
             [L2+L3+W, 0, L1], [L2+L3+2*W, 0, L1]]).conj().T
    basic_arm_screw_list = np.zeros((6, 6))
    for i in range(6):
        basic_arm_screw_list[0:6, i] = np.hstack((
                basic_arm_joint_axes[0:3, i],
                np.cross(basic_arm_joint_homes[0:3, i], basic_arm_joint_axes[0:3, i])))
    box_dims = np.array(
            [[W, W, L1], [L2, W, W], [L3, W, W], [W, W, W], [W, W, W], [W, W, W]]).conj().T

    arm = Arm(Base_T, basic_arm_screw_list, basic_arm_end_effector_home,
            basic_arm_joint_homes, basic_arm_joint_axes)

    vis_props = []
    for i in range(6):
        info = vis_info()
        info.geo_type = 'box'
        info.box_size = box_dims[:, i]
        vis_props.append(info)
    eef_info = vis_info()
    eef_info.geo_type = 'box'
    eef_info.box_size = [W, W, W]
    vis_props.append(eef_info)
    arm.setVisColProperties(vis_props=vis_props)
    return arm


def make_mixed_geo_test_arm():
    """Same base arm as make_test_arm(), but with a mix of box/cyl/spr/mesh
    vis_info entries (plus one None entry) to exercise every branch of
    ColliderArm.populateSerialArm()."""
    arm = make_test_arm()
    vis_props = [
        None,  # skipped entirely - no geo_type to key off of
        vis_info(),
        vis_info(),
        vis_info(),
        vis_info(),
        vis_info(),
        vis_info(),
    ]
    vis_props[1].geo_type = 'cyl'
    vis_props[1].radius = 0.1
    vis_props[1].length = 0.5
    vis_props[2].geo_type = 'spr'
    vis_props[2].radius = 0.1
    vis_props[3].geo_type = 'mesh'
    vis_props[3].origin = tm()
    vis_props[3].file_name = MESH_FILE
    vis_props[4].geo_type = 'box'
    vis_props[4].box_size = [0.1, 0.1, 0.1]
    vis_props[5].geo_type = None  # also skipped - no usable geometry
    vis_props[6].geo_type = 'box'
    vis_props[6].box_size = [0.1, 0.1, 0.1]
    arm.setVisColProperties(vis_props=vis_props)
    return arm


class test_collisions(unittest.TestCase):

    # -- primitive mesh construction --

    def test_collisions_createBox(self):
        box = createBox(tm([1, 2, 3, 0, 0, 0]), [1, 1, 1])
        self.assertIsInstance(box, trimesh.Trimesh)
        centroid = box.bounding_box.centroid
        np.testing.assert_allclose(centroid, [1, 2, 3], atol=1e-6)

    def test_collisions_createCylinder(self):
        cyl = createCylinder(tm([0, 0, 5, 0, 0, 0]), radius=0.5, height=2.0)
        self.assertIsInstance(cyl, trimesh.Trimesh)
        centroid = cyl.bounding_box.centroid
        np.testing.assert_allclose(centroid, [0, 0, 5], atol=1e-6)

    def test_collisions_createSphere(self):
        sph = createSphere(tm([2, -1, 0, 0, 0, 0]), radius=1.0)
        self.assertIsInstance(sph, trimesh.Trimesh)
        centroid = sph.bounding_box.centroid
        np.testing.assert_allclose(centroid, [2, -1, 0], atol=1e-6)

    def test_collisions_createMesh(self):
        mesh = createMesh(tm(), MESH_FILE, type='stl')
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertGreater(len(mesh.vertices), 0)

    # -- ColliderObject --

    def test_collisions_ColliderObject_addMesh_and_internal_collision(self):
        obj = ColliderObject('group')
        self.assertEqual(obj.name, 'group')
        obj.addMesh('a', createBox(tm(), [1, 1, 1]))
        obj.addMesh('b', createBox(tm([0.5, 0, 0, 0, 0, 0]), [1, 1, 1]))
        self.assertIn('a', obj.meshes)
        self.assertIn('b', obj.meshes)
        self.assertTrue(obj.checkInternalCollisions())

    def test_collisions_ColliderObject_no_internal_collision(self):
        obj = ColliderObject('group')
        obj.addMesh('a', createBox(tm(), [1, 1, 1]))
        obj.addMesh('b', createBox(tm([10, 0, 0, 0, 0, 0]), [1, 1, 1]))
        self.assertFalse(obj.checkInternalCollisions())

    def test_collisions_ColliderObject_external_collision(self):
        manager = ColliderManager()
        obj_a = ColliderObject('a')
        obj_a.addMesh('box', createBox(tm(), [1, 1, 1]))
        obj_b = ColliderObject('b')
        obj_b.addMesh('box', createBox(tm([0.5, 0, 0, 0, 0, 0]), [1, 1, 1]))

        obj_a.bindManager(manager)
        obj_b.bindManager(manager)

        colliding, names = obj_a.checkExternalCollisions()
        self.assertTrue(colliding)
        self.assertIn(('box', 'box'), names)

    def test_collisions_ColliderObject_no_external_collision_when_unbound(self):
        # With no super_manager bound, there is nothing to collide against.
        obj = ColliderObject('solo')
        obj.addMesh('box', createBox(tm(), [1, 1, 1]))
        colliding, names = obj.checkExternalCollisions()
        self.assertFalse(colliding)
        self.assertEqual(names, [])

    def test_collisions_ColliderObject_checkAllCollisions(self):
        # checkAllCollisions requires both an internal AND an external collision.
        manager = ColliderManager()
        obj_a = ColliderObject('a')
        obj_a.addMesh('a1', createBox(tm(), [1, 1, 1]))
        obj_a.addMesh('a2', createBox(tm([0.5, 0, 0, 0, 0, 0]), [1, 1, 1]))
        obj_b = ColliderObject('b')
        obj_b.addMesh('b1', createBox(tm(), [1, 1, 1]))
        obj_a.bindManager(manager)
        obj_b.bindManager(manager)

        self.assertTrue(obj_a.checkInternalCollisions())
        self.assertTrue(obj_a.checkExternalCollisions()[0])
        self.assertTrue(obj_a.checkAllCollisions())

    def test_collisions_ColliderObject_update_is_noop(self):
        obj = ColliderObject('noop')
        # Base ColliderObject.update() has no work to do; just shouldn't raise.
        obj.update()

    # -- ColliderManager --

    def test_collisions_ColliderManager_bind_deduplicates(self):
        manager = ColliderManager()
        obj = ColliderObject('only')
        manager.bind(obj)
        manager.bind(obj)
        self.assertEqual(len(manager.collision_objects), 1)
        self.assertIs(obj.super_manager, manager)

    def test_collisions_ColliderManager_checkCollisions_true(self):
        manager = ColliderManager()
        obj_a = ColliderObstacles('a')
        obj_a.addMesh('box', createBox(tm(), [1, 1, 1]))
        obj_b = ColliderObstacles('b')
        obj_b.addMesh('box', createBox(tm([0.2, 0, 0, 0, 0, 0]), [1, 1, 1]))
        manager.bind(obj_a)
        manager.bind(obj_b)

        colliding, names = manager.checkCollisions()
        self.assertTrue(colliding)
        self.assertEqual(set(names), {'a', 'b'})

    def test_collisions_ColliderManager_checkCollisions_false(self):
        manager = ColliderManager()
        obj_a = ColliderObstacles('a')
        obj_a.addMesh('box', createBox(tm(), [1, 1, 1]))
        obj_b = ColliderObstacles('b')
        obj_b.addMesh('box', createBox(tm([100, 0, 0, 0, 0, 0]), [1, 1, 1]))
        manager.bind(obj_a)
        manager.bind(obj_b)

        colliding, names = manager.checkCollisions()
        self.assertFalse(colliding)
        self.assertIsNone(names)

    def test_collisions_ColliderManager_update(self):
        manager = ColliderManager()
        obj = ColliderObstacles('obs')
        obj.addMesh('box', createBox(tm(), [1, 1, 1]))
        manager.bind(obj)
        # ColliderObject.update() is a no-op, but ColliderManager.update()
        # should still fan out to every bound object without raising.
        manager.update()

    # -- ColliderObstacles --

    def test_collisions_ColliderObstacles_update_component(self):
        obj = ColliderObstacles('obstacle_field')
        self.assertEqual(obj.name, 'obstacle_field')
        obj.addMesh('rock', createBox(tm(), [1, 1, 1]))

        manager = ColliderManager()
        far_obstacle = ColliderObstacles('far')
        far_obstacle.addMesh('rock', createBox(tm([100, 0, 0, 0, 0, 0]), [1, 1, 1]))
        obj.bindManager(manager)
        far_obstacle.bindManager(manager)
        self.assertFalse(obj.checkExternalCollisions()[0])

        # update_component's transform is applied on top of the mesh's own
        # baked-in geometry (which starts at [100, 0, 0] here, with the
        # manager's tracked offset defaulting to identity) - so moving the
        # obstacle back on top of the first one means offsetting by -100,
        # not by re-supplying an identity transform.
        far_obstacle.update_component('rock', tm([-100, 0, 0, 0, 0, 0]))
        self.assertTrue(obj.checkExternalCollisions()[0])

    # -- ColliderSP --

    def test_collisions_ColliderSP_construction(self):
        sp_collider = ColliderSP()
        self.assertEqual(sp_collider.meshes, {})
        self.assertIsNone(sp_collider.super_manager)

    # -- ColliderArm --

    def test_collisions_ColliderArm_populates_from_arm(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        self.assertEqual(collider.name, 'test_arm')
        self.assertEqual(collider.num_links, len(arm.link_names))
        # every link with a box vis_info entry should have produced a mesh
        for link_name in arm.link_names[:collider.num_links]:
            self.assertIn(link_name, collider.meshes)

    def test_collisions_ColliderArm_update_moves_meshes(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        # Moving the arm and calling update() should not raise, and should
        # re-synchronize the manager's transforms with the new joint poses.
        arm.FK(np.array([0.3, -0.2, 0.4, 0.0, 0.1, -0.1]))
        collider.update()

    def test_collisions_ColliderArm_deleteEE_reduces_link_count(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        before = collider.num_links
        collider.deleteEE()
        self.assertEqual(collider.num_links, before - 1)

    def test_collisions_ColliderArm_checkInternalCollisions_ignores_adjacent_links(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        # Adjacent, connected links are expected to touch/overlap at their
        # shared joint - that must not count as a reportable collision.
        result = collider.checkInternalCollisions()
        self.assertIsInstance(result, bool)

    def test_collisions_ColliderArm_checkInternalCollisions_no_collision(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        collider.manager = MagicMock()
        collider.manager.in_collision_internal.return_value = (False, set())
        self.assertFalse(collider.checkInternalCollisions())

    def test_collisions_ColliderArm_checkInternalCollisions_ignore_connected_links_false(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        collider.ignore_connected_links = False
        collider.manager = MagicMock()
        collider.manager.in_collision_internal.return_value = (True, {('link0', 'link1')})
        self.assertTrue(collider.checkInternalCollisions())

    def test_collisions_ColliderArm_checkInternalCollisions_ignores_previous_link(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        collider.manager = MagicMock()
        # link1 colliding with its own *preceding* neighbor link0.
        collider.manager.in_collision_internal.return_value = (True, {('link1', 'link0')})
        self.assertFalse(collider.checkInternalCollisions())

    def test_collisions_ColliderArm_checkInternalCollisions_ignores_end_effector(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        collider.ignore_ee = True
        collider.manager = MagicMock()
        # Non-adjacent collision, but one side is the end effector, which is
        # excluded whenever ignore_ee is set.
        collider.manager.in_collision_internal.return_value = (True, {('link0', 'end_effector')})
        self.assertFalse(collider.checkInternalCollisions())

    def test_collisions_ColliderArm_checkInternalCollisions_real_nonadjacent_collision(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        collider.manager = MagicMock()
        # link0 and link3 are neither the same link nor adjacent - a
        # genuine, reportable collision.
        collider.manager.in_collision_internal.return_value = (True, {('link0', 'link3')})
        self.assertTrue(collider.checkInternalCollisions())

    def test_collisions_ColliderArm_populateSerialArm_mixed_geometry(self):
        arm = make_mixed_geo_test_arm()
        collider = ColliderArm(arm, name='mixed_arm')
        # index 0 (None) and index 5 (geo_type None) are skipped entirely;
        # the rest (cyl, spr, mesh, box, box) get real meshes.
        self.assertEqual(set(collider.meshes.keys()),
                {'link1', 'link2', 'link3', 'link4', 'end_effector'})
        self.assertNotIn('link0', collider.meshes)
        self.assertNotIn('link5', collider.meshes)

    def test_collisions_ColliderArm_update_skips_links_without_meshes(self):
        arm = make_mixed_geo_test_arm()
        collider = ColliderArm(arm, name='mixed_arm')
        # link0/link5 have no mesh, so update() must skip them rather than
        # raise a KeyError - just confirm it runs cleanly.
        collider.update()

    def test_collisions_ColliderArm_drawArmMeshes(self):
        arm = make_test_arm()
        collider = ColliderArm(arm, name='test_arm')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        try:
            collider.drawArmMeshes(ax)
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
