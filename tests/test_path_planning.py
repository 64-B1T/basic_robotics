import unittest

import numpy as np

from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Arm
from basic_robotics.path_planning.pathplanner import (
    R6Tree, Tree6Node, PathNode, Graph, RRTStar,
)


def _make_arm_for_obstruction_test():
    # Same 6-DOF arm shape used by tests/test_kinematics_arm.py and
    # tests/test_collisions.py's make_test_arm(), but with link_dimensions
    # set directly so armObstruction() can call arm.getLinkDimensions().
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
    arm.setVisColProperties(link_dimensions=box_dims)
    return arm


class test_path_planning(unittest.TestCase):

    # -- PathNode --

    def test_path_planning_PathNode_position_and_parent(self):
        parent = PathNode(tm([0, 0, 0, 0, 0, 0]))
        child = PathNode(tm([1, 0, 0, 0, 0, 0]))
        self.assertIsNone(child.getParent())

        child.setParent(parent)
        self.assertIs(child.getParent(), parent)
        self.assertIn(child, parent.children)

        parent.removeChild(child)
        self.assertNotIn(child, parent.children)

    def test_path_planning_PathNode_getDistance_to_other(self):
        a = PathNode(tm([0, 0, 0, 0, 0, 0]))
        b = PathNode(tm([3, 4, 0, 0, 0, 0]))
        self.assertAlmostEqual(float(a.getDistance(b)), 5.0, places=6)

    def test_path_planning_PathNode_setCost_uses_parent(self):
        parent = PathNode(tm([0, 0, 0, 0, 0, 0]))
        parent.cost = 10.0
        child = PathNode(tm([1, 0, 0, 0, 0, 0]))
        child.setParent(parent)

        child.setCost()

        self.assertAlmostEqual(float(child.getCost()), 11.0, places=6)

    def test_path_planning_PathNode_equality(self):
        a = PathNode(tm([1, 2, 3, 0, 0, 0]))
        b = PathNode(tm([1, 2, 3, 0, 0, 0]))
        c = PathNode(tm([9, 9, 9, 0, 0, 0]))

        self.assertTrue(bool(a == b))
        self.assertFalse(bool(a == c))
        # __eq__ swallows exceptions from non-PathNode comparisons and
        # reports inequality rather than raising.
        self.assertFalse(a == "not a node")

    def test_path_planning_PathNode_arc_mode_distance(self):
        # mode=0 exercises the ArcDistance branch of getDistance(), both
        # against another node and against a node's parent.
        a = PathNode(tm([0, 0, 0, 0, 0, 0]), mode=0)
        b = PathNode(tm([1, 0, 0, 0, 0, 0]), mode=0)
        dist = float(np.asarray(a.getDistance(b)).flatten()[0])
        self.assertGreater(dist, 0.0)

        parent = PathNode(tm([0, 0, 0, 0, 0, 0]), mode=0)
        parent.cost = 5.0
        child = PathNode(tm([1, 0, 0, 0, 0, 0]), mode=0)
        child.setParent(parent)
        child.setCost()

        self.assertGreater(float(np.asarray(child.getCost()).flatten()[0]), 5.0)

    # -- Graph --

    def test_path_planning_Graph_empty_and_seeded(self):
        empty = Graph()
        self.assertEqual(empty.node_list, [])

        seed = PathNode(tm())
        seeded = Graph(seed)
        self.assertEqual(seeded.node_list, [seed])
        self.assertIs(seeded.getNode(0), seed)

    def test_path_planning_Graph_findClosest(self):
        origin = PathNode(tm([0, 0, 0, 0, 0, 0]))
        graph = Graph(origin)
        near = PathNode(tm([1, 0, 0, 0, 0, 0]))
        far = PathNode(tm([100, 0, 0, 0, 0, 0]))
        graph.node_list.append(far)
        graph.node_list.append(near)

        query = PathNode(tm([1.1, 0, 0, 0, 0, 0]))
        closest_index = graph.findClosest(query)

        self.assertIs(graph.getNode(closest_index), near)

    # -- R6Tree --

    def test_path_planning_R6Tree_place_and_count(self):
        tree = R6Tree(6)
        self.assertEqual(tree.getCount(), 0)
        for i in range(5):
            tree.place(PathNode(tm([float(i), 0, 0, 0, 0, 0])))
        self.assertEqual(tree.getCount(), 5)

    def test_path_planning_R6Tree_nearestNeighbors(self):
        tree = R6Tree(6)
        origin_node = PathNode(tm([0, 0, 0, 0, 0, 0]))
        far_node = PathNode(tm([50, 50, 50, 0, 0, 0]))
        tree.place(origin_node)
        tree.place(far_node)

        result = tree.nearestNeighbors(PathNode(tm([0.1, 0.1, 0.1, 0, 0, 0])), 1)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].object.getPosition()[0], 0.0, places=6)

    def test_path_planning_R6Tree_intersection(self):
        tree = R6Tree(6)
        node = PathNode(tm([2, 2, 2, 0, 0, 0]))
        tree.place(node)

        hit = tree.intersection(PathNode(tm([2, 2, 2, 0, 0, 0])))
        miss = tree.intersection(PathNode(tm([200, 200, 200, 0, 0, 0])))

        self.assertEqual(len(hit), 1)
        self.assertEqual(len(miss), 0)

    def test_path_planning_R6Tree_containsBounded(self):
        tree = R6Tree(6)
        tree.place(PathNode(tm([0, 0, 0, 0, 0, 0])))

        inside = tree.containsBounded(
                tm([-1, -1, -1, -10, -10, -10]), tm([1, 1, 1, 10, 10, 10]))
        outside = tree.containsBounded(
                tm([100, 100, 100, -10, -10, -10]), tm([101, 101, 101, 10, 10, 10]))

        self.assertTrue(inside)
        self.assertFalse(outside)

    def test_path_planning_R6Tree_getAll(self):
        tree = R6Tree(6)
        placed = [PathNode(tm([float(i), 0, 0, 0, 0, 0])) for i in range(4)]
        for node in placed:
            tree.place(node)

        self.assertEqual(len(tree.getAll()), len(placed))

    def test_path_planning_R6Tree_getNeighborChain(self):
        tree = R6Tree(6)
        root = PathNode(tm([0, 0, 0, 0, 0, 0]))
        mid = PathNode(tm([1, 0, 0, 0, 0, 0]))
        mid.setParent(root)
        leaf = PathNode(tm([2, 0, 0, 0, 0, 0]))
        leaf.setParent(mid)
        tree.place(root)
        tree.place(mid)
        tree.place(leaf)

        chain = tree.getNeighborChain(leaf, 5)

        self.assertEqual(chain, [mid, root])

    def test_path_planning_R6Tree_3d_mode(self):
        # R6Tree also supports a 3-dimensional index (dimension != 6).
        tree = R6Tree(3)
        tree.place(PathNode(tm([1, 1, 1, 0, 0, 0])))
        result = tree.nearestNeighbors(PathNode(tm([1, 1, 1, 0, 0, 0])), 1)
        self.assertEqual(len(result), 1)
        self.assertTrue(tree.containsBounded(tm([0, 0, 0, 0, 0, 0]), tm([2, 2, 2, 0, 0, 0])))
        hit = tree.intersection(PathNode(tm([1, 1, 1, 0, 0, 0])))
        miss = tree.intersection(PathNode(tm([200, 200, 200, 0, 0, 0])))
        self.assertEqual(len(hit), 1)
        self.assertEqual(len(miss), 0)

    # -- Tree6Node (alternate/experimental octree-style node, smoke test only) --

    def test_path_planning_Tree6Node_place_and_size(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        self.assertEqual(root.getSize(), 1)
        root.place(PathNode(tm([1, 1, 1, 1, 1, 1])))
        self.assertEqual(root.getSize(), 2)

    def test_path_planning_Tree6Node_find(self):
        root_node = PathNode(tm([0, 0, 0, 0, 0, 0]))
        root = Tree6Node(root_node)
        child_node = PathNode(tm([1, 1, 1, 1, 1, 1]))
        root.place(child_node)

        found = root.find(child_node)

        self.assertEqual(found, child_node)

    def test_path_planning_Tree6Node_find_missing_child_returns_negative_one(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        missing = PathNode(tm([-1, -1, -1, -1, -1, -1]))

        self.assertEqual(root.find(missing), -1)

    def test_path_planning_Tree6Node_find_recurses_through_deeper_branch(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        mid_node = PathNode(tm([1, 1, 1, 1, 1, 1]))
        root.place(mid_node)
        leaf_node = PathNode(tm([2, 2, 2, 2, 2, 2]))
        root.place(leaf_node)

        found = root.find(leaf_node)

        self.assertEqual(found, leaf_node)

    def test_path_planning_Tree6Node_delete_returns_false_when_not_found_in_deeper_branch(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        mid_node = PathNode(tm([1, 1, 1, 1, 1, 1]))
        root.place(mid_node)
        leaf_node = PathNode(tm([2, 2, 2, 2, 2, 2]))
        root.place(leaf_node)
        # Lands in the same top-level child slot as mid_node/leaf_node but
        # doesn't match anything further down that branch.
        phantom = PathNode(tm([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

        self.assertFalse(root.delete(phantom))
        self.assertEqual(root.getSize(), 3)

    def test_path_planning_Tree6Node_printStatus_smoke(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        root.place(PathNode(tm([1, 1, 1, 1, 1, 1])))

        root.printStatus()  # Should not raise.

    def test_path_planning_Tree6Node_find_no_match_returns_negative_one(self):
        root_node = PathNode(tm([0, 0, 0, 0, 0, 0]))
        root = Tree6Node(root_node)
        placed_node = PathNode(tm([1, 1, 1, 1, 1, 1]))
        root.place(placed_node)

        # Same child index as placed_node (all dims >= root's) but not equal to it.
        mismatched_query = PathNode(tm([2, 2, 2, 2, 2, 2]))

        self.assertEqual(root.find(mismatched_query), -1)

    def test_path_planning_Tree6Node_delete_missing_child_returns_false(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        missing_node = PathNode(tm([-1, -1, -1, -1, -1, -1]))

        self.assertFalse(root.delete(missing_node))

    def test_path_planning_Tree6Node_delete_leaf_child(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        leaf_node = PathNode(tm([1, 1, 1, 1, 1, 1]))
        root.place(leaf_node)
        self.assertEqual(root.getSize(), 2)

        deleted = root.delete(leaf_node)

        self.assertTrue(deleted)
        self.assertEqual(root.getSize(), 1)
        ind = root.findChildInd(leaf_node)
        self.assertIsNone(root.children[ind])

    def test_path_planning_Tree6Node_delete_recurses_into_grandchild(self):
        root = Tree6Node(PathNode(tm([0, 0, 0, 0, 0, 0])))
        mid_node = PathNode(tm([1, 1, 1, 1, 1, 1]))
        root.place(mid_node)
        leaf_node = PathNode(tm([2, 2, 2, 2, 2, 2]))
        root.place(leaf_node)
        self.assertEqual(root.getSize(), 3)

        deleted = root.delete(leaf_node)

        self.assertTrue(deleted)
        self.assertEqual(root.getSize(), 2)

    # -- RRTStar --

    def test_path_planning_RRTStar_default_construction(self):
        planner = RRTStar()
        self.assertEqual(planner.r6_tree_graph.getCount(), 1)

    def test_path_planning_RRTStar_obstruction(self):
        planner = RRTStar(tm())
        planner.addObstruction([-1, -1, -1], [1, 1, 1])

        blocked = planner.obstruction(
                PathNode(tm([-5, 0, 0, 0, 0, 0])), PathNode(tm([5, 0, 0, 0, 0, 0])))
        clear = planner.obstruction(
                PathNode(tm([-5, 10, 10, 0, 0, 0])), PathNode(tm([5, 10, 10, 0, 0, 0])))

        self.assertTrue(blocked)
        self.assertFalse(clear)

    def test_path_planning_RRTStar_obstruction_early_axis_rejections(self):
        # Exercises each of the separating-axis "continue" branches in
        # obstruction() (segment cleanly misses the box along a single
        # world axis) that the basic blocked/clear test above doesn't reach.
        planner = RRTStar(tm())
        planner.addObstruction([-1, -1, -1], [1, 1, 1])

        # Misses along x.
        self.assertFalse(planner.obstruction(
                PathNode(tm([-13, -11, -13, 0, 0, 0])),
                PathNode(tm([-12, 10, -7, 0, 0, 0]))))
        # Misses along z.
        self.assertFalse(planner.obstruction(
                PathNode(tm([1, 6, -7, 0, 0, 0])),
                PathNode(tm([-13, -11, -13, 0, 0, 0]))))

    def test_path_planning_RRTStar_obstruction_cross_axis_rejections(self):
        # Exercises the three separating-axis "continue" branches based on
        # cross products of the segment and box-center offsets.
        planner = RRTStar(tm())
        planner.addObstruction([-1, -1, -1], [1, 1, 1])
        base = PathNode(tm([1, 6, -7, 0, 0, 0]))

        self.assertFalse(planner.obstruction(base, PathNode(tm([0, -3, 0, 0, 0, 0]))))
        self.assertFalse(planner.obstruction(base, PathNode(tm([-14, -5, 9, 0, 0, 0]))))
        self.assertFalse(planner.obstruction(base, PathNode(tm([-6, -10, 14, 0, 0, 0]))))

    def test_path_planning_RRTStar_armObstruction_short_circuits_on_direct_obstruction(self):
        planner = RRTStar(tm())
        planner.addObstruction([-1, -1, -1], [1, 1, 1])
        arm = _make_arm_for_obstruction_test()
        start = PathNode(tm([-5, 0, 0, 0, 0, 0]))
        end = PathNode(tm([5, 0, 0, 0, 0, 0]))

        self.assertTrue(planner.armObstruction(arm, start, end))

    def test_path_planning_RRTStar_armObstruction_walks_arm_links(self):
        planner = RRTStar(tm())
        arm = _make_arm_for_obstruction_test()
        start = PathNode(tm([-5, 10, 10, 0, 0, 0]))
        end = PathNode(tm([5, 10, 10, 0, 0, 0]))

        # No obstructions are registered, and the per-segment corner check
        # inside armObstruction always raises internally (it compares raw
        # corner indices rather than points, see final report) and is
        # swallowed by a bare except, so this always resolves to False.
        self.assertFalse(planner.armObstruction(arm, start, end))

    def test_path_planning_RRTStar_armObstruction_advances_start_index_and_skips_none_pose(self):
        # A duck-typed stand-in for Arm: its first screw column has a linear
        # part summing to exactly 1, which advances armObstruction's
        # start_index loop once, and its joint transforms include a None
        # entry to exercise the "if poses[i] is None: continue" branch.
        class FakeArm:
            def getScrewList(self):
                return np.array([
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ], dtype=float)

            def getJointTransforms(self):
                return [tm(), tm(), None, tm([2, 0, 0, 0, 0, 0])]

            def getLinkDimensions(self):
                return np.ones((3, 3))

        planner = RRTStar(tm())
        start = PathNode(tm([-5, 10, 10, 0, 0, 0]))
        end = PathNode(tm([5, 10, 10, 0, 0, 0]))

        self.assertFalse(planner.armObstruction(FakeArm(), start, end))

    def test_path_planning_RRTStar_generateTerrain_adds_grid_of_obstructions(self):
        planner = RRTStar(tm())
        self.assertEqual(len(planner.obstructions), 0)

        planner.generateTerrain(xd=4, yd=4, xc=2, yc=2, zvar=1)

        self.assertEqual(len(planner.obstructions), 4)

    def test_path_planning_RRTStar_generalGenerateTree_retries_on_invalid_sample(self):
        planner = RRTStar(tm())
        planner.iterations = 2

        calls = {'n': 0}

        def gen():
            calls['n'] += 1
            if calls['n'] == 1:
                # Too far from the (single) existing node: forces a retry.
                return PathNode(tm([1000, 0, 0, 0, 0, 0]))
            return PathNode(tm([float(calls['n']), 0, 0, 0, 0, 0]))

        planner.generalGenerateTree(gen, planner.distance, lambda a, b: False)

        self.assertEqual(calls['n'], 3)
        self.assertEqual(planner.r6_tree_graph.getCount(), 3)

    def test_path_planning_RRTStar_generateTreeDual_is_confirmed_broken(self):
        # generateTreeDual() calls the undefined self.obstructed(...) (only
        # self.obstruction(...) exists) on its very first iteration -- see
        # final report for the full analysis of why this isn't a simple
        # rename fix. Documented here rather than silently left untested.
        planner = RRTStar(tm())
        planner.iterations = 3

        with self.assertRaises(AttributeError):
            planner.generateTreeDual()

    def test_path_planning_RRTStar_findPathGeneral(self):
        planner = RRTStar(tm())
        planner.iterations = 5
        goal = tm([1, 1, 1, 0, 0, 0])

        path = planner.findPathGeneral(planner.generateTree, goal)

        self.assertGreaterEqual(len(path), 2)
        self.matrix_close(path[-1].gTAA(), goal.gTAA())

    def test_path_planning_RRTStar_findPathDual_with_stubbed_tree_generation(self):
        # generateTreeDual() itself is confirmed broken (see final report:
        # it calls the undefined self.obstructed(...) on a list of rtree
        # hits rather than PathNode objects), so it is stubbed out here to
        # exercise findPathDual()'s own bookkeeping in isolation.
        planner = RRTStar(tm())
        planner.generateTreeDual = lambda: None
        goal = tm([2, 2, 2, 0, 0, 0])

        path = planner.findPathDual(goal)

        self.assertEqual(planner.r6_tree_graph.getCount(), 2)
        self.assertGreaterEqual(len(path), 2)
        self.matrix_close(path[-1].gTAA(), goal.gTAA())

    def test_path_planning_RRTStar_distance_modes(self):
        planner = RRTStar(tm())
        planner.dmode = 0
        straight = float(planner.distance(tm(), tm([3, 4, 0, 0, 0, 0])))
        self.assertAlmostEqual(straight, 5.0, places=6)

        planner.dmode = 1
        arc = float(np.asarray(planner.distance(tm(), tm([1, 0, 0, 0, 0, 0]))).flatten()[0])
        self.assertGreater(arc, 0.0)

    def test_path_planning_RRTStar_randomPos_within_bounds(self):
        planner = RRTStar(tm())
        for _ in range(20):
            node = planner.randomPos()
            pos = node.getPosition()
            for i in range(6):
                self.assertGreaterEqual(pos[i], planner.bounds[i][0])
                self.assertLessEqual(pos[i], planner.bounds[i][1])

    def test_path_planning_RRTStar_generateTree_and_findPath(self):
        # Keep this fast: a handful of iterations and no obstructions is
        # enough to exercise generateTree()/findPath() end to end.
        planner = RRTStar(tm())
        planner.iterations = 8
        planner.generateTree()

        self.assertGreater(planner.r6_tree_graph.getCount(), 1)

        goal = tm([1, 1, 1, 0, 0, 0])
        path = planner.findPath(goal)

        self.assertGreaterEqual(len(path), 2)
        self.matrix_close(path[-1].gTAA(), goal.gTAA())

    def matrix_close(self, a, b, atol=1e-6):
        np.testing.assert_allclose(np.asarray(a).flatten(), np.asarray(b).flatten(), atol=atol)


if __name__ == '__main__':
    unittest.main()
