import unittest

import numpy as np

from basic_robotics.general import tm
from basic_robotics.path_planning.pathplanner import (
    R6Tree, Tree6Node, PathNode, Graph, RRTStar,
)


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
