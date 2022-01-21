from basic_robotics.faser_math import faser_general as fsr
from basic_robotics.faser_math import tm
import numpy as np
from rtree import index
import random
from faser_utils.disp.disp import *

#Created by Liam Chapin


class R6Tree:
    """
    Six dimensional easily searched storage system for nodes
    Args:
        dimension (Int): dimensionality of the space

    Attributes:
        idx (rtree.index): Rtree index
        count (Int): PathNode Count
        dimension: dimensionality of the space

    """

    def __init__(self, dimension = 6):
        """
        Init new R6Tree
        Args:
            dimension (Int): dimensionality of the space

        Returns:
            R6Tree: R6Tree object instance

        """
        p = index.Property()
        self.dimension = dimension
        p.dimension = self.dimension
        self.idx = index.Index(properties=p)
        self.count = 0

    def place(self, node):
        """
        Place a new node into the R6Tree
        Args:
            node (PathNode): PathNode to operate on
        """
        b = node.getPosition()
        #self.idx.insert(100,
        #       (b[0], b[0], b[1], b[1], b[2], b[2], b[3], b[3], b[4], b[4], b[5], b[5]), node)
        if self.dimension == 6:
            self.idx.insert(100,
                (b[0], b[1], b[2], b[3], b[4], b[5], b[0], b[1], b[2], b[3], b[4], b[5]), node)
        else:
            self.idx.insert(100,
                (b[0], b[1], b[2], b[0], b[1], b[2]), node)
        self.count+=1

    def nearestNeighbors(self, node, n):
        """
        Get the next n nearest Neighbors to the current node
        Args:
            node (PathNode): PathNode to operate on
            n (Int): Number of nearest neighbors to gather

        Returns:
            List: List of Pathnode nearest neihbors n

        """
        b = node.getPosition()
        #return list(self.idx.nearest(
        #    (b[0], b[0], b[1], b[1], b[2], b[2], b[3], b[3], b[4], b[4], b[5], b[5]), n))
        if self.dimension == 6:
            return list(
                self.idx.nearest(
                (b[0], b[1], b[2], b[3], b[4], b[5], b[0], b[1], b[2], b[3], b[4], b[5]),
                n, objects = True))
        else:
            return list(self.idx.nearest((b[0], b[1], b[2], b[0], b[1], b[2]), n, objects = True))

    def intersection(self, node):
        """
        Get the list of nodes intersecting the current in any dimension
        Args:
            node (PathNode): PathNode to operate on

        Returns:
            List: list of nodes intersecting in some fashion

        """
        b = node.getPosition()
        #return list(self.idx.intersection(
        #   (b[0], b[0], b[1], b[1], b[2], b[2], b[3], b[3], b[4], b[4], b[5], b[5])))
        if self.dimension == 6:
            return list(
                self.idx.intersection(
                (b[0], b[1], b[2], b[3], b[4], b[5], b[0], b[1], b[2], b[3], b[4], b[5])))
        else:
            return list(self.idx.intersection((b[0], b[1], b[2], b[0], b[1], b[2])))

    def containsBounded(self, L, R):
        """
        Determine if there are nodes in the region bounded by L and R
        Args:
            L (tm): Lower Left bound of rectangular obstruction region
            R (tm): Upper Right bound of rectangular obstruction region

        Returns:
            Bool: Success

        """
        intersects = []
        if self.dimension == 6:
            intersects =  list(
                self.idx.intersection(
                (L[0], L[1], L[2], L[3], L[4], L[5], R[0], R[1], R[2], R[3], R[4], R[5])))
        else:
            intersects =  list(
                self.idx.intersection(
                (L[0], L[1], L[2], R[0], R[1], R[2])))
        return not (len(intersects) == 0)

    def getAll(self):
        """
        Get all nodes in the graph
        Returns:
            List(PathNode): All Nodes in the graph

        """
        return self.nearestNeighbors(PathNode(tm()), self.count)

    def getNeighborChain(self, node, num):
        """
        Get the chain of parent nodes to number or to origin
        Args:
            node (PathNode): PathNode to operate on
            num (Int): Number of neighbors to return

        Returns:
            List(PathNode): Chain back to origin or to number

        """
        chain = []
        for i in range(num):
            if(node.getParent() is not None):
                chain.append(node.getParent())
                node = node.getParent()
            else:
                break
        return chain

    def getCount(self):
        """
        Get count of nodes in the graph
        Returns:
            Int: Count of nodes in graph

        """
        return self.count

class Tree6Node:
    """
    Tree6Node, alternate node implementation
    Args:
        node (PathNode): PathNode to operate on
        parent (PathNode): Parent of the specified PathNode

    Attributes:
        data (PathNode): `data`.
        size (Int): `size`.
        level (Int): `level`.
        children (List(PathNode)): `children`.
        parent

    """

    def __init__(self, node, parent = None):
        """
        Create new Tree6Node
        Args:
            node (PathNode): PathNode to operate on
            parent (PathNode): Parent of the specified PathNode

        Returns:
            Tree6Node: Tree6Node Object

        """
        self.data = node
        self.parent = parent
        self.size = 1
        self.level = 1
        self.children = [None] * 64

    def place(self, node):
        """
        Place new node onto current as child
        Args:
            node (PathNode): PathNode to operate on
        """
        ind = self.findChildInd(node)
        if self.children[ind] == None:
            self.children[ind] = Tree6Node(node, self)
            self.children[ind].level = self.level + 1
            self.size += 1
            return
        else:
            self.children[ind].place(node)
            self.size += 1
            return

    def getSize(self):
        """
        Get size of nodes dependent on this one
        Returns:
            Int: size of node children list

        """
        return self.size

    def find(self, node):
        """
        Find a specific child
        Args:
            node (PathNode): PathNode to operate on

        Returns:
            PathNode: Find a child that matches node

        """
        ind = self.findChildInd(node)
        if (self.children[ind] == None):
            return -1
        if (self.children[ind].getSize() == 1):
            if (self.children[ind].data == node):
                return self.children[ind].data
            print("Didn't Match")
            print(node.getPosition())
            print(self.children[ind].data.getPosition())
            return -1
        return self.children[ind].find(node)


    def delete(self, node):
        """
        Delete a node, if it is one of this node's children
        Args:
            node (PathNode): PathNode to operate on

        Returns:
            Boolean: Success

        """
        ind = self.findChildInd(self, node)
        if (self.children[ind] == None):
            return False
        if (self.children[ind].getSize() == 1):
            self.children[ind] == None
            self.size -= 1
            return True
        if self.children[ind].delete(node):
            self.size -= 1
            return True
        return False

    def findChildInd(self, node):
        """
        Find index of specific child node
        Args:
            node (PathNode): PathNode to operate on

        Returns:
            Int: Index of desired child

        """
        ind = 0
        if(node.getPosition()[0] < self.data.getPosition()[0]):
            ind += 32
        if(node.getPosition()[1] < self.data.getPosition()[1]):
            ind += 16
        if(node.getPosition()[2] < self.data.getPosition()[2]):
            ind += 8
        if(node.getPosition()[3] < self.data.getPosition()[3]):
            ind += 4
        if(node.getPosition()[4] < self.data.getPosition()[4]):
            ind += 2
        if(node.getPosition()[5] < self.data.getPosition()[5]):
            ind += 1
        return ind

    def printStatus(self):
        """
        Print current setup
        """
        print(self.printHelper(""))


    def printHelper(self, strfx):
        """
        Helper function for printStatus
        Args:
            strfx (String): current string data

        Returns:
            String: Current string data

        """
        strx = strfx + self.data.getPosition().__str__() + "\n"
        for i in range(64):
            if (self.children[i] == None):
                #strx += strfx + "Empty" + "\n"
                a = 0
            else:
                strx += self.children[i].printHelper(strfx + "  ")
        return strx




class PathNode:

    def __init__(self, position = None, parent = None, mode = 3):
        """
        Creates a new PathNode object
        Args:
            position (tm): transformation of the pathnode
            parent (PathNode): Parent of the specified PathNode
            mode (Int): Distance mode

        Returns:
            PathNode: New PathNode instance

        """
        self.position = position
        self.parent = parent
        self.mode = mode
        self.children = []
        self.cost = 0
        self.type = 0

    def setChild(self, child):
        """
        Appends a child to this node.
        Args:
            child (PathNode): Child of specified PathNode
        """
        self.children.append(child)

    def setParent(self, parent):
        """
        Set the parent of this node
        Args:
            parent (PathNode): Parent of the specified PathNode
        """
        parent.setChild(self)
        self.parent = parent

    def removeChild(self, child):
        """
        Remove a child from this node
        Args:
            child (PathNode): Child of specified PathNode
        """
        if child in self.children:
            self.children.remove(child)

    def setCost(self):
        """
        Set the cost of this node
        """
        self.cost = self.parent.getCost() +self.getDistance()

    def getDistance(self, other = None):
        """
        Get distance from this node to another node
        Args:
            other (PathNode): Other PathNode to get distance to

        Returns:
            Float: Distance between this and other

        """
        if other is not None:
            if(self.mode == 3):
                return fsr.Distance(self.getPosition(), other.getPosition())
            else:
                return fsr.ArcDistance(self.getPosition(), other.getPosition())
        if(self.mode == 3):
            self.cost = (self.parent.getCost() +
                fsr.Distance(self.position, self.parent.getPosition()))

    def getCost(self):
        """
        Get the current cost of this node
        Returns:
            Float: current cost of reaching node

        """
        return self.cost

    def getPosition(self):
        """
        Get the position of this node
        Returns:
            tm: position of node

        """
        return self.position

    def getParent(self):
        """
        Get this node's parent
        Returns:
            PathNode: parent of this node

        """
        return self.parent

    def __eq__(self, a):
        """
        Determine if this node is equal to another node
        Args:
            a (PathNode): Other PathNode

        Returns:
            Bool: equality to other node

        """
        try:
            eq = sum(abs(self.getPosition().gTAA()-a.getPosition().gTAA())) < .0001
            return eq
        except:
            return False

class Graph:

    def __init__(self, init = None):
        """
        Creates a new Graph object
        Args:
            init (List(PathNode)): Initialization list

        Returns:
            Graph: New Graph Object

        """
        self.node_list = []

        if init is not None:
            self.node_list.append(init)

    def findClosest(self, node):
        """
        Find the closest node to a specified node within the graph
        Args:
            node (PathNode): PathNode to operate on

        Returns:
            Int: Index of closest node

        """
        close_index = 0
        max_distance = 9999999999
        for i in range(len(self.node_list)):
            dist = self.node_list[i].getDistance(node)
            if dist < max_distance:
                max_distance = dist
                close_index = i
        return close_index

    def getNode(self, ind):
        """
        get a node at a specific index in the node list
        Args:
            ind (Int): Index of PathNode in node_list

        Returns:
            PathNode: Node at index in the node_list

        """
        return self.node_list[ind]

class RRTStar:
    """
    Create complex paths through unpredictable terrain for robots to follow
    Args:
        origin (tm): Origin point (Start location)

    Attributes:
        dimension (Int): dimensionality of the space
        dmode (Int): Distance Mode
        r6_tree_graph (R6Tree): R6Tree Graph object
        obstructions (List): List of obstructions in the space
        bounds (List): Bounds of the search space
        path_rejection_distance (Float): what is too far to connect?
        nearest_neighbors_limit (Int): amount to limit nearest neighbors to
        maximum_distance (Float): maximum distance to path to
        minimum_distance (Float): minimim distance to spawn new point
        iterations (Int): Number of allowable iterations

    """
    #Adapted from here https://arxiv.org/pdf/1105.1186.pdf
    def __init__(self, origin = None):
        """
        Initialize new RRTStar object
        Args:
            origin (tm): Origin point (Start location)

        Returns:
            RRTStar: New RRTStar object

        """

        self.dimension = 6
        self.dmode = 0
        self.r6_tree_graph = R6Tree(self.dimension)
        self.obstructions = []
        self.bounds = [[-10, 10],
                [-10, 10],
                [-10, 10],
                [-2*np.pi, 2*np.pi],
                [-2*np.pi, 2*np.pi],
                [-2*np.pi, 2*np.pi]]
        self.path_rejection_distance = 5
        self.nearest_neighbors_limit = 15
        self.maximum_distance = 100
        self.minimum_distance = 0.1
        self.iterations = 1500
        if origin is not None:
            self.r6_tree_graph.place(PathNode(origin))
        else:
            self.r6_tree_graph.place(PathNode(tm()))

    def obstructed(self, node, node2):
        """
        Return if path between nodes is obstructed
        Args:
            node (PathNode): PathNode to operate on
            node2 (PathNode): Desired second node

        Returns:
            Bool: Whether or not given node pairing is obstructed

        """
        for obstruction in self.obstructions:
            pos = node.getPosition()
            if (pos[0] >= obstruction[0][0] and pos[0] <= obstruction[1][0]
             and pos[1] >= obstruction[0][1] and pos[1] <= obstruction[1][1]
             and pos[2] >= obstruction[0][2] and pos[2] <= obstruction[1][2]
             and pos[3] >= obstruction[0][3] and pos[3] <= obstruction[1][3]
             and pos[4] >= obstruction[0][4] and pos[4] <= obstruction[1][4]
             and pos[5] >= obstruction[0][5] and pos[5] <= obstruction[1][5]):
                return True
        return False

    def obstruction(self, n1, n2):
        """
        return if path between nodes is obstructed
        Args:
            n1 (PathNode): First Path Node
            n2 (PathNode): Second Path Node

        Returns:
            Bool: whether or not given node pairing is obstructed

        """
        for obstruction_object in self.obstructions:
            point_1 = n1.getPosition()
            point_2 = n2.getPosition()
            bounds_1 = obstruction_object[0]
            bounds_2 = obstruction_object[1]

            mid = np.array([(bounds_2[0] + bounds_1[0])/2,
                (bounds_2[1] + bounds_1[1])/2,
                (bounds_2[2] + bounds_1[2])/2])
            a = np.array([point_1[0] - mid[0], point_1[1] - mid[1], point_1[2] - mid[2]])
            b = np.array([point_2[0] - mid[0], point_2[1] - mid[1], point_2[2] - mid[2]])

            extents = np.abs(bounds_2[0:3].reshape((3))-mid)

            midpoint_ab = (a + b) / 2
            L = (a - midpoint_ab)
            abs_obstruct = np.abs(L)

            if abs(midpoint_ab[0]) > extents[0] + abs_obstruct[0]:
                continue
            if abs(midpoint_ab[1]) > extents[1] + abs_obstruct[1]:
                continue
            if abs(midpoint_ab[2]) > extents[2] + abs_obstruct[2]:
                continue

            if (abs(midpoint_ab[1] * L[2] - midpoint_ab[2] * L[1]) >
                (extents[1] * abs_obstruct[2] + extents[2] * abs_obstruct[1])):
                continue
            if (abs(midpoint_ab[0] * L[2] - midpoint_ab[2] * L[0]) >
                (extents[0] * abs_obstruct[2] + extents[2] * abs_obstruct[0])):
                continue
            if (abs(midpoint_ab[0] * L[1] - midpoint_ab[1] * L[0]) >
                (extents[0] * abs_obstruct[1] + extents[1] * abs_obstruct[0])):
                continue
            return True
        return False

    def armObstruction(self, arm, x, y):
        """
        Determine if path between nodes is obstructed when constrained to serial arm kinematics
        Args:
            arm (Arm): FASER-Arm object
            x (PathNode): Path Node Start
            y (PathNode): Path Node Finish

        Returns:
            Bool: whether or not given node pairing is obstructed

        """
        if(self.obstruction(x, y)):
            return True
        start_index = 0

        while (sum(arm.getScrewList()[3:6, start_index]) == 1):
            start_index = start_index + 1

        poses = arm.getJointTransforms()
        for i in range(start_index, len(poses[start_index:])):
            if poses[i] is None:
                continue
        link_dimensions = arm.getLinkDimensions().T
        degrees_of_freedom = arm.getScrewList().shape[1]
        for i in range(start_index, degrees_of_freedom):
            try:
                link_to_link_next_midpoint = fsr.TMMidPoint(poses[i], poses[i+1])
                link_midpoint_facing_next = fsr.TMMidRotAdjust(
                        link_to_link_next_midpoint, poses[i], poses[i+1], mode = 1)
                link_dimensions_next = link_dimensions[i+1, 0:3]
                dx = link_dimensions_next[0]
                dy = link_dimensions_next[1]
                dz = link_dimensions_next[2]
                corners = .5 * np.array([
                    [-dx, -dy, -dz],
                    [dx, -dy, -dz],
                    [-dx, dy, -dz],
                    [dx, dy, -dz],
                    [-dx, -dy, dz],
                    [dx, -dy, dz],
                    [-dx, dy, dz],
                    [dx, dy, dz]]).T
                link_collision_corners = np.zeros((3, 8))
                for i in range(0, 8):
                    h = link_midpoint_facing_next.gTM() @ np.array([
                        [corners[0, i]],
                        [corners[1, i]],
                        [corners[2, i]],
                        [1]])
                    link_collision_corners[0:3, i] = np.squeeze(h[0:3])
                segments = np.array([
                    [1, 2], [1, 3], [2, 4],
                    [3, 4], [1, 5], [2, 6],
                    [3, 7], [4, 8], [5, 6],
                    [5, 7], [6, 8], [7, 8]])-1
                #disp(link_collision_corners)
                for i in range(12):
                    a = segments[i, 0]
                    b = segments[i, 1]
                    if self.obstruction(a, b):
                        return True
            except:
                pass
        return False

    def addObstruction(self, L, R):
        """
        Add an obstruction bounded rectangularly by L and R
        Args:
            L (tm): Lower Left bound of rectangular obstruction region
            R (tm): Upper Right bound of rectangular obstruction region
        """
        self.obstructions.append([
            tm([L[0], L[1], L[2], -2*np.pi, -2*np.pi, -2*np.pi]),
            tm([R[0], R[1], R[2], 2*np.pi, 2*np.pi, 2*np.pi])])

    def generateTerrain(self, xd, yd, xc, yc, zvar, xs=0, ys=0):
        """
        Generate some random terrain to use as obstructions
        Args:
            xd (Float): terrain block x dimension
            yd (Float): terrain block y dimension
            xc (Float): terrain total distance x
            yc (Float): terrain total distance y
            zvar (Float): terrain height varaince
            xs (Float): terrain x minimum
            ys (Float): terrain y minimum
        """
        cx = int(xd/xc)
        cy = int(yd/yc)
        for i in range(cx):
            for j in range(cy):
                h = random.uniform(0.1, zvar + .1)
                self.addObstruction(
                        [xc * i + xs,
                        yc * j + ys, 0.1],
                        [xc * (i+1) + xs,
                        yc*(j+1) + ys, h])

    def randomPos(self):
        """
        Create a random pathNode
        Returns:
            PathNode: Randomly placed new node

        """
        pos = [None] * 6
        for i in range(6):
            pos[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
        pose_transform = tm(pos)
        new_path_node = PathNode(pose_transform)
        return new_path_node

    def distance(self, pos1, pos2):
        """
        Get distance from one point to another dependent on distance mode
        Args:
            pos1 (tm): transform 1
            pos2 (tm): transform 2

        Returns:
            Float: distance between two transforms

        """
        if self.dmode == 1:
            return fsr.ArcDistance(pos1, pos2)
        else:
            return fsr.Distance(pos1, pos2)

    def generalGenerateTree(self, randomGenerator, distanceFunction, collisionDetector):
        """
        generate a general RRT
        Args:
            randomGenerator (Func): random pathNode generation function handle
            distanceFunction (Func): distance calculation function
            collisionDetector (Func): collision detection function
        """
        for i in range(self.iterations):
            progressBar(i, self.iterations-1)
            new_node = randomGenerator()
            nearest = self.r6_tree_graph.nearestNeighbors(new_node, 1)
            dist = distanceFunction(new_node.getPosition(), nearest[0].object.getPosition())
            while (dist > self.maximum_distance or
                    dist < self.minimum_distance or
                    collisionDetector(new_node, nearest[0].object)):
                new_node = randomGenerator()
                nearest = self.r6_tree_graph.nearestNeighbors(new_node, 1)
                dist = distanceFunction(new_node.getPosition(), nearest[0].object.getPosition())
            if len(nearest) == 0:
                self.r6_tree_graph.place(new_node)
                continue
            new_node_cost = (distanceFunction(new_node.getPosition(),
                nearest[0].object.getPosition()) + nearest[0].object.getCost())
            new_node.cost = new_node_cost
            new_node.setParent(nearest[0].object)
            nearest = self.r6_tree_graph.nearestNeighbors(new_node, self.nearest_neighbors_limit)
            #nearest = self.r6_tree_graph.getNeighborChain(new_node, 15)
            for j in range(0, len(nearest)):
                #if new_node.cost + self.Distance(new_node.getPosition(),
                #       nearest[i].object.getPosition()) < nearest[i].object.getCost():
                #    nearest[i].object.cost = new_node.cost + self.Distance(new_node.getPosition(),
                #       nearest[i].object.getPosition())
                #    nearest[i].object.setParent(new_node)
                if (distanceFunction(
                        new_node.getPosition(), nearest[j].object.getPosition()) +
                        nearest[j].object.getCost() < new_node.cost and not
                        collisionDetector(new_node, nearest[j].object)):
                    new_node.cost = (distanceFunction(
                            new_node.getPosition(), nearest[j].object.getPosition()) +
                            nearest[j].object.getCost())
                    new_node.setParent(nearest[j].object)
            #for i in range(0, len(nearest)):
            #    if new_node.cost + self.Distance(new_node.getPosition(),
            #nearest[i].getPosition()) < nearest[i].getCost():
            #        nearest[i].cost = new_node.cost + self.Distance(new_node.getPosition(),
            #nearest[i].getPosition())
            #        nearest[i].setParent(new_node)
            self.r6_tree_graph.place(new_node)

    def generateTree(self):
        """
        Generate the tree based on bound functions
        """
        self.generalGenerateTree(
            lambda : self.randomPos(),
            lambda x, y : self.distance(x, y),
            lambda x, y : self.obstruction(x, y))

    def generateTreeDual(self):
        """
        Generate a dual searching tree from origin and goal that attempts to unite in the middle
        """
        for i in range(self.iterations):
            progressBar(i, self.iterations)
            new_node = self.randomPos()
            nearest = self.r6_tree_graph.nearestNeighbors(new_node, 1)
            while (not self.obstructed(new_node)
                    or self.distance(new_node.getPosition(),
                    nearest[0].object.getPosition()) > self.maximum_distance):
                new_node = self.randomPos()
                nearest = self.r6_tree_graph.nearestNeighbors(new_node, 1)
            if len(nearest) == 0:
                self.r6_tree_graph.place(new_node)
                continue
            new_node_cost = (self.distance(new_node.getPosition(),
                    nearest[0].object.getPosition()) +
                    nearest[0].object.getCost())
            new_node.cost = new_node_cost
            new_node.setParent(nearest[0].object)
            new_node.type = nearest[0].object.type

            nearest = self.r6_tree_graph.nearestNeighbors(new_node, self.nearest_neighbors_limit)
            #nearest = self.r6_tree_graph.getNeighborChain(new_node, 15)
            for j in range(0, len(nearest)):
                #if new_node.cost + self.distance(new_node.getPosition(),
                #nearest[i].object.getPosition()) < nearest[i].object.getCost():
                #    nearest[i].object.cost = new_node.cost + self.distance(new_node.getPosition(),
                #nearest[i].object.getPosition())
                #    nearest[i].object.setParent(new_node)
                if (self.distance(
                        new_node.getPosition(), nearest[j].object.getPosition()) +
                        nearest[j].object.getCost() < new_node.cost):
                    if(new_node.type == 0 and nearest[j].object.type == 1):
                        temp = nearest[j].object.getParent()
                        temp_old = nearest[j].object
                        prev = new_node
                        while(temp is not None):
                            temp_old.setParent(prev)
                            prev = temp_old
                            temp_old.type = 0
                            temp_old = temp
                            print(temp.getCost())
                            temp = temp.getParent()

                        break
                    new_node.cost = (self.distance(new_node.getPosition(),
                            nearest[j].object.getPosition()) +
                            nearest[j].object.getCost())
                    new_node.setParent(nearest[j].object)
                    new_node.type = nearest[j].object.type
            #for i in range(0, len(nearest)):
            #    if new_node.cost + self.distance(new_node.getPosition(),
            #nearest[i].getPosition()) < nearest[i].getCost():
            #        nearest[i].cost = new_node.cost + self.distance(new_node.getPosition(),
            #nearest[i].getPosition())
            #        nearest[i].setParent(new_node)
            self.r6_tree_graph.place(new_node)

    def findPath(self, goal):
        """

        Args:
            goal (tm): Goal position to reach

        Returns:
            List(tm): Path from start to goal

        """
        self.generateTree()
        closest = self.r6_tree_graph.nearestNeighbors(PathNode(goal), 1)[0].object
        pose_list = []
        while closest is not None:
            pose_list.insert(0, closest.getPosition())
            closest = closest.getParent()
        pose_list.append(goal)
        return pose_list

    def findPathGeneral(self, treeMethod, goal):
        """

        Args:
            treeMethod (Class Init): Tree generation method handle
            goal (tm): Goal position to reach

        Returns:
            List(tm): Path froms tart to goal

        """
        treeMethod()
        closest = self.r6_tree_graph.nearestNeighbors(PathNode(goal), 1)[0].object
        pose_list = []
        while closest is not None:
            pose_list.insert(0, closest.getPosition())
            closest = closest.getParent()
        pose_list.append(goal)
        return pose_list

    def findPathDual(self, goal):
        """

        Args:
            goal (tm): Goal position to reach

        Returns:
            List(tm): path from start to goal

        """
        goal_path = PathNode(goal)
        goal_path.int = 1
        self.r6_tree_graph.place(goal_path)
        self.generateTreeDual()
        closest = self.r6_tree_graph.nearestNeighbors(PathNode(goal), 1)[0].object
        pose_list = []
        while closest is not None:
            pose_list.insert(0, closest.getPosition())
            closest = closest.getParent()
        pose_list.append(goal)
        return pose_list
