"""Alpha shape (concave hull) construction for point clouds."""
import alphashape
from collections import defaultdict
import numpy as np
from scipy.spatial import Delaunay


class AlphaShape:
    """Calculate an alpha shape from a set of points."""

    def __init__(self, pos, alpha=None, mode=0):
        """
        Build an alpha shape conforming to required specifications.

        Args:
            pos: points to create the shape out of
            alpha: Optional alpha parameter
            mode: mode select switch for alpha shape backend

        Returns:
            AlphaShape: alpha shape of requeted point cloud
        """
        shape = None
        if mode == 1:
            # if Mode 1, use library-less solution, less polished, slightly faster
            if alpha is not None:
                verts, edges, triangles = alpha_shape_3d(pos, 1 / alpha)
            else:
                verts, edges, triangles = alpha_shape_3d(pos, 1 / 1.98)
                #Default alpha value of 1.98 chosen for convenience
            self.verts = np.array(pos)[verts, :]
            self.pos = pos
            self.edges = edges
            # `triangles` indexes into the full `pos` array, but `self.verts` has
            # been reindexed down to only the vertices actually used by the shape
            # (`verts`, sorted ascending). Remap so `self.triangle_inds` indexes
            # into `self.verts`, as callers such as `draw()` expect.
            self.triangle_inds = np.searchsorted(verts, triangles)
            self.triangles = [np.array(pos)[t] for t in triangles]

        else:
            # if mode 0 (Default) use the regular alphashape library
            if alpha is not None:
                shape = alphashape.alphashape(pos, alpha)  # Manually Specify Alpha Value
            else:
                shape = alphashape.alphashape(pos)  # Automatically determine Alpha Value
                # Warning: Automatic determination of alpha values is extremely slow for complex
                #   shapes consisting of many points
            self.alpha = alpha
            self.verts = shape.vertices
            self.edges = None
            self.triangle_inds = shape.faces
            self.triangles = self.process_triangles()
        self.bounds = self.calculate_bounds()

    def draw(self, ax, transparency=0.2):
        """
        Draws the Alpha Shape.
        Args:
            ax: matplotlib axis to plot on
            transparency: optional parameter for transparency of resultant plot
        """
        ax.plot_trisurf(*zip(*self.verts),
                        triangles=self.triangle_inds, alpha=transparency, edgecolor='black')

    def process_triangles(self):
        """
        Calculate traingles within vertices, creating a useable list

        Returns:
            List: triangles (list containing three points)
        """
        triangles = []
        for tri in self.triangle_inds:
            triangles.append(self.verts[tri])
        return triangles

    def calculate_bounds(self):
        """
        Calculate the X, Y, and Z bounds of the resultant alpha shape

        Returns:
            [[],[],[]]: X, Y, and Z bounds arranged by dimension, min and then max
        """
        bounds_x = [np.inf, -np.inf]
        bounds_y = [np.inf, -np.inf]
        bounds_z = [np.inf, -np.inf]
        for vert in self.verts:
            if vert[0] < bounds_x[0]:
                bounds_x[0] = vert[0]
            if vert[0] > bounds_x[1]:
                bounds_x[1] = vert[0]
            if vert[1] < bounds_y[0]:
                bounds_y[0] = vert[1]
            if vert[1] > bounds_y[1]:
                bounds_y[1] = vert[1]
            if vert[2] < bounds_z[0]:
                bounds_z[0] = vert[2]
            if vert[2] > bounds_z[1]:
                bounds_z[1] = vert[2]
        return [bounds_x, bounds_y, bounds_z]

def alpha_shape_3d(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    https://stackoverflow.com/questions/26303878/alpha-shapes-in-3d
    Parameters:
        pos: np.array of shape (n,3) points.
        alpha: alpha value.
    Returns
        outer surface vertex indices, edge indices, and triangle indices
    """
    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    # Note: scipy renamed Delaunay.vertices to Delaunay.simplices.
    tetrapos = np.take(pos, tetra.simplices, axis=0)
    normsq = np.sum(tetrapos**2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    det_x = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    det_y = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    det_z = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
    r = np.sqrt(det_x**2+det_y**2+det_z**2-4*a*c)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra.simplices[r < alpha, :]
    # triangles
    triangle_comb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    triangles = tetras[:, triangle_comb].reshape(-1, 3)
    triangles = np.sort(triangles, axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    triangles_dict = defaultdict(int)
    for tri in triangles:
        triangles_dict[tuple(tri)] += 1
    triangles = np.array([tri for tri in triangles_dict if triangles_dict[tri] == 1])
    # edges
    edge_comb = np.array([(0, 1), (0, 2), (1, 2)])
    edges = triangles[:, edge_comb].reshape(-1, 2)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    vertices = np.unique(edges)
    return vertices, edges, triangles
