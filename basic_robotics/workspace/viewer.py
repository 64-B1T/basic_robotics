"""Interactive matplotlib viewer for previously saved workspace analysis results."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pickle

from .alpha_shape import AlphaShape
from ..utilities.disp import progressBar
from .helpers import score_point, sort_cloud

TRANSPARENCY_CONSTANT = .7

def view_workspace(image,
                  draw_alpha_shape=False,
                  alpha_shape_file=None,
                  draw_slices=True):  # pragma: no cover
    """
    Handler function for the WorkspaceViewer Class
    Args:
        image: filename of the workspace cloud
        draw_alpha_shape: [Optional Boolean] draw alpha shape
        alpha_shape_file: [Optional String] filename of the alpha shape skin
        draw_slices: [Optional Boolean] Draw 3D slices of workspace
    """
    #fig, ax = plt.subplots(1, 1)
    #fig, axes = plt.subplots(2, 2)
    #axillary = plt.add_subplot(1,1,projection='3d')
    fig = plt.figure()
    axis_1 = fig.add_subplot(1, 3, 1)
    axis_2 = fig.add_subplot(1, 3, 2)
    axis_3 = fig.add_subplot(1, 3, 3)
    #ax4 = fig.add_subplot(2,2,4, projection='3d')

    axes = np.array([[axis_2, axis_3], [axis_1, 0]])

    tracker = WorkspaceViewer(image, axes, draw_alpha_shape,
                              alpha_shape_file, draw_slices)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()


def draw_square(point, dims, excluded=2, col='darkred', axis_2=None):  # pragma: no cover
    """
    Draw a 2D Square for slicing.

    Args:
        point: point at which to center the Square
        dims: dimensions of the square
        excluded: dimension to exlude
        col: color of the square
        axis_2: axis to plot the square on
    """
    if excluded == 2:
        vert_1 = point[0] - dims[0] / 2
        vert_2 = point[0] + dims[0] / 2
        vert_3 = point[1] - dims[1] / 2
        vert_4 = point[1] + dims[1] / 2
        f = point[2] #Done
        xs = np.array([vert_1, vert_1, vert_2, vert_2])
        ys = np.array([vert_4, vert_3, vert_3, vert_4])
        zs = np.array([f, f, f, f])
        verts = [list(zip(xs, ys, zs))]
        axis_2.add_collection3d(
            Poly3DCollection(verts, facecolors=col, alpha=TRANSPARENCY_CONSTANT))
    elif excluded == 1:
        vert_1 = point[0] - dims[0] / 2
        vert_2 = point[0] + dims[0] / 2
        vert_3 = point[2] - dims[1] / 2
        vert_4 = point[2] + dims[1] / 2
        f = point[1]
        xs = np.array([vert_1, vert_1, vert_2, vert_2])
        zs = np.array([vert_4, vert_3, vert_3, vert_4])
        ys = np.array([f, f, f, f])
        verts = [list(zip(xs, ys, zs))]
        axis_2.add_collection3d(
            Poly3DCollection(verts, facecolors=col, alpha=TRANSPARENCY_CONSTANT))
    elif excluded == 0:
        vert_1 = point[1] - dims[0] / 2
        vert_2 = point[1] + dims[0] / 2
        vert_3 = point[2] - dims[1] / 2
        vert_4 = point[2] + dims[1] / 2
        f = point[0]
        ys = np.array([vert_1, vert_1, vert_2, vert_2])
        zs = np.array([vert_4, vert_3, vert_3, vert_4])
        xs = np.array([f, f, f, f])
        verts = [list(zip(xs, ys, zs))]
        axis_2.add_collection3d(
            Poly3DCollection(verts, facecolors=col, alpha=TRANSPARENCY_CONSTANT))


class WorkspaceViewer:
    """
    Used to View Saved ROM Files
    """

    def __init__(self,
                 file_name,
                 ax,
                 draw_alpha_shape=False,
                 alpha_file_name=None,
                 plot_3d_slice=True):
        """
        Initialize the Workspace Viewer
        Args:
            file_name: name of file for workspace point cloud
            ax: matplotlib.axes object
            draw_alpha_shape: [Optional boolean] whether or not to plot the alpha-shape
            alpha_file_name: [Optional boolean] filename of the optional alpha-shape
            plot_3d_slice: [Optional boolean] create a 3d rendered workspace slice graph
        """
        self.file_name = file_name
        print('Loading Data')
        with open(self.file_name, 'rb') as fp:
            self.work_space = pickle.load(fp)

        self.ax = ax

        ax[0, 0].set_xlabel('Meters X')
        ax[0, 0].set_ylabel('Meters Y')

        ax[0, 1].set_xlabel('Meters Y')
        ax[0, 1].set_ylabel('Meters Z')

        ax[1, 0].set_xlabel('Meters X')
        ax[1, 0].set_ylabel('Meters Z')

        #ax[1,1].set_xlabel('Meters X')
        #ax[1,1].set_ylabel('Meters Y')
        #ax[1,1].set_zlabel('Meters Z')
        self.real_bounds = []
        self.scalars = []
        self.cloud_data = self.process_slices()
        rows, columns, zed = self.cloud_data.shape
        self.slices = max([rows, columns, zed])
        self.ind = self.slices // 2

        x_layer = round(
            min(self.ind, self.cloud_data.shape[0] - 1) * self.scalars[0] +
            self.real_bounds[0], 2)
        y_layer = round(
            min(self.ind, self.cloud_data.shape[1] - 1) * self.scalars[1] +
            self.real_bounds[2], 2)
        z_layer = round(
            min(self.ind, self.cloud_data.shape[2] - 1) * self.scalars[2] +
            self.real_bounds[4], 2)
        self.ax[0, 0].set_title('Top Down, Z Slice = ' + str(z_layer) + 'm')
        self.ax[0, 1].set_title('Right View, X Slice = ' + str(x_layer) + 'm')
        self.ax[1, 0].set_title('Front View, Y Slice = ' + str(y_layer) + 'm')
        #self.ax[1,1].set_title('3D Total Workspace')

        self.im1 = ax[0, 0].imshow(self.cloud_data[:, :,
                                          min(self.ind, self.cloud_data.shape[2] - 1)],
                                   extent=self.real_bounds[0:4],
                                   cmap='inferno')
        yset = self.real_bounds[0:2] + [self.real_bounds[5]
                                        ] + [self.real_bounds[4]]
        self.im2 = ax[0, 1].imshow(self.cloud_data[:,
                                          min(self.ind, self.cloud_data.shape[1] -
                                              1), :].T,
                                   extent=yset,
                                   cmap='inferno')
        xset = self.real_bounds[2:4] + [self.real_bounds[5]
                                        ] + [self.real_bounds[4]]
        self.im3 = ax[1, 0].imshow(self.cloud_data[min(self.ind, self.cloud_data.shape[0] -
                                          1), :, :].T,
                                   extent=xset,
                                   cmap='inferno')
        if draw_alpha_shape:
            self.draw_alpha_shape(alpha_file_name)
        if plot_3d_slice:
            self.coord_str = ''
            self.scoords = [
                self.cloud_data.shape[0] // 2, self.cloud_data.shape[1] // 2,
                self.cloud_data.shape[2] // 2
            ]
            figure_2 = plt.figure(2)
            self.axis_2 = plt.axes(projection='3d')
            figure_2.canvas.mpl_connect('key_press_event', self.on_key)
            self.plot_3d_slices()
        self.update()
        self.ax[0, 1].invert_yaxis()
        self.ax[1, 0].invert_yaxis()

    def process_slices(self):
        """
        Process a slice.
        """
        sliced = [[[]]]
        workspace_len = len(self.work_space)
        x_min_dim, y_min_dim, z_min_dim = np.inf, np.inf, np.inf
        x_max_dim, y_max_dim, z_max_dim = -np.inf, -np.inf, -np.inf
        x_min_prox, y_min_prox, z_min_prox = np.inf, np.inf, np.inf
        for i in range(workspace_len):
            progressBar(i, workspace_len - 1, prefix='Finding Bounds')
            p = self.work_space[i][0]
            if i < workspace_len - 1:
                pn = self.work_space[i + 1][0]
                if abs(p[0] - pn[0]) < x_min_prox and abs(p[0] - pn[0]) > .05:
                    x_min_prox = abs(p[0] - pn[0])
                if abs(p[1] - pn[1]) < y_min_prox and abs(p[1] - pn[1]) > .05:
                    y_min_prox = abs(p[1] - pn[1])
                if abs(p[2] - pn[2]) < z_min_prox and abs(p[2] - pn[2]) > .05:
                    z_min_prox = abs(p[2] - pn[2])
            if p[0] < x_min_dim:
                x_min_dim = p[0]
            if p[1] < y_min_dim:
                y_min_dim = p[1]
            if p[2] < z_min_dim:
                z_min_dim = p[2]
            if p[0] > x_max_dim:
                x_max_dim = p[0]
            if p[1] > y_max_dim:
                y_max_dim = p[1]
            if p[2] > z_max_dim:
                z_max_dim = p[2]
        print('Creating Graph Skeleton')
        x_min = (x_min_dim * (1 / x_min_prox))
        y_min = (y_min_dim * (1 / y_min_prox))
        z_min = (z_min_dim * (1 / z_min_prox))

        x_max = (x_max_dim * (1 / x_min_prox))
        y_max = (y_max_dim * (1 / y_min_prox))
        z_max = (z_max_dim * (1 / z_min_prox))

        # +1: bounds are inclusive of both the min and max scaled index, so the
        # binned array needs one more slot than the raw span to avoid an
        # out-of-bounds write when a point lands exactly on the max index.
        x_bound = int(np.ceil(x_max - x_min)) + 1
        y_bound = int(np.ceil(y_max - y_min)) + 1
        z_bound = int(np.ceil(z_max - z_min)) + 1
        # print(x_min, x_max, x_bound)
        # print(y_min, y_max, y_bound)
        # print(z_min, z_max, z_bound)
        # print(x_min_prox, y_min_prox, z_min_prox)
        self.real_bounds = [
            x_min_dim, x_max_dim, y_min_dim, y_max_dim, z_min_dim, z_max_dim
        ]
        self.scaled_maxes = [x_bound, y_bound, z_bound]
        self.scalars = [x_min_prox, y_min_prox, z_min_prox]
        # self.ax[0,0].axis(scalex = x_min_prox, scaley = y_min_prox)

        sliced = np.zeros((y_bound, x_bound, z_bound))
        for i in range(workspace_len):
            progressBar(i, workspace_len - 1, prefix='Organizing Data')
            pd = self.work_space[i]
            p = pd[0]
            c = pd[1]
            yind = int(round(-x_min + (p[0] * (1 / x_min_prox))))
            xind = int(round(-y_min + (p[1] * (1 / y_min_prox))))
            zind = int(round(-z_min + (p[2] * (1 / z_min_prox))))

            sliced[xind, yind, zind] = c

        return sliced

    def plot_3d_slices_helper(self, xi, yi, zi):  # pragma: no cover
        """
        Helper function for plotting a 3d slice
        Args:
            xi: x array
            yi: y array
            zi: z array
        """
        s = np.shape(self.cloud_data)
        for x in range(s[0]):
            for y in range(s[1]):
                score = self.cloud_data[x, y, zi]
                if score <= .001:
                    continue
                col = score_point(score)
                draw_square([x, y, zi], [1, 1], 2, col, self.axis_2)
            for z in range(s[2]):
                score = self.cloud_data[x, yi, z]
                if score <= .001:
                    continue
                col = score_point(score)
                draw_square([x, yi, z], [1, 1], 1, col, self.axis_2)
        for y in range(s[1]):
            for z in range(s[2]):
                score = self.cloud_data[xi, y, z]
                if score <= .001:
                    continue
                col = score_point(score)
                draw_square([xi, y, z], [1, 1], 0, col, self.axis_2)

    def plot_3d_slices(self):  # pragma: no cover
        """
        Plot a set of 3d Slices
        """
        self.axis_2.clear()
        xi = self.scoords[0]
        yi = self.scoords[1]
        zi = self.scoords[2]
        self.axis_2.set_xlim3d(0, self.scaled_maxes[0])
        self.axis_2.set_ylim3d(0, self.scaled_maxes[1])
        self.axis_2.set_zlim3d(0, self.scaled_maxes[2])
        self.axis_2.set_xlabel('Units X')
        self.axis_2.set_ylabel('Units Y')
        self.axis_2.set_zlabel('Units Z')

        self.plot_3d_slices_helper(xi, yi, zi)

    def on_scroll(self, event):  # pragma: no cover
        """
        Update MRI slices based on mouse scrollwheel
        Args:
            event: event (scroll)
        """
        # print("%s %s" % (event.button, event.step))
        x_layer = round(
            min(self.ind, self.cloud_data.shape[0] - 1) * self.scalars[0] +
            self.real_bounds[0], 2)
        y_layer = round(
            min(self.ind, self.cloud_data.shape[1] - 1) * self.scalars[1] +
            self.real_bounds[2], 2)
        z_layer = round(
            min(self.ind, self.cloud_data.shape[2] - 1) * self.scalars[2] +
            self.real_bounds[4], 2)
        self.ax[0, 0].set_title('Top Down, Z Slice = ' + str(z_layer) + 'm')
        self.ax[0, 1].set_title('Right View, X Slice = ' + str(x_layer) + 'm')
        self.ax[1, 0].set_title('Front View, Y Slice = ' + str(y_layer) + 'm')
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def on_key(self, event):  # pragma: no cover
        """
        Enter coordinates for 3D slice visualization.
        Args:
            event: event (keystroke)
        """
        if (event.key == '1' or event.key == '2' or event.key == '3'
                or event.key == '4' or event.key == '5' or event.key == '6'
                or event.key == '7' or event.key == '8' or event.key == '9'
                or event.key == '0' or event.key == ','):
            self.coord_str += event.key
            #print(self.coord_str)
            self.axis_2.set_title(self.coord_str)
            self.axis_2.figure.canvas.draw()
        if event.key == 'enter':
            strl = self.coord_str.split(',')
            self.scoords = list(map(int, strl))
            self.plot_3d_slices()
            self.coord_str = ''
            self.axis_2.figure.canvas.draw()

    def update(self):
        """
        Update slice imager.
        """
        self.im1.set_data(self.cloud_data[:, :, min(self.ind, self.cloud_data.shape[2] - 1)])
        self.im2.set_data(self.cloud_data[:, min(self.ind, self.cloud_data.shape[1] - 1), :].T)
        self.im3.set_data(self.cloud_data[min(self.ind, self.cloud_data.shape[0] - 1), :, :].T)
        # self.ax[0,0].set_ylabel('slice %s' % self.ind)

        self.im1.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()
        self.im3.axes.figure.canvas.draw()
        # self.plot_3d_slices()

    def draw_alpha_shape(self, file_name):  # pragma: no cover
        """
        Draws an alpha shape in a separate window
        Args:
            file_name: name of the alpha shape to draw
        """
        fig = plt.figure(3)
        ax = plt.axes(projection='3d')
        ax.set_xlabel('Meters X')
        ax.set_ylabel('Meters Y')
        ax.set_zlabel('Meters Z')
        ax.set_title('Total Reachability Volume')
        arr = 0
        if file_name is not None:
            print('Loading Cloud Data')
            with open(file_name, 'rb') as fp:
                arr = sort_cloud(pickle.load(fp))
        else:
            print('Parsing data to produce Alpha Shape Cloud')
            workspace_len = len(self.work_space)
            pruned = []
            for i in range(workspace_len):
                progressBar(i, workspace_len - 1, prefix='Parsing')
                p = self.work_space[i][0]
                pruned.append([p[0], p[1], p[2]])

            arr = np.array(pruned)
        print('Processing Cloud')
        alpha = AlphaShape(arr, 1.2)
        alpha.draw(ax)
