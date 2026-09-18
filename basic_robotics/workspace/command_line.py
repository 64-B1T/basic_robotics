#!/usr/bin/env python
"""Interactive command line front-end for the workspace analyzer."""

#Utility Imports
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

#Other basic_robotics imports
from ..general import tm
from ..plotting.vis_matplotlib import drawRectangle, drawAxes, drawMesh
from ..kinematics import loadArmFromURDF
from ..utilities.disp import disp, progressBar

#This module imports
from .robot_link import RobotLink
from .alpha_shape import AlphaShape
from .analyzer import WorkspaceAnalyzer
from .helpers import score_point, post_flag
from .helpers import load_point_cloud_from_file, sort_cloud
from .viewer import view_workspace

#The ALPHA_VALUE is the alpha parameter which determines which points are included or
# excluded to create an alpha shape. This is technically an optional parameter,
# however when the alpha shape optimizes this itself, for large clouds of points
# it can take many hours to reach a solution. 1.2 is a convenient constant tested on a variety
# of robot sizes.
ALPHA_VALUE = 1.2

#The TRANSPARENCY_CONSTANT determines the degree of transparency at which 3d objects are shown
# when plotted within matplotlib.
TRANSPARENCY_CONSTANT = .5


done = False

cmd_str_help = 'help'
cmd_str_exit = 'exit'
cmd_str_load_robot = 'loadRobot'
cmd_str_analyze_task_manipulability = 'analyzeTaskSpaceManipulability'
cmd_str_analyze_total_workspace_exhaustive = 'exhaustiveMethodTotalWorkspace'
cmd_str_analyze_total_workspace_alpha = 'alphaMethodTotalWorkspace'
cmd_str_analyze_unit_shell_manipulability = 'unitShellManipulability'
cmd_str_analyze_manipulability_object_surface = 'objectSurfaceManipulability'
cmd_str_manipulability_within_volume = 'manipulabilityWithinVolume'
cmd_str_manipulability_over_trajectory = 'manipulabilityOverTrajectory'
cmd_str_manipulability_over_trajectory_cloud = 'manipulabilityFromCloudTrajectory'
cmd_str_view_manipulability_space = 'viewManipulabilitySpace'
cmd_str_brute_manipulability = 'analyzeBruteManipulability'
cmd_str_analyze_force_norm = 'analyzeForceNorm'
cmd_str_analyze_velocity_norm = 'analyzeVelocityNorm'
cmd_str_analyze_wrench_torques = 'analyzeWrenchTorques'
cmd_str_schedule_sequence = 'newSequence'
cmd_str_start_sequence = 'startSequence'
cmd_str_view_sequence = 'viewSequence'
cmd_str_load_sequence = 'loadSequence'
cmd_str_save_sequence = 'saveSequence'


valid_commands = [
    cmd_str_help,
    cmd_str_exit,
    cmd_str_load_robot,
    cmd_str_analyze_task_manipulability,
    cmd_str_analyze_total_workspace_exhaustive,
    cmd_str_analyze_total_workspace_alpha,
    cmd_str_analyze_unit_shell_manipulability,
    cmd_str_analyze_manipulability_object_surface,
    cmd_str_manipulability_over_trajectory_cloud,
    cmd_str_brute_manipulability,
    cmd_str_manipulability_within_volume,
    cmd_str_manipulability_over_trajectory,
    cmd_str_view_manipulability_space,
    cmd_str_schedule_sequence,
    cmd_str_analyze_force_norm,
    cmd_str_analyze_velocity_norm,
    cmd_str_analyze_wrench_torques,
    cmd_str_start_sequence,
    cmd_str_view_sequence,
    cmd_str_load_sequence,
    cmd_str_save_sequence
]

# General_Flags
cmd_flag_save_results = '-o'
cmd_flag_input_file = '-f'
cmd_flag_num_iters = '-numIterations'
cmd_flag_plot_results = '-plot'
cmd_flag_parallel = '-parallel'
cmd_flag_manip_res = '-manipulationResolution'
cmd_flag_collision_detect = '-checkCollisions'
# Manipulability now defaults to the closed-form Jacobian isotropy score
# everywhere it's computed; this flag opts back into sampling IK across a
# unit sphere of orientations at each point instead (slower, but reports
# actual reachable orientations rather than a local linearization).
cmd_flag_use_unit_sphere = '-unitSphereManipulability'

# Robot Loading Flags
cmd_flag_from_urdf = '-fromURDF'
cmd_flag_from_file = '-fromPyFile'
cmd_flag_from_directory = '-dir'

# 6DOF Manipulability Flags
cmd_flag_shells_range = '-range'
cmd_flag_shells_num_shells = '-numShells'
cmd_flag_shells_points_per_shell = '-numShellPoints'

# Object Surface Flags
cmd_flag_object_scale = '-scale'
cmd_flag_object_pose = '-pose'
cmd_flag_object_exempt_ee = '-exemptEE'
cmd_flag_object_bound_volume = '-boundVolume'
cmd_flag_object_min_dist = '-minDist'
cmd_flag_object_approx_collide = '-approxCollisions'

# Within Volume Flags
cmd_flag_volume_grid_res = '-gridResolution'

# Over Trajectory Flags
cmd_flag_trajectory_interpolation = '-interpolationDist'
cmd_flag_trajectory_arc_interp = '-arcInterpolation'

#Workspace Viewer Flags
cmd_flag_viewer_draw_alpha = '-alphaFile'
cmd_flag_viewer_draw_slices = '-draw3DSlices'

#Vels and Torques Flags
cmd_flag_jac_norm = '-targetNorm'
cmd_flag_jac_grav = '-grav'
cmd_flag_jac_origin = '-massCG'
cmd_flag_jac_mass = '-mass'
cmd_flag_jac_angular = '-angular'

#Brute Manipulability Joints Flags
cmd_flag_brute_excluded = '-exclude'

#Trajectory point cloud flags
cmd_flag_point_traj_cloud = '-pointCloud'
cmd_flag_point_traj_trajectory = '-trajectory'

class CommandExecutor:
    """Superclass for workspace command line, to enable use in other applications more easily"""

    def __init__(self):
        """Initialize Command Executor"""
        self.analyzer = None
        self.ready = False
        self.done = False
        self.exhaustive_pose_cloud = None
        self.functional_pose_cloud = None
        self.pose_results_6dof_shells = None
        self.pose_results_object_surface = None
        self.pose_results_manipulability_volume = None
        self.pose_results_manipulability_trajectory = None
        self.sequence = []

    def load_args(self, args):
        if '-sequence' in args:
            sequence_file_name = post_flag('-sequence', args)
            with open(sequence_file_name) as input_file:
                lines = input_file.readlines()
                for line in lines:
                    self.sequence.append(line.strip())
            self.cmd_start_sequence()
            return
    def set_sequence_at_launch(self, args):
        """
        Sets a launch sequence so that it can be executed straight from the command line
        Args:
            args: Argument sequence to set up
        """
        self.load_args(args)
        args = sys.argv[1:]
        self.load_args(args)

    def load_robot_from_urdf(self, fname):
        """
        Load a robot from URDF using basic_robotics conventions.
        Args:
            fname: file name of target file
        Returns:
            arm link instance
        """
        arm = loadArmFromURDF(fname)
        link = RobotLink(arm)

        def get_ee():
            return link.robot.getEEPos()

        def do_fk(x):
            return (link.robot.FK(x), True)

        def do_ik(x):
            return link.robot.IK(x, protect=True)

        def get_jt():
            return link.robot.getJointTransforms()

        link.bind_ee(get_ee)
        link.bind_fk(do_fk)
        link.bind_ik(do_ik)
        link.bind_jt(get_jt)
        link.joint_mins = link.robot.joint_mins
        link.joint_maxs = link.robot.joint_maxs

        #Collision Support
        link.link_names = arm.link_names
        link.vis_props = arm._vis_props
        link.col_props = arm._col_props
        return link

    def load_robot_from_module(self, dir_name, file_name):
        """
        Load a robot from a module file
        Args:
            dir_name: directory path .
            file_name: module file name
        Returns:
            RobotLink: Resulting RobotLink Object
        """
        sys.path.append(dir_name)
        file_name_sanitized = file_name.replace('.py', '')
        temp_module = importlib.import_module(file_name_sanitized)
        return temp_module.load_arm()

    def cmd_load_robot(self, cmds):
        """
        loadRobot -from___ file_name.
        Loads a robot for use in the workspace analyzer
        -fromURDF loads from a URDF file using modern_robotics/basic_robotics
        -fromPyFile loads from a user-defined python file specifying the robot
            see the example_robot_module.py file for additional details
        """
        if cmd_flag_from_urdf in cmds:
            file_name = post_flag(cmd_flag_from_urdf, cmds)
            link = self.load_robot_from_urdf(file_name)
        if cmd_flag_from_directory in cmds and cmd_flag_from_file in cmds:
            dir_name = post_flag(cmd_flag_from_directory, cmds)
            file_name = post_flag(cmd_flag_from_file, cmds)
            link = self.load_robot_from_module(dir_name, file_name)
        self.analyzer = WorkspaceAnalyzer(link)
        self.ready = True
        return link

    def load_point_cloud_in_cmds(self, cmds):
        """
        Load pose cloud from a file:
        Args:
            cmds: command list
        Returns:
            [tm] tm list
        """
        if cmd_flag_input_file not in cmds:
            disp('A trajectory file is required')
            return None
        else:
            point_list = load_point_cloud_from_file(post_flag(cmd_flag_input_file, cmds))
        return point_list

    def save_results_flag(self, cmds):
        """
        Determine whether or not to save results
        Args:
            cmds: command list
        Returns:
            output boolean, output file name
        """
        out_file_name = ''
        save_output = False
        if cmd_flag_save_results in cmds:
            out_file_name = post_flag(cmd_flag_save_results, cmds)
            save_output = True
        return save_output, out_file_name

    def save_to_file(self, results, out_file_name):
        """
        Saves results to a file
        Args:
            results: results to save
            out_file_name: output file name
        """
        with open(out_file_name, 'wb') as fp:
            pickle.dump(results, fp)

    def plot_joints_to_ee(self, results, grid_rez=0.2):  # pragma: no cover
        """
        Plots to matplotlib instance
        Args:
            results: results super-list
            grid_rez: size of cubes to plot
        """
        plt.figure()
        ax = plt.axes(projection='3d')
        for r in results:
            min = np.inf
            for j in r:
                if min > max(j[1]):
                    min = max(j[1])
            col = score_point(1 - min)
            drawRectangle(
                tm([r[0][0], r[0][1], r[0][2], 0, 0, 0]),
                [grid_rez]*3, ax, c=col, a=TRANSPARENCY_CONSTANT)
        plt.show()

    def plot_reachability_space(self, pose_cloud):  # pragma: no cover
        """
        Plots reachability space givin a pose cloud
        Args:
            pose_cloud: pose cloud to plot for
        """
        plt.figure()
        ax = plt.axes(projection='3d')
        alpha = AlphaShape(sort_cloud(pose_cloud), ALPHA_VALUE)
        my_cmap = plt.get_cmap('jet')
        ax.plot_trisurf(*zip(*alpha.verts),
            triangles=alpha.triangle_inds, cmap=my_cmap,alpha =.2, edgecolor='black')
        plt.show()

    def plot_manipulability_grid(self, results, grid_rez):  # pragma: no cover
        """
        Plots a manipulability grid given results and a grid resolution

        Args:
            results: Results list
            grid_rez: grid resolution, in meters
        """
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-4,4)
        ax.set_ylim3d(-4,4)
        ax.set_zlim3d(-4,4)
        for r in results:
            score = r[1]
            col = score_point(score)
            drawRectangle(
                tm([r[0][0], r[0][1], r[0][2], 0, 0, 0]),
                [grid_rez]*3, ax, c=col, a=TRANSPARENCY_CONSTANT)
        plt.show()

    def cmd_analyze_task_manipulability(self, cmds):
        """
        Analyze manipulability space
        -o : save output to file specified
        -plot: displays a basic plot of the results
        -manipResolution: desired manipulation resolution (number of unit sphere points).
            Only used with -unitSphereManipulability. Default 25
        -parallel: Calculate in parallel for increased efficiency. Default false
        -unitSphereManipulability: sample IK across a unit sphere of orientations at each
            point instead of the (now default) closed-form Jacobian isotropy score. Slower,
            but reports actual reachable orientations rather than a local linearization.
        -checkCollisions: enable collision checking
        """
        point_list = []
        manip_resolution = 25
        parallel = False
        use_jacobian = True
        collision_detect = False
        plot = False


        point_list = load_point_cloud_from_file(post_flag(cmd_flag_input_file, cmds))
        if point_list is None:
            return

        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))
        parallel = cmd_flag_parallel in cmds
        use_jacobian = cmd_flag_use_unit_sphere not in cmds
        collision_detect = cmd_flag_collision_detect in cmds

        save_output, out_file_name = self.save_results_flag(cmds)
        plot = cmd_flag_plot_results in cmds

        results = self.analyzer.analyze_task_space_manipulability(point_list,
                manip_resolution, parallel, use_jacobian, collision_detect)

        self.pose_results_manipulability_volume = results
        if plot:
            self.plot_manipulability_grid(results, 0.1)
        if save_output:
            self.save_to_file(results, out_file_name)
        return results


    def cmd_analyze_total_workspace_exhaustive(self, cmds):
        """
        Analyze a robot's total reachability using joint variation (exponential time).
        -numIterations: number of interpolated positions per joint
        -o : save output to file specified
        -plot: displays a basic plot of the results
        """
        num_iterations = 12
        plot = False
        if cmd_flag_num_iters in cmds:
            num_iters_str = post_flag(cmd_flag_num_iters, cmds)
            num_iterations = int(num_iters_str)
        save_output, out_file_name = self.save_results_flag(cmds)
        plot = cmd_flag_plot_results in cmds
        pose_cloud = self.analyzer.analyze_total_workspace_exhaustive_point_cloud(num_iterations)
        if plot:
            self.plot_reachability_space(pose_cloud)
        self.exhaustive_pose_cloud = pose_cloud
        if save_output:
            self.save_to_file(pose_cloud, out_file_name)
        return pose_cloud

    def cmd_analyze_total_workspace_functional(self, cmds):
        """
        Analyze a robot's total reachability using single joint variation alpha shape projection.
        Generally runs in linear time
        -numIterations: number of interpolated positions per joint
        -o: save output to file specified
        -plot: displays a basic plot of the results
        """
        num_iterations = 25
        plot = False
        if cmd_flag_num_iters in cmds:
            num_iters_str = post_flag(cmd_flag_num_iters, cmds)
            num_iterations = int(num_iters_str)
        save_output, out_file_name = self.save_results_flag(cmds)
        plot = cmd_flag_plot_results in cmds
        pose_cloud = self.analyzer.analyze_total_workspace_functional(num_iterations)
        if plot:
            self.plot_reachability_space(pose_cloud)
        self.functional_pose_cloud = pose_cloud

        if save_output:
            self.save_to_file(pose_cloud, out_file_name)
        return pose_cloud

    def cmd_analyze_manipulability_unit_shells(self, cmds):
        """
        Analyzes workspace manipulability through IK and unit spheres.
        -range: maximum range (meters) of the largest unit sphere from robot origin. Default 4
        -numShells: number of concentric shells from the robot base to range. Default 30
        -numShellPoints: number of points to analyze per shell. Default 1000
        -checkCollisions: enable collision checking
        -plot: displays a basic plot of the results
        """
        shells_range = 4
        num_shells = 30
        points_per_shell = 1000
        plot = False
        if cmd_flag_shells_range in cmds:
            num_iters_str = post_flag(cmd_flag_shells_range, cmds)
            shells_range = int(num_iters_str)
        if cmd_flag_shells_num_shells in cmds:
            num_iters_str = post_flag(cmd_flag_shells_num_shells, cmds)
            num_shells = int(num_iters_str)
        if cmd_flag_shells_points_per_shell  in cmds:
            num_iters_str = post_flag(cmd_flag_shells_points_per_shell, cmds)
            points_per_shell = int(num_iters_str)
        collision_detect = cmd_flag_collision_detect in cmds
        save_output, out_file_name = self.save_results_flag(cmds)
        plot = cmd_flag_plot_results in cmds

        results = self.analyzer.analyze_6dof_manipulability(
            shells_range, num_shells, points_per_shell, collision_detect)
        self.pose_results_6dof_shells = results
        if plot:
            self.plot_manipulability_grid(results, 0.25)
        if save_output:
            self.save_to_file(results, out_file_name)
        return results

    def cmd_analyze_manipulability_object_surface(self, cmds):
        """
        Analyzes manipulability on the surface of an object
        -f: filename of the stl object to analyze
        -o: Save output to file. Default to unused
        -scale: scale of the object Default 1
        -pose: pose of the object. TAA such as [x,y,z,xr,yr,zr]. M/Rad. No spaces. Default identity
        -manipulationResolution: number of points in manipulation unit sphere.
            Only used with -unitSphereManipulability. Default 25
        -checkCollisions: make sure no part of the arm intersects the object. Default True
        -exemptEE: exempt the end effector of the robot from collision checking. Default True
        -parallel: run in parallel for increased efficiency. Default False.
        -unitSphereManipulability: sample IK across a unit sphere of orientations at each
            vertex instead of the (now default) closed-form Jacobian isotropy score.
        -plot: displays a basic plot of the results
        -boundVolume: Optional bounds volume
        -minDist: Minimum distance between vertices to consider analyzing
        """
        object_file_name = ''
        object_scale = 1
        object_pose = tm()
        manip_resolution = 25
        collision_detect = True
        exempt_ee = True
        parallel = False

        plot = False
        use_jacobian = True
        shape = None
        min_dist = -1.0

        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
        if cmd_flag_object_scale in cmds:
            obj_scale_str = post_flag(cmd_flag_object_scale, cmds)
            object_scale = float(obj_scale_str)
        if cmd_flag_object_pose in cmds:
            pose_str = post_flag(cmd_flag_object_pose, cmds)
            object_pose = tm([float(item) for item in pose_str[1:-1].split(',')])
        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))
        collision_detect = cmd_flag_collision_detect in cmds
        exempt_ee = cmd_flag_object_exempt_ee in cmds
        parallel = cmd_flag_parallel in cmds
        save_output, out_file_name = self.save_results_flag(cmds)
        plot = cmd_flag_plot_results in cmds
        if cmd_flag_use_unit_sphere in cmds:
            use_jacobian = False
        if cmd_flag_object_bound_volume in cmds:
            shape_fname = post_flag(cmd_flag_object_bound_volume, cmds)
            with open(shape_fname, 'rb') as fp:
                shape = AlphaShape(sort_cloud(pickle.load(fp)), ALPHA_VALUE)
        if cmd_flag_object_min_dist in cmds:
            min_dist = float(post_flag(cmd_flag_object_min_dist, cmds))

        results, mesh = self.analyzer.analyze_manipulability_on_object_surface(
            object_file_name, object_scale,
            object_pose, manip_resolution,
            collision_detect, exempt_ee,
            parallel, use_jacobian,
            min_dist, shape)

        if plot:  # pragma: no cover
            plt.figure()
            ax = plt.axes(projection='3d')
            for r in results:
                score = r[1]
                if score < .1:
                    continue
                else:
                    col = score_point(score)
                    drawRectangle(tm([r[0][0], r[0][1], r[0][2], 0, 0, 0]),
                        [.25] * 3, ax, c=col, a=TRANSPARENCY_CONSTANT)
            drawMesh(mesh, ax)
            plt.show()
        self.pose_results_object_surface = results
        if save_output:
            self.save_to_file(results, out_file_name)
        return results

    def cmd_analyze_manipulability_within_volume(self, cmds):
        """
        Analyze manipulability within the bounds of a volume.
        -f: filename of shape. Defaults to exhaustive or alpha total workspace if run prior.
            Required if not (.npy)
        -o: output file, if desired
        -gridResolution: desired grid resolution in meters. defaults to .25m
        -manipResolution: desired manipulation resolution (number of unit sphere points).
            Only used with -unitSphereManipulability. Default 25
        -parallel: Calculate in parallel for increased efficiency. Default false
        -unitSphereManipulability: sample IK across a unit sphere of orientations at each
            point instead of the (now default) closed-form Jacobian isotropy score.
        -plot: displays a basic plot of the results
        """
        shape_fname = ''
        shape = None
        if self.exhaustive_pose_cloud is not None:
            res = input('Would you like to use prior calculated exhaustive pose cloud? Y/N >')
            if res.lower() == 'y':
                shape = AlphaShape(sort_cloud(self.exhaustive_pose_cloud), ALPHA_VALUE)
        if self.functional_pose_cloud is not None:
            res = input('Would you like to use prior calculated functional pose cloud? Y/N >')
            if res.lower() == 'y':
                shape = AlphaShape(sort_cloud(self.functional_pose_cloud), ALPHA_VALUE)

        if shape is None:
            if cmd_flag_input_file in cmds:
                shape_fname = post_flag(cmd_flag_input_file, cmds)
            with open(shape_fname, 'rb') as fp:
                arr = sort_cloud(pickle.load(fp))
            shape = AlphaShape(arr, ALPHA_VALUE)
        if shape is None:
            disp('This function requires an alpha shape cloud as an input\n' +
                'Please either specify a .npy file name or run a pose cloud calculator')
            return

        manip_resolution = 25
        grid_resolution = .25

        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))
        if cmd_flag_volume_grid_res in cmds:
            grid_resolution = float(post_flag(cmd_flag_volume_grid_res, cmds))
        parallel = cmd_flag_parallel in cmds
        save_output, out_file_name = self.save_results_flag(cmds)
        collision_detect = cmd_flag_collision_detect in cmds
        use_jacobian = cmd_flag_use_unit_sphere not in cmds
        plot = cmd_flag_plot_results in cmds
        results = self.analyzer.analyze_manipulability_within_volume(
            shape, grid_resolution, manip_resolution, parallel, use_jacobian, collision_detect)
        if plot:
            self.plot_manipulability_grid(results, grid_resolution)
        if save_output:
            self.save_to_file(results, out_file_name)
        return results

    def cmd_analyze_manipulability_over_trajectory(self, cmds):
        """
        Analyzes manipulability over a given trajectory.
        -f: required file name for trajectory in TAA format ([x,y,z,xr,yr,zr]) one per line
        -o: optional output file for saving results
        -unitSphereManipulability: use unit sphere manipulability instead of jacobian manipulability
        -manipResolution: desired manipulation resolution (number of unit sphere points) Default 25
        -interpolationDist: interpolate between points in unit trajectory by a given amount (meters)
        -arcInterpolation: use arc interpolation instead of linear interpolation
        -plot: displays a basic plot of the results
        """
        point_list = []
        manipulability_mode = 1
        manip_resolution = 25
        point_interpolation_dist = -1
        point_interpolation_mode = 1


        plot = False

        if cmd_flag_input_file not in cmds:
            disp('A trajectory file is required')
            return
        else:
            point_list = load_point_cloud_from_file(post_flag(cmd_flag_input_file, cmds))
        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))
        if cmd_flag_use_unit_sphere in cmds:
            manipulability_mode = 2
        if cmd_flag_trajectory_interpolation in cmds:
            point_interpolation_dist = float(post_flag(cmd_flag_trajectory_interpolation, cmds))
        if cmd_flag_trajectory_arc_interp in cmds:
            point_interpolation_mode = 2
        collision_detect = cmd_flag_collision_detect in cmds
        save_output, out_file_name = self.save_results_flag(cmds)
        plot = cmd_flag_plot_results in cmds

        results = self.analyzer.analyze_manipulability_over_trajectory(
            point_list, manipulability_mode, manip_resolution,
            point_interpolation_dist, point_interpolation_mode, collision_detect)

        self.pose_results_manipulability_trajectory = results

        if plot:  # pragma: no cover
            plt.figure()
            ax = plt.axes(projection='3d')
            for r in results:
                drawAxes(r[0], r[1] / 2, ax)
                ax.scatter3D(r[0][0], r[0][1], r[0][2], c=score_point(r[1]), s=25)
            plt.show()
        if save_output:
            self.save_to_file(results, out_file_name)
        return results

    def cmd_view_manipulability_space(self, cmds):  # pragma: no cover
        """
        Views a previously saved workspace
        -f: Results file to load (.dat)
        -alphafile: alpha shape file to load in (Optional)
        -draw3DSlices: Draw an interactive 3d representation of the alpha slices (Optional)
        """
        object_file_name = ''
        draw_alpha_shape = False
        alpha_shape_file = ''
        draw_slices = False
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
        if cmd_flag_viewer_draw_alpha in cmds:
            alpha_shape_file = post_flag(cmd_flag_viewer_draw_alpha, cmds)
            draw_alpha_shape = True
        if cmd_flag_viewer_draw_slices in cmds:
            draw_slices = True

        view_workspace(object_file_name, draw_alpha_shape, alpha_shape_file, draw_slices)

    def _relate_joints_to_ee(self, cmds):
        """
        Helper function for cmd_analyze_velocity_norm and cmd_analyze_torque_norms

        Args:
            cmds: command strings
        Returns:
            list: target values related to givens
        """
        target_norm = 0
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
        if cmd_flag_jac_norm in cmds:
            target_norm = float(post_flag(cmd_flag_jac_norm))
        angular = cmd_flag_jac_angular in cmds
        with open(object_file_name, 'rb') as fp:
            data = pickle.load(fp)
        results = self.analyzer.analyze_joint_related_to_end_effector_vals(
                data, target_norm, angular)
        save_output, out_file_name = self.save_results_flag(cmds)
        if save_output:
            self.save_to_file(results, out_file_name)
        if cmd_flag_plot_results in cmds:  # pragma: no cover
            self.plot_points_to_ee(self, results)
        return results

    def cmd_analyze_velocity_norm(self, cmds):
        """
        Determine minimum joint velocities for a given end effector velocity norm
        -f: filename of previously created manipulability file
        -targetNorm: float target norm for EE velocity (linear or angular)
        -angular: use angular calculations instead of linear
        -plot: plot results
        -o: output file name
        """
        return self._relate_joints_to_ee(cmds)

    def cmd_analyze_force_norm(self, cmds):
        """
        Determine minimum joint torques for a given end effector force norm
        -f: filename of previously created manipulability file
        -targetNorm: float target norm for EE Torques (linear or angular)
        -angular: use angular calculations instead of linear
        -plot: plot results
        -o: output file name
        """
        return self._relate_joints_to_ee(cmds)

    def cmd_analyze_wrench_torques(self, cmds):
        """
        Given a mass, gravity (or force vector) and mass origin,
        calculate the joint torques of a robot at each point in manipulability
        -f: filename of previously created manipulability file
        -targetNorm: float target norm for EE Torques (linear or angular)
        -grav: gravity vector of the three-form [0,0,-9.81]
        -massCG: mass cg transform of the TAA form [0,1,2,np.pi/4,np.pi/7,np.pi/8]
        -mass: mass amount to apply (Kg) or N if using a force vector instead of gravity
        -plot: plot results
        -o: output file name
        """
        data = None
        mass = 10
        origin = tm()
        grav_vector = np.array([0, 0, -9.81])
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
        if cmd_flag_jac_grav in cmds:
            pose_str = post_flag(cmd_flag_jac_grav, cmds)
            grav_vector = np.array([float(item) for item in pose_str[1:-1].split(',')])
        if cmd_flag_jac_origin in cmds:
            pose_str = post_flag(cmd_flag_jac_origin, cmds)
            origin = tm([float(item) for item in pose_str[1:-1].split(',')])
        if cmd_flag_jac_mass in cmds:
            mass = float(post_flag(cmd_flag_jac_mass, cmds))
        with open(object_file_name, 'rb') as fp:
            data = pickle.load(fp)
        results = self.analyzer.analyze_matching_joint_torques(data, origin, mass, grav_vector)
        save_output, out_file_name = self.save_results_flag(cmds)
        if cmd_flag_plot_results in cmds:  # pragma: no cover
            self.plot_points_to_ee(results, 0.1)
        if save_output:
            self.save_to_file(results, out_file_name)
        return results

    def cmd_analyze_brute_manipulability(self, cmds):
        """
        Analyze Manipulability using a brute-force FK based approach
        -numIterations: number of iterations per joint to use
        -exclude: list of joints [x,y,z...] to lock at 0 if desired.
        -checkCollisions: check collisions
        -parallel: Calculate in parallel for increased efficiency. Default false
        -plot: plot results
        -o: output file name
        """
        excluded = []
        num_iters = 15
        parallel = cmd_flag_parallel in cmds
        collision_detect = cmd_flag_collision_detect in cmds
        if cmd_flag_brute_excluded in cmds:
            exclude_str = post_flag(cmd_flag_brute_excluded, cmds)
            excluded = [int(item) for item in exclude_str[1:-1].split(',')]
        if cmd_flag_num_iters in cmds:
            num_iters = int(post_flag(cmd_flag_num_iters, cmds))
        results = self.analyzer.analyze_brute_manipulability_on_joints(
                num_iters, excluded, parallel, collision_detect)
        save_output, out_file_name = self.save_results_flag(cmds)
        if save_output:
            disp('saving results')
            self.save_to_file(results, out_file_name)
        if cmd_flag_plot_results in cmds:  # pragma: no cover
            disp('plotting')
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlim3d(-4,4)
            ax.set_ylim3d(-4,4)
            ax.set_zlim3d(-4,4)
            result_len = len(results)
            for i in range(result_len):
                r = results[i]
                progressBar(i, result_len-1, 'plotting')
                score = r[1]
                col = score_point(score)
                ax.scatter3D(r[0][0], r[0][1], r[0][2], s=1, color=col)
            plt.show()
        return results

    def cmd_analyze_point_cloud_with_trajectory(self, cmds):
        """
        Analyze the effects of a trajectory on a set of points, useful for
        determining whether or not objects can be removed from a surface or something similar

        -pointCloud: the point cloud used as the origin of trajectories
        -trajectory: the local trajectory applied to each point
        -o: optional output file for saving results
        -unitSphereManipulability: use unit sphere manipulability instead of jacobian manipulability
        -manipResolution: desired manipulation resolution (number of unit sphere points) Default 25
        -interpolationDist: interpolate between points in unit trajectory by a given amount (meters)
        -arcInterpolation: use arc interpolation instead of linear interpolation
        -plot: displays a basic plot of the results
        """
        point_list = []
        point_interpolation_dist = -1
        point_interpolation_mode = 1
        manipulability_mode = 1
        manip_resolution = 25
        plot = False
        parallel = False

        if cmd_flag_point_traj_cloud not in cmds:
            disp('A point cloud file is required')
            return
        else:
            point_list = load_point_cloud_from_file(post_flag(cmd_flag_point_traj_cloud, cmds))
        if cmd_flag_point_traj_trajectory not in cmds:
            disp('A trajectory cloud file is required')
            return
        else:
            trajectory = load_point_cloud_from_file(post_flag(cmd_flag_point_traj_trajectory, cmds))
        if cmd_flag_trajectory_interpolation in cmds:
            point_interpolation_dist = float(post_flag(cmd_flag_trajectory_interpolation, cmds))
        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))
        if cmd_flag_trajectory_arc_interp in cmds:
            point_interpolation_mode = 2
        if cmd_flag_use_unit_sphere in cmds:
            manipulability_mode = 2
        parallel = cmd_flag_parallel in cmds
        collision_detect = cmd_flag_collision_detect in cmds

        results = self.analyzer.analyze_manipulability_point_cloud_with_trajectory(
            point_list, trajectory,
            manip_resolution, point_interpolation_dist, manipulability_mode,
            point_interpolation_mode, collision_detect, parallel)

        save_output, out_file_name = self.save_results_flag(cmds)
        plot = cmd_flag_plot_results in cmds
        if plot:  # pragma: no cover
            plt.figure()
            ax = plt.axes(projection='3d')
            for traj in results:
                for r in traj[2]:
                    drawAxes(r[0], r[1] / 2, ax)
                    ax.scatter3D(r[0][0], r[0][1], r[0][2], c=score_point(r[1]), s=25)
            plt.show()
        if save_output:
            self.save_to_file(results, out_file_name)

    def cmd_start_sequence(self):
        """
        Starts or repeats a sequence of commands
        previously defined by the user
        """
        disp('Starting Sequence')
        for cmd in self.sequence:
            self.cmd_parser(cmd)
        disp('Sequence Complete')

    def cmd_prepare(self, cmds):
        cmd_no_eq = cmds.replace('=', ' ')
        cmds_parsed = cmd_no_eq.split(' ')
        return cmds_parsed

    def cmd_parser(self, cmd):
        """
        Parse a command string.

        Args:
            cmd: command string passed in
        """
        cmds_parsed = self.cmd_prepare(cmd)
        if not cmds_parsed[0] in valid_commands:
            disp('Unrecognized command. Please try again. You can use \'help\' for help')
            return
        if cmds_parsed[0] == cmd_str_help:
            self.cmd_help(cmds_parsed)
            return
        if cmds_parsed[0] == cmd_str_exit:
            self.done = True
            return False
        elif cmds_parsed[0] == cmd_str_schedule_sequence:
            self.cmd_schedule_sequence()
            return
        elif cmds_parsed[0] == cmd_str_start_sequence:
            self.cmd_start_sequence()
            return
        elif cmds_parsed[0] == cmd_str_view_sequence:
            self.cmd_view_sequence()
            return
        elif cmds_parsed[0] == cmd_str_save_sequence:
            self.cmd_save_sequence(cmds_parsed)
            return
        elif cmds_parsed[0] == cmd_str_load_sequence:
            self.cmd_load_sequence(cmds_parsed)
            return
        elif cmds_parsed[0] == cmd_str_load_robot:
            self.cmd_load_robot(cmds_parsed)
            return
        if not self.ready:
            disp('Please load a robot first')
            return
        if cmds_parsed[0] == cmd_str_analyze_total_workspace_exhaustive:
            return self.cmd_analyze_total_workspace_exhaustive(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_analyze_task_manipulability:
            return self.cmd_analyze_task_manipulability(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_analyze_total_workspace_alpha:
            return self.cmd_analyze_total_workspace_functional(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_analyze_unit_shell_manipulability:
            return self.cmd_analyze_manipulability_unit_shells(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_analyze_manipulability_object_surface:
            return self.cmd_analyze_manipulability_object_surface(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_manipulability_within_volume:
            return self.cmd_analyze_manipulability_within_volume(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_manipulability_over_trajectory:
            return self.cmd_analyze_manipulability_over_trajectory(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_view_manipulability_space:
            return self.cmd_view_manipulability_space(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_brute_manipulability:
            return self.cmd_analyze_brute_manipulability(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_manipulability_over_trajectory_cloud:
            return self.cmd_analyze_point_cloud_with_trajectory(cmds_parsed)


class WorkspaceCommandLine(CommandExecutor):  # pragma: no cover
    """
    Provide command line access to workspace_analyzer .
    """

    def __init__(self, args = []):
        super().__init__()
        self.analyzer = None
        self.ready = False
        self.done = False
        self.exhaustive_pose_cloud = None
        self.functional_pose_cloud = None
        self.pose_results_6dof_shells = None
        self.pose_results_object_surface = None
        self.pose_results_manipulability_volume = None
        self.pose_results_manipulability_trajectory = None
        self.sequence = []
        self.set_sequence_at_launch(args)
        self.main_loop()

    def cmd_schedule_sequence(self):
        """
        Allows for scheduling a sequence of commands
        Presents a prompt to the user
        """
        self.sequence = []
        sequence_done = False
        disp('Welcome to sequence scheduler. To exit, type \'done\'')
        while not sequence_done:
            cmd = input('Sequence Command>')
            if cmd.strip().lower() == 'done':
                disp('Exiting. Use \'startSequence\' to start sequence')
                sequence_done = True
                continue
            confirm = input('Confirm Command?(Y/N)>')
            if confirm.strip().lower() != 'y':
                continue
            self.sequence.append(cmd)

    def cmd_load_sequence(self, cmds):
        """
        Loads a command sequence from file
        -f: file to load from
        """
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
            with open(object_file_name) as input_file:
                lines = input_file.readlines()
                for line in lines:
                    self.sequence.append(line.strip())
        elif len(cmds) > 1:
            object_file_name = cmds[1]
            with open(object_file_name) as input_file:
                lines = input_file.readlines()
                for line in lines:
                    self.sequence.append(line.strip())
        else:
            disp('Please specify a file')

    def cmd_save_sequence(self, cmds):
        """
        Saves a command sequence to file
        -f: file to save to
        """
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
            with open(object_file_name, 'w+') as output_file:
                for line in self.sequence:
                    output_file.write(line + '\n')
        else:
            disp('Please specify a file')


    def cmd_view_sequence(self):
        """
        If a user has previously defined a sequence of commands,
        this function will display them for the user to review
        """
        if len(self.sequence) == 0:
            disp('Nothing to show')
        else:
            for i in range(len(self.sequence)):
                disp(str(i+1) + ') ' + self.sequence[i])


    def main_loop(self):
        """Start Main Loop for Command Line Program."""
        self.done = False
        while not self.done:
            cmd = input('Command> ')
            self.cmd_parser(cmd)
            continue
            try:
                cmd = input('Command> ')
                self.cmd_parser(cmd)
            except Exception as e:
                disp('Command failure. Details: ' + str(e))

    def cmd_help(self, cmds):
        """
        Assist the user in function understanding.

        Args:
            cmds: command passed through, for specific help if needed
        """
        if len(cmds) > 1:
            if cmds[1] == cmd_str_exit:
                disp('Exits the program')
            elif cmds[1] == cmd_str_load_robot:
                disp(self.cmd_load_robot.__doc__)
            elif cmds[1] == cmd_str_analyze_task_manipulability:
                disp(self.cmd_analyze_task_manipulability.__doc__)
            elif cmds[1] == cmd_str_analyze_total_workspace_exhaustive:
                disp(self.cmd_analyze_total_workspace_exhaustive.__doc__)
            elif cmds[1] == cmd_str_analyze_total_workspace_alpha:
                disp(self.cmd_analyze_total_workspace_functional.__doc__)
            elif cmds[1] == cmd_str_analyze_unit_shell_manipulability:
                disp(self.cmd_analyze_manipulability_unit_shells.__doc__)
            elif cmds[1] == cmd_str_analyze_manipulability_object_surface:
                disp(self.cmd_analyze_manipulability_object_surface.__doc__)
            elif cmds[1] == cmd_str_manipulability_within_volume:
                disp(self.cmd_analyze_manipulability_within_volume.__doc__)
            elif cmds[1] == cmd_str_manipulability_over_trajectory:
                disp(self.cmd_analyze_manipulability_over_trajectory.__doc__)
            elif cmds[1] == cmd_str_manipulability_over_trajectory_cloud:
                disp(self.cmd_analyze_point_cloud_with_trajectory.__doc__)
            elif cmds[1] == cmd_str_schedule_sequence:
                disp(self.cmd_schedule_sequence.__doc__)
            elif cmds[1] == cmd_str_start_sequence:
                disp(self.cmd_start_sequence.__doc__)
            elif cmds[1] == cmd_str_view_sequence:
                disp(self.cmd_view_sequence.__doc__)
            elif cmds[1] == cmd_str_load_sequence:
                disp(self.cmd_load_sequence.__doc__)
            elif cmds[1] == cmd_str_save_sequence:
                disp(self.cmd_save_sequence.__doc__)
            elif cmds[1] == cmd_str_view_manipulability_space:
                disp(self.cmd_view_manipulability_space.__doc__)
            elif cmds[1] == cmd_str_brute_manipulability:
                disp(self.cmd_analyze_brute_manipulability.__doc__)
        else:
            disp('\nhelp [cmd] for details\nValid Commands:')
            for command in valid_commands:
                disp('\t'+command)
            disp('\n')


if __name__ == '__main__':  # pragma: no cover
    command_line = WorkspaceCommandLine()
