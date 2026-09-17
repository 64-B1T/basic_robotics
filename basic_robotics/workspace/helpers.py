"""Helper functions shared by the workspace analysis, viewing, and command line tools."""
import numpy as np
import sys
import time
import json

from ..general import fsr, tm


def filter_manipulability_at_threshold(results, score_threshold=0.5):
    """
    Filter Manipulability At THreshold
    (See only the parts of a workspace that have desired manipulability)

    Args:
        results: a pre-existing results list, generated from an earlier calculation
        score_threshold: desired threshold to filter above (e.g .75 = above 75% range 0-1)

    Returns:
        type: Truncated Results List
    """
    filtered_results = []
    for result in results:
        if result[1] > score_threshold:  # Should be range 0-1
            filtered_results.append(result)
    return filtered_results

def convert_to_json(results, json_file_name):
    """
    Convert a results list into a human-readable json file

    Args:
        results: Results list from workspace analyzer
        json_file_name: filename string of desired json file
    """
    json_dict = {}
    for result in results:
        id_str = str(result[0])  # Coordinate of result, used only as a unique key
        successes = result[2]
        thetas = result[3]
        json_dict[id_str] = {}
        # The coordinate itself must also be stored as data: its str() form (used
        # above only to key the dict) can't be parsed back into a tm by convert_from_json.
        json_dict[id_str]['point'] = result[0].TAA.flatten().tolist()
        json_dict[id_str]['score'] = result[1]
        json_dict[id_str]['successes'] = {}
        json_dict[id_str]['thetas'] = {}
        for i in range(len(successes)):
            json_dict[id_str]['successes'][str(i)] = successes[i].TAA.flatten().tolist()
        for i in range(len(thetas)):
            json_dict[id_str]['thetas'][str(i)] = thetas[i].flatten().tolist()
    with open(json_file_name, 'w') as json_file:
        json.dump(json_dict, json_file)

def convert_from_json(json_file_name):
    """
    Convert results from a json file into something useable

    Args:
        json_file_name: json file name containing results

    Returns:
        list: results as loaded from JSON
    """
    with open(json_file_name, 'r') as json_file:
        data = json.load(json_file)

    results = []
    for item in data:
        result = []
        result.append(tm(data[item]['point']))
        result.append(data[item]['score'])
        successes = []
        thetas = []
        for success in data[item]['successes']:
            successes.append(tm(data[item]['successes'][success]))
        for theta in data[item]['thetas']:
            thetas.append(np.array(data[item]['thetas'][theta]))
        result.append(successes)
        result.append(thetas)
        results.append(result)
    return results



def score_point(score):
    """
    Scores a point for use in plotting.

    Args:
        score: float score of point
    Returns:
        color: matplotlib color
    """
    col = 'darkred'
    if score > .9:
        col = 'limegreen'
    elif score > .8:
        col = 'green'
    elif score > .7:
        col = 'teal'
    elif score > .6:
        col = 'dodgerblue'
    elif score > .5:
        col = 'blue'
    elif score > .4:
        col = 'yellow'
    elif score > .3:
        col = 'orange'
    elif score > .2:
        col = 'peru'
    elif score > .1:
        col = 'red'
    return col

def complete_trajectory(point_list, point_interpolation_dist, point_interpolation_mode):
    """
    Interpolate a given trajectory to user specifications (IK)

    Args:
        point_list: List of waypoints that comprise the trajectory
        point_interpolation_dist: maximum distance apart to place interpolated points
        point_interpolation_mode: type of interpolation to perform

    Returns:
        [tm]: list of transformation objects
    """
    if point_interpolation_dist > 0:
        new_points = []
        point_list_len = len(point_list)
        for i in range(point_list_len):
            if i == point_list_len - 1:  # Add the last point, ensuring at least one point
                new_points.append(point_list[i])
                continue
            start_point = point_list[i]
            end_point = point_list[i + 1]
            distance = fsr.arcDistance(start_point, end_point)
            if distance < point_interpolation_dist:
                continue
            while distance > point_interpolation_dist:
                # Linear gap closes by directly interpolating between waypoints and scaling by
                #   a distance, creating a linear path.
                # Arc gap creates a more curved or 'arced' trajectory from waypoints A to B
                if point_interpolation_mode == 1:
                    start_point = fsr.closeLinearGap(start_point, end_point,
                                               point_interpolation_dist)
                else:
                    start_point = fsr.closeArcGap(start_point, end_point,
                                             point_interpolation_dist)
                distance = fsr.arcDistance(start_point, end_point)
                new_points.append(start_point)
        point_list = new_points
    return point_list

def load_point_cloud_from_file(fname):
    """
    Load a point cloud from a file.

    Args:
        fname: file to load from
    Returns:
        point cloud (tm)
    """
    def safe_float(val):
        if 'np.pi/' in val:
            return np.pi/float(val[6:])
        elif 'np.pi*' in val:
            return np.pi/float(val[6:])
        else:
            return float(val)

    with open(fname, 'r') as f_handle:
        lines = f_handle.readlines()
        tms = []
        for line in lines:
            tms.append(tm([safe_float(item.strip()) for item in line[1:-2].strip().split(',')]))
        return tms

def post_flag(flag, cmds):
    """
    Return the argument of a specified flag in a list of commands.

    Args:
        flag: string to seach for
        cmds: list of strings
    Returns:
        argument to flag
    """
    return cmds[cmds.index(flag) + 1]


def sort_cloud(cloud):
    """
    Sort a cloud of points to only have unique points.

    Args:
        cloud: cloud of points
    Returns:
        sorted points
    """
    list_array_input = []
    for point in cloud:
        p = point.TAA.flatten()
        list_array_input.append([p[0], p[1], p[2]])
    array_to_process = np.array(list_array_input)
    processed_array = np.unique(array_to_process.round(decimals=1), axis=0)
    return processed_array

def wait_for(result, message):
    """
    Wait for an asynchronous result to complete proessing and also display a waiting message

    Args:
        result: asynchronous result
        message: message to display while waiting
    """
    waiter = WaitText(message)
    while not result.ready():
        waiter.print()
        time.sleep(0.5)
    waiter.done()


class WaitText:
    """
    Simple helper class to provide feedback to user
    """

    def __init__(self, text='Processing', iter_limit=5):
        """
        Initialize WaitText
        Args:
            text: [Optional String] text to display
            iter_limit: number of '.' to print before resetting
        """
        self.text = text
        self.iter = 0
        self.iter_limit = iter_limit
        sys.stdout.write(self.text)

    def print(self):
        """
        Updates the waiting message
        """
        if self.iter == self.iter_limit:
            sys.stdout.write('\b' * self.iter_limit)
            sys.stdout.write('.')
            self.iter = 0
        else:
            sys.stdout.write('.')
        self.iter += 1

    def done(self):
        """
        Prints 'done' and advances to the next line
        """
        print(' Done')
