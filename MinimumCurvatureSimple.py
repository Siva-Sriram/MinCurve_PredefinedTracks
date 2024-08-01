import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline

import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly.express as px

import random
import csv

#For interactive matplotlib use within pycharm
mpl.use('TkAgg')

#csv files contain x,y coods of centreline, track widths to the left and right
#note that the csv files use smoothed centre points, so does not lie exactly on the centre of the track

with open('Austin.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    column = {h: [] for h in headers}
    for row in reader:
        for h, v, in zip(headers, row):
            column[h].append(float(v))

# Separating each column into its own list
    x_coord_cp = column['x_m']
    y_coord_cp = column['y_m']
    track_width_right = column['w_tr_right_m']
    track_width_left = column['w_tr_left_m']




#The following function generates random x coordinates for the control points based on the centre point and track width
def generate_random_x_coordinate(x_coordinates, left_width, right_width):

    x_coord_randomised = []

    for i in range(len(x_coordinates)):

        #finding the range in x direction allowed for random control points to lie on (confined to track width)
        x_range_minimum = x_coordinates[i] - left_width[i]
        x_range_maximum = x_coordinates[i] + right_width[i]

        #randomising x coordinates of control point with 0.5m proximity restriction to boundaries
        random_x_coordinate = random.uniform(x_range_minimum + 0.5, x_range_maximum - 0.5)
        x_coord_randomised.append(random_x_coordinate)

    return x_coord_randomised


def spline_interpolation(x_coordinate, y_coordinate):
    # interpolate random control pts
    tck_random = interp1d(x_coordinate, y_coordinate, kind='cubic')
    new_y_random = tck_random(x_coordinate)

    return new_y_random


def main():

    # CREATING, INTERPOLATING, AND PLOTTING BOUNDARIES
    x_left_boundary = []
    x_right_boundary = []

    for i in range(len(x_coord_cp)):
        # finding the range in x direction allowed for random control points to lie on (confined to track width)
        x_range_minimum = x_coord_cp[i] - track_width_left[i]
        x_range_maximum = x_coord_cp[i] + track_width_right[i]

        x_left_boundary.append(x_range_minimum)
        x_right_boundary.append(x_range_maximum)

    tck_left_boundary = interp1d(x_left_boundary, y_coord_cp, kind='cubic')
    new_y_left_boundary = tck_left_boundary(x_left_boundary)
    plt.figure(figsize=(8, 6))
    plt.plot(x_left_boundary, new_y_left_boundary, label="Left Boundary", color="black")
    plt.legend()

    tck_right_boundary = interp1d(x_right_boundary, y_coord_cp, kind='cubic')
    new_y_right_boundary = tck_right_boundary(x_right_boundary)
    plt.plot(x_right_boundary, new_y_right_boundary, label="Right Boundary", color="black")
    plt.legend()

    # INTERPOLATING AND PLOTTING THE CENTRE LINE
    # interpolate centre line
    tck = interp1d(x_coord_cp, y_coord_cp, kind='cubic')
    new_y = tck(x_coord_cp)
    plt.plot(x_coord_cp, new_y, label="Centre Line")


    # CREATING AND INTERPOLATING THE RANDOM POINTS

    #generate random x coordinate for a given y based on track dimensions
    x_coordinate_random_controlPt = generate_random_x_coordinate(x_coord_cp, track_width_left, track_width_right)

    #intepolate spline of randomised x coordinate for unchanged y coordinate
    interpolated_y = spline_interpolation(x_coordinate_random_controlPt, y_coord_cp)

    plt.plot(x_coordinate_random_controlPt, interpolated_y, label="Random track")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
















#for each centre coordinate, generate offset. Perhaps a loop or something

#     offsets = np.random.uniform(-widthLeft/2, widthRight/2, len(center_points))
#     return offsets
#
# def get_center_line(self):
#     # ... (existing code to get center points) ...
#
#     # Apply random offsets to center points
#     offsets = generate_random_offsets(coods, track_width=10)  # Adjust track_width as needed
#     modified_coods = [(x, y + offset) for (x, y), offset in zip(coods, offsets)]
#
#     # Generate cubic spline curve from modified center points
#     x_coords, y_coords = zip(*modified_coods)
#     cubic_spline = CubicSpline(x_coords, y_coords)
#
#     # Generate evenly spaced points along the cubic spline curve
#     path_points = [(x, cubic_spline(x)) for x in np.linspace(x_coords[0], x_coords[-1], 100)]
#
#     return path_points
#
# def calculate_curvature(path):
#     # Calculate the curvature along the given path.
#
#     # Args:
#     #     path (list): A list of (x, y) coordinates representing the path.
#
#     # Returns:
#     #     list: A list of curvature values corresponding to each point on the path.
#
#     # Convert the path to numpy arrays
#     path = np.array(path)
#     x = path[:, 0]
#     y = path[:, 1]
#
#     # Calculate the first and second derivatives using central difference
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#
#     # Calculate the curvature using the formula
#     curvature = (dx * ddy - dy * ddx) / np.power(d x* *2 + d y* *2, 3/ 2)
#
#     return curvature.tolist()
#
#
# def find_minimum_curvature_path(paths):
#     # Implement quadratic programming or other optimization technique
#     # to find the path with minimum curvature
#     # ...
#     return optimal_path
#
#
# # Generate multiple paths
# num_paths = 100
# paths = [get_center_line(self) for _ in range(num_paths)]
#
# # Find the path with minimum curvature
# optimal_path = find_minimum_curvature_path(paths)
#
# # Publish the optimal path
# optimal_path_pose_array = self.pack_to_pose_array(optimal_path)
# self.best_traj_pub.publish(optimal_path_pose_array)
#
