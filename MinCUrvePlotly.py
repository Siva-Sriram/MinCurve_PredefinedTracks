import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import random
import csv

# Read data from CSV file
with open('Austin.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    column = {h: [] for h in headers}
    for row in reader:
        for h, v in zip(headers, row):
            column[h].append(float(v))

# Extract data from CSV columns
x_coord_cp = column['x_m']
y_coord_cp = column['y_m']
track_width_right = column['w_tr_right_m']
track_width_left = column['w_tr_left_m']


def calculate_boundary_points(x_centerline, y_centerline, track_width, negate=False):
    """
    Calculate boundary points given the centerline and track width.
    """
    if negate:
        track_width = np.array(track_width) * -1

    dx = np.gradient(x_centerline)
    dy = np.gradient(y_centerline)
    normal_dx = -dy / np.sqrt(dx ** 2 + dy ** 2)
    normal_dy = dx / np.sqrt(dx ** 2 + dy ** 2)

    x_boundary = np.array(x_centerline) + np.array(track_width) * normal_dx
    y_boundary = np.array(y_centerline) + np.array(track_width) * normal_dy

    # Connect the last point to the first point
    x_boundary = np.append(x_boundary, x_boundary[0])
    y_boundary = np.append(y_boundary, y_boundary[0])

    return x_boundary, y_boundary


def generate_random_x_coordinate(x_coordinates, left_width, right_width):
    """
    Generate random x coordinates for the control points based on the center point and track width.
    """
    x_coord_randomised = []
    for i in range(len(x_coordinates)):
        # finding the range in x direction allowed for random control points to lie on (confined to track width)
        x_range_minimum = x_coordinates[i] - left_width[i]
        x_range_maximum = x_coordinates[i] + right_width[i]
        # randomizing x coordinates of control point with 0.5m proximity restriction to boundaries
        random_x_coordinate = random.uniform(x_range_minimum + 0.5, x_range_maximum - 0.5)
        x_coord_randomised.append(random_x_coordinate)
    return x_coord_randomised


def spline_interpolation(x_coordinate, y_coordinate):
    """
    Interpolate random control points.
    """
    tck_random = interp1d(x_coordinate, y_coordinate, kind='cubic')
    new_y_random = tck_random(x_coordinate)
    return new_y_random


def main():
    # Creating and interpolating the center line
    fig = go.Figure()

    # Calculate left and right boundary points
    x_left_boundary, y_left_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_left)
    x_right_boundary, y_right_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_right,
                                                                   negate=True)

    # Plotting the left boundary
    fig.add_trace(
        go.Scatter(x=x_left_boundary, y=y_left_boundary, mode='lines', name='Left Boundary', line=dict(color='black')))

    # Plotting the right boundary in reverse order to maintain the order of the points
    fig.add_trace(go.Scatter(x=x_right_boundary[::-1], y=y_right_boundary[::-1], mode='lines', name='Right Boundary',
                             line=dict(color='black')))

    # Generating and interpolating the random points
    x_coordinate_random_controlPt = generate_random_x_coordinate(x_coord_cp, track_width_left, track_width_right)
    interpolated_y = spline_interpolation(x_coordinate_random_controlPt, y_coord_cp)
    fig.add_trace(go.Scatter(x=x_coordinate_random_controlPt, y=interpolated_y, mode='lines', name='Random track'))

    # Ensure closure for centerline and random track
    x_coord_cp_closed = np.append(x_coord_cp, x_coord_cp[0])
    y_coord_cp_closed = np.append(y_coord_cp, y_coord_cp[0])
    interpolated_y_closed = np.append(interpolated_y, interpolated_y[0])
    fig.add_trace(go.Scatter(x=x_coord_cp_closed, y=y_coord_cp_closed, mode='lines', name='Closed Center Line'))

    fig.show()


if __name__ == "__main__":
    main()
