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

# Extract data
x_coord_cp = column['x_m']
y_coord_cp = column['y_m']
track_width_right = column['w_tr_right_m']
track_width_left = column['w_tr_left_m']


# The following function generates random x coordinates for the control points based on the center point and track width
def generate_random_x_coordinate(x_coordinates, left_width, right_width):
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
    # interpolate random control points
    tck_random = interp1d(x_coordinate, y_coordinate, kind='cubic')
    new_y_random = tck_random(x_coordinate)
    return new_y_random


def main():
    # Creating and interpolating the center line
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_coord_cp, y=y_coord_cp, mode='lines', name='Center Line'))

    # Creating, interpolating, and plotting boundaries
    x_left_boundary = [x_coord_cp[i] - track_width_left[i] for i in range(len(x_coord_cp))]
    x_right_boundary = [x_coord_cp[i] + track_width_right[i] for i in range(len(x_coord_cp))]

    tck_left_boundary = interp1d(x_left_boundary, y_coord_cp, kind='cubic')
    new_y_left_boundary = tck_left_boundary(x_left_boundary)
    fig.add_trace(go.Scatter(x=x_left_boundary, y=new_y_left_boundary, mode='lines', name='Left Boundary'))

    tck_right_boundary = interp1d(x_right_boundary, y_coord_cp, kind='cubic')
    new_y_right_boundary = tck_right_boundary(x_right_boundary)
    fig.add_trace(go.Scatter(x=x_right_boundary, y=new_y_right_boundary, mode='lines', name='Right Boundary'))

    # Generating and interpolating the random points
    x_coordinate_random_controlPt_left = generate_random_x_coordinate(x_coord_cp, track_width_left,
                                                                      [0] * len(track_width_right))
    x_coordinate_random_controlPt_right = generate_random_x_coordinate(x_coord_cp, [0] * len(track_width_left),
                                                                       track_width_right)

    interpolated_y_left = spline_interpolation(x_coordinate_random_controlPt_left, y_coord_cp)
    interpolated_y_right = spline_interpolation(x_coordinate_random_controlPt_right, y_coord_cp)

    fig.add_trace(
        go.Scatter(x=x_coordinate_random_controlPt_left, y=interpolated_y_left, mode='lines', name='Random Left track'))
    fig.add_trace(go.Scatter(x=x_coordinate_random_controlPt_right, y=interpolated_y_right, mode='lines',
                             name='Random Right track'))

    fig.show()


if __name__ == "__main__":
    main()

# import numpy as np
# import pandas as pd
# from scipy.interpolate import interp1d
# import plotly.graph_objects as go
# import random
# import csv
#
# # csv files contain x,y coords of centerline, track widths to the left and right
# # note that the csv files use smoothed center points, so does not lie exactly on the center of the track
# with open('Austin.csv', 'r') as f:
#     reader = csv.reader(f)
#     headers = next(reader)
#     column = {h: [] for h in headers}
#     for row in reader:
#         for h, v in zip(headers, row):
#             column[h].append(float(v))
#
# # Separating each column into its own list
# x_coord_cp = column['x_m']
# y_coord_cp = column['y_m']
# track_width_right = column['w_tr_right_m']
# track_width_left = column['w_tr_left_m']
#
# # The following function generates random x coordinates for the control points based on the center point and track width
# def generate_random_x_coordinate(x_coordinates, left_width, right_width):
#     x_coord_randomised = []
#     for i in range(len(x_coordinates)):
#         # finding the range in x direction allowed for random control points to lie on (confined to track width)
#         x_range_minimum = x_coordinates[i] - left_width[i]
#         x_range_maximum = x_coordinates[i] + right_width[i]
#         # randomizing x coordinates of control point with 0.5m proximity restriction to boundaries
#         random_x_coordinate = random.uniform(x_range_minimum + 0.5, x_range_maximum - 0.5)
#         x_coord_randomised.append(random_x_coordinate)
#     return x_coord_randomised
#
# def spline_interpolation(x_coordinate, y_coordinate):
#     # interpolate random control points
#     tck_random = interp1d(x_coordinate, y_coordinate, kind='cubic')
#     new_y_random = tck_random(x_coordinate)
#     return new_y_random
#
# def main():
#     # Creating and interpolating the center line
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=x_coord_cp, y=y_coord_cp, mode='lines', name='Center Line'))
#
#     # Creating, interpolating, and plotting boundaries
#     x_left_boundary = [x_coord_cp[i] - track_width_left[i] for i in range(len(x_coord_cp))]
#     x_right_boundary = [x_coord_cp[i] + track_width_right[i] for i in range(len(x_coord_cp))]
#
#     tck_left_boundary = interp1d(x_left_boundary, y_coord_cp, kind='cubic')
#     new_y_left_boundary = tck_left_boundary(x_left_boundary)
#     fig.add_trace(go.Scatter(x=x_left_boundary, y=new_y_left_boundary, mode='lines', name='Left Boundary'))
#
#     tck_right_boundary = interp1d(x_right_boundary, y_coord_cp, kind='cubic')
#     new_y_right_boundary = tck_right_boundary(x_right_boundary)
#     fig.add_trace(go.Scatter(x=x_right_boundary, y=new_y_right_boundary, mode='lines', name='Right Boundary'))
#
#     # Generating and interpolating the random points
#     x_coordinate_random_controlPt = generate_random_x_coordinate(x_coord_cp, track_width_left, track_width_right)
#     interpolated_y = spline_interpolation(x_coordinate_random_controlPt, y_coord_cp)
#     fig.add_trace(go.Scatter(x=x_coordinate_random_controlPt, y=interpolated_y, mode='lines', name='Random track'))
#
#     fig.show()
#
# if __name__ == "__main__":
#     main()