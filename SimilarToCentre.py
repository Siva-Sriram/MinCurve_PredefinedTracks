import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
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


def calculate_curvature(x, y):
    """
    Calculate curvature at each point of the track.
    """
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs((dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / ((dx_dt ** 2 + dy_dt ** 2) ** 1.5))
    return curvature


def generate_control_points(x_centerline, y_centerline, track_width_left, track_width_right, deviation_factor=0.5, smoothness=0.5):
    """
    Generate control points for the spline interpolation based on deviation from the centerline and desired smoothness.
    """
    curvature = calculate_curvature(x_centerline, y_centerline)
    dx = np.gradient(x_centerline)
    dy = np.gradient(y_centerline)
    tangent_angle = np.arctan2(dy, dx)

    x_control_points = []
    y_control_points = []

    for i in range(len(x_centerline)):
        # Calculate the desired offset from the centerline based on curvature, smoothness, and deviation factor
        offset = curvature[i] * smoothness + deviation_factor

        # Calculate the x and y coordinates of the control point based on the offset and tangent angle
        x_control_point = x_centerline[i] + offset * np.cos(tangent_angle[i])
        y_control_point = y_centerline[i] + offset * np.sin(tangent_angle[i])

        # Ensure the control point is within the track boundaries
        x_control_point = np.clip(x_control_point,
                                  x_centerline[i] - track_width_left[i],
                                  x_centerline[i] + track_width_right[i])

        x_control_points.append(x_control_point)
        y_control_points.append(y_control_point)

    return x_control_points, y_control_points


def spline_interpolation(x_coordinate, y_coordinate):
    """
    Interpolate control points.
    """
    tck = interp1d(x_coordinate, y_coordinate, kind='cubic')
    new_y = tck(x_coordinate)
    return new_y


def adjust_track(x_coordinates, y_coordinates, curvature):
    """
    Adjust track based on curvature.
    """
    # Scale curvature to target maximum curvature
    max_curvature = np.max(curvature)
    target_curvature = max_curvature / 2.0  # Arbitrary reduction factor
    scaled_curvature = curvature * (target_curvature / max_curvature)

    # Apply adjustment to y coordinates
    y_adjusted = y_coordinates - scaled_curvature

    return x_coordinates, y_adjusted


def main():
    best_curvature = float('inf')
    best_track = None
    best_deviation_factor = None

    # Iterative optimization of track curvature and deviation factor
    for generation in range(100):
        # Try different deviation factor values
        for deviation_factor in np.linspace(0.0, 5.0, 11):
            # Generate control points based on deviation from the centerline and smoothness
            x_control_points, y_control_points = generate_control_points(x_coord_cp, y_coord_cp, track_width_left, track_width_right, deviation_factor=deviation_factor)

            # Interpolate control points
            interpolated_y = spline_interpolation(x_control_points, y_control_points)

            # Calculate curvature
            curvature = calculate_curvature(x_control_points, interpolated_y)

            # Adjust track based on curvature
            x_adjusted, y_adjusted = adjust_track(x_control_points, interpolated_y, curvature)

            # Calculate total curvature
            total_curvature = np.sum(curvature)

            # Update best track and deviation factor if current track has lower curvature
            if total_curvature < best_curvature:
                best_curvature = total_curvature
                best_track = (x_adjusted, y_adjusted)
                best_deviation_factor = deviation_factor

    # Plot the best track
    fig = go.Figure()

    # Calculate left and right boundary points
    x_left_boundary, y_left_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_left)
    x_right_boundary, y_right_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_right, negate=True)

    # Plotting the left boundary
    fig.add_trace(go.Scatter(x=x_left_boundary, y=y_left_boundary, mode='lines', name='Left Boundary', line=dict(color='black')))

    # Plotting the right boundary in reverse order to maintain the order of the points
    fig.add_trace(go.Scatter(x=x_right_boundary[::-1], y=y_right_boundary[::-1], mode='lines', name='Right Boundary', line=dict(color='black')))

    # Plot the best track
    x_best, y_best = best_track
    fig.add_trace(go.Scatter(x=x_best, y=y_best, mode='lines', name='Best Track'))

    # Ensure closure for centerline and final track
    x_coord_cp_closed = np.append(x_coord_cp, x_coord_cp[0])
    y_coord_cp_closed = np.append(y_coord_cp, y_coord_cp[0])
    fig.add_trace(go.Scatter(x=x_coord_cp_closed, y=y_coord_cp_closed, mode='lines', name='Closed Center Line'))

    fig.show()

    # Calculate and print total curvature of the best track
    final_curvature = calculate_curvature(x_best, y_best)
    total_curvature_final_track = np.sum(final_curvature)
    print("Total curvature of the best track:", total_curvature_final_track)
    print("Optimal deviation factor:", best_deviation_factor)

    # Calculate and print total curvature of the centerline
    centerline_curvature = calculate_curvature(x_coord_cp, y_coord_cp)
    total_curvature_centerline = np.sum(centerline_curvature)
    print("Total curvature of the centerline:", total_curvature_centerline)

if __name__ == "__main__":
    main()