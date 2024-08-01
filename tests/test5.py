import numpy as np
import plotly.graph_objects as go
import csv
import os
import scipy.special
import time

try:
    import cvxpy as cp
except ImportError:
    print("Please install cvxpy: pip install cvxpy")
    exit(1)

# Read data from CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Montreal.csv')
try:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        column = {h: [] for h in headers}
        for row in reader:
            for h, v in zip(headers, row):
                column[h].append(float(v))
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

# Extract data from CSV columns
x_coord_cp = column['x_m']
y_coord_cp = column['y_m']
track_width_right = column['w_tr_right_m']
track_width_left = column['w_tr_left_m']

def calculate_curvature(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    epsilon = 1e-9
    curvature = np.abs(dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / (dx_dt**2 + dy_dt**2 + epsilon)**1.5
    return curvature  # Return array of curvatures

def quadratic_programming_optimization(x, y, num_points, track_width_left, track_width_right):
    y = np.array(y)
    track_width_left = np.array(track_width_left)
    track_width_right = np.array(track_width_right)

    # Calculate gradients and second derivatives
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    epsilon = 1e-9

    y_var = cp.Variable(num_points)

    # Ensure the track is a closed loop
    objective = cp.Minimize(10 * cp.sum_squares(y_var[2:] - 2 * y_var[1:-1] + y_var[:-2]) +
                            10.0 * cp.sum_squares(y_var[1:] - y_var[:-1]) +
                            10.0 * cp.sum_squares(y_var[-1] - y_var[0]) +
                            1e3 * cp.sum(cp.abs(dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / (dx_dt**2 + dy_dt**2 + epsilon)**1.5))

    constraints = [y_var[i] >= y[i] - track_width_left[i] + 1.0 for i in range(num_points)]
    constraints += [y_var[i] <= y[i] + track_width_right[i] - 1.0 for i in range(num_points)]

    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    problem.solve()
    end_time = time.time()

    # Ensure the track is a closed loop
    y_opt = np.concatenate((y_var.value, [y_var.value[0]]))

    return x, y_opt, end_time - start_time  # Return execution time

def calculate_boundary_points(x_centerline, y_centerline, track_width, negate=False):
    if negate:
        track_width = np.array(track_width) * -1

    dx = np.gradient(x_centerline)
    dy = np.gradient(y_centerline)
    norm = np.sqrt(dx ** 2 + dy ** 2)
    norm = np.where(norm == 0, np.finfo(float).eps, norm)
    normal_dx = -dy / norm
    normal_dy = dx / norm

    x_boundary = np.array(x_centerline) + np.array(track_width) * normal_dx
    y_boundary = np.array(y_centerline) + np.array(track_width) * normal_dy

    x_boundary = np.append(x_boundary, x_boundary[0])
    y_boundary = np.append(y_boundary, y_boundary[0])

    return x_boundary, y_boundary

def smooth_track_with_bezier(track, num_points=100):
    smooth_track = []
    num_control_points = len(track)
    i = 0
    while i < num_control_points:
        p0 = track[i]
        p1 = track[min(i + 1, num_control_points - 1)]
        p2 = track[min(i + 2, num_control_points - 1)]

        control_points = [p0, p1, p2]
        bezier_points = bezier_curve(control_points, num_points)

        if i + 2 < num_control_points:
            smooth_track.extend(bezier_points[:-1])
        else:
            smooth_track.extend(bezier_points[:-1])  # Exclude last point to avoid duplicate

        i += 2

    # Ensure the track loops back to the start
    smooth_track.append(smooth_track[0])

    return smooth_track

def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(n + 1):
        curve += np.outer((scipy.special.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))), control_points[i])
    return curve

def interpolate_track_widths(track_width, original_points, new_points):
    return np.interp(new_points, np.linspace(0, 1, len(original_points)), track_width)

def calculate_track_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def calculate_global_curvature(x, y):
    curvature = calculate_curvature(x, y)
    track_length = calculate_track_length(x, y)
    return np.sum(np.abs(curvature)), len(curvature), track_length

def main():
    num_points = len(x_coord_cp)

    # Smooth the original path
    smooth_centerline = smooth_track_with_bezier(list(zip(x_coord_cp, y_coord_cp)), num_points=200)
    x_smooth_centerline, y_smooth_centerline = zip(*smooth_centerline)

    # Interpolate track widths to match the number of points in the smoothed path
    track_width_left_interp = interpolate_track_widths(track_width_left, x_coord_cp, x_smooth_centerline)
    track_width_right_interp = interpolate_track_widths(track_width_right, x_coord_cp, x_smooth_centerline)

    # Optimize the smoothed path
    optimized_x, optimized_y, optimization_time = quadratic_programming_optimization(x_smooth_centerline, y_smooth_centerline, len(x_smooth_centerline), track_width_left_interp, track_width_right_interp)

    # Calculate track boundaries
    x_left_boundary, y_left_boundary = calculate_boundary_points(x_smooth_centerline, y_smooth_centerline, track_width_left_interp)
    x_right_boundary, y_right_boundary = calculate_boundary_points(x_smooth_centerline, y_smooth_centerline, track_width_right_interp, negate=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=x_left_boundary, y=y_left_boundary, mode='lines', name='Left Boundary', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=x_right_boundary[::-1], y=y_right_boundary[::-1], mode='lines', name='Right Boundary', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=optimized_x, y=optimized_y, mode='lines', name='Optimized Track'))
    fig.add_trace(go.Scatter(x=x_smooth_centerline, y=y_smooth_centerline, mode='lines', name='Smooth Centerline'))

    fig.update_layout(
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )

    fig.show()

    global_curvature_opt, points_opt, length_opt = calculate_global_curvature(optimized_x, optimized_y)
    print(f"Optimized track:")
    print(f"Global curvature: {global_curvature_opt:.4f} m^-1")
    print(f"Number of points: {points_opt}")
    print(f"Track length: {length_opt:.2f} m")
    print(f"Optimization time: {optimization_time:.4f} seconds")
    print()

    global_curvature_smooth_centerline, points_smooth_centerline, length_smooth_centerline = calculate_global_curvature(x_smooth_centerline, y_smooth_centerline)
    print(f"Smooth centerline:")
    print(f"Global curvature: {global_curvature_smooth_centerline:.4f} m^-1")
    print(f"Number of points: {points_smooth_centerline}")
    print(f"Track length: {length_smooth_centerline:.2f} m")
    print()

if __name__ == "__main__":
    main()
