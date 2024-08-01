import numpy as np
import plotly.graph_objects as go
import csv
import os
import scipy.special  # Import scipy.special

try:
    import cvxpy as cp
    import scipy.interpolate as interp
except ImportError:
    print("Please install cvxpy: pip install cvxpy")
    exit(1)

# Read data from CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Austin.csv')

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
    denominator = (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    curvature = np.abs((dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / denominator)
    return curvature


def quadratic_programming_optimization(x, y, num_points, track_width_left, track_width_right):
    y = np.array(y)
    track_width_left = np.array(track_width_left)
    track_width_right = np.array(track_width_right)

    y_var = cp.Variable(num_points)

    if num_points >= 3:
        d2y_dt2 = y_var[2:] - 2 * y_var[1:-1] + y_var[:-2]
        objective = cp.Minimize(cp.sum_squares(d2y_dt2) + 10.0 * cp.sum_squares(y_var[1:] - y_var[:-1]))
    else:
        objective = cp.Minimize(10.0 * cp.sum_squares(y_var[1:] - y_var[:-1]))

    constraints = [y_var[i] >= y[i] - track_width_left[i] + 1.0 for i in range(num_points)]
    constraints += [y_var[i] <= y[i] + track_width_right[i] - 1.0 for i in range(num_points)]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x, y_var.value


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


def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(n + 1):
        curve += np.outer((scipy.special.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))), control_points[i])
    return curve


def smooth_track_with_bezier(track, num_points=100):
    smooth_track = []
    num_control_points = len(track)
    i = 0
    while i < num_control_points:
        p0 = track[i]
        p1 = track[(i + 1) % num_control_points]
        p2 = track[(i + 2) % num_control_points]

        control_points = [p0, p1, p2]

        # Adjust control points if curvature is too high
        curvature = calculate_curvature([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])
        if np.max(curvature) > 0.1:  # Adjust threshold as needed
            p1 = [(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]

        bezier_points = bezier_curve(control_points, num_points)

        if i + 2 < num_control_points:
            smooth_track.extend(bezier_points[:-1])
        else:
            smooth_track.extend(bezier_points)

        i += 2  # Skip the control point to connect every third point

    return smooth_track


def main():
    num_points = len(x_coord_cp)

    optimized_x, optimized_y = quadratic_programming_optimization(x_coord_cp, y_coord_cp, num_points, track_width_left,
                                                                  track_width_right)

    smooth_optimized_track = smooth_track_with_bezier(list(zip(optimized_x, optimized_y)), num_points=500)

    fig = go.Figure()

    x_left_boundary, y_left_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_left)
    x_right_boundary, y_right_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_right,
                                                                   negate=True)

    smooth_left_boundary = smooth_track_with_bezier(list(zip(x_left_boundary, y_left_boundary)), num_points=500)
    smooth_right_boundary = smooth_track_with_bezier(list(zip(x_right_boundary, y_right_boundary)), num_points=500)

    x_smooth_left, y_smooth_left = zip(*smooth_left_boundary)
    x_smooth_right, y_smooth_right = zip(*smooth_right_boundary)

    fig.add_trace(go.Scatter(x=x_smooth_left, y=y_smooth_left, mode='lines', name='Smooth Left Boundary',
                             line=dict(color='black')))
    fig.add_trace(go.Scatter(x=x_smooth_right[::-1], y=y_smooth_right[::-1], mode='lines', name='Smooth Right Boundary',
                             line=dict(color='black')))

    x_smooth_opt, y_smooth_opt = zip(*smooth_optimized_track)
    fig.add_trace(go.Scatter(x=x_smooth_opt, y=y_smooth_opt, mode='lines', name='Smooth Optimized Track'))

    x_coord_cp_closed = np.append(x_coord_cp, x_coord_cp[0])
    y_coord_cp_closed = np.append(y_coord_cp, y_coord_cp[0])
    smooth_centerline = smooth_track_with_bezier(list(zip(x_coord_cp_closed, y_coord_cp_closed)), num_points=500)
    x_smooth_center, y_smooth_center = zip(*smooth_centerline)
    fig.add_trace(go.Scatter(x=x_smooth_center, y=y_smooth_center, mode='lines', name='Smooth Center Line'))

    fig.show()

    total_curvature_smooth_opt_track = np.sum(calculate_curvature(x_smooth_opt, y_smooth_opt))
    print("Total curvature of the smooth optimized track:", total_curvature_smooth_opt_track)

    centerline_curvature = np.sum(calculate_curvature(x_coord_cp, y_coord_cp))
    print("Curvature of the centerline:", centerline_curvature)


if __name__ == "__main__":
    main()
