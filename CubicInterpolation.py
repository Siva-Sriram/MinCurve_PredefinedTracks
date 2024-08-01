import numpy as np
import plotly.graph_objects as go
import csv
import os
import time
import cvxpy as cp

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
    epsilon = 1e-9
    curvature = np.abs(dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / (dx_dt**2 + dy_dt**2 + epsilon)**1.5
    return curvature

def quadratic_programming_optimization(x, y, num_points, track_width_left, track_width_right):
    x = np.array(x)
    y = np.array(y)
    track_width_left = np.array(track_width_left)
    track_width_right = np.array(track_width_right)

    # Variables for x and y coordinates
    x_var = cp.Variable(num_points)
    y_var = cp.Variable(num_points)

    # Compute arc length (approximate)
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate(([0], np.cumsum(ds)))

    # Second derivative matrices
    D = np.zeros((num_points-2, num_points))
    for i in range(num_points-2):
        D[i, i:i+3] = [1, -2, 1]

    # Objective: minimize curvature
    objective = cp.Minimize(cp.sum_squares(D @ x_var) + cp.sum_squares(D @ y_var))

    # Constraints
    constraints = [
        x_var[0] == x[0],  # Start point
        y_var[0] == y[0],
        x_var[-1] == x[-1],  # End point
        y_var[-1] == y[-1],
        cp.norm(cp.vstack([x_var[1:] - x_var[:-1], y_var[1:] - y_var[:-1]]), axis=0) <= 1.05 * ds,  # Max step size
        cp.norm(cp.vstack([x_var[1:] - x_var[:-1], y_var[1:] - y_var[:-1]]), axis=0) >= 0.95 * ds,  # Min step size
    ]

    # Track boundary constraints
    for i in range(num_points):
        normal = np.array([-np.diff(y)[i], np.diff(x)[i]])
        normal = normal / np.linalg.norm(normal)
        constraints.append(normal @ (cp.vstack([x_var[i] - x[i], y_var[i] - y[i]])) <= track_width_right[i])
        constraints.append(normal @ (cp.vstack([x_var[i] - x[i], y_var[i] - y[i]])) >= -track_width_left[i])

    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    problem.solve()
    end_time = time.time()

    return x_var.value, y_var.value, end_time - start_time

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

def calculate_track_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def calculate_global_curvature(x, y):
    curvature = calculate_curvature(x, y)
    track_length = calculate_track_length(x, y)
    return np.sum(np.abs(curvature)), len(curvature), track_length

def main():
    num_points = len(x_coord_cp)

    optimized_x, optimized_y, optimization_time = quadratic_programming_optimization(x_coord_cp, y_coord_cp, num_points, track_width_left, track_width_right)

    fig = go.Figure()

    x_left_boundary, y_left_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_left)
    x_right_boundary, y_right_boundary = calculate_boundary_points(x_coord_cp, y_coord_cp, track_width_right, negate=True)

    fig.add_trace(go.Scatter(x=x_left_boundary, y=y_left_boundary, mode='lines', name='Left Boundary', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=x_right_boundary, y=y_right_boundary, mode='lines', name='Right Boundary', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=optimized_x, y=optimized_y, mode='lines', name='Optimized Track', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_coord_cp, y=y_coord_cp, mode='lines', name='Centerline', line=dict(color='blue')))

    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    fig.show()

    global_curvature_opt, points_opt, length_opt = calculate_global_curvature(optimized_x, optimized_y)
    print(f"Optimized track:")
    print(f"Global curvature: {global_curvature_opt:.4f} m^-1")
    print(f"Number of points: {points_opt}")
    print(f"Track length: {length_opt:.2f} m")
    print(f"Optimization time: {optimization_time:.4f} seconds")
    print()

    global_curvature_centerline, points_centerline, length_centerline = calculate_global_curvature(x_coord_cp, y_coord_cp)
    print(f"Centerline:")
    print(f"Global curvature: {global_curvature_centerline:.4f} m^-1")
    print(f"Number of points: {points_centerline}")
    print(f"Track length: {length_centerline:.2f} m")
    print()

if __name__ == "__main__":
    main()