import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import csv
import math
import scipy.interpolate as interp

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


def calculate_curvature(track):
    """
    Calculate curvature at each point of the track.
    """
    x, y = zip(*track)
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs((dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / ((dx_dt ** 2 + dy_dt ** 2) ** 1.5))
    return curvature


def adjust_track(x_coordinates, y_coordinates, curvature):
    """
    Adjust track based on curvature.
    """
    # Scale curvature to target maximum curvature
    max_curvature = np.max(curvature)
    target_curvature = max_curvature / 3.0  # Adjusted reduction factor
    scaled_curvature = curvature * (target_curvature / max_curvature)

    # Apply adjustment to y coordinates
    y_adjusted = y_coordinates - scaled_curvature

    return x_coordinates, y_adjusted


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


def generate_smooth_random_track(x_centerline, y_centerline, left_width, right_width, num_points=200):
    """
    Generate a smooth random track within the given track width bounds.
    """
    random_track = []
    for x, y, left, right in zip(x_centerline, y_centerline, left_width, right_width):
        # Randomly select a y-coordinate within the track width bounds with smaller variation
        y_random = random.uniform(y - 0.5 * left, y + 0.5 * right)
        random_track.append((x, y_random))

    # Smooth the random track using spline interpolation
    x_random, y_random = zip(*random_track)
    t = np.linspace(0, 1, num_points)
    t_random = np.linspace(0, 1, len(random_track))
    x_smooth = np.interp(t, t_random, x_random)
    y_smooth = np.interp(t, t_random, y_random)

    return list(zip(x_smooth, y_smooth))


def curvature_penalty(track, target_curvature=3):
    """
    Calculate the penalty for curvature values above the target value.
    """
    curvature = calculate_curvature(track)
    return np.sum(np.maximum(curvature - target_curvature, 0))


def evaluate_track(track, target_curvature=3, penalty_weight=100):
    """
    Evaluate the curvature of a track with a penalty for high curvature values.
    """
    curvature = calculate_curvature(track)
    total_curvature = np.sum(curvature)
    curvature_penalty_value = curvature_penalty(track, target_curvature=target_curvature)
    return total_curvature + penalty_weight * curvature_penalty_value


def crossover(parent1, parent2):
    """
    Crossover operation to generate offspring.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def mutate(track, mutation_rate=0.05):
    """
    Mutate a track by randomly adjusting its y-coordinates.
    """
    mutated_track = []
    for x, y in track:
        if random.random() < mutation_rate:
            mutation_amount = random.uniform(-0.1, 0.1)  # Smaller mutation range
            y += mutation_amount
        mutated_track.append((x, y))
    return mutated_track


def select_parents(population, target_curvature, penalty_weight):
    """
    Select two parents from the population using tournament selection with curvature penalty.
    """
    tournament_size = 3
    selected_parents = random.sample(population, tournament_size)
    return sorted(selected_parents, key=lambda x: evaluate_track(x, target_curvature, penalty_weight))[:2]


def genetic_algorithm(num_generations, population_size, target_curvature, penalty_weight):
    """
    Genetic algorithm to evolve tracks with minimal curvature and penalized for high curvature values.
    """
    population = [generate_smooth_random_track(x_coord_cp, y_coord_cp, track_width_left, track_width_right)
                  for _ in range(population_size)]

    for generation in range(num_generations):
        # Select parents
        parent1, parent2 = select_parents(population, target_curvature, penalty_weight)

        # Crossover and mutate to create offspring
        child = crossover(parent1, parent2)
        child = mutate(child)

        # Replace worst track in population with offspring
        worst_index = np.argmax([evaluate_track(track, target_curvature, penalty_weight) for track in population])
        population[worst_index] = child

    # Select the best track from the final population
    best_track = min(population, key=lambda x: evaluate_track(x, target_curvature, penalty_weight))
    return best_track


def smooth_track_with_spline(track, num_points):
    """
    Smooth a track using cubic spline interpolation.
    """
    x_track, y_track = zip(*track)
    tck, _ = interp.splprep([x_track, y_track], s=0.5, per=True)  # Adjust s for smoothing
    x_smooth, y_smooth = interp.splev(np.linspace(0, 1, num_points), tck)
    return list(zip(x_smooth, y_smooth))


def main():
    num_generations = 250
    population_size = 75
    target_curvature = 0.5  # Lower target curvature value
    penalty_weight = 300  # Increased penalty weight for high curvature values
    mutation_rate = 0.05
    mutation_range = 0.1

    best_track = genetic_algorithm(num_generations, population_size, target_curvature, penalty_weight)

    # Smooth the best track using cubic spline interpolation with reduced num_points
    smooth_best_track = smooth_track_with_spline(best_track, num_points=500)  # Decreased num_points

    # Calculate curvature of the centerline
    centerline_curvature = evaluate_track(list(zip(x_coord_cp, y_coord_cp)), target_curvature, penalty_weight)

    # Plotting the best track and boundaries
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

    # Plot the smooth best track
    x_smooth_best, y_smooth_best = zip(*smooth_best_track)
    fig.add_trace(go.Scatter(x=x_smooth_best, y=y_smooth_best, mode='lines', name='Smooth Best Track'))

    # Ensure closure for centerline and final track
    x_coord_cp_closed = np.append(x_coord_cp, x_coord_cp[0])
    y_coord_cp_closed = np.append(y_coord_cp, y_coord_cp[0])
    fig.add_trace(go.Scatter(x=x_coord_cp_closed, y=y_coord_cp_closed, mode='lines', name='Closed Center Line'))

    fig.show()

    # Calculate and print total curvature of the smooth best track
    total_curvature_smooth_best_track = evaluate_track(smooth_best_track, target_curvature, penalty_weight)
    print("Total curvature of the smooth best track:", total_curvature_smooth_best_track)

    # Print total curvature of the centerline
    print("Total curvature of the centerline:", centerline_curvature)


if __name__ == "__main__":
    main()



