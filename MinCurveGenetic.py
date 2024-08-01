import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import csv
import math

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
        # Randomly select a y-coordinate within the track width bounds
        y_random = random.uniform(y - left, y + right)
        random_track.append((x, y_random))

    # Smooth the random track using spline interpolation
    x_random, y_random = zip(*random_track)
    t = np.linspace(0, 1, num_points)
    t_random = np.linspace(0, 1, len(random_track))
    x_smooth = np.interp(t, t_random, x_random)
    y_smooth = np.interp(t, t_random, y_random)

    return list(zip(x_smooth, y_smooth))


def evaluate_track(track):
    """
    Evaluate the curvature of a track.
    """
    x_track, y_track = zip(*track)
    curvature = calculate_curvature(x_track, y_track)
    return np.sum(curvature)


def crossover(parent1, parent2):
    """
    Crossover operation to generate offspring.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def mutate(track, mutation_rate=0.1):
    """
    Mutate a track by randomly adjusting its y-coordinates.
    """
    mutated_track = []
    for x, y in track:
        if random.random() < mutation_rate:
            mutation_amount = random.uniform(-1, 1)
            y += mutation_amount
        mutated_track.append((x, y))
    return mutated_track


def select_parents(population):
    """
    Select two parents from the population using tournament selection.
    """
    tournament_size = 3
    selected_parents = random.sample(population, tournament_size)
    return sorted(selected_parents, key=lambda x: evaluate_track(x))[:2]


def genetic_algorithm(num_generations, population_size):
    """
    Genetic algorithm to evolve tracks with minimal curvature.
    """
    population = [generate_smooth_random_track(x_coord_cp, y_coord_cp, track_width_left, track_width_right)
                  for _ in range(population_size)]

    for generation in range(num_generations):
        # Select parents
        parent1, parent2 = select_parents(population)

        # Crossover and mutate to create offspring
        child = crossover(parent1, parent2)
        child = mutate(child)

        # Replace worst track in population with offspring
        worst_index = np.argmax([evaluate_track(track) for track in population])
        population[worst_index] = child

    # Select the best track from the final population
    best_track = min(population, key=lambda x: evaluate_track(x))
    return best_track


def main():
    num_generations = 100
    population_size = 50

    best_track = genetic_algorithm(num_generations, population_size)

    # Plot the best track
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

    # Plot the best track
    x_best, y_best = zip(*best_track)
    fig.add_trace(go.Scatter(x=x_best, y=y_best, mode='lines', name='Best Track'))

    # Ensure closure for centerline and final track
    x_coord_cp_closed = np.append(x_coord_cp, x_coord_cp[0])
    y_coord_cp_closed = np.append(y_coord_cp, y_coord_cp[0])
    fig.add_trace(go.Scatter(x=x_coord_cp_closed, y=y_coord_cp_closed, mode='lines', name='Closed Center Line'))

    fig.show()

    # Calculate and print total curvature of the best track
    total_curvature_final_track = evaluate_track(best_track)
    print("Total curvature of the best track:", total_curvature_final_track)


if __name__ == "__main__":
    main()
