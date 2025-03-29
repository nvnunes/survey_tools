#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

#########################################################################
# Symmetric Equal-Area Voronoi Tessellation
#########################################################################
#
# This is a version of Centroidal Voronoi Tessellation (CVT). In a CVT, you
# choose a set of generator points and then partition the domain (in this case,
# the circle) into regions so that each region consists of all the points closer
# to its generator than to any other. The twist is that you then require each
# generator to be the centroid (or “center of mass”) of its own region.
#
# Also performs Lloyd's relaxation with area equalization.
# This is a variant of Lloyd's algorithm that adjusts the movement of the
# centroids based on the area of their corresponding Voronoi cells. The
# algorithm iteratively moves the centroids towards the centroid of their
# Voronoi cells, but the step size is adjusted based on the area of the cells.
# This can help to create a more uniform distribution of points, especially in
# cases where the Voronoi cells have significantly different areas.
#
# Optionally, the points can be constrained to a hexagonal grid for symmetry.
# This is done by adjusting the points back onto an approximate hexagonal lattice
# after each iteration of the Lloyd relaxation. The hexagonal grid is a common
# choice for generating a symmetric distribution of points, as it minimizes the
# distance between points and maximizes the area of the Voronoi cells.
#
#########################################################################

class AOSimException(Exception):
    pass

def closest_num_hex_rings(N):
    """
    Estimate the number of hexagonal rings needed for N.
    """
    if N == 1:
        return 0  # Only center point

    R = (-1 + np.sqrt(1 + 4 * (N - 1) / 3)) / 2
    return int(np.round(R))  # Round to nearest integer

def to_polar_coordinates(points):
    """
    Convert Cartesian coordinates to polar coordinates.
    """
    r = np.linalg.norm(points, axis=1)
    theta = (np.arctan2(points[:, 1], points[:, 0]) + 2*np.pi) % (2*np.pi)
    return r, theta

def _sort_points(points, radius):
    """
    Sort points by angle and distance from the origin.
    """
    r, theta = to_polar_coordinates(points)

    # Round to nearest hexagonal grid radius to account for slight variations
    N = len(points)
    dr = radius / closest_num_hex_rings(N)
    sort_r = np.round(r/dr)

    # Sort by radial distance then angle
    sorted_indices = np.lexsort((theta, sort_r))

    # Align point 1 with theta=0
    if theta[sorted_indices[1]] != 0.0:
        theta = (theta - theta[sorted_indices[1]]) % (2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack((x, y))

    return points[sorted_indices]

def _random_points_in_circle(N, radius):
    """
    Generate random points uniformly in a circle of given radius.
    """
    r = radius * np.sqrt(np.random.rand(N))
    theta = 2 * np.pi * np.random.rand(N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))

    return _sort_points(points, radius)

def _hexagonal_grid_in_circle(N, radius):
    """
    Generate points in a circle of given radius using a hexagonal grid for symmetry.
    """
    area_per_cell = np.pi * radius**2 / N
    spacing = np.sqrt(2 * area_per_cell / np.sqrt(3))  # Approximate hex spacing

    points = []
    for i in range(-int(radius // spacing), int(radius // spacing) + 1):
        for j in range(-int(radius // spacing), int(radius // spacing) + 1):
            x = i * spacing
            y = j * spacing * np.sqrt(3) / 2
            if j % 2 != 0:  # Offset every other row
                x += spacing / 2

            if np.sqrt(x**2 + y**2) <= radius:  # Only keep points inside circle
                points.append((x, y))

    points = np.array(points)

    return _sort_points(points, radius)[:N]  # Trim to exactly N points

def _fibonacci_lattice_in_circle(N, radius):
    """
    Generate points in a circule of given radius using a Fibonacci spiral with centre point always included.
    """
    if N == 1:
        return np.array([[0, 0]])  # If only one point, it's at the center

    golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees
    r_values = radius * np.sqrt(np.arange(N-1) / (N-1)) # Exclude center
    theta_values = np.arange(N-1) * golden_angle

    x = r_values * np.cos(theta_values)
    y = r_values * np.sin(theta_values)

    points = np.vstack(([0, 0], np.column_stack((x, y))))

    return _sort_points(points, radius)

def _make_circle_polygon(radius, num_edges=256):
    """
    Approximate the circle boundary by a polygon.
    """
    angles = np.linspace(0, 2*np.pi, num_edges, endpoint=False)
    circle_coords = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
    return Polygon(circle_coords)

def _generate_ghost_points(radius, num_ghost=50):
    """
    Generate ghost points evenly spaced on the circle's boundary.
    """
    angles = np.linspace(0, 2*np.pi, num_ghost)
    ghost = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    return ghost

def _voronoi_cells_in_circle(points, ghost_points, circle_poly):
    """
    Compute the Voronoi diagram for the combined set of points (real and ghost).
    Returns a list of polygons corresponding only to the real points.
    Intersects the Voronoi cells with the circle polygon to ensure they are within the circle.
    """
    vor = Voronoi(np.vstack([points, ghost_points]))
    cells = []
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region_vertices = vor.regions[region_idx]
        poly_coords = [tuple(vor.vertices[v]) for v in region_vertices]
        cell_poly = Polygon(poly_coords)
        clipped_cell = circle_poly.intersection(cell_poly)
        cells.append(clipped_cell)
    return cells

def _compute_centroids_areas(cells):
    """
    Compute centroids and areas for a set of Voronoi cells.
    """
    centroids = []
    areas = []
    for cell in cells:
        centroids.append((cell.centroid.x, cell.centroid.y))
        areas.append(cell.area)
    return np.array(centroids), np.array(areas)

def _lloyd_relaxation_in_circle(points, radius, method='fibonacci', max_iterations=20, alpha=0.5, verbose=False):
    """
    Perform Lloyd relaxation with area equalization inside a circle.
      - points: the interior points to be updated.
      - fixed ghost points are added on the circle's boundary.
      - The Voronoi diagram is computed for the union.
      - If equal area, move the points towards their centroids, adjusting the step size based on the area of the cells.
        = If a cell is too large, it moves toward its centroid more aggressively.
        = If a cell is too small, movement is more restricted.
    """
    N = len(points)
    is_symmetric = 'rand' not in method

    target_area = np.pi * radius**2 / N
    estimated_num_edge_cells = int(2*np.sqrt(np.pi*N)-np.pi) # num target area cells in annulus sqrt(target area) thick
    circle_poly = _make_circle_polygon(radius, num_edges=max(256, estimated_num_edge_cells))

    ghost_radius = radius + np.sqrt(target_area)
    ghost_points = _generate_ghost_points(ghost_radius, estimated_num_edge_cells)

    for i in range(max_iterations):
        cells = _voronoi_cells_in_circle(points, ghost_points, circle_poly)
        centroids, areas = _compute_centroids_areas(cells)

        # Ensure center point stays fixed at (0,0)
        if is_symmetric:
            centroids[0] = [0, 0]

        # Adjust step size based on area discrepancy
        scaling_factor = (target_area / areas) ** alpha  # Gradual adjustment
        new_points = points + (centroids - points) * scaling_factor[:, None]

        # Ensure points stay inside the circle
        r, theta = to_polar_coordinates(new_points)
        max_distances = np.minimum(r, radius * 0.99)

        if is_symmetric:
            # Enforce radial symmetry by keeping points in approximate hex grid positions
            new_points[:,0] = max_distances * np.cos(theta)
            new_points[:,1] = max_distances * np.sin(theta)

            # Keep center point unchanged
            new_points[0] = [0, 0]
        else:
            new_points *= max_distances[:, None] / r[:, None]

        if np.mean(np.linalg.norm(new_points - points, axis=1)) / radius < 1/np.sqrt(N)/1e3:
            if verbose:
                print(f"Converged after {i} iterations")
            break

        points = _sort_points(new_points, radius)

    relative_areas = areas / target_area

    return centroids, cells, relative_areas

def plot_circle_tessellation(points, cells, radius, method=None):
    _, ax = plt.subplots(figsize=(6,6))

    # Plot the circle boundary
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    # Plot each cell
    for poly in cells:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.3)

    # Plot center of each cell
    if len(points) <= 256:
        for i, (x, y) in enumerate(points):
            ax.text(x, y, str(i), fontsize=8, ha='center', va='center')
    else:
        ax.scatter(points[:,0], points[:,1], c='red', s=10, zorder=5)

    ax.set_aspect('equal')
    ax.set_xlim(-radius*1.1, radius*1.1)
    ax.set_ylim(-radius*1.1, radius*1.1)
    if method is not None:
        plt.title(f"Voronoi Tessellation in a Circle (N={len(points)}, method={method})")
    else:
        plt.title(f"Voronoi Tessellation in a Circle (N={len(points)})")
    plt.show()

def tessellate_circle(N, radius=1.0, method='hexagon', max_iterations=1000, verbose=False):
    """
    Generate a Voronoi tessellation inside a circle with given radius and number of points
    """

    if N < 5:
        raise AOSimException('Minimum number of cells is 5')

    if verbose:
        print('Parameters:')
        print(f"  points = {N}")
        print(f"  radius = {radius}")

    # Generate random points
    match method:
        case 'rand' | 'random':
            initial_points = _random_points_in_circle(N, radius)
        case 'fib' | 'fibonacci':
            initial_points = _fibonacci_lattice_in_circle(N, radius)
        case _: # hexagonal
            initial_points = _hexagonal_grid_in_circle(N, radius)

    # Produce Voronoi diagram and run Lloyd's relaxation to optimize
    final_points, final_cells, relative_areas = _lloyd_relaxation_in_circle(initial_points, radius, method=method, max_iterations=max_iterations, verbose=verbose)

    # Output results
    if verbose:
        if len(final_points) != N:
            print(f"Warning: Number of points changed from {N} to {len(final_points)}")

        print('Relative Areas:')
        print(f"  mean   = {np.mean(relative_areas):.1%}")
        print(f"  std    = {np.std(relative_areas):.1%}")

    return final_points, final_cells

def tessellate_circle_with_hexagons(num_rings, radius=1.0,max_iterations=1000, verbose=False):
    """
    Hexagons tessellate efficiently, approximate circles well, and offer optimal connectivity.
    This is why they are widely used in nature (beehives, molecular structures), technology
    (network grids, imaging), and mathematics (Voronoi diagrams, game maps).

    For complete rings of hexagons, use N = 3n(n+1)+1 where n is the number of rings. So:
    n=1 -> N = 7
    n=2 -> N = 19
    n=3 -> N = 37
    n=4 -> N = 61
    n=5 -> N = 91
    n=6 -> N = 127 (use 131 as a starting number to land on 127)
    """

    if num_rings < 1:
        raise AOSimException('Minimum number of rings is 1')

    N = 3*num_rings*(num_rings+1)+1
    return tessellate_circle(N, radius=radius, method='hex', max_iterations=max_iterations, verbose=verbose)
