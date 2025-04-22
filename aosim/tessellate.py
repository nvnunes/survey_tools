#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

#region Globals

class TessellateException(Exception):
    pass

class StructType:
    pass

#endregion

#region Veronoi Tesselation

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

def closest_num_hex_rings(N):
    """
    Estimate the number of hexagonal rings needed for N.
    """
    if N == 1:
        return 0  # Only center point

    R = (-1 + np.sqrt(1 + 4 * (N - 1) / 3)) / 2
    return int(np.round(R))  # Round to nearest integer

def num_points_using_hex_rings(num_rings):
    """
    Calculate the number of points in a hexagonal grid with the given number of rings.
    """
    return 3 * num_rings * (num_rings + 1) + 1

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

def _hexagonal_grid_in_circle(N, radius, omit_centre=False):
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
    points = _sort_points(points, radius)[:N]  # Trim to exactly N points

    if omit_centre:
        points = points[1:]

    return points

def _fibonacci_lattice_in_circle(N, radius, omit_centre=False):
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

    points = np.column_stack((x, y))

    if not omit_centre:
        points = np.vstack(([0, 0], points))

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

def _lloyd_relaxation_in_circle(points, radius, method='fibonacci', omit_centre=False, max_iterations=20, alpha=0.5, verbose=False):
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
        if not omit_centre and is_symmetric:
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
            if not omit_centre:
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

def tessellate_circle(N, radius=1.0, method='hexagon', omit_centre=False, max_iterations=1000, verbose=False):
    """
    Generate a Voronoi tessellation inside a circle with given radius and number of points
    """

    if N < 5:
        raise TessellateException('Minimum number of cells is 5')

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
            initial_points = _hexagonal_grid_in_circle(N, radius, omit_centre=omit_centre)

    # Produce Voronoi diagram and run Lloyd's relaxation to optimize
    final_points, final_cells, relative_areas = _lloyd_relaxation_in_circle(initial_points, radius, omit_centre=omit_centre, method=method, max_iterations=max_iterations, verbose=verbose)

    # Output results
    if verbose:
        if len(final_points) != N:
            print(f"Warning: Number of points changed from {N} to {len(final_points)}")

        print('Relative Areas:')
        print(f"  mean   = {np.mean(relative_areas):.1%}")
        print(f"  std    = {np.std(relative_areas):.1%}")

    return final_points, final_cells

def tessellate_circle_with_hexagons(num_rings, radius=1.0, omit_centre=False, max_iterations=1000, verbose=False):
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
        raise TessellateException('Minimum number of rings is 1')

    N = 3*num_rings*(num_rings+1)+1
    return tessellate_circle(N, radius=radius, method='hex', omit_centre=omit_centre, max_iterations=max_iterations, verbose=verbose)

#endregion

#region Symmetric Polar Grid

def generate_points_from_first_quadrant(points=None, r=None, theta=None, return_polar=False):
    if points is None and (r is None or theta is None):
        raise ValueError("Either 'points' or both 'r' and 'theta' must be provided.")

    if points is not None:
        r = np.linalg.norm(points, axis=1)
        theta = np.arctan2(points[:, 1], points[:, 0])

    r_new = np.concatenate([r, r, r, r])
    theta_new = np.concatenate([theta, theta+np.pi/2, theta+np.pi, theta+3*np.pi/2])

    if return_polar:
        return r_new, theta_new

    x = r_new * np.cos(theta_new)
    y = r_new * np.sin(theta_new)
    return np.column_stack([x, y])

def generate_symmetric_polar_grid(n_rings, n_first_ring, r_min, r_max, delta_n=1, return_polar=False):
    """
    Generate a 4-quadrant symmetric grid from a Q1 base using polar coordinates.

    Parameters:
    - n_rings: number of concentric rings
    - n_first_ring: number of angular divisions in the first ring (Q1 only)
    - r_min: radius of the innermost ring
    - r_max: radius of the outermost ring
    - delta_n: number of additional angular points to add per ring (optional)
    - return_polar: if True, return (r_all, theta_all); else return Cartesian xy points

    Returns:
    - r_all, theta_all if return_polar is True
    - np.column_stack([x, y]) if return_polar is False
    """
    if r_min >= r_max or n_first_ring <= 0 or n_rings <= 0:
        raise ValueError("Invalid parameters: check radii and number of rings.")

    # Linearly spaced radii
    radii = np.linspace(r_min, r_max, n_rings)
    
    r_list = []
    theta_list = []

    for i, r_val in enumerate(radii):
        n_points = n_first_ring + i * delta_n
        angles = np.linspace(0, np.pi / 2, n_points, endpoint=False)
        r_list.extend([r_val] * n_points)
        theta_list.extend(angles)

    r = np.array(r_list)
    theta = np.array(theta_list)

    # 4-quadrant symmetry
    r_all = np.tile(r, 4)
    theta_all = np.concatenate([
        theta,
        theta + np.pi / 2,
        theta + np.pi,
        theta + 3 * np.pi / 2
    ])

    if return_polar:
        return r_all, theta_all

    x = r_all * np.cos(theta_all)
    y = r_all * np.sin(theta_all)

    return np.column_stack([x, y])

#endregion

#region Asterisms

Asterism = namedtuple("Triangle", ["r", "theta", "x", "y", "area", "scale", "score"], defaults=(None, None, None, None, None, None, None))

def create_asterism(r, theta):
    coords = polar_to_cartesian(r, theta)
    center_coords = get_triangle_incenter(coords)
    area = get_triangle_area(coords)
    mean_dist = get_triangle_mean_distance(coords, center_coords=center_coords)
    score = get_normalized_compactness_score(coords, mean_dist=mean_dist, area=area)
    return Asterism(r=r, theta=theta, x=coords[0], y=coords[1], area=area, scale=2*mean_dist, score=score)

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def get_triangle_area(coords):
    """
    Compute the area of a triangle defined by 3 points.
    """
    x, y = coords[:, 0], coords[:, 1]
    area = 0.5 * np.abs(
        x[0]*(y[1] - y[2]) +
        x[1]*(y[2] - y[0]) +
        x[2]*(y[0] - y[1])
    )
    return area

def get_triangle_incenter(coords):
    """
    Compute the incenter (ux, uy) of a triangle given 3 Cartesian points.
    Input: coords is (3, 2) array [[x1,y1], [x2,y2], [x3,y3]]
    """
    ax, ay = coords[0]
    bx, by = coords[1]
    cx, cy = coords[2]

    d1 = np.sqrt((bx - cx)**2 + (by - cy)**2)  # side opposite A
    d2 = np.sqrt((cx - ax)**2 + (cy - ay)**2)  # side opposite B
    d3 = np.sqrt((ax - bx)**2 + (ay - by)**2)  # side opposite C
    p = d1 + d2 + d3

    ux = (d1 * ax + d2 * bx + d3 * cx) / p
    uy = (d1 * ay + d2 * by + d3 * cy) / p
    return ux, uy

def is_incenter_within_radius(r, theta, max_radius):
    coords = polar_to_cartesian(r, theta)
    ux, uy = get_triangle_incenter(coords)
    return np.hypot(ux, uy) <= max_radius

def get_triangle_mean_distance(coords, center_coords=None):
    """
    Computes an angle-invariant scale for a triangle:
    the maximum distance from a center point (default: incenter) to its vertices.
    """
    if center_coords is None:
        cx, cy = get_triangle_incenter(coords)
    else:
        cx, cy = center_coords

    dists = np.linalg.norm(coords - np.array([cx, cy]), axis=1)
    return np.mean(dists)

def get_normalized_compactness_score(coords, center_coords=None, mean_dist=None, area=None):
    """
    Compute a normalized compactness score for a triangle:
    - Area efficiency relative to mean distance from center
    - Normalized so equilateral triangle → score = 1
    """
    if mean_dist is None:
        if center_coords is None:
            center_coords = get_triangle_incenter(coords)
        mean_dist = get_triangle_mean_distance(coords, center_coords)

    if area is None:
        area = get_triangle_area(coords)

    raw_score = area / (mean_dist ** 2)
    equil_ref = 3 * np.sqrt(3) / 4  # equilateral score baseline

    return raw_score / equil_ref

def is_degenerate_triangle(r, theta, tol=1e-3):
    """
    Returns True if the triangle is degenerate (area ≈ 0).
    Inputs are arrays of r and theta (length 3).
    """
    coords = polar_to_cartesian(r, theta)
    x, y = coords[:, 0], coords[:, 1]
    
    # Shoelace formula for area
    area = 0.5 * abs(
        x[0]*(y[1] - y[2]) +
        x[1]*(y[2] - y[0]) +
        x[2]*(y[0] - y[1])
    )
    return area < tol

def canonicalize_asterism(indices, r, theta):
    pts = polar_to_cartesian(r[indices], theta[indices])
    
    def apply_transform(mat, pts):
        return np.dot(pts, mat.T)

    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    rotation_matrices = [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angles]
    reflection_matrices = [
        np.array([[1, 0], [0, -1]]),
        np.array([[-1, 0], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1], [-1, 0]])
    ]

    all_variants = []
    for R in rotation_matrices:
        rotated = apply_transform(R, pts)
        all_variants.append(np.sort(rotated, axis=0))
        for M in reflection_matrices:
            reflected = apply_transform(M @ R, pts)
            all_variants.append(np.sort(reflected, axis=0))

    signatures = [tuple(np.round(v.flatten(), 6)) for v in all_variants]
    return min(signatures)

def rotate_first_point_to_q1(asterism, r, theta):
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    rotation_matrices = [
        np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for a in angles
    ]

    coords = polar_to_cartesian(r[asterism], theta[asterism])
    best_rotated = None
    best_first_index = None

    for R in rotation_matrices:
        rotated_coords = np.dot(coords, R.T)
        q1_flags = (rotated_coords[:, 0] > 0) & (rotated_coords[:, 1] > 0)
        q1_positions = np.where(q1_flags)[0]

        if len(q1_positions) == 0:
            continue

        min_index = np.argmin(asterism[q1_positions])
        min_q1_position = q1_positions[min_index]
        rolled = np.roll(asterism, -min_q1_position)
        first_index = rolled[0]

        if (best_first_index is None) or (first_index < best_first_index):
            best_first_index = first_index
            best_rotated = rolled

    return best_rotated if best_rotated is not None else asterism

def find_asterism(asterisms, r, theta):
    for i, ast in enumerate(asterisms):
        if np.all(np.isclose(np.sort(ast.r), np.sort(r))) and np.all(np.isclose(np.sort(ast.theta), np.sort(theta))):
            return i
    return None

def make_asterism_changes(asterisms):
    additions = [
        (np.array([10, 10, 10]), np.deg2rad(np.array([0, 120, 240]))),
        (np.array([20, 20, 20]), np.deg2rad(np.array([0, 120, 240]))),
        (np.array([40, 40, 40]), np.deg2rad(np.array([0, 120, 240]))),
        (np.array([50, 50, 50]), np.deg2rad(np.array([0, 120, 240]))),
    ]

    for addition in additions:
        asterisms.append(create_asterism(addition[0], addition[1]))

    deletions = [
       (np.array([60, 60, 60]), np.deg2rad(np.array([15, 135, 255]))),
    ]

    for deletion in deletions:
        idx = find_asterism(asterisms, deletion[0], deletion[1])
        if idx is not None:
            asterisms.pop(idx)

    return asterisms

def generate_asterisms(r, theta, max_incentre_distance=2.0, return_stats=False):
    N = len(r)

    # --- Main Asterism Loop ---
    triplets = np.array(np.meshgrid(*[np.arange(N)]*3)).T.reshape(-1, 3)
    num_triplets = len(triplets)

    # --- Filter duplicate indices ---
    triplets = triplets[np.all(np.diff(np.sort(triplets, axis=1), axis=1) > 0, axis=1)]
    num_non_repeat_triplets = len(triplets)

    seen_signatures = {}
    num_degenerate = 0
    num_not_centered = 0

    for triplet in triplets:
        if is_degenerate_triangle(r[triplet], theta[triplet]):
            num_degenerate += 1
            continue

        if not is_incenter_within_radius(r[triplet], theta[triplet], max_incentre_distance):
            num_not_centered += 1
            continue

        sig = canonicalize_asterism(triplet, r, theta)
        if sig not in seen_signatures:
            seen_signatures[sig] = triplet

    unique_triplets = np.array([rotate_first_point_to_q1(ast, r, theta) for ast in seen_signatures.values()])
    asterisms = [create_asterism(r[triplet], theta[triplet]) for triplet in unique_triplets]
    asterisms = make_asterism_changes(asterisms)
    asterisms.sort(key=lambda t: (-(t.score), abs(t.scale - 60)))

    if return_stats:
        stats = StructType()
        stats.max_incentre_distance = max_incentre_distance
        stats.num_triplets = num_triplets
        stats.num_non_repeat_triplets = num_non_repeat_triplets
        stats.num_degenerate = num_degenerate
        stats.num_not_centered = num_not_centered
        stats.num_asterisms = len(asterisms)
        return asterisms, stats
    else:
        return asterisms

#endregion