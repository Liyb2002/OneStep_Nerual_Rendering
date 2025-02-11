import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline


from itertools import product






# ------------------------------------------------------------------------------------# 


def vis_feature_lines(feature_lines):
    """
    Visualize only the feature_line strokes in 3D space.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Loop through all feature_line strokes
    for stroke in feature_lines:
        geometry = stroke["geometry"]

        if len(geometry) < 2:
            continue  # Ensure there are enough points to plot

        # Plot each segment of the stroke
        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            # Extract coordinates for plotting
            x_values = [start[0], end[0]]
            y_values = [start[1], end[1]]
            z_values = [start[2], end[2]]

            # Plot the stroke as a black line
            ax.plot(x_values, y_values, z_values, color='black', linewidth=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def vis_permuted_points(vertices_matrix, midpoints_matrix, permuted_points, feature_lines):
    """
    Visualizes four kinds of 3D points in different colors along with feature-line strokes.
    Background and axes are removed for a clean visualization.

    Parameters:
    - vertices_matrix (numpy.ndarray): (n, 3) Original feature-line points.
    - midpoints_matrix (numpy.ndarray): (m, 3) Midpoints of strokes.
    - permuted_points (numpy.ndarray): (p, 3) Generated points from point_permutations.
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    """
    # Initialize 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove background and axis
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Plot original feature-line points in black
    if vertices_matrix.shape[0] > 0:
        ax.scatter(vertices_matrix[:, 0], vertices_matrix[:, 1], vertices_matrix[:, 2], 
                   color='black', s=10, label="Feature-line Points")

    # Plot midpoints in red
    if midpoints_matrix.shape[0] > 0:
        ax.scatter(midpoints_matrix[:, 0], midpoints_matrix[:, 1], midpoints_matrix[:, 2], 
                   color='red', s=10, label="Midpoints")

    # Plot permuted points in blue
    if permuted_points.shape[0] > 0:
        ax.scatter(permuted_points[:, 0], permuted_points[:, 1], permuted_points[:, 2], 
                   color='blue', s=5, label="Permuted Points")

    # Plot feature lines as green lines
    for stroke in feature_lines:
        geometry = stroke["geometry"]
        if len(geometry) >= 2:
            x_values = [point[0] for point in geometry]
            y_values = [point[1] for point in geometry]
            z_values = [point[2] for point in geometry]
            ax.plot(x_values, y_values, z_values, color='green', linewidth=1, label="Feature Lines" if "Feature Lines" not in ax.get_legend_handles_labels()[1] else "")


    # Show plot
    plt.show()


# ------------------------------------------------------------------------------------# 


def extract_feature_lines(final_edges_data):
    """
    Extracts strokes from final_edges_data where type is 'feature_line'.

    Parameters:
    - final_edges_data (dict): A dictionary where keys are stroke IDs and values contain stroke properties.

    Returns:
    - list: A list of strokes that are labeled as 'feature_line'.
    """
    feature_lines = []

    for key, stroke in final_edges_data.items():
        stroke_type = stroke['type']

        if stroke_type == 'feature_line' or stroke_type == 'extrude_line' or stroke_type == 'fillet_line':
            feature_lines.append(stroke)

    return feature_lines



def extract_xyz_sets(feature_lines):
    """
    Extracts unique min and max x, y, z values from feature_line strokes.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).

    Returns:
    - x_dict (set): Unique x values (min and max per stroke).
    - y_dict (set): Unique y values (min and max per stroke).
    - z_dict (set): Unique z values (min and max per stroke).
    """
    x_dict, y_dict, z_dict = set(), set(), set()

    # Iterate through each stroke to find min and max x, y, z
    for stroke in feature_lines:
        geometry = stroke["geometry"]

        # Extract min/max for x, y, z
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')

        for point in geometry:
            x, y, z = point[0], point[1], point[2]

            # Update min/max values
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_z, max_z = min(min_z, z), max(max_z, z)

        # Store unique values in sets
        x_dict.update([min_x, max_x])
        y_dict.update([min_y, max_y])
        z_dict.update([min_z, max_z])

    return x_dict, y_dict, z_dict



def point_permutations(x_dict, y_dict, z_dict, feature_lines):
    """
    Generates new 3D points by permuting values in x_dict, y_dict, or z_dict 
    based on the changing direction of the stroke.

    Parameters:
    - x_dict (set): Unique x values.
    - y_dict (set): Unique y values.
    - z_dict (set): Unique z values.
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).

    Returns:
    - numpy.ndarray: A (n, 3) matrix of newly generated points.
    """
    new_points = set()

    for stroke in feature_lines:
        geometry = stroke["geometry"]

        if len(geometry) == 2:
            p1, p2 = geometry[0], geometry[1]
            x1, y1, z1 = p1
            x2, y2, z2 = p2

            # Identify the changing direction
            if x1 != x2 and y1 == y2 and z1 == z2:
                changing_dict, fixed_y, fixed_z = x_dict, y1, z1
                exclude_values = {x1, x2, (x1 + x2) / 2}
                new_points.update((x, fixed_y, fixed_z) for x in changing_dict if x not in exclude_values)

            elif y1 != y2 and x1 == x2 and z1 == z2:
                changing_dict, fixed_x, fixed_z = y_dict, x1, z1
                exclude_values = {y1, y2, (y1 + y2) / 2}
                new_points.update((fixed_x, y, fixed_z) for y in changing_dict if y not in exclude_values)

            elif z1 != z2 and x1 == x2 and y1 == y2:
                changing_dict, fixed_x, fixed_y = z_dict, x1, y1
                exclude_values = {z1, z2, (z1 + z2) / 2}
                new_points.update((fixed_x, fixed_y, z) for z in changing_dict if z not in exclude_values)

    return np.array(list(new_points), dtype=np.float32) if new_points else np.empty((0, 3), dtype=np.float32)
 

def extract_vertices(feature_lines):
    """
    Extracts unique 3D points from feature_line strokes.
    - If a stroke has exactly 2 points, both points are used.
    - If a stroke has more than 2 points, min/max values of x, y, z are used to generate unique vertices.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).

    Returns:
    - numpy.ndarray: A (n, 3) matrix containing unique (x, y, z) points.
    """
    unique_vertices = set()

    # Iterate through all feature_line strokes
    for stroke in feature_lines:
        geometry = stroke["geometry"]

        if len(geometry) == 2:
            # Directly add the two endpoints
            unique_vertices.add(tuple(geometry[0]))
            unique_vertices.add(tuple(geometry[1]))

        elif len(geometry) > 2:
            # Compute min/max for x, y, z
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            min_z, max_z = float('inf'), float('-inf')

            for point in geometry:
                x, y, z = point[0], point[1], point[2]
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)

            # Generate 8 possible corner points from min/max combinations
            corner_points = {
                (min_x, min_y, min_z), (min_x, min_y, max_z),
                (min_x, max_y, min_z), (min_x, max_y, max_z),
                (max_x, min_y, min_z), (max_x, min_y, max_z),
                (max_x, max_y, min_z), (max_x, max_y, max_z)
            }

            # Add only unique corner points
            unique_vertices.update(corner_points)

    # Convert to a NumPy array of shape (n, 3)
    return np.array(list(unique_vertices))



def extract_midpoints(feature_lines, vertices_matrix):
    """
    Extracts unique midpoints from feature_line strokes and creates a midpoint-to-endpoints relationship matrix.

    Parameters:
    - feature_lines (list): List of stroke dictionaries containing geometry (list of 3D points).
    - vertices_matrix (numpy.ndarray): (n, 3) matrix of unique 3D vertices.

    Returns:
    - numpy.ndarray: (m, 3) matrix of unique midpoints.
    - numpy.ndarray: (n, m) matrix where each midpoint is connected to its two endpoints.
    """
    unique_midpoints = set()
    midpoint_relations = []

    # Iterate through strokes to compute midpoints
    for stroke in feature_lines:
        geometry = stroke["geometry"]

        if len(geometry) == 2:
            # Compute midpoint
            midpoint = tuple((np.array(geometry[0]) + np.array(geometry[1])) / 2)

            # Find indices of geometry[0] and geometry[1] in vertices_matrix
            idx_0 = np.where((vertices_matrix == geometry[0]).all(axis=1))[0]
            idx_1 = np.where((vertices_matrix == geometry[1]).all(axis=1))[0]

            # Ensure valid indices and uniqueness
            if len(idx_0) > 0 and len(idx_1) > 0:
                idx_0, idx_1 = idx_0[0], idx_1[0]

                if midpoint not in unique_midpoints:
                    unique_midpoints.add(midpoint)
                    midpoint_idx = len(unique_midpoints) - 1  # Current midpoint index
                    midpoint_relations.append((idx_0, midpoint_idx))
                    midpoint_relations.append((idx_1, midpoint_idx))

    # Convert unique midpoints to (m, 3) matrix
    midpoints_matrix = np.array(list(unique_midpoints), dtype=np.float32) if unique_midpoints else np.empty((0, 3), dtype=np.float32)

    # Create (n, m) midpoint relation matrix
    matrix_midPoint_relation = np.zeros((vertices_matrix.shape[0], midpoints_matrix.shape[0]), dtype=int)

    for v_idx, m_idx in midpoint_relations:
        matrix_midPoint_relation[v_idx, m_idx] = 1

    return midpoints_matrix, matrix_midPoint_relation
