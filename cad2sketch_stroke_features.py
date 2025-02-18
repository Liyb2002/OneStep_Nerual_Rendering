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


def extract_all_lines(final_edges_data):
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

        feature_lines.append(stroke)

    return feature_lines

