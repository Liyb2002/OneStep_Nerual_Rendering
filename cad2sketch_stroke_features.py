import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline


from itertools import product


import json
import os



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


# ------------------------------------------------------------------------------------# 
def extract_input_json(final_edges_data, subfolder_path):
    """
    Extracts stroke data from final_edges_data and saves it as 'input.json' in the specified subfolder.

    Parameters:
    - final_edges_data: Dictionary containing stroke information.
    - subfolder_path: Path where the JSON file should be saved.
    """
    strokes = []
    stroke_id_mapping = {}  # Maps stroke keys to index IDs
    current_id = 0

    for key, stroke in final_edges_data.items():
        stroke_type = stroke["type"]

        # Only consider feature, extrude, and fillet lines
        if stroke_type in ["feature_line", "extrude_line", "fillet_line"]:
            geometry = stroke["geometry"]

            if len(geometry) == 2:
                # Straight line: (x1, y1, z1, x2, y2, z2)
                stroke_data = {
                    "id": current_id,
                    "type": "line",
                    "coords": [*geometry[0], *geometry[1]]  # Flatten start & end points
                }
            else:
                # Curve line: (x1, y1, z1, x2, y2, z2, cx, cy, cz)
                start = geometry[0]
                end = geometry[-1]
                control = geometry[1]  # Assuming single control point for now

                stroke_data = {
                    "id": current_id,
                    "type": "curve",
                    "coords": [*start, *end, *control]
                }

            strokes.append(stroke_data)
            stroke_id_mapping[key] = current_id
            current_id += 1

    # Extract intersections based on geometry proximity (to be implemented)
    intersections = extract_intersections(strokes)

    dataset_entry = {
        "strokes": strokes,
        "intersections": intersections,
        "construction_lines": []  # Placeholder until we define a method
    }

    # Ensure the folder exists before saving
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        
    json_path = os.path.join(subfolder_path, "input.json")

    # Save to file
    with open(json_path, "w") as f:
        json.dump(dataset_entry, f, indent=4)

    print(f"JSON dataset saved successfully at: {json_path}")

# Function to extract intersections (to be implemented)
def extract_intersections(strokes):
    """
    Placeholder function to determine intersections based on stroke coordinates.
    Returns a list of index pairs.
    """
    return []  # Implement intersection logic later
