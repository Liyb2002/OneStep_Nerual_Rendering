import os
import json
import torch
from torch.utils.data import Dataset
import cad2sketch_stroke_features


class cad2sketch_dataset_loader(Dataset):
    def __init__(self):
        """
        Initializes the dataset generator by setting paths and loading the dataset.
        """
        self.data_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch')
        self.subfolder_paths = []  # Store all subfolder paths
        self.load_dataset()

    def load_dataset(self):
        """
        Loads the dataset by iterating over all subfolders and storing their paths.
        """
        if not os.path.exists(self.data_path):
            print(f"Dataset path '{self.data_path}' not found.")
            return

        folders = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]

        if not folders:
            print("No folders found in the dataset directory.")
            return

        for folder in folders:
            folder_path = os.path.join(self.data_path, folder)
            subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]

            if not subfolders:
                print(f"No subfolders found in '{folder}'. Skipping...")
                continue

            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolders[0])
                self.subfolder_paths.append(subfolder_path)  # Store paths instead of processing

    def process_subfolder(self, subfolder_path):
        """
        Processes an individual subfolder by reading JSON files and extracting relevant data.
        """
        final_edges_file_path = os.path.join(subfolder_path, 'final_edges.json')
        all_edges_file_path = os.path.join(subfolder_path, 'unique_edges.json')
        strokes_dict_path = os.path.join(subfolder_path, 'strokes_dict.json')

        # Check if required JSON files exist, printing which one is missing
        missing_files = []
        
        if not os.path.exists(final_edges_file_path):
            missing_files.append("final_edges.json")
        if not os.path.exists(all_edges_file_path):
            missing_files.append("unique_edges.json")
        if not os.path.exists(strokes_dict_path):
            missing_files.append("strokes_dict.json")

        if missing_files:
            # print(f"Skipping {subfolder_path}: Missing files: {', '.join(missing_files)}")
            return None, None, None


        # Load and visualize final edges
        final_edges_data = self.read_json(final_edges_file_path)
        feature_lines = cad2sketch_stroke_features.extract_feature_lines(final_edges_data)

        # Type 1) : vertices
        vertices_matrix = cad2sketch_stroke_features.extract_vertices(feature_lines)

        # Type 2) : midpoints
        midpoints_matrix, matrix_midPoint_relation = cad2sketch_stroke_features.extract_midpoints(feature_lines, vertices_matrix)
        
        # Type 3) : permuted_points
        x_dict, y_dict, z_dict = cad2sketch_stroke_features.extract_xyz_sets(feature_lines)
        permuted_points = cad2sketch_stroke_features.point_permutations(x_dict, y_dict, z_dict, feature_lines)

        print("vertices_matrix", vertices_matrix.shape)
        print("midpoints_matrix", midpoints_matrix.shape)
        print("permuted_points", permuted_points.shape)
        print("matrix_midPoint_relation", matrix_midPoint_relation.shape)
        print("-------------")


        cad2sketch_stroke_features.vis_permuted_points(vertices_matrix, midpoints_matrix, permuted_points, feature_lines)


        return None


    def __getitem__(self, index):
        """
        Loads and processes the next subfolder when requested.
        If a subfolder has missing files, find the next available subfolder.
        Returns a tuple (intersection_matrix, all_edges_matrix, final_edges_matrix).
        """
        while index < len(self.subfolder_paths):
            subfolder_path = self.subfolder_paths[index]
            result = self.process_subfolder(subfolder_path)
            
            if result is not None and all(item is not None for item in result):
                return result  # Return valid data

            # If missing files, move to the next available subfolder
            # print(f"Skipping index {index} due to missing files. Trying next index.")
            index += 1  

        raise IndexError("No valid subfolders left in the dataset.")

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.subfolder_paths)

    def read_json(self, file_path):
        """
        Reads a JSON file and returns its contents.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None
