import os
import json
import torch
from torch.utils.data import Dataset
import cad2sketch_stroke_features
import shutil

from tqdm import tqdm
from pathlib import Path

class cad2sketch_dataset_loader(Dataset):
    def __init__(self):
        """
        Initializes the dataset generator by setting paths and loading the dataset.
        """
        self.data_path = os.path.join(os.getcwd(), 'dataset', 'cad2sketch')
        self.subfolder_paths = []  # Store all subfolder paths

        self.base_input_directory = Path.cwd() / "annotated_input"


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

        for subfolder_path in tqdm(self.subfolder_paths, desc=f"Cleaning Data",):
            self.process_subfolder( subfolder_path)


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


        # Load and visualize only feature lines version
        final_edges_data = self.read_json(final_edges_file_path)
        feature_lines = cad2sketch_stroke_features.extract_feature_lines(final_edges_data)
        # cad2sketch_stroke_features.vis_feature_lines(feature_lines)


        # Load and visualize only final edges (feature + construction lines)
        all_lines = cad2sketch_stroke_features.extract_all_lines(final_edges_data)
        # cad2sketch_stroke_features.vis_feature_lines(all_lines)


        # create new directory
        last_two_dirs = os.path.join(*subfolder_path.rstrip(os.sep).split(os.sep)[-2:])
        new_directory = os.path.join(self.base_input_directory, last_two_dirs)
        os.makedirs(new_directory, exist_ok=True)

        # Prepare the input data
        cad2sketch_stroke_features.extract_input_json(final_edges_data, new_directory)

        return None


    def __getitem__(self, index):
        """
        Loads and processes the next subfolder when requested.
        If a subfolder has missing files, find the next available subfolder.
        Returns a tuple (intersection_matrix, all_edges_matrix, final_edges_matrix).
        """
        pass

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
