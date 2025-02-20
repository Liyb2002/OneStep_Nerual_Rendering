import sys
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataloader import cad2sketch_dataset_loader
from torch.utils.data import DataLoader

import helper
import os

import numpy as np

import cad2sketch_stroke_features


import json

dataset = cad2sketch_dataset_loader()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


data_path = os.path.join(os.getcwd(), 'annotated_input')  # Where input.json & gt_output.json are stored
formatted_data_path = os.path.join(os.getcwd(), 'dataset', 'formatted_data')  # Where training_dataset.json is saved

def collect_formatted_data(data_path, formatted_data_path):
    """
    Goes through all processed subfolders inside `annotated_input`, reads `input.json` & `gt_output.json`,
    formats them for DeepSeek fine-tuning, and saves a single `training_dataset.json`.
    """
    formatted_data = []

    # Recursively find all subfolders
    for root, dirs, files in os.walk(data_path):
        if "input.json" in files and "gt_output.json" in files:
            input_path = os.path.join(root, "input.json")
            output_path = os.path.join(root, "gt_output.json")

            with open(input_path, "r") as f:
                input_json = json.load(f)
            with open(output_path, "r") as f:
                output_json = json.load(f)

            
            formatted_data.append({
                "input": f"Given the stroke cloud data:\n{json.dumps(input_json, indent=2)}, what are the construction line relations?",
                "output": json.dumps(output_json, indent=2)
            })

    # Ensure the formatted_data_path exists
    os.makedirs(formatted_data_path, exist_ok=True)

    # Save the final dataset
    dataset_path = os.path.join(formatted_data_path, "training_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Final formatted dataset saved at: {dataset_path}")

# Run after all folders are processed
collect_formatted_data(data_path, formatted_data_path)
