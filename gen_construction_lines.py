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




def train():
    # Initialize dataset
    dataset = cad2sketch_dataset_loader()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    graphs = []
    final_edges_mask = []

    # Load data
    for data in tqdm(dataloader, desc="Building Graphs"):
        pass

# ------------------------------------------------------------------------------# 
train()

