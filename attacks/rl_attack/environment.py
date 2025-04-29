
# Import required libraries
from ultralytics import YOLO 
from PIL import Image
import numpy as np
import torch
import cv2
import os
from attacks import attack_fn_map


img_dim = 640  # Image dimension for YOLO model
patch_dim = 64


patch_index = [i for i in range((img_dim // patch_dim)**2)]
attack_index = [i for i in range(len(attack_fn_map))]

# action space is all possible (patches, attack) tuples
action_space = [(i, j) for i in patch_index for j in attack_index]



def reward(state, next_state, model):
    
    """
    _params_:
    state: current state of the environment i.e image
    action: action taken by the agent i.e a tuple (patch, perturbation)
    next_state: next state of the environment i.e image after applying the action i.e select patch and apply perturbation
    """

    state = state.unsqueeze(0)  # Add batch dimension
    # Perform inference using the raw PyTorch model
    output = model(state)  # Returns raw tensor outputs

    output = output[0] 

    # Compute loss (use highest confidence detection as target)
    reward = 1 - output[..., 4].max()  # Objectness score (simplified attack)
    
    rmse = torch.sqrt(torch.mean((state.cpu() - next_state) ** 2))  # Calculate RMSE between original and adversarial image
    reward = reward - rmse  # Adjust reward based on RMSE
    
    return reward

