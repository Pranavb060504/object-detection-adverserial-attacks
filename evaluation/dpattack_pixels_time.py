# Written by: Ankit

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.ops as ops
from PIL import Image
import cv2
from tqdm import tqdm
from time import time

def compute_pixels_time_dpattack_faster_rcnn(image_path, model, device, epsilon=1, iterations=100, grid_size=10, target_confidence=0.25, confidence_threshold=0.5, nms_threshold=0.3):

    # Load and preprocess the image
    transform = transforms.Compose([transforms.ToTensor()])
    org_image = Image.open(image_path).convert('RGB')
    org_tensor = transform(org_image).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]

    # Forward pass to detect objects
    with torch.no_grad():
        output = model(org_tensor)

    # Extract bounding boxes for detected objects
    boxes = output[0]['boxes'].detach()
    scores = output[0]['scores'].detach()
    keep = scores > confidence_threshold
    boxes, scores = boxes[keep], scores[keep]
    keep = ops.nms(boxes, scores, nms_threshold)
    boxes, scores = boxes[keep], scores[keep]

    # Create binary mask M (grid-based within bounding boxes)
    M = torch.zeros_like(org_tensor)  # Initialize M as zero (no perturbation)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        y_len = y_max - y_min
        x_len = x_max - x_min
        
        try:
            M[:, :, y_min:y_max:y_len//(grid_size+1), x_min:x_max] = 1
            M[:, :, y_min:y_max, x_min:x_max:x_len//(grid_size+1)] = 1

        except:
            M[:, :, y_min:y_max:y_len//2, x_min:x_max] = 1
            M[:, :, y_min:y_max, x_min:x_max:x_len//2] = 1
            
    M.requires_grad = False  # Mask is not trainable


    # Create trainable adversarial perturbation δ
    delta = torch.zeros_like(org_tensor, requires_grad=True)

    # Adversarial training loop
    start_time = time()
    for _ in range(iterations):
        
        # Apply masked perturbation
        patched_tensor = org_tensor * (1 - M) + delta * M

        # Forward pass through the model
        output = model(patched_tensor)
        if len(output[0]['boxes']) == 0:
            break

        # Compute loss based on confidences
        loss = 0
        for i in range(len(output[0]['scores'])):
            confidence = output[0]['scores'][i]
            loss += torch.max(torch.tensor(0.0), confidence - target_confidence)
        
        # Backpropagation
        loss.backward()
        delta = delta - epsilon * delta.grad.sign()
        delta = delta.detach()
        delta = torch.clamp(delta, 0, 1)
        delta.requires_grad = True  
    end_time = time()

    # Apply the final optimized patch to the image
    adv_image_tensor = org_tensor * (1 - M) + delta * M
    adv_image = transforms.ToPILImage()(adv_image_tensor.squeeze(0))
    
    return np.count_nonzero(M[0, 0].cpu().numpy()), end_time - start_time

def compute_pixels_time_dpattack_yolo(model_train, model_inf, device, image_path, image_id, epsilon=1, iterations=10, grid_size=3, target_confidence=0.25):


    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to YOLO input size
        transforms.ToTensor()
    ])
    org_image = Image.open(image_path).convert('RGB')
    org_tensor = transform(org_image).unsqueeze(0).to(device)
    org_tensor.requires_grad = True

    # Forward pass to detect objects
    with torch.no_grad():
        output = model_inf([org_image.resize((640, 640))])

    # Extract bounding boxes for detected objects
    boxes = output.xyxy[0]

    # Create binary mask M (grid-based within bounding boxes)
    M = torch.zeros_like(org_tensor)  # Initialize M as zero (no perturbation)
    for box in boxes:
        x_min, y_min, x_max, y_max, _, _ = map(int, box)
        y_len = y_max - y_min
        x_len = x_max - x_min
        try:
            M[:, :, y_min:y_max:y_len//(grid_size+1), x_min:x_max] = 1
            M[:, :, y_min:y_max, x_min:x_max:x_len//(grid_size+1)] = 1
        except:
            M[:, :, y_min:y_max:y_len//2, x_min:x_max] = 1
            M[:, :, y_min:y_max, x_min:x_max:x_len//2] = 1
    M.requires_grad = False  # Mask is not trainable

    # Create trainable adversarial perturbation δ
    delta = torch.zeros_like(org_tensor, requires_grad=True)
    
    start_time = time()
    # Adversarial training loop
    for _ in tqdm(range(iterations)):
        
        # Apply masked perturbation
        patched_tensor = org_tensor * (1 - M) + delta * M

        # Forward pass through the model
        output = model_train(patched_tensor)[0]

        # Compute loss based on confidences
        confidence = output[0, :, 5:]
        loss = torch.max(torch.tensor(0.0), confidence - target_confidence).sum()

        # Backpropagation
        loss.backward()
        delta = delta - epsilon * delta.grad.sign()
        delta = delta.detach()
        delta = torch.clamp(delta, 0, 1)
        delta.requires_grad = True  
        
    end_time = time()
    
    # Apply the final optimized patch to the image
    adv_image_tensor = org_tensor * (1 - M) + delta * M
    adv_image = transforms.ToPILImage()(adv_image_tensor.squeeze(0))
    
    return np.count_nonzero(M[0, 0].cpu().numpy()), end_time - start_time