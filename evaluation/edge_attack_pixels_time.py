import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.ops as ops
from PIL import Image
import cv2
from tqdm import tqdm
from time import time

def compute_pixels_time_edge(image_path, model, device, epsilon=1, iterations=100, target_confidence=0.25, confidence_threshold=0.5, nms_threshold=0.3):

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

    # Extract edges
    cv2_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(cv2_image, (5, 5), 1)
    edges = cv2.Canny(blurred, threshold1=200, threshold2=250)
    binary_edge_mask = torch.tensor((edges > 0).astype(np.uint8), device=device).repeat(3, 1, 1).unsqueeze(0)

    # Create binary mask M
    M = torch.zeros_like(org_tensor)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        M[..., y_min:y_max, x_min:x_max] = 1
    M *= binary_edge_mask
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