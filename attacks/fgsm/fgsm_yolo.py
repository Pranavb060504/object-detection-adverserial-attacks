import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
model.eval()

# image_id = '000000000662'
image_id = '000000000191'
image_path = f'/home/ankit/fiftyone/coco-2017/test/data/{image_id}.jpg'

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLOv5 input size
    transforms.ToTensor()
])
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
input_tensor.requires_grad = True  # Enable gradients

# Perform inference
output = model(input_tensor)  # Get detections
output = output[0]  # Extract the first output tensor

# Compute loss (use the highest confidence detection as target)
loss = output[..., 4].max()  # Objectness score (simplified attack)
model.zero_grad()
loss.backward()  # Compute gradients

# Generate FGSM adversarial perturbation
epsilon = 0.05  # Small perturbation magnitude
perturbation = epsilon * input_tensor.grad.sign()

# Create adversarial image
adv_image_tensor = input_tensor + perturbation
adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)  # Keep pixel values valid

# Convert adversarial image tensor to NumPy
adv_image_np = adv_image_tensor.squeeze().permute(1, 2, 0).detach().numpy()
adv_image_np = (adv_image_np * 255).astype(np.uint8)
cv2.imwrite(f'adv_image_{image_id}.jpg', cv2.cvtColor(adv_image_np, cv2.COLOR_RGB2BGR))
