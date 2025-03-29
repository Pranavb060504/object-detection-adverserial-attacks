import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.ops as ops
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Load the pretrained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define attack parameters
EPSILON = 1    # Perturbation step size
ITERATIONS = 100  # Number of attack iterations
TARGET_CONFIDENCE = 0.25  # Threshold to suppress detection
GRID_SIZE = 10

# Load and preprocess the image
image_id = '000000000191'
org_path = f'/home/ankit/fiftyone/coco-2017/test/data/{image_id}.jpg'
transform = transforms.Compose([transforms.ToTensor()])
org_image = Image.open(org_path).convert('RGB')
org_tensor = transform(org_image).unsqueeze(0)  # Shape: [1, 3, H, W]

# Forward pass to detect objects
with torch.no_grad():
    output = model(org_tensor)

# Extract bounding boxes for detected objects
boxes = output[0]['boxes'].detach()
scores = output[0]['scores'].detach()
keep = scores > 0.5
boxes, scores = boxes[keep], scores[keep]
keep = ops.nms(boxes, scores, 0.3)
boxes, scores = boxes[keep], scores[keep]

# Create binary mask M (grid-based within bounding boxes)
M = torch.zeros_like(org_tensor)  # Initialize M as zero (no perturbation)

for box in boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    y_len = y_max - y_min
    x_len = x_max - x_min
    # M[:, :, y_min:y_max:5, x_min:x_max:5] = 1  # Grid-based perturbation
    M[:, :, y_min:y_max:y_len//(GRID_SIZE+1), x_min:x_max] = 1
    M[:, :, y_min:y_max, x_min:x_max:x_len//(GRID_SIZE+1)] = 1

M.requires_grad = False  # Mask is not trainable

# Create trainable adversarial perturbation δ
delta = torch.zeros_like(org_tensor, requires_grad=True)

# Optimizer for updating δ
# optimizer = torch.optim.Adam([delta], lr=0.01)
losses = []
# Adversarial training loop
for _ in tqdm(range(ITERATIONS)):
    # optimizer.zero_grad()
    
    # Apply masked perturbation
    patched_tensor = org_tensor * (1 - M) + delta * M

    # Forward pass through the model
    output = model(patched_tensor)

    # Compute loss based on confidences
    loss = 0
    for i in range(len(output[0]['scores'])):
        # for c in range(len(output[0]['labels'])):
        confidence = output[0]['scores'][i]
        loss += torch.max(torch.tensor(0.0), confidence - TARGET_CONFIDENCE)
    
    # Backpropagation
    loss.backward()
    # optimizer.step()
    delta = delta - EPSILON * delta.grad.sign()
    delta = delta.detach()
    delta = torch.clamp(delta, 0, 1)
    delta.requires_grad = True  
    losses.append(loss.item())
    # Clamp δ values to ensure pixel range is valid
    # delta.data = torch.clamp(delta.data, -EPSILON, EPSILON)

# Apply the final optimized patch to the image
adv_image_tensor = org_tensor * (1 - M) + delta * M
adv_image = transforms.ToPILImage()(adv_image_tensor.squeeze(0))
adv_image.save(f'adv_image_{image_id}.jpg')

with open('losses.pkl', 'wb') as f:
    pickle.dump(losses, f)

plt.plot(losses)
plt.grid()
plt.show()