import torch
import torchvision.transforms as transforms
import torchvision.ops as ops
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Load the pretrained YOLO11 model
model = YOLO('yolov5s.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
model.eval()

# Define attack parameters
EPSILON = 1    # Perturbation step size
ITERATIONS = 10  # Number of attack iterations
TARGET_CONFIDENCE = 0.25  # Threshold to suppress detection
GRID_SIZE = 10

# Load and preprocess the image
image_id = '000000000191'
org_path = f'/home/ankit/fiftyone/coco-2017/test/data/{image_id}.jpg'
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
org_image = Image.open(org_path).convert('RGB')
org_tensor = transform(org_image).unsqueeze(0)  # Shape: [1, 3, H, W]

# Forward pass to detect objects
with torch.no_grad():
    # output = model(org_tensor, conf=0.5)[0]
    output = model(org_tensor)[0]
    # print(output.shape)
    # print(output[0][0, :, 0])
    # exit()

# Create binary mask M (grid-based within bounding boxes)
M = torch.zeros_like(org_tensor)  # Initialize M as zero (no perturbation)

for box in output.boxes:
    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].detach().numpy())
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
    # output = model(patched_tensor, conf=0.5)[0]
    output = model.model(patched_tensor)[0]

    # Compute loss based on confidences
    loss = torch.sum(torch.max(torch.tensor(0.0), output - TARGET_CONFIDENCE))
    # loss = 0
    # for i in range(output.shape[2]):
    #     for c in range(output.shape[1]):
    #         confidence = output[0, c, i]
    #         loss += torch.max(torch.tensor(0.0), confidence - TARGET_CONFIDENCE)

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
adv_image.save(f'yolo_adv_image_{image_id}.jpg')

with open('yolo_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)

plt.plot(losses)
plt.grid()
plt.show()