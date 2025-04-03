import torch
import torchvision.transforms as transforms
from ultralytics import YOLO 
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def dpattack_yolo(image_path, model_name, epsilon=1, iterations=10, grid_size=3, target_confidence=0.25):

    model_inf = YOLO(f'{model_name}.pt')  # Use the YOLO model
    model_inf.model.eval()  # Set the model in evaluation mode

    model_train = YOLO(f'{model_name}.pt')  # Use the YOLO model for training
    model_train.model.train()  # Set the model in training mode

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train.model.to(device)
    model_inf.model.to(device)

    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to YOLO input size
        transforms.ToTensor()
    ])
    org_image = Image.open(image_path).convert('RGB')
    org_tensor = transform(org_image).unsqueeze(0).to(device)
    org_tensor.requires_grad = True

    # Forward pass to detect objects
    with torch.no_grad():
        output = model_inf.predict(org_tensor)

    # Extract bounding boxes for detected objects
    boxes = output[0].boxes.xyxy
    scores = output[0].boxes.conf
    print(boxes)
    exit()

    # Create binary mask M (grid-based within bounding boxes)
    M = torch.zeros_like(org_tensor)  # Initialize M as zero (no perturbation)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        y_len = y_max - y_min
        x_len = x_max - x_min
        M[:, :, y_min:y_max:y_len//(grid_size+1), x_min:x_max] = 1
        M[:, :, y_min:y_max, x_min:x_max:x_len//(grid_size+1)] = 1
    M.requires_grad = False  # Mask is not trainable

    # Create trainable adversarial perturbation δ
    delta = torch.zeros_like(org_tensor, requires_grad=True)
    # output = model_inf.model(org_tensor)[0]
    # print(output)
    # return

    # Adversarial training loop
    for _ in tqdm(range(iterations)):
        
        # Apply masked perturbation
        patched_tensor = org_tensor * (1 - M) + delta * M

        # Forward pass through the model
        output = model_train.model(patched_tensor)

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

    # Apply the final optimized patch to the image
    adv_image_tensor = org_tensor * (1 - M) + delta * M
    adv_image = transforms.ToPILImage()(adv_image_tensor.squeeze(0))
    
    save_path = f'dpattack_images/faster_rcnn/epsilon_{epsilon}_iterations_{iterations}_gridsize_{grid_size}/'
    os.makedirs(save_path, exist_ok=True)
    adv_image.save(os.path.join(save_path, image_path.split('/')[-1]))

# dpattack_yolo('/home/ankit/fiftyone/coco-2017/test/data/000000000662.jpg', 'yolov9e')

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (grayscale)
image = cv2.imread('/home/ankit/fiftyone/coco-2017/test/data/000000000662.jpg', cv2.IMREAD_GRAYSCALE)
org_image = Image.open('/home/ankit/fiftyone/coco-2017/test/data/000000000662.jpg').convert('RGB')
print(np.array(org_image).shape)
exit()
# Apply Gaussian Blur (optional but recommended)
blurred = cv2.GaussianBlur(image, (5, 5), 1)

# Apply Canny edge detector
edges = cv2.Canny(blurred, threshold1=200, threshold2=250)

# `edges` is a binary mask already (0: non-edge, 255: edge), convert to 0/1
binary_edge_mask = (edges > 0).astype(np.uint8)

boxes = [[ 31.4337,  32.0351, 630.1816, 436.4275],
        [155.7590,  40.1051, 209.9881,  74.0234],
        [447.6165,  28.3950, 508.9240,  64.3876]]
M = np.zeros_like(image)
for box in boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    M[..., y_min:y_max, x_min:x_max] = 1

# Visualize
plt.imshow(M*binary_edge_mask, cmap='gray')
plt.show()


