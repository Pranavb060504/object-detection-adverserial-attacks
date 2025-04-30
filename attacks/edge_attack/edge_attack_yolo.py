import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_attack_yolo(image_path, epsilon=1, iterations=10, grid_size=3, target_confidence=0.25):

    model_train = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
    model_inf = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=True)
    
    model_train.eval()
    model_inf.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train.to(device)
    model_inf.to(device)

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

    # Extract edges
    cv2_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2_image = cv2.resize(cv2_image, (640, 640))
    blurred = cv2.GaussianBlur(cv2_image, (5, 5), 1)
    edges = cv2.Canny(blurred, threshold1=200, threshold2=250)
    binary_edge_mask = torch.tensor((edges > 0).astype(np.uint8)).repeat(3, 1, 1).unsqueeze(0)

    # Create binary mask M (grid-based within bounding boxes)
    M = torch.zeros_like(org_tensor)  # Initialize M as zero (no perturbation)
    for box in boxes:
        x_min, y_min, x_max, y_max, _, _ = map(int, box)
        M[..., y_min:y_max, x_min:x_max] = 1
    M *= binary_edge_mask
    M.requires_grad = False  # Mask is not trainable

    # Create trainable adversarial perturbation δ
    delta = torch.zeros_like(org_tensor, requires_grad=True)

    # Adversarial training loop
    for _ in tqdm(range(iterations)):
        
        # Apply masked perturbation
        patched_tensor = org_tensor * (1 - M) + delta * M

        # Forward pass through the model
        output = model_train(patched_tensor)[0]

        # Compute loss based on confidences
        confidence = output[0, :, 5:]
        loss = torch.max(torch.tensor(0.0), confidence - target_confidence).sum()
        if not torch.is_nonzero(loss):
            break

        # Backpropagation
        loss.backward()
        delta = delta - epsilon * delta.grad.sign()
        delta = delta.detach()
        delta = torch.clamp(delta, 0, 1)
        delta.requires_grad = True  

    # Apply the final optimized patch to the image
    adv_image_tensor = org_tensor * (1 - M) + delta * M
    adv_image = transforms.ToPILImage()(adv_image_tensor.squeeze(0))
    
    save_path = f'edge_attack_images/yolo/epsilon_{epsilon}_iterations_{iterations}_gridsize_{grid_size}/'
    os.makedirs(save_path, exist_ok=True)
    adv_image.save(os.path.join(save_path, image_path.split('/')[-1]))

if __name__ == '__main__':
    edge_attack_yolo('/home/ankit/fiftyone/coco-2017/test/data/000000000191.jpg')