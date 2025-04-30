import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

def dpattack_yolo(image_path, epsilon=1, iterations=10, grid_size=3, target_confidence=0.25):

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

    # Apply the final optimized patch to the image
    adv_image_tensor = org_tensor * (1 - M) + delta * M
    adv_image = transforms.ToPILImage()(adv_image_tensor.squeeze(0))
    
    save_path = f'dpattack_images/yolov5s/epsilon_{epsilon}_iterations_{iterations}_gridsize_{grid_size}/'
    os.makedirs(save_path, exist_ok=True)
    adv_image.save(os.path.join(save_path, image_path.split('/')[-1]))

if __name__ == '__main__':
    dpattack_yolo('/home/ankit/fiftyone/coco-2017/test/data/000000000191.jpg')