""" 
Run the script using the following command:

    python dpattack_yolo_eval.py  --max_samples "max_samples" --epsilon "epsilon" --gridsize "gridsize" --iterations "iterations"
    
where:
max_samples: any integer value, default is 1000
epsilon: any float value, default is 1
gridsize: any float value, default is 3
iterations: any integer value, default is 10

This script performs a DP attack on images from the COCO dataset using a YOLO model. It saves the adversarial images and evaluates the model's performance on these images, comparing it to the original images. The results are saved in a specified directory.
# -*- coding: utf-8 -*-
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import fiftyone.zoo as foz
from tqdm import tqdm
import os
import argparse
import sys
import io
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import cv2
from ultralytics import YOLO
import fiftyone as fo

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run yolov5s model inference on COCO dataset")
parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to evaluate (default: 1000)")
parser.add_argument("--epsilon", type=float, default=1, help="Epsilon value for dpattack (default: 1)")
parser.add_argument("--grid_size", type=int, default=3, help="gridsize value for dpattack (default: 3)")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for dpattack (default: 10)")
args = parser.parse_args()


epsilon = args.epsilon
gridsize = args.grid_size
iterations = args.iterations

# Load Dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples = args.max_samples,
    # Use the provided max_samples argument
    label_types=["detections"],
    )

with open("instances_val2017.json") as f:
    ground_truth = json.load(f)
    
gt_categories = ground_truth['categories']

# COCO mapping from category name to category id
gt = defaultdict(list)
for category in gt_categories:
    gt[category['name']] = category['id']

# COCO mapping from category id to category name
gt_rev = {v: k for k, v in gt.items()}

# YOLO mapping from class index to class name
yolo_classes = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 
    8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model_train = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
model_inf = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=True)
model = YOLO('yolov5s.pt')  # Load the YOLO model


model_train.eval()
model_inf.eval()
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_train.to(device)
model_inf.to(device)
model.to(device)

def dpattack_yolo(model_train, model_inf, device, image_path, image_id, epsilon=1, iterations=10, grid_size=3, target_confidence=0.25):


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
    
    adv_image.save(f'dpattack_images/{name}/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/adv_image_{image_id}.jpg')




name = 'yolov5s'

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLO input size
    transforms.ToTensor()
])

os.makedirs(f'dpattack_images/{name}/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/', exist_ok=True)

predictions = []

for sample in dataset[:100]:
    
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    
    if not os.path.exists(f'dpattack_images/{name}/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/adv_image_{image_id}.jpg'):
        
        image_path = sample.filepath
        dpattack_yolo(model_train, model_inf, device, image_path, image_id, epsilon=epsilon, iterations=iterations, grid_size=gridsize)
        
    image = Image.open(f'dpattack_images/{name}/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/adv_image_{image_id}.jpg')

    # Perform inference
    results = model.predict(image, imgsz=640)  # Specify the input size if needed
    detections = results[0].boxes  # Extract bounding boxes

    # Convert detections to the required format
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy[0].tolist()  # Bounding box coordinates
        conf = detection.conf[0].item()  # Confidence score
        cls = int(detection.cls[0].item())  # Class index
        width, height = x2 - x1, y2 - y1
        predictions.append({
            "image_id": image_id,
            "category_id": gt[yolo_classes[cls]],
            "bbox": [float(x1), float(y1), float(width), float(height)],
            "score": float(conf)
        })

os.makedirs(f'predictions/dpattack/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/', exist_ok=True)
# Save predictions
with open(f"predictions/dpattack/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/predictions_{name}.json", "w") as f:
    json.dump(predictions, f, indent=4)

# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"predictions/dpattack/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/predictions_{name}.json")
imgIds=sorted(coco_gt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# Run COCO evaluation
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.params.imgIds  = imgIds
coco_eval.evaluate()  # Ensure evaluation runs
coco_eval.accumulate()

# Redirect stdout to capture the printed output
output_buffer = io.StringIO()
sys.stdout = output_buffer

# Run COCO evaluation and print results
coco_eval.summarize()

# Restore stdout
sys.stdout = sys.__stdout__
os.makedirs(f'evaluation/dpattack/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/', exist_ok=True)
# Save the captured output to a file
with open(f"evaluation/dpattack/epsilon_{epsilon}_iterations_{iterations}_gridsize_{gridsize}/evaluation_results_{name}.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_{name}.txt")

