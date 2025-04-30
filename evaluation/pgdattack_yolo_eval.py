""" 
Run the script using the following command:

    python pgdattack_yolo_eval.py  --max_samples "max_samples" --epsilon "epsilon" --alpha "alpha" --steps "steps"
    
where:
max_samples: any integer value, default is 1000
epsilon: any float value, default is 0.031
alpha: any float value, default is 0.007843
steps: any integer value, default is 10

This script performs a PGD attack on images from the COCO dataset using a Faster R-CNN model. It saves the adversarial images and evaluates the model's performance on these images, comparing it to the original images. The results are saved in a specified directory.
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
parser.add_argument("--epsilon", type=float, default=0.031, help="Epsilon value for PGD attack (default: 0.031)")
parser.add_argument("--alpha", type=float, default=0.007843, help="Alpha value for PGD attack (default: 0.007843)")
parser.add_argument("--steps", type=int, default=10, help="Number of steps for PGD attack (default: 10)")
args = parser.parse_args()


epsilon = args.epsilon
alpha = args.alpha
steps = args.steps

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
# Load a pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False).train()

import torch
import torchvision.transforms.functional as F
from PIL import Image
import os
from torchvision.ops import box_iou

def pgd_attack(model, image_path, image_id, eps=8/255, alpha=2/255, steps=10, device='cuda', name='yolov5s'):
    """
    Perform PGD attack on the image using the YOLOv5 model, minimizing detection confidence near target box.
    """

    model.eval()  # Inference mode for Ultralytics YOLOv5

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    x = F.to_tensor(img).unsqueeze(0).to(device)

    _, H, W = x.shape[1:]

    # Target box and label (can be modified)
    target_box = torch.tensor([[0., 0., float(W), float(H)]], device=device)  # Full image
    target_label = torch.tensor([1], device=device)

    # Initialize adversarial image
    x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)

    for _ in range(steps):
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        preds = model(x_adv)[0]  # YOLOv5 returns list of predictions per image
        loss = torch.tensor(0.0, device=device)

        if preds.shape[0] > 0:
            pred_boxes = preds[:, :4]   # [x1, y1, x2, y2]
            confidences = preds[:, 4]   # objectness scores

            # Compute IoU of each predicted box with target box
            ious = box_iou(pred_boxes, target_box)[:, 0]  # [N]
            loss = (confidences * ious).sum()  # weighted sum

        # Minimize detection confidence near target
        loss = -loss

        loss.backward()

        with torch.no_grad():
            x_adv += alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv.requires_grad_(True)

    # Save adversarial image
    x_adv_np = x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
    x_adv_np = (x_adv_np * 255).astype("uint8")

    save_dir = f"pgdattack_images/{name}/epsilon_{eps}_alpha_{alpha}_iterations_{steps}"
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(x_adv_np).save(f"{save_dir}/adv_image_{image_id}.jpg")


# Load the YOLO model
# Use the provided model name
name = 'yolov5s'
model_inf = YOLO(f'{name}.pt')  # Use the YOLO model
model_inf.model.eval()  # Set the model in evaluation mode

model_inf.model.to(device)

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLO input size
    transforms.ToTensor()
])

os.makedirs(f'pgdattack_images/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/', exist_ok=True)

predictions = []

for sample in dataset:
    
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    
    if not os.path.exists(f'pgdattack_images/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/adv_image_{image_id}.jpg'):
        
        img = Image.open(sample.filepath).convert("RGB")
        x = F.to_tensor(img).unsqueeze(0).to(device) 

        _, H, W = x.shape[1:]
        targets = [{
            "boxes": torch.tensor([[0., 0., W, H]], device=device),
            "labels": torch.tensor([1],   device=device)
        }]

        x_adv = x.detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)


        for i in range(steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            losses = model(x_adv, targets)
            loss = losses.values()[..., 4].max() 
            
            loss.backward()
            
            with torch.no_grad():
                x_adv += alpha * x_adv.grad.sign()
                x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)

                x_adv = torch.clamp(x_adv, 0, 1)
            
                x_adv.requires_grad_(True)
        # Save the adversarial image
        x_adv = x_adv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        x_adv = (x_adv * 255).astype("uint8")
        Image.fromarray(x_adv).save(f"pgdattack_images/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/adv_image_{image_id}.jpg")
        
    image = Image.open(f'pgdattack_images/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/adv_image_{image_id}.jpg')

    # Perform inference
    results = model_inf.predict(image, imgsz=640)  # Specify the input size if needed
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

os.makedirs(f'predictions/pgdattack/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/', exist_ok=True)
# Save predictions
with open(f"predictions/pgdattack/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/predictions_{name}.json", "w") as f:
    json.dump(predictions, f, indent=4)


# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"predictions/pgdattack/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/predictions_{name}.json")
imgIds=sorted(coco_gt.getImgIds())
imgIds=imgIds[0:1000]
imgId = imgIds[np.random.randint(1000)]

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
os.makedirs(f'evaluation/pgdattack/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/', exist_ok=True)
# Save the captured output to a file
with open(f"evaluation/pgdattack/{name}/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/evaluation_results_{name}.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_{name}.txt")

