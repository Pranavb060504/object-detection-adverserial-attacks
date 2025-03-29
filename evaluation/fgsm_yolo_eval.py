"""
Run this script using the following command:

    python fgsm_yolo_eval.py --model_name "name" --max_samples "max_samples" --epsilon "epsilon"

list of names:[
                    "yolov5s",
                    "yolov8s",
                    "yolov8x",
                    "yolov9e",
                    "yolov10b",
                    "yolov10x",
                    "yolo11n",
                    "yolo11m",
                    "yolo11x",
            ]
            
max_samples: any integer value, default is 1000
epsilon: any float value, default is 0.05
"""

# Import required libraries
from ultralytics import YOLO 
import fiftyone.zoo as foz
from collections import defaultdict
from PIL import Image
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import sys
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
import os


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run YOLO model inference on COCO dataset")
parser.add_argument("--model_name", type=str, required=True, help="Name of the YOLO model file (without .pt)")
parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to evaluate (default: 1000)")
parser.add_argument("--epsilon", type=float, default=0.05, help="Perturbation magnitude for FGSM (default: 0.05)")
args = parser.parse_args()


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

# Load the YOLO model
# Use the provided model name
name = args.model_name
model_inf = YOLO(f'{name}.pt')  # Use the YOLO model
model_inf.model.eval()  # Set the model in evaluation mode

model_train = YOLO(f'{name}.pt')  # Use the YOLO model for training
model_train.model.train()  # Set the model in training mode

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_train.model.to(device)
model_inf.model.to(device)

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLO input size
    transforms.ToTensor()
])

os.makedirs(f'fgsm_images/{name}/epsilon_{args.epsilon}/', exist_ok=True)

predictions = []

for sample in dataset:
    
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    
    if not os.path.exists(f'fgsm_images/{name}/epsilon_{args.epsilon}/adv_image_{image_id}.jpg'):
        
        image = Image.open(sample.filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        input_tensor.requires_grad = True  # Enable gradients

        # Perform inference using the raw PyTorch model
        output = model_train.model(input_tensor)  # Returns raw tensor outputs

        # Ensure the output requires gradients
        if isinstance(output, (list, tuple)):
            output = output[0]  # Extract first tensor if multiple

        output = output.requires_grad_()  # Ensure gradient tracking

        # Compute loss (use highest confidence detection as target)
        loss = output[..., 4].max()  # Objectness score (simplified attack)
        model_train.model.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients

        # Generate FGSM adversarial perturbation
        perturbation = args.epsilon * input_tensor.grad.sign()

        # Create adversarial image
        adv_image_tensor = input_tensor + perturbation
        adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)  # Keep pixel values valid

        # Convert adversarial image tensor to NumPy
        adv_image_np = adv_image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        adv_image_np = (adv_image_np * 255).astype(np.uint8)

        # Save adversarial image
        cv2.imwrite(f'fgsm_images/{name}/epsilon_{args.epsilon}/adv_image_{image_id}.jpg', cv2.cvtColor(adv_image_np, cv2.COLOR_RGB2BGR))
        
    image = Image.open(f'fgsm_images/{name}/epsilon_{args.epsilon}/adv_image_{image_id}.jpg')

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

os.makedirs(f'fgsm_predictions/{name}/epsilon_{args.epsilon}/', exist_ok=True)
# Save predictions
with open(f"fgsm_predictions/{name}/epsilon_{args.epsilon}/fgsm_predictions_{name}_{args.epsilon}.json", "w") as f:
    json.dump(predictions, f, indent=4)


# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"fgsm_predictions/{name}/epsilon_{args.epsilon}/fgsm_predictions_{name}_{args.epsilon}.json")
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
os.makedirs(f'evaluation/fgsm/{name}/epsilon_{args.epsilon}/', exist_ok=True)
# Save the captured output to a file
with open(f"evaluation/fgsm/{name}/epsilon_{args.epsilon}/evaluation_results_{name}.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_{name}.txt")
