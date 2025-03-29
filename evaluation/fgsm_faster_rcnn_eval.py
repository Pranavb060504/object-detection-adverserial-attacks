"""
Run this script using the following command:

    python fgsm_faster_rcnn_eval.py  --max_samples "max_samples" --epsilon "epsilon"
            
max_samples: any integer value, default is 1000
epsilon: any float value, default is 0.05
"""

# Import required libraries
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
import torch
import torchvision.transforms as transforms
import os
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run faster-rcnn model inference on COCO dataset")
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

# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device), image  # Move tensor to same device as model


# Perform inference
def detect_objects(image_path, model, threshold=0.5):
    image_tensor, image = load_image(image_path)
    image_tensor = image_tensor.to(device)  # Move input tensor to same device as model
    
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Extract boxes, labels, and scores
    boxes = predictions['boxes'].cpu().numpy()  # Move to CPU before converting to NumPy
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # Filter based on threshold
    valid_indices = scores > threshold
    filtered_boxes = boxes[valid_indices]
    filtered_labels = labels[valid_indices]
    filtered_scores = scores[valid_indices]
    
    return image, filtered_boxes, filtered_labels, filtered_scores

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


model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])

os.makedirs(f'fgsm_images/faster_rcnn/epsilon_{args.epsilon}/', exist_ok=True)

predictions = []

for sample in dataset:
    
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    
    if not os.path.exists(f'fgsm_images/faster_rcnn/epsilon_{args.epsilon}/adv_image_{image_id}.jpg'):
        
        image_path = sample.filepath

        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        input_tensor.requires_grad = True  # Enable gradients

        # Perform inference
        output = model(input_tensor)  # Get detections

        # Compute loss (use the highest confidence detection as target)
        loss = -torch.sum(output[0]['scores'])
        loss.requires_grad_(True)  # Ensure it can be backpropagated
        loss.backward()  # Backpropagate to get gradients

        # Generate FGSM adversarial perturbation  
        perturbation = args.epsilon * input_tensor.grad.sign()

        # Create adversarial image
        adv_image_tensor = input_tensor + perturbation
        adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)  # Keep pixel values valid
        
        adv_image_np = adv_image_tensor.cpu().squeeze().permute(1, 2, 0).detach().numpy()
        adv_image_np = (adv_image_np * 255).astype(np.uint8)

        cv2.imwrite(f'fgsm_images/faster_rcnn/epsilon_{args.epsilon}/adv_image_{image_id}.jpg', cv2.cvtColor(adv_image_np, cv2.COLOR_RGB2BGR))
    
    image, boxes, labels, scores = detect_objects(f'fgsm_images/faster_rcnn/epsilon_{args.epsilon}/adv_image_{image_id}.jpg', model)
    # Convert boxes to COCO format
    for i, bbox in enumerate(boxes):
        # Convert from (x1, y1, x2, y2) to (x, y, width, height)
        x1, y1, x2, y2 = bbox
        conf = scores[i]
        cls = labels[i]
        width, height = x2 - x1, y2 - y1
        # Convert class index to COCO category ID
        predictions.append({
            "image_id": image_id,
            "category_id": int(cls),
            "bbox": [float(x1), float(y1), float(width), float(height)],
            "score": float(conf)
        })

os.makedirs(f'predictions/fgsm/epsilon_{args.epsilon}/', exist_ok=True)
# Save predictions
with open(f"predictions/fgsm/epsilon_{args.epsilon}/predictions_faster_rcnn.json", "w") as f:
    json.dump(predictions, f, indent=4)


# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"predictions/fgsm/epsilon_{args.epsilon}/predictions_faster_rcnn.json")
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
os.makedirs(f'evaluation/fgsm/epsilon_{args.epsilon}/', exist_ok=True)
# Save the captured output to a file
with open(f"evaluation/fgsm/epsilon_{args.epsilon}/evaluation_results_faster_rcnn.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_faster_rcnn.txt")
