""" 
Run the script using the following command:

    python pgdattack_faster_rcnn_eval.py  --max_samples "max_samples" --epsilon "epsilon" --alpha "alpha" --steps "steps"
    
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run faster-rcnn model inference on COCO dataset")
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
model_train = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT').to(device).train()  


# Load model
model_inf = fasterrcnn_resnet50_fpn(pretrained=True)
model_inf.eval()
model_inf.to(device)  # Move model to GPU/CPU


# Transformation
transform = transforms.Compose([transforms.ToTensor()])


def pgd_attack(image_path, image_id, epsilon, alpha, steps):
    
    """
    Perform PGD attack on the input image.
    """
    
    img = Image.open(image_path).convert("RGB")
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

        losses = model_train(x_adv, targets)
        loss = sum(losses.values())
        
        loss.backward()
        
        with torch.no_grad():
            x_adv += alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)

            x_adv = torch.clamp(x_adv, 0, 1)
        
            x_adv.requires_grad_(True)
    # Save the adversarial image
    x_adv = x_adv.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    x_adv = (x_adv * 255).astype("uint8")
    Image.fromarray(x_adv).save(f"pgdattack_images/faster_rcnn/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/adv_image_{image_id}.jpg")
    
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

predictions = []
os.makedirs(f'pgdattack_images/faster_rcnn/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/', exist_ok=True)


for sample in tqdm(dataset):
    
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    
    if not os.path.exists(f'pgdattack_images/faster_rcnn/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/adv_image_{image_id}.jpg'):
        
        image_path = sample.filepath
        pgd_attack(image_path, image_id, epsilon, alpha, steps)
        # Perform inference on the original image

    # Perform inference on the adversarial image
    image, boxes, labels, scores = detect_objects(f'pgdattack_images/faster_rcnn/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/adv_image_{image_id}.jpg', model_inf)

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


os.makedirs(f'predictions/pgdattack/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/', exist_ok=True)

# Save predictions
with open(f"predictions/pgdattack/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/predictions_faster_rcnn.json", "w") as f:
    json.dump(predictions, f, indent=4)


# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"predictions/pgdattack/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/predictions_faster_rcnn.json")
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
os.makedirs(f'evaluation/pgdattack/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/', exist_ok=True)
# Save the captured output to a file
with open(f"evaluation/pgdattack/epsilon_{epsilon}_alpha_{alpha}_iterations_{steps}/evaluation_results_faster_rcnn.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_faster_rcnn.txt")