""" 
Run the script using the following command:

    python faster_rcnn_eval.py  --max_samples "max_samples"

max_samples: any integer value, default is 1000
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
import argparse
import sys
import io

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run faster-rcnn model inference on COCO dataset")
parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to evaluate (default: 1000)")
args = parser.parse_args()

# Load Dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples = args.max_samples,
    # Use the provided max_samples argument
    label_types=["detections"],
    )

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True, require_grad=True)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

with open("instances_val2017.json") as f:
    ground_truth = json.load(f)
    
gt_categories = ground_truth['categories']

# COCO mapping from category name to category id
gt = defaultdict(list)
for category in gt_categories:
    gt[category['name']] = category['id']

# COCO mapping from category id to category name
gt_rev = {v: k for k, v in gt.items()}


predictions = []


# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

# Perform inference
def detect_objects(image_path, threshold=0.5):
    image_tensor, image = load_image(image_path)
    
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Extract boxes, labels, and scores
    boxes = predictions['boxes'].cpu().numpy()  # Convert to NumPy
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # Filter based on threshold
    valid_indices = scores > threshold
    filtered_boxes = boxes[valid_indices]
    filtered_labels = labels[valid_indices]
    filtered_scores = scores[valid_indices]
    
    return image, filtered_boxes, filtered_labels, filtered_scores


for sample in tqdm(dataset):
    
    # Get image ID from the sample filepath
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    image_path = sample.filepath
    image, boxes, labels, scores = detect_objects(image_path)
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
        
# Save predictions
with open(f"predictions_faster_rcnn.json", "w") as f:
    json.dump(predictions, f, indent=4)       
    
    
# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"predictions_faster_rcnn.json")
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

# Save the captured output to a file
with open(f"evaluation/baseline/evaluation_results_faster_rcnn.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_faster_rcnn.txt")
 