"""
Run this script using the following command:

    python eval.py --model_name "name"

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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run YOLO model inference on COCO dataset")
parser.add_argument("--model_name", type=str, required=True, help="Name of the YOLO model file (without .pt)")
args = parser.parse_args()


# Load Dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples = 1000,
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
model = YOLO(f'{name}.pt')  # Use the YOLO model

predictions = []

for sample in dataset:
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    image = Image.open(sample.filepath)

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

# Save predictions
with open(f"predictions_{name}.json", "w") as f:
    json.dump(predictions, f, indent=4)


# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"predictions_{name}.json")
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
with open(f"evaluation_results_{name}.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_{name}.txt")
