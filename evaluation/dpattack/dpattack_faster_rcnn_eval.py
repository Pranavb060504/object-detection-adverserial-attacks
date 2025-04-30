import fiftyone.zoo as foz
from PIL import Image
import numpy as np
import json
import os
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
from collections import defaultdict
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import io
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Run dpattack on Faster R-CNN model")
parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples")
parser.add_argument("--epsilon", type=float, default=1, help="Perturbation magnitude for FGSM")
parser.add_argument("--grid_size", type=int, default=3, help="Grid size for perturbation")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
args = parser.parse_args()


# Load dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=args.max_samples,
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

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU/CPU


# Transformation
transform = transforms.Compose([transforms.ToTensor()])

# Attack parameters
epsilon = args.epsilon
iterations = args.iterations
grid_size = args.grid_size

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

def dpattack_faster_rcnn(image_path, image_id, epsilon=1, iterations=10, grid_size=3, target_confidence=0.25, confidence_threshold=0.5, nms_threshold=0.3):

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load and preprocess the image
    transform = transforms.Compose([transforms.ToTensor()])
    org_image = Image.open(image_path).convert('RGB')
    org_tensor = transform(org_image).unsqueeze(0)  # Shape: [1, 3, H, W]

    # Forward pass to detect objects
    with torch.no_grad():
        output = model(org_tensor)

    # Extract bounding boxes for detected objects
    boxes = output[0]['boxes'].detach()
    scores = output[0]['scores'].detach()
    keep = scores > confidence_threshold
    boxes, scores = boxes[keep], scores[keep]
    keep = nms(boxes, scores, nms_threshold)
    boxes, scores = boxes[keep], scores[keep]

    # Create binary mask M (grid-based within bounding boxes)
    M = torch.zeros_like(org_tensor)  # Initialize M as zero (no perturbation)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
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
        output = model(patched_tensor)
        if len(output[0]['boxes']) == 0:
            break

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
    
    adv_image.save(f'dpattack_images/faster_rcnn/epsilon_{epsilon}_iterations_{iterations}_gridsize_{grid_size}/adv_image_{image_id}.jpg')



predictions = []
os.makedirs(f'dpattack_images/faster_rcnn/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/', exist_ok=True)

max_samples = 100

for sample in dataset[:max_samples]:
    
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    
    if not os.path.exists(f'dpattack_images/faster_rcnn/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/adv_image_{image_id}.jpg'):
        
        image_path = sample.filepath
        dpattack_faster_rcnn(image_path, epsilon=args.epsilon, iterations=args.iterations, grid_size=args.grid_size, image_id=image_id)

    # Perform inference on the adversarial image
    image, boxes, labels, scores = detect_objects(f'dpattack_images/faster_rcnn/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/adv_image_{image_id}.jpg', model)

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


os.makedirs(f'predictions/dpattack/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/', exist_ok=True)

# Save predictions
with open(f"predictions/dpattack/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/predictions_faster_rcnn.json", "w") as f:
    json.dump(predictions, f, indent=4)


# Load ground truth
coco_gt = COCO("instances_val2017.json")

# Load predictions
coco_dt = coco_gt.loadRes(f"predictions/dpattack/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/predictions_faster_rcnn.json")
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
os.makedirs(f'evaluation/dpattack/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/', exist_ok=True)
# Save the captured output to a file
with open(f"evaluation/dpattack/epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_{args.grid_size}/evaluation_results_faster_rcnn.txt", "w") as f:
    f.write(output_buffer.getvalue())

# Optionally, print a message to confirm
print(f"Evaluation results saved to evaluation_results_faster_rcnn.txt")
