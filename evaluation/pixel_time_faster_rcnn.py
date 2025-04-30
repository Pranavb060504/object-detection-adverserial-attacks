import fiftyone.zoo as foz
from PIL import Image
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from edge_attack_pixels_time import compute_pixels_time_edge_faster_rcnn
from dpattack_pixels_time import compute_pixels_time_dpattack_faster_rcnn
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run faster-rcnn model inference on COCO dataset")
parser.add_argument("--epsilon", type=float, default=0.05, help="Perturbation magnitude for FGSM (default: 0.05)")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for the attack (default: 10)")
parser.add_argument("--grid_size", type=int, default=3, help="Grid size for the attack (default: 3)")
args = parser.parse_args()


# Load Dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples = 1000,
    # Use the provided max_samples argument
    label_types=["detections"],
    )


# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image  # Move tensor to same device as model


# Perform inference
def detect_objects(image_path, model, threshold=0.5):
    image_tensor, image = load_image(image_path)
    image_tensor = image_tensor  # Move input tensor to same device as model
    
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



transform = transforms.Compose([
    transforms.ToTensor()
])

predictions = []

max_samples =100

num_pixels_time ={
    'dpattack': {},
    'edge': {}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

for sample in tqdm(dataset[:max_samples]):
    
    image_id = int(sample.filepath.split('\\')[-1].split('.')[0])
    
    num_pixel, t = compute_pixels_time_dpattack_faster_rcnn(sample.filepath, model, device, args.epsilon, args.iterations, args.grid_size) 
    
    num_pixels_time['dpattack'][image_id] = {
        'num_pixels': num_pixel,
        'time': t
    }
   
    
    num_pixel, t = compute_pixels_time_edge_faster_rcnn(sample.filepath, model, device, args.epsilon, args.iterations, args.grid_size)
    num_pixels_time['edge'][image_id] = {
        'num_pixels': num_pixel,
        'time': t
    }

# Save results to JSON files
with open(f'num_pixels_time_epsilon_{args.epsilon}_iterations_{args.iterations}_gridsize_3.json', 'w') as f:
    json.dump(num_pixels_time, f)
