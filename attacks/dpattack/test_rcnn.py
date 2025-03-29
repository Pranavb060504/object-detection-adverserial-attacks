import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.ops as ops
import cv2
from PIL import Image
from pprint import pprint

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# image_id = '000000000662'
image_id = '000000000191'
org_path = f'/home/ankit/fiftyone/coco-2017/test/data/{image_id}.jpg'
adv_path = f'adv_image_{image_id}.jpg'

transform = transforms.Compose([
    transforms.ToTensor()
])
org_image = Image.open(adv_path).convert('RGB')
org_tensor = transform(org_image).unsqueeze(0)

with torch.no_grad():
    # output = model([org_image, adv_image])
    output = model(org_tensor)
    # pprint(output[0], sort_dicts=False)
    print(len(output[0]['boxes']))

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet',
    'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def plot_detections(image, boxes, labels, scores):
    fig, ax = plt.subplots(1, figsize=(10, 6))

    # Convert PIL image to NumPy array
    image_np = np.array(image)
    ax.imshow(image_np)

    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]  # Ensure integers
        width, height = x_max - x_min, y_max - y_min

        # Get class name from COCO labels
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label] if label < len(COCO_INSTANCE_CATEGORY_NAMES) else f"Class {label}"

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f'{class_name}: {score:.2f}', color='red', fontsize=12, backgroundcolor='white')

    plt.show(block=True) 

boxes = output[0]['boxes']
scores = output[0]['scores']
labels = output[0]['labels']
keep = scores > 0.5
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
print(len(boxes))
keep = ops.nms(boxes, scores, 0.3)
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
print(len(boxes))
plot_detections(org_image, boxes, labels, scores)