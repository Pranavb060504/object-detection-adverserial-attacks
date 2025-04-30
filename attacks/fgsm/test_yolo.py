# Written by: Ankit

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# image_id = '000000000662'
image_id = '000000000191'
org_path = f'/home/ankit/fiftyone/coco-2017/test/data/{image_id}.jpg'
org_image = cv2.imread(org_path)
org_image = cv2.resize(org_image, (640, 640))

adv_path = f'adv_image_{image_id}.jpg'
adv_image = cv2.imread(adv_path)
output = model([org_image, adv_image])

print('-'*100, '\nOriginal Image Detection Results')
print(output.pandas().xyxy[0])
print('-'*100, '\nAdversarial Image Detection Results')
print(output.pandas().xyxy[1])
print('-'*100)
output.show()