from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_MobileNetV1
from PIL import Image
from tog.attacks import *
import os
from models.frcnn import FRCNN
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils_tog import *

def frcnn_attack_tog(input_img_path_list, output_path_list, attack_type):
    print('Loading FRCNN model...')
    weights = 'model_weights/FRCNN.pth'
    detector = FRCNN().cuda(device=0).load(weights)
    print('FRCNN Model loaded successfully')

    print('Running attack...')
    for input_img_path, output_path in zip(input_img_path_list, output_path_list):
        if attack_type == 'untargeted':
            tog_untargeted_attack(detector, input_img_path, output_path)
        elif attack_type == 'vanishing':
            tog_vanishing_attack(detector, input_img_path, output_path)
        elif attack_type == 'fabrication':
            tog_fabrication_attack(detector, input_img_path, output_path)
        elif attack_type == 'mislabeling_ml':
            tog_mislabeling_attack_most_likely(detector, input_img_path, output_path)
        elif attack_type == 'mislabeling_ll':
            tog_mislabeling_attack_least_likely(detector, input_img_path, output_path)
        else:
            raise ValueError(f'Unknown attack type: {attack_type}')



def yolo_attack_tog(input_img_path_list, output_path_list, attack_type):
    print('Loading YOLOv3 model...')
    # K.clear_session()

    weights = 'model_weights/YOLOv3_MobileNetV1.h5'  # TODO: Change this path to the victim model's weights
    detector = YOLOv3_MobileNetV1(weights=weights)
    print('YOLOv3 Model loaded successfully')

    print('Running attack...')
    for input_img_path, output_path in zip(input_img_path_list, output_path_list):
        if attack_type == 'untargeted':
            tog_untargeted_attack(detector, input_img_path, output_path)
        elif attack_type == 'vanishing':
            tog_vanishing_attack(detector, input_img_path, output_path)
        elif attack_type == 'fabrication':
            tog_fabrication_attack(detector, input_img_path, output_path)
        elif attack_type == 'mislabeling_ml':
            tog_mislabeling_attack_most_likely(detector, input_img_path, output_path)
        elif attack_type == 'mislabeling_ll':
            tog_mislabeling_attack_least_likely(detector, input_img_path, output_path)
        else:
            raise ValueError(f'Unknown attack type: {attack_type}')


if __name__ == '__main__':
    # Example usage
    input_img_path_list = ['./assets/example_1.jpg']
    output_path_list = ['output_image_1.jpg']
    attack_type = 'vanishing'  # Choose from: 'untargeted', 'vanishing', 'fabrication', 'mislabeling_ml', 'mislabeling_ll'

    # Uncomment the following lines to run the attacks
    # yolo_attack_tog(input_img_path_list, output_path_list, attack_type)
    frcnn_attack_tog(input_img_path_list, output_path_list, attack_type)