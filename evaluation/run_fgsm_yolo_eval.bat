@echo off
for %%M in (yolov5s yolov8s yolov8x yolov9e yolo11n yolo11m yolo11x) do (
    python fgsm_yolo_eval.py --model_name %%M --max_samples 1000 --epsilon 0.05
)
