@echo off
for %%M in (yolov5s yolov8s yolov8x yolov9e yolo11n yolo11m yolo11x) do (
    python yolo_baseline_eval.py --model_name %%M --max_samples 1000
)