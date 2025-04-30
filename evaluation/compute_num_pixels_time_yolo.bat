@echo off
for %%E in (1  0.3) do (
    for %%I in (10 20) do (
        python pixel_time_yolov5s.py --epsilon %%E --grid_size 3 --iterations %%I
    )
)