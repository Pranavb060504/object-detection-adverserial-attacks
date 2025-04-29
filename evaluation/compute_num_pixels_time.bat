@echo off
for %%E in (1  0.3) do (
    for %%I in (10 20) do (
        python pixel_time_faster_rcnn.py --epsilon %%E --grid_size 3 --iterations %%I
    )
)