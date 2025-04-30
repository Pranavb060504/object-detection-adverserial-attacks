@echo off
for %%E in (1  0.3) do (
    for %%I in (10 20) do (
        python dpattack_yolo_eval.py --max_samples 1000 --epsilon %%E --grid_size 3 --iterations %%I
    )
)