@echo off
for %%E in (1  0.3) do (
    for %%I in (10 20 70) do (
        python edge_attack_faster_rcnn_eval.py --max_samples 1000 --epsilon %%E --grid_size 3 --iterations %%I
    )
)