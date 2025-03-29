@echo off
for %%E in (0.01 0.05 0.1) do (
    python fgsm_faster_rcnn_eval.py --max_samples 1000 --epsilon %%E
)
