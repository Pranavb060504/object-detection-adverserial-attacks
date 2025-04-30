@echo off
setlocal enabledelayedexpansion

for %%E in (0.031 0.062) do (
    for %%A in (0.007843 0.015686) do (
        for %%I in (10 20) do (
            set "run=true"
            if "%%E"=="0.031" if not "%%A"=="0.007843" set "run=false"
            if "%%E"=="0.062" if not "%%A"=="0.015686" set "run=false"

            if "!run!"=="true" (
                echo Running: epsilon=%%E alpha=%%A steps=%%I
                python pgdattack_yolo_eval.py --max_samples 1000 --epsilon %%E --alpha %%A --steps %%I
            )
        )
    )
)
