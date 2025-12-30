#!/bin/bash

CONFIGS=(
    "--config"
    "configs/trainer.yaml"
    "--config"
    "configs/data.yaml"
)

if [ $# -gt 0 ]; then
    for arg in "$@"; do
        CONFIGS+=("--config")
        CONFIGS+=("$arg")
    done
fi

echo ${CONFIGS[@]}
python main.py test "${CONFIGS[@]}"