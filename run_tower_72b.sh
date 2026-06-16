#!/bin/bash
# Script to run Tower-Plus-72B GGUF model with llama.cpp using GPU offload
# P40 (CUDA1) as main GPU for model layers, RTX 3050 (CUDA0) for KV cache split
# Usage: ./run_tower_72b.sh [model_path]
# If model_path not provided, looks for .gguf file in current directory

# Set default model path (relative to script directory)
MODEL_PATH="${1:-./tower-plus-72b.gguf}"

# If no argument and default not found, search for any .gguf file in current directory
if [[ ! -f "$MODEL_PATH" && "$1" == "" ]]; then
    SHUFFLE=$(find . -maxdepth 1 -name "*.gguf" -type f | head -n 1)
    if [[ -n "$SHUFFLE" ]]; then
        MODEL_PATH="$SHUFFLE"
    fi
fi

# Check if model file exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found at '$MODEL_PATH'"
    echo "Please provide the path to the Tower-Plus-72B GGUF model as an argument."
    echo "Example: ./run_tower_72b.sh /path/to/model.gguf"
    exit 1
fi

# GPU configuration (based on system mapping: CUDA0=RTX3050 8GB, CUDA1=Tesla P40 24GB)
# We use split mode for layers and KV cache:
#   - Model layers: 10% on CUDA0 (3050), 90% on CUDA1 (P40)  [tensor-split]
#   - KV cache:     90% on CUDA0 (3050), 10% on CUDA1 (P40)  [vgpu-split]
# GPU memory limits set to full capacity of each card

# llama.cpp server parameters
# Adjust port/host as needed; ensure llama-server is in PATH or provide full path
llama-server \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 131072 \
    --ngl 999 \
    --split-mode layer \
    --tensor-split 0.1,0.9 \
    --vgpu-split 0.9,0.1 \
    --gpu-memory 8192,24576 \
    --log-disable \
    ${@:2}

# Note:
#   --ctx-size 131072 matches the model's context length
#   --ngl 999 attempts to offload all layers (split via tensor-split)
#   --log-disable reduces output; remove for debugging
#   Additional arguments after model path are passed to llama-server (e.g., -c 2048)