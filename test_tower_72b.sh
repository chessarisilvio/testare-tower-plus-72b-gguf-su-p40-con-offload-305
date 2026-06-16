#!/bin/bash
# Test script for Tower-Plus-72B GGUF model with progressive offload fallback
# Tries different GPU offload configurations and logs attempts to test_log.txt

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
    echo "Example: ./test_tower_72b.sh /path/to/model.gguf"
    exit 1
fi

# Log file
LOG_FILE="./test_log.txt"
echo "=== Test started at $(date) ===" > "$LOG_FILE"
echo "Model path: $MODEL_PATH" >> "$LOG_FILE"

# Function to log attempt
log_attempt() {
    local config_name="$1"
    local params="$2"
    local result="$3"
    echo "[$(date)] $config_name: $params -> $result" >> "$LOG_FILE"
}

# Function to test a configuration
test_config() {
    local config_name="$1"
    local tensor_split="$2"   # format: "x,y" for CUDA0,CUDA1
    local vgpu_split="$3"     # format: "x,y" for CUDA0,CUDA1
    local ngl="$4"
    local gpu_mem="$5"        # format: "8192,24576" (fixed for now)

    # Build command
    local cmd="llama-server -m \"$MODEL_PATH\" --host 0.0.0.0 --port 8080 --ctx-size 131072 --ngl $ngl --split-mode layer --tensor-split $tensor_split --vgpu-split $vgpu_split --gpu-memory $gpu_mem --log-disable"

    # Run in background, capture output
    local tmp_log="/tmp/llama_test_$$.$RANDOM.log"
    eval "$cmd" > "$tmp_log" 2>&1 &
    local server_pid=$!

    # Wait for server to start or fail
    sleep 10

    # Check if process is still running
    if kill -0 $server_pid 2>/dev/null; then
        # Server is running, kill it
        kill $server_pid
        wait $server_pid 2>/dev/null
        # Check log for success indicators
        if grep -q "server started\|listening on\|llama_server:" "$tmp_log"; then
            rm -f "$tmp_log"
            echo "SUCCESS"
            return 0
        else
            rm -f "$tmp_log"
            echo "FAILED (no success indicators)"
            return 1
        fi
    else
        # Process died, check log for errors
        wait $server_pid 2>/dev/null
        if grep -iq "cannot allocate\|out of memory\|failed to allocate\|ERROR\|failed" "$tmp_log"; then
            rm -f "$tmp_log"
            echo "FAILED (OOM/allocation error)"
            return 1
        else
            rm -f "$tmp_log"
            echo "FAILED (unknown error)"
            return 1
        fi
    fi
}

# Define configurations to try (from most aggressive to most conservative)
# Each config: name, tensor_split, vgpu_split, ngl, gpu_memory
declare -a configs=(
    # Original configuration from run_tower_72b.sh
    "original" "0.1,0.9" "0.9,0.1" "999" "8192,24576"
    # Try more model on P40 (increase tensor-split for P40)
    "more_model_on_p40" "0.2,0.8" "0.8,0.2" "999" "8192,24576"
    # Try all model on P40, KV cache on 3050
    "all_model_on_p40" "0.0,1.0" "0.9,0.1" "999" "8192,24576"
    # Try all model on P40, all KV on P40
    "all_on_p40" "0.0,1.0" "0.0,1.0" "999" "8192,24576"
    # Reduce ngl significantly
    "reduced_ngl_50" "0.1,0.9" "0.9,0.1" "50" "8192,24576"
    "reduced_ngl_20" "0.1,0.9" "0.9,0.1" "20" "8192,24576"
    "reduced_ngl_10" "0.1,0.9" "0.9,0.1" "10" "8192,24576"
    # Try offloading nothing (CPU only) as last resort
    "cpu_only" "0.0,0.0" "0.0,0.0" "0" "8192,24576"
)

# Try each configuration
success=0
for config in "${configs[@]}"; do
    # Parse config
    IFS=' ' read -r name tsplit vsplit ng mem <<< "$config"
    echo "Testing configuration: $name (tensor-split: $tsplit, vgpu-split: $vsplit, ngl: $ng)" | tee -a "$LOG_FILE"
    result=$(test_config "$name" "$tsplit" "$vsplit" "$ng" "$mem")
    log_attempt "$name" "tensor-split=$tsplit, vgpu-split=$vsplit, ngl=$ng" "$result"
    if [[ "$result" == "SUCCESS" ]]; then
        echo "SUCCESS: Found working configuration: $name" | tee -a "$LOG_FILE"
        echo "Working configuration:" >> "$LOG_FILE"
        echo "  tensor-split: $tsplit" >> "$LOG_FILE"
        echo "  vgpu-split: $vsplit" >> "$LOG_FILE"
        echo "  ngl: $ng" >> "$LOG_FILE"
        echo "  gpu-memory: $mem" >> "$LOG_FILE"
        success=1
        break
    else
        echo "FAILED: $result" | tee -a "$LOG_FILE"
    fi
done

if [[ $success -eq 0 ]]; then
    echo "ERROR: No working configuration found" | tee -a "$LOG_FILE"
    exit 1
fi

echo "=== Test completed at $(date) ===" >> "$LOG_FILE"
exit 0