#!/bin/bash
# Manual benchmark script for Tower-Plus-72B GGUF model
# Measures tok/s, latency, and VRAM usage under real load
# Usage: ./benchmark_tower_72b.sh [model_path]
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
    echo "Example: ./benchmark_tower_72b.sh /path/to/model.gguf"
    exit 1
fi

# GPU configuration (based on system mapping: CUDA0=RTX3050 8GB, CUDA1=Tesla P40 24GB)
# We use split mode for layers and KV cache:
#   - Model layers: 10% on CUDA0 (3050), 90% on CUDA1 (P40)  [tensor-split]
#   - KV cache:     90% on CUDA0 (3050), 10% on CUDA1 (P40)  [vgpu-split]
# GPU memory limits set to full capacity of each card

# Port for the server (choose a port unlikely to conflict)
PORT=8080

# Function to start the server
start_server() {
    echo "Starting llama-server..."
    llama-server \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --ctx-size 131072 \
        --ngl 999 \
        --split-mode layer \
        --tensor-split 0.1,0.9 \
        --vgpu-split 0.9,0.1 \
        --gpu-memory 8192,24576 \
        --log-disable \
        &
    SERVER_PID=$!
    echo "Server started with PID $SERVER_PID"

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:$PORT/health > /dev/null; then
            echo "Server is ready!"
            return 0
        fi
        sleep 1
    done
    echo "Error: Server did not become ready in time"
    kill $SERVER_PID 2>/dev/null
    return 1
}

# Function to stop the server
stop_server() {
    echo "Stopping server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}

# Function to run llama-perf benchmark
run_llama_perf() {
    echo "Running llama-perf benchmark..."
    # Run llama-perf for 100 tokens, using a simple prompt
    # Adjust the prompt and parameters as needed
    LLAMA_PERF_OUTPUT=$(llama-perf \
        -m "$MODEL_PATH" \
        -p "Hello, my name is" \
        -n 100 \
        --temp 0.8 \
        --top-p 0.95 \
        --ctx-size 131072 \
        --ngl 999 \
        --split-mode layer \
        --tensor-split 0.1,0.9 \
        --vgpu-split 0.9,0.1 \
        --gpu-memory 8192,24576 \
        2>&1)
    echo "$LLAMA_PERF_OUTPUT"

    # Extract tokens per second and latency from llama-perf output
    # This parsing may need adjustment based on actual llama-perf output format
    TOK_S=$(echo "$LLAMA_PERF_OUTPUT" | grep -oP 'tokens per second: \K\d+\.?\d*' | head -1)
    LATENCY=$(echo "$LLAMA_PERF_OUTPUT" | grep -oP 'latency: \K\d+\.?\d*' | head -1)

    if [[ -z "$TOK_S" ]]; then TOK_S="N/A"; fi
    if [[ -z "$LATENCY" ]]; then LATENCY="N/A"; fi

    echo "Tok/s: $TOK_S"
    echo "Latency (ms): $LATENCY"
}

# Function to run curl-based benchmark (fallback if llama-perf not available)
run_curl_benchmark() {
    echo "Running curl-based benchmark (llama-perf not found or failed)..."
    PROMPT="Hello, my name is"
    N_PREDICT=100

    # Time the request
    START_TIME=$(date +%s.%N)
    RESPONSE=$(curl -s -X POST http://localhost:$PORT/completion \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$PROMPT\",
            \"n_predict\": $N_PREDICT,
            \"temperature\": 0.8,
            \"top_p\": 0.95,
            \"stream\": false
        }")
    END_TIME=$(date +%s.%N)

    # Calculate elapsed time
    ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    # Extract the generated text and count tokens (approximate by splitting words)
    # Note: This is a rough estimate; for accurate token count, we'd need to use the tokenizer
    GENERATED_TEXT=$(echo "$RESPONSE" | grep -oP '(?<="content": ").*?(?=")' | head -1)
    WORD_COUNT=$(echo "$GENERATED_TEXT" | wc -w)

    # Tokens per second (approximate)
    if [[ "$ELAPSED_TIME" > 0 ]]; then
        TOK_S=$(echo "scale=2; $WORD_COUNT / $ELAPSED_TIME" | bc)
    else
        TOK_S="0"
    fi

    # Latency per token (ms)
    if [[ "$WORD_COUNT" > 0 ]]; then
        LATENCY=$(echo "scale=2; ($ELAPSED_TIME * 1000) / $WORD_COUNT" | bc)
    else
        LATENCY="0"
    fi

    echo "Tok/s (approx): $TOK_S"
    echo "Latency per token (ms, approx): $LATENCY"
    echo "Generated words: $WORD_COUNT"
    echo "Elapsed time: $ELAPSED_TIME s"
}

# Function to monitor VRAM usage with nvidia-smi dmon
monitor_vram() {
    echo "Starting VRAM monitoring..."
    # Run nvidia-smi dmon for a short period, capturing VRAM usage
    # We'll run it in the background and kill it after the benchmark
    NVSMI_LOG=$(mktemp)
    nvidia-smi dmon -s u -o DT -i 0,1 1 > "$NVSMI_LOG" 2>/dev/null &
    NVSMI_PID=$!
    echo "nvidia-smi dmon started with PID $NVSMI_PID, logging to $NVSMI_LOG"

    # Return the log file path and PID for later cleanup
    echo "$NVSMI_LOG"
    echo "$NVSMI_PID"
}

# Function to stop VRAM monitoring and get max VRAM usage
stop_vram_monitoring() {
    local NVSMI_LOG="$1"
    local NVSMI_PID="$2"

    echo "Stopping VRAM monitoring (PID: $NVSMI_PID)..."
    kill $NVSMI_PID 2>/dev/null
    wait $NVSMI_PID 2>/dev/null

    # Parse the log to find max VRAM usage (column 3 is memory usage % for GPU 0 and 1?)
    # nvidia-smi dmon -s u outputs: # gpu  sm  mem   enc   dec  ...
    # We want the memory column (mem) for each GPU
    # We'll compute the max across both GPUs
    if [[ -f "$NVSMI_LOG" ]]; then
        # Skip the first line (header) and get the max of the third column (memory %)
        MAX_MEM=$(tail -n +2 "$NVSMI_LOG" | awk '{print $3}' | sort -nr | head -1)
        echo "Max VRAM usage (%): $MAX_MEM"
        # Also get raw memory usage in MB if available? The 's u' option gives utilization and memory in %
        # For MB, we would need to use a different query. We'll stick with % for simplicity.
        rm -f "$NVSMI_LOG"
    else
        echo "VRAM log not found"
    fi
}

# Main script execution
trap 'stop_server; stop_vram_monitoring "$NVSMI_LOG" "$NVSMI_PID"' EXIT INT TERM

if start_server; then
    # Start VRAM monitoring
    read NVSMI_LOG NVSMI_PID < <(monitor_vram)

    # Run benchmark
    if command -v llama-perf > /dev/null 2>&1; then
        run_llama_perf
    else
        echo "llama-perf not found, falling back to curl benchmark"
        run_curl_benchmark
    fi

    # Stop VRAM monitoring and get results
    stop_vram_monitoring "$NVSMI_LOG" "$NVSMI_PID"
else
    echo "Failed to start server. Exiting."
    exit 1
fi

echo "Benchmark completed."