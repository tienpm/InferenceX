#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.6-27B}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/qwen3.6-27b}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen36-27b-sglang}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612}"
PORT="${PORT:-30000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
SHM_SIZE="${SHM_SIZE:-32g}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

detect_accelerator_count() {
    if command -v amd-smi >/dev/null 2>&1; then
        local count
        count="$(amd-smi list 2>/dev/null | grep -c '^GPU: ' || true)"
        if [[ "$count" -gt 0 ]]; then
            printf '%s\n' "$count"
            return
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
        if [[ "$count" -gt 0 ]]; then
            printf '%s\n' "$count"
            return
        fi
    fi

    printf '1\n'
}

TP="${TP:-$(detect_accelerator_count)}"

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "docker is not installed. Run scripts/install_docker_ubuntu.sh first." >&2
        exit 1
    fi
}

build_gpu_args() {
    GPU_ARGS=()
    if [[ -e /dev/kfd && -d /dev/dri ]]; then
        GPU_ARGS+=(--device=/dev/kfd --device=/dev/dri)
        GPU_ARGS+=(--group-add=video --group-add=render)
        GPU_ARGS+=(--security-opt=seccomp=unconfined --cap-add=SYS_PTRACE)
    elif command -v nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS+=(--gpus all)
    else
        echo "No supported GPU device found. Expected AMD /dev/kfd+/dev/dri or NVIDIA nvidia-smi." >&2
        exit 1
    fi
}

pull_image() {
    require_docker
    docker pull "$SGLANG_DOCKER_IMAGE"
}

download_model() {
    require_docker
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is required to download $MODEL_ID." >&2
        exit 1
    fi

    mkdir -p "$MODEL_DIR"
    pull_image

    docker run --rm \
        --network host \
        -e HF_TOKEN \
        -e MODEL_ID="$MODEL_ID" \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        -v "$MODEL_DIR:/model" \
        "$SGLANG_DOCKER_IMAGE" \
        bash -lc 'set -euo pipefail
if ! command -v hf >/dev/null 2>&1; then
    python3 -m pip install -q --no-cache-dir "huggingface_hub[hf_transfer]" || \
        python3 -m pip install -q --no-cache-dir huggingface_hub
fi
hf download "$MODEL_ID" --local-dir /model'
}

start_service() {
    require_docker
    build_gpu_args

    if [[ ! -d "$MODEL_DIR" || -z "$(find "$MODEL_DIR" -maxdepth 1 -type f -print -quit)" ]]; then
        echo "Model directory is empty: $MODEL_DIR" >&2
        echo "Run: HF_TOKEN=... $0 download" >&2
        exit 1
    fi

    mkdir -p "$RESULT_DIR"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

    local extra_args=()
    if [[ -n "$EXTRA_SGLANG_ARGS" ]]; then
        read -r -a extra_args <<< "$EXTRA_SGLANG_ARGS"
    fi

    docker run -d \
        --name "$CONTAINER_NAME" \
        --network host \
        --ipc host \
        --shm-size "$SHM_SIZE" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        "${GPU_ARGS[@]}" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e PYTHONUNBUFFERED=1 \
        -v "$PROJECT_ROOT:/workspace/InferenceX" \
        -v "$MODEL_DIR:/models/qwen36-27b:ro" \
        -v "$RESULT_DIR:/workspace/results" \
        "$SGLANG_DOCKER_IMAGE" \
        python3 -m sglang.launch_server \
            --model-path /models/qwen36-27b \
            --served-model-name "$MODEL_ID" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tensor-parallel-size "$TP" \
            --trust-remote-code \
            --context-length "$MAX_MODEL_LEN" \
            --max-running-requests "$MAX_RUNNING_REQUESTS" \
            --mem-fraction-static "$MEM_FRACTION_STATIC" \
            --disable-radix-cache \
            "${extra_args[@]}"

    wait_service
}

wait_service() {
    require_docker
    local url="http://127.0.0.1:$PORT/health"
    local deadline=$((SECONDS + ${SGLANG_START_TIMEOUT:-1800}))

    until curl -fsS "$url" >/dev/null 2>&1; do
        if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
            echo "SGLang container exited before becoming healthy." >&2
            docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for $url." >&2
            docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
            exit 1
        fi
        sleep 5
    done

    echo "SGLang is ready at $url"
}

stop_service() {
    require_docker
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

case "${1:-start}" in
    pull)
        pull_image
        ;;
    download)
        download_model
        ;;
    start)
        start_service
        ;;
    wait)
        wait_service
        ;;
    logs)
        require_docker
        docker logs -f "$CONTAINER_NAME"
        ;;
    status)
        require_docker
        docker ps --filter "name=$CONTAINER_NAME"
        ;;
    stop)
        stop_service
        ;;
    *)
        echo "Usage: $0 {pull|download|start|wait|logs|status|stop}" >&2
        exit 2
        ;;
esac
