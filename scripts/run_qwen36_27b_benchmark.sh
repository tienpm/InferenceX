#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SCRIPT="$PROJECT_ROOT/scripts/qwen36_sglang_service.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.6-27B}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/qwen3.6-27b}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612}"
PORT="${PORT:-30000}"
BENCHMARK_MATRIX="${BENCHMARK_MATRIX:-1024:256:1 1024:256:4}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-10}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KEEP_SERVER="${KEEP_SERVER:-0}"
SHM_SIZE="${SHM_SIZE:-32g}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed. Run scripts/install_docker_ubuntu.sh first." >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"

if [[ "$KEEP_SERVER" != "1" ]]; then
    cleanup() {
        "$SERVICE_SCRIPT" stop >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
fi

"$SERVICE_SCRIPT" download
"$SERVICE_SCRIPT" start

run_case() {
    local isl="$1"
    local osl="$2"
    local conc="$3"
    local num_prompts=$((conc * NUM_PROMPTS_MULTIPLIER))
    local result_name="qwen36_27b_sglang_isl${isl}_osl${osl}_conc${conc}"
    local result_json="$RESULT_DIR/${result_name}.json"

    if [[ "$SKIP_EXISTING" == "1" && -f "$result_json" ]]; then
        echo "Skipping existing result: $result_json"
        return
    fi

    echo "Benchmarking ISL=$isl OSL=$osl CONC=$conc NUM_PROMPTS=$num_prompts"

    docker run --rm \
        --network host \
        --ipc host \
        --shm-size "$SHM_SIZE" \
        -e PYTHONUNBUFFERED=1 \
        -v "$PROJECT_ROOT:/workspace/InferenceX" \
        -v "$MODEL_DIR:/models/qwen36-27b:ro" \
        -v "$RESULT_DIR:/workspace/results" \
        -w /workspace/InferenceX \
        "$SGLANG_DOCKER_IMAGE" \
        python3 utils/bench_serving/benchmark_serving.py \
            --model "$MODEL_ID" \
            --tokenizer /models/qwen36-27b \
            --trust-remote-code \
            --backend vllm \
            --base-url "http://127.0.0.1:$PORT" \
            --dataset-name random \
            --random-input-len "$isl" \
            --random-output-len "$osl" \
            --random-range-ratio "$RANDOM_RANGE_RATIO" \
            --num-prompts "$num_prompts" \
            --max-concurrency "$conc" \
            --request-rate inf \
            --ignore-eos \
            --save-result \
            --num-warmups "$((2 * conc))" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --result-dir /workspace/results \
            --result-filename "${result_name}.json"
}

for case_spec in $BENCHMARK_MATRIX; do
    IFS=: read -r isl osl conc <<< "$case_spec"
    if [[ -z "${isl:-}" || -z "${osl:-}" || -z "${conc:-}" ]]; then
        echo "Invalid BENCHMARK_MATRIX entry: $case_spec. Expected ISL:OSL:CONC." >&2
        exit 1
    fi
    run_case "$isl" "$osl" "$conc"
done

python3 "$PROJECT_ROOT/scripts/export_qwen36_metrics.py" \
    --results-dir "$RESULT_DIR" \
    --output "$RESULT_DIR/metrics.md" \
    --model-id "$MODEL_ID" \
    --image "$SGLANG_DOCKER_IMAGE"

echo "Metrics written to $RESULT_DIR/metrics.md"
