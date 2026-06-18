# Qwen3.6-27B SGLang Docker Benchmark

This runbook starts an SGLang Docker service for [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B), runs a short random-prompt benchmark, and exports metrics to Markdown.

## Prerequisites

Run once on the benchmark host:

```bash
cd /root/InferenceX
bash scripts/install_docker_ubuntu.sh
```

Use `HF_TOKEN` from your shell. Do not write it into repo files.

## Run

```bash
cd /root/InferenceX
export HF_TOKEN=...
bash scripts/run_qwen36_27b_benchmark.sh
```

The default matrix is `1024:256:1 1024:256:4` (`ISL:OSL:CONC`). Override it for longer sweeps:

```bash
BENCHMARK_MATRIX="1024:1024:1 1024:1024:4 8192:1024:1" \
NUM_PROMPTS_MULTIPLIER=10 \
bash scripts/run_qwen36_27b_benchmark.sh
```

Useful overrides:

```bash
SGLANG_DOCKER_IMAGE=lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612
MODEL_DIR=/root/InferenceX/models/Qwen3.6-27B
RESULT_DIR=/root/InferenceX/results/qwen3.6-27b
PORT=30000
TP=1
MAX_MODEL_LEN=32768
KEEP_SERVER=1
EXTRA_SGLANG_ARGS="--attention-backend triton"
```

## Outputs

- `results/qwen3.6-27b/*.json` - raw benchmark results.
- `results/qwen3.6-27b/metrics.md` - Markdown summary of throughput, latency, and token metrics.

## Service Controls

```bash
bash scripts/qwen36_sglang_service.sh status
bash scripts/qwen36_sglang_service.sh logs
bash scripts/qwen36_sglang_service.sh stop
```

## Sample Result

The initial short run on `root@165.245.169.222` used `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` on AMD MI350X/gfx950:

| Case | Completed | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024:256 c1 | 10/10 | 33.11 | 0.30 | 77.32 | 386.60 | 87.60 | 88.21 | 12.64 | 12.68 |
| 1024:256 c4 | 40/40 | 36.96 | 1.08 | 277.04 | 1385.22 | 259.47 | 282.67 | 13.47 | 13.54 |
