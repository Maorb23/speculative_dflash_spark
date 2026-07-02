#!/usr/bin/env bash
# Launch baseline target-only serving.
#
# Examples:
#   FRAMEWORK=sglang PROFILE=cheap TARGET_MODEL=Qwen/Qwen3-4B ./scripts/launch_baseline.sh
#   FRAMEWORK=vllm PROFILE=h100 TARGET_MODEL=Qwen/Qwen3-4B PORT=30000 ./scripts/launch_baseline.sh
#
# Optional vLLM-only env vars:
#   MAX_NUM_BATCHED_TOKENS=32768 ATTENTION_BACKEND=FLASHINFER FRAMEWORK=vllm ./scripts/launch_baseline.sh

set -euo pipefail

FRAMEWORK=${FRAMEWORK:-sglang}
PROFILE=${PROFILE:-cheap}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
DTYPE=${DTYPE:-auto}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-}
CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS_DECODE:-}
DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-8}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-15}
USE_AGGRESSIVE_BACKENDS=${USE_AGGRESSIVE_BACKENDS:-0}
USE_QWEN35_ADVANCED_BACKENDS=${USE_QWEN35_ADVANCED_BACKENDS:-0}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-}

case "$PROFILE" in
  cheap)
    MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.65}
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}
    ;;
  h100)
    MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
    CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS_DECODE:-32}
    if [[ "$DTYPE" == "auto" ]]; then
      DTYPE=bfloat16
    fi
    ;;
  *)
    echo "ERROR: unknown PROFILE='$PROFILE'. Use PROFILE=cheap or PROFILE=h100." >&2
    exit 2
    ;;
esac

print_versions() {
  python - <<'PY' || true
import importlib.metadata as md
for pkg in ["torch", "transformers", "sglang", "vllm", "wandb"]:
    try:
        print(f"{pkg}: {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}: not installed")
PY
}

print_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf "\n=== nvidia-smi ===\n"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  else
    printf "\n=== nvidia-smi ===\n"
    echo "nvidia-smi not found"
  fi
}

cat <<CFG
=== Baseline launch config ===
FRAMEWORK=$FRAMEWORK
PROFILE=$PROFILE
TARGET_MODEL=$TARGET_MODEL
DRAFT_MODEL=$DRAFT_MODEL
HOST=$HOST
PORT=$PORT
TP_SIZE=$TP_SIZE
DTYPE=$DTYPE
MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC
MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS
CUDA_GRAPH_MAX_BS_DECODE=$CUDA_GRAPH_MAX_BS_DECODE
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
ATTENTION_BACKEND=$ATTENTION_BACKEND
CFG

print_gpu_info

printf "\n=== Python package versions ===\n"
print_versions

case "$FRAMEWORK" in
  sglang)
    cmd=(
      python -m sglang.launch_server
      --model-path "$TARGET_MODEL"
      --trust-remote-code
      --tp-size "$TP_SIZE"
      --host "$HOST"
      --port "$PORT"
      --mem-fraction-static "$MEM_FRACTION_STATIC"
    )
    if [[ "$DTYPE" != "auto" ]]; then
      cmd+=(--dtype "$DTYPE")
    fi
    ;;

  vllm)
    cmd=(
      vllm serve "$TARGET_MODEL"
      --host "$HOST"
      --port "$PORT"
      --trust-remote-code
    )
    if [[ "$DTYPE" != "auto" ]]; then
      cmd+=(--dtype "$DTYPE")
    fi
    if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
      cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    fi
    if [[ -n "$ATTENTION_BACKEND" ]]; then
      cmd+=(--attention-backend "$ATTENTION_BACKEND")
    fi
    ;;

  *)
    echo "ERROR: unknown FRAMEWORK='$FRAMEWORK'. Use FRAMEWORK=sglang or FRAMEWORK=vllm." >&2
    exit 2
    ;;
esac

printf "\n=== Final command ===\n"
printf '%q ' "${cmd[@]}"
echo

exec "${cmd[@]}"
