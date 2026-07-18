#!/usr/bin/env bash
# Launch the non-speculative baseline using the same isolated runtime as DSpark.
#
# SGLang H100 example:
#   FRAMEWORK=sglang PROFILE=h100 TARGET_MODEL=Qwen/Qwen3-4B PORT=30000 ./scripts/launch_baseline.sh
#
# vLLM H100 example:
#   FRAMEWORK=vllm PROFILE=h100 TARGET_MODEL=Qwen/Qwen3-4B PORT=30000 ./scripts/launch_baseline.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

FRAMEWORK=${FRAMEWORK:-sglang}
PROFILE=${PROFILE:-h100}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
DTYPE=${DTYPE:-auto}

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}

SGLANG_PYTHON=${SGLANG_PYTHON:-$REPO_ROOT/.venv-sglang-dspark/bin/python}
VLLM_BIN=${VLLM_BIN:-$REPO_ROOT/.venv-vllm-dspark/bin/vllm}

case "$PROFILE" in
  cheap)
    MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.65}
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
    ;;
  h100)
    MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
    CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-32}
    if [[ "$DTYPE" == "auto" ]]; then
      DTYPE=bfloat16
    fi
    ;;
  *)
    echo "ERROR: unknown PROFILE='$PROFILE'. Use cheap or h100." >&2
    exit 2
    ;;
esac

print_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf '\n=== GPU ===\n'
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  fi
}

print_python_version() {
  local python_bin=$1
  "$python_bin" - <<'PY' || true
import importlib.metadata as md
for pkg in ("torch", "transformers", "sglang", "vllm"):
    try:
        print(f"{pkg}: {md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
PY
}

cat <<CFG
=== Baseline launch config ===
FRAMEWORK=$FRAMEWORK
PROFILE=$PROFILE
TARGET_MODEL=$TARGET_MODEL
HOST=$HOST
PORT=$PORT
TP_SIZE=$TP_SIZE
DTYPE=$DTYPE
MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS
MAX_NUM_SEQS=$MAX_NUM_SEQS
CUDA_GRAPH_MAX_BS=$CUDA_GRAPH_MAX_BS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
DISABLE_RADIX_CACHE=$DISABLE_RADIX_CACHE
CFG

print_gpu_info

case "$FRAMEWORK" in
  sglang)
    if [[ ! -x "$SGLANG_PYTHON" ]]; then
      echo "ERROR: SGLang Python not found: $SGLANG_PYTHON" >&2
      echo "Run ./scripts/setup_dspark_envs.sh first or set SGLANG_PYTHON." >&2
      exit 2
    fi
    printf '\n=== SGLang environment ===\n'
    print_python_version "$SGLANG_PYTHON"

    cmd=(
      "$SGLANG_PYTHON" -m sglang.launch_server
      --model-path "$TARGET_MODEL"
      --trust-remote-code
      --tp-size "$TP_SIZE"
      --host "$HOST"
      --port "$PORT"
      --mem-fraction-static "$MEM_FRACTION_STATIC"
      --max-running-requests "$MAX_RUNNING_REQUESTS"
    )
    if [[ "$DTYPE" != "auto" ]]; then
      cmd+=(--dtype "$DTYPE")
    fi
    if [[ -n "$CUDA_GRAPH_MAX_BS" ]]; then
      cmd+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
    fi
    if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
      cmd+=(--disable-radix-cache)
    fi
    ;;

  vllm)
    if [[ ! -x "$VLLM_BIN" ]]; then
      echo "ERROR: vLLM executable not found: $VLLM_BIN" >&2
      echo "Run ./scripts/setup_dspark_envs.sh first or set VLLM_BIN." >&2
      exit 2
    fi
    VLLM_PYTHON=$(dirname "$VLLM_BIN")/python
    printf '\n=== vLLM environment ===\n'
    print_python_version "$VLLM_PYTHON"

    cmd=(
      "$VLLM_BIN" serve "$TARGET_MODEL"
      --host "$HOST"
      --port "$PORT"
      --trust-remote-code
      --tensor-parallel-size "$TP_SIZE"
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
      --max-num-seqs "$MAX_NUM_SEQS"
    )
    if [[ "$DTYPE" != "auto" ]]; then
      cmd+=(--dtype "$DTYPE")
    fi
    if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
      cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    fi
    ;;

  *)
    echo "ERROR: unknown FRAMEWORK='$FRAMEWORK'. Use sglang or vllm." >&2
    exit 2
    ;;
esac

printf '\n=== Final command ===\n'
printf '%q ' "${cmd[@]}"
echo

exec "${cmd[@]}"
