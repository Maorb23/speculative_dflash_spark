#!/usr/bin/env bash
# Launch Qwen3 + DSpark using SGLang or vLLM.
#
# SGLang H100:
#   FRAMEWORK=sglang PROFILE=h100 \
#   TARGET_MODEL=Qwen/Qwen3-4B \
#   DRAFT_MODEL=deepseek-ai/dspark_qwen3_4b_block7 \
#   ./scripts/launch_dspark.sh
#
# vLLM H100:
#   FRAMEWORK=vllm PROFILE=h100 \
#   TARGET_MODEL=Qwen/Qwen3-4B \
#   DRAFT_MODEL=deepseek-ai/dspark_qwen3_4b_block7 \
#   ./scripts/launch_dspark.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

FRAMEWORK=${FRAMEWORK:-sglang}
PROFILE=${PROFILE:-h100}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-deepseek-ai/dspark_qwen3_4b_block7}
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

DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-7}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-7}
DSPARK_ATTENTION_BACKEND=${DSPARK_ATTENTION_BACKEND:-FLASH_ATTN}
DRAFT_SAMPLE_METHOD=${DRAFT_SAMPLE_METHOD:-probabilistic}
DSPARK_SPS_TABLE_PATH=${DSPARK_SPS_TABLE_PATH:-}
DSPARK_STS_PATH=${DSPARK_STS_PATH:-}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
SKIP_FEATURE_CHECK=${SKIP_FEATURE_CHECK:-0}

SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}
SGLANG_ENABLE_METRICS_DEVICE_TIMER=${SGLANG_ENABLE_METRICS_DEVICE_TIMER:-1}
export SGLANG_RAGGED_VERIFY_MODE
export SGLANG_ENABLE_METRICS_DEVICE_TIMER

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

if [[ "$DRAFT_MODEL" == *"dspark_qwen3_4b"* && "$TARGET_MODEL" != *"Qwen3-4B"* ]]; then
  echo "WARNING: the selected draft checkpoint is for Qwen3-4B, but TARGET_MODEL is '$TARGET_MODEL'." >&2
fi

if [[ "$SGLANG_RAGGED_VERIFY_MODE" != "static" && "$SGLANG_RAGGED_VERIFY_MODE" != "compact" && "$SGLANG_RAGGED_VERIFY_MODE" != "cap-accept" ]]; then
  echo "ERROR: SGLANG_RAGGED_VERIFY_MODE must be static, compact, or cap-accept." >&2
  exit 2
fi

if [[ "$SGLANG_RAGGED_VERIFY_MODE" == "compact" && -z "$DSPARK_SPS_TABLE_PATH" ]]; then
  echo "WARNING: compact mode has no SPS table; scheduling may degenerate to the full verify window." >&2
fi

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

check_sglang_features() {
  local help_text
  help_text=$("$SGLANG_PYTHON" -m sglang.launch_server --help 2>&1 || true)
  if ! grep -q -- "--speculative-dspark-block-size" <<<"$help_text"; then
    echo "ERROR: this SGLang build does not expose DSpark support." >&2
    echo "Install SGLang 0.5.15.post1+ or use the official dev-dspark image." >&2
    exit 3
  fi
}

check_vllm_features() {
  local vllm_python
  vllm_python=$(dirname "$VLLM_BIN")/python
  "$vllm_python" - <<'PY'
import importlib.metadata as md
from packaging.version import Version
v = Version(md.version("vllm"))
if v < Version("0.25.0"):
    raise SystemExit(f"ERROR: vLLM 0.25.0+ is required for DSpark; found {v}")
print(f"vLLM DSpark version check passed: {v}")
PY
  local help_text
  help_text=$("$VLLM_BIN" serve --help 2>&1 || true)
  if ! grep -Eq -- "--speculative-config|--speculative_config" <<<"$help_text"; then
    echo "ERROR: this vLLM build does not expose --speculative-config." >&2
    exit 3
  fi
}

cat <<CFG
=== DSpark launch config ===
FRAMEWORK=$FRAMEWORK
PROFILE=$PROFILE
TARGET_MODEL=$TARGET_MODEL
DRAFT_MODEL=$DRAFT_MODEL
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
DSPARK_BLOCK_SIZE=$DSPARK_BLOCK_SIZE
NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS
DSPARK_ATTENTION_BACKEND=$DSPARK_ATTENTION_BACKEND
DRAFT_SAMPLE_METHOD=$DRAFT_SAMPLE_METHOD
SGLANG_RAGGED_VERIFY_MODE=$SGLANG_RAGGED_VERIFY_MODE
DSPARK_SPS_TABLE_PATH=$DSPARK_SPS_TABLE_PATH
DSPARK_STS_PATH=$DSPARK_STS_PATH
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
    if [[ "$SKIP_FEATURE_CHECK" != "1" ]]; then
      check_sglang_features
    fi

    cmd=(
      "$SGLANG_PYTHON" -m sglang.launch_server
      --model-path "$TARGET_MODEL"
      --trust-remote-code
      --speculative-algorithm DSPARK
      --speculative-draft-model-path "$DRAFT_MODEL"
      --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
      --tp-size "$TP_SIZE"
      --max-running-requests "$MAX_RUNNING_REQUESTS"
      --mem-fraction-static "$MEM_FRACTION_STATIC"
      --host "$HOST"
      --port "$PORT"
    )
    if [[ "$DTYPE" != "auto" ]]; then
      cmd+=(--dtype "$DTYPE")
    fi
    if [[ -n "$CUDA_GRAPH_MAX_BS" ]]; then
      cmd+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
    fi
    if [[ -n "$DSPARK_SPS_TABLE_PATH" ]]; then
      cmd+=(--speculative-dspark-sps-table-path "$DSPARK_SPS_TABLE_PATH")
    fi
    if [[ -n "$DSPARK_STS_PATH" ]]; then
      cmd+=(--speculative-dspark-confidence-sts-path "$DSPARK_STS_PATH")
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
    if [[ "$SKIP_FEATURE_CHECK" != "1" ]]; then
      check_vllm_features
    fi

    speculative_config=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":%s,"attention_backend":"%s","draft_sample_method":"%s"}' \
      "$DRAFT_MODEL" "$NUM_SPECULATIVE_TOKENS" "$DSPARK_ATTENTION_BACKEND" "$DRAFT_SAMPLE_METHOD")

    cmd=(
      "$VLLM_BIN" serve "$TARGET_MODEL"
      --host "$HOST"
      --port "$PORT"
      --trust-remote-code
      --tensor-parallel-size "$TP_SIZE"
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
      --max-num-seqs "$MAX_NUM_SEQS"
      --speculative-config "$speculative_config"
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
