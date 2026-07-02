#!/usr/bin/env bash
# Launch DFlash speculative serving: target model + DFlash draft model.
#
# Examples:
#   FRAMEWORK=sglang PROFILE=cheap TARGET_MODEL=Qwen/Qwen3-4B DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 ./scripts/launch_dflash.sh
#   FRAMEWORK=sglang PROFILE=h100 DTYPE=bfloat16 MAX_RUNNING_REQUESTS=32 CUDA_GRAPH_MAX_BS_DECODE=32 ./scripts/launch_dflash.sh
#   FRAMEWORK=vllm PROFILE=h100 TARGET_MODEL=Qwen/Qwen3-4B DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 ./scripts/launch_dflash.sh

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
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
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
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}
    ;;
  *)
    echo "ERROR: unknown PROFILE='$PROFILE'. Use PROFILE=cheap or PROFILE=h100." >&2
    exit 2
    ;;
esac

warn_pair_mismatch() {
  if [[ "$DRAFT_MODEL" == *"Qwen3-4B-DFlash"* && "$TARGET_MODEL" != *"Qwen3-4B"* ]]; then
    echo "WARNING: DRAFT_MODEL looks like Qwen3-4B-DFlash but TARGET_MODEL does not contain Qwen3-4B." >&2
    echo "         TARGET_MODEL=$TARGET_MODEL" >&2
    echo "         DRAFT_MODEL=$DRAFT_MODEL" >&2
  fi
  if [[ "$DRAFT_MODEL" == *"Qwen3.5-4B-DFlash"* && "$TARGET_MODEL" != *"Qwen3.5-4B"* ]]; then
    echo "WARNING: DRAFT_MODEL looks like Qwen3.5-4B-DFlash but TARGET_MODEL does not contain Qwen3.5-4B." >&2
    echo "         TARGET_MODEL=$TARGET_MODEL" >&2
    echo "         DRAFT_MODEL=$DRAFT_MODEL" >&2
  fi
}

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

warn_pair_mismatch

cat <<CFG
=== DFlash launch config ===
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
DFLASH_BLOCK_SIZE=$DFLASH_BLOCK_SIZE
NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS
USE_AGGRESSIVE_BACKENDS=$USE_AGGRESSIVE_BACKENDS
USE_QWEN35_ADVANCED_BACKENDS=$USE_QWEN35_ADVANCED_BACKENDS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
ATTENTION_BACKEND=$ATTENTION_BACKEND
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-}
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
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "$DRAFT_MODEL"
      --speculative-dflash-block-size "$DFLASH_BLOCK_SIZE"
      --tp-size "$TP_SIZE"
      --max-running-requests "$MAX_RUNNING_REQUESTS"
      --mem-fraction-static "$MEM_FRACTION_STATIC"
      --host "$HOST"
      --port "$PORT"
    )

    if [[ "$DTYPE" != "auto" ]]; then
      cmd+=(--dtype "$DTYPE")
    fi

    if [[ "$PROFILE" == "h100" ]]; then
      cmd+=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE")
      if [[ "$USE_AGGRESSIVE_BACKENDS" == "1" ]]; then
        cmd+=(--attention-backend fa3)
      fi
      if [[ "$TARGET_MODEL" == *"Qwen3.5-4B"* && "$USE_QWEN35_ADVANCED_BACKENDS" == "1" ]]; then
        cmd+=(
          --speculative-draft-attention-backend fa4
          --attention-backend trtllm_mha
          --linear-attn-prefill-backend flashinfer
          --linear-attn-decode-backend flashinfer
          --mamba-scheduler-strategy extra_buffer
          --cuda-graph-backend-prefill tc_piecewise
          --enable-flashinfer-allreduce-fusion
        )
      fi
    fi
    ;;

  vllm)
    echo "WARNING: vLLM DFlash support may require a recent source build or PR branch. If this fails, use FRAMEWORK=sglang first." >&2
    speculative_config=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s}' "$DRAFT_MODEL" "$NUM_SPECULATIVE_TOKENS")
    cmd=(
      vllm serve "$TARGET_MODEL"
      --host "$HOST"
      --port "$PORT"
      --trust-remote-code
      --speculative-config "$speculative_config"
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
