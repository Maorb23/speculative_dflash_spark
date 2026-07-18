#!/usr/bin/env bash
# Create isolated client, SGLang DSpark, and vLLM DSpark environments.
# Run from the repository root:
#   chmod +x scripts/setup_dspark_envs.sh
#   ./scripts/setup_dspark_envs.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.11 SGLANG_VERSION=0.5.15.post1 VLLM_VERSION=0.25.0 ./scripts/setup_dspark_envs.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.15.post1}
VLLM_VERSION=${VLLM_VERSION:-0.25.0}
CLIENT_VENV=${CLIENT_VENV:-$REPO_ROOT/.venv}
SGLANG_VENV=${SGLANG_VENV:-$REPO_ROOT/.venv-sglang-dspark}
VLLM_VENV=${VLLM_VENV:-$REPO_ROOT/.venv-vllm-dspark}
CLIENT_REQUIREMENTS=${CLIENT_REQUIREMENTS:-$REPO_ROOT/requirements-client.txt}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' was not found. Install Python 3.10+ or set PYTHON_BIN." >&2
  exit 2
fi

printf '=== Host ===\n'
"$PYTHON_BIN" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi was not found. Run this on the GPU VM." >&2
fi

create_venv() {
  local path=$1
  if [[ ! -d "$path" ]]; then
    echo "Creating $path"
    "$PYTHON_BIN" -m venv "$path"
  else
    echo "Reusing existing $path"
  fi
  "$path/bin/python" -m pip install --upgrade pip setuptools wheel packaging
}

create_venv "$CLIENT_VENV"
if [[ ! -f "$CLIENT_REQUIREMENTS" ]]; then
  echo "ERROR: client requirements not found: $CLIENT_REQUIREMENTS" >&2
  exit 2
fi
"$CLIENT_VENV/bin/python" -m pip install -r "$CLIENT_REQUIREMENTS"

create_venv "$SGLANG_VENV"
"$SGLANG_VENV/bin/python" -m pip install "sglang==$SGLANG_VERSION"

create_venv "$VLLM_VENV"
"$VLLM_VENV/bin/python" -m pip install "vllm==$VLLM_VERSION"

printf '\n=== Feature checks ===\n'
"$SGLANG_VENV/bin/python" - <<'PY'
import importlib.metadata as md
print("sglang:", md.version("sglang"))
PY

sglang_help=$("$SGLANG_VENV/bin/python" -m sglang.launch_server --help 2>&1 || true)
if ! grep -q -- "--speculative-dspark-block-size" <<<"$sglang_help"; then
  echo "ERROR: installed SGLang does not expose DSpark CLI arguments." >&2
  echo "Expected --speculative-dspark-block-size in launch_server --help." >&2
  exit 3
fi

"$VLLM_VENV/bin/python" - <<'PY'
import importlib.metadata as md
from packaging.version import Version
v = Version(md.version("vllm"))
print("vllm:", v)
if v < Version("0.25.0"):
    raise SystemExit("ERROR: vLLM 0.25.0+ is required for DSpark")
PY

vllm_help=$("$VLLM_VENV/bin/vllm" serve --help 2>&1 || true)
if ! grep -Eq -- "--speculative-config|--speculative_config" <<<"$vllm_help"; then
  echo "ERROR: installed vLLM does not expose --speculative-config." >&2
  exit 3
fi

cat <<OUT

Environments are ready:
  Client:  $CLIENT_VENV
  SGLang:  $SGLANG_VENV
  vLLM:    $VLLM_VENV

Use the client environment for prompt generation and benchmarking:
  source "$CLIENT_VENV/bin/activate"

The launch scripts automatically use the two server environments above.
OUT
