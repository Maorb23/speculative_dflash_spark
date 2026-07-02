# DFlash Speculative Decoding Benchmark

Minimal benchmark setup for comparing normal Qwen serving against DFlash speculative serving through an OpenAI-compatible API.

The main goal is to measure whether DFlash improves throughput / latency when serving Qwen 4B-class models.

## What this repo tests

We compare:

1. **Baseline serving**
   Target model only.

2. **DFlash speculative serving**
   Target model + DFlash draft model.

Supported serving frameworks:

* **SGLang** — recommended / safer path
* **vLLM** — optional / experimental path

The benchmark client sends requests to a running OpenAI-compatible server and saves local JSON result files.

## Important concept

DFlash draft models are **not normal standalone language models**.

Do not benchmark them by loading them as ordinary draft LMs in a custom PyTorch speculative loop.

The correct flow is:

```text
Launch DFlash-capable inference server
        ↓
Send OpenAI-compatible API requests
        ↓
Measure speed / latency / tokens per second
```

## Repo structure

```text
scripts/
  make_prompts.py
  launch_baseline.sh
  launch_dflash.sh
  bench_openai_server.py
  compare_results.py

results/
  .gitkeep

requirements.txt
README.md
.gitignore
```

## Supported model pairs

### Pair 1 — safer first benchmark

Target model:

```text
Qwen/Qwen3-4B
```

DFlash draft model:

```text
z-lab/Qwen3-4B-DFlash-b16
```

### Pair 2 — Qwen3.5 benchmark

Target model:

```text
Qwen/Qwen3.5-4B
```

DFlash draft model:

```text
z-lab/Qwen3.5-4B-DFlash
```

## Install

Use a clean Python environment.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Install SGLang separately:

```bash
pip install "sglang[all]"
```

Optional, only if testing vLLM:

```bash
pip install vllm
```

Verify versions:

```bash
python - <<'PY'
import sys
print("python:", sys.executable)

for pkg in ["torch", "transformers", "datasets", "openai", "sglang", "vllm", "wandb"]:
    try:
        import importlib.metadata as m
        print(pkg, m.version(pkg))
    except Exception as e:
        print(pkg, "not installed:", e)
PY
```

Make scripts executable:

```bash
chmod +x scripts/*.sh
```

## Optional: Hugging Face login

Unauthenticated downloads work for public models, but login helps avoid rate limits.

```bash
huggingface-cli login
```

Or:

```bash
export HF_TOKEN="your_hf_token_here"
```

## Optional: W&B setup

W&B is disabled unless `--wandb` is passed.

Login once:

```bash
wandb login
```

Optional environment variables:

```bash
export WANDB_PROJECT=dflash-benchmark
export WANDB_MODE=online
export WANDB_TAGS=dflash,qwen,sglang
```

If using a team workspace:

```bash
export WANDB_ENTITY=your_team_or_username
```

## Generate prompts

FineWeb-Edu prompts are used only as benchmark prefixes.

For Qwen3:

```bash
python scripts/make_prompts.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --split train \
  --tokenizer Qwen/Qwen3-4B \
  --num-prompts 100 \
  --num-prefix-tokens 128 \
  --output prompts.jsonl
```

For Qwen3.5:

```bash
python scripts/make_prompts.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --split train \
  --tokenizer Qwen/Qwen3.5-4B \
  --num-prompts 100 \
  --num-prefix-tokens 128 \
  --output prompts_qwen35_4b.jsonl
```

Smoke-test prompts:

```bash
python scripts/make_prompts.py \
  --tokenizer Qwen/Qwen3.5-4B \
  --num-prompts 5 \
  --num-prefix-tokens 128 \
  --output prompts_qwen35_4b_smoke.jsonl
```

Validate a prompt file:

```bash
python - <<'PY'
import json

path = "prompts_qwen35_4b.jsonl"
with open(path) as f:
    rows = [json.loads(line) for line in f]

print("num rows:", len(rows))
print("first keys:", rows[0].keys())
print("first prompt preview:", rows[0]["prompt"][:300])
PY
```

## Recommended benchmark flow

Use two terminals.

Terminal 1 runs the server.

Terminal 2 runs the benchmark client.

## Qwen3.5-4B on H100 — recommended H100-safe setup

This is the recommended H100 path after our testing.

It uses:

```text
SGLang
Qwen/Qwen3.5-4B
z-lab/Qwen3.5-4B-DFlash
bfloat16
DFlash block size 8
concurrency 32
CUDA graph decode
```

Why block size 8?

For high-concurrency serving, block size 8 was faster in our run than block size 16. Block size 16 may be useful for lower-concurrency or longer single-request workloads, but for concurrency 32 we use block size 8.

### Terminal 1 — baseline server

```bash
cd ~/speculative_dflash_spark
source .venv/bin/activate

PROFILE=h100 \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3.5-4B \
DTYPE=bfloat16 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS_DECODE=32 \
PORT=30000 \
./scripts/launch_baseline.sh
```

Keep this terminal running.

### Terminal 2 — benchmark baseline

```bash
cd ~/speculative_dflash_spark
source .venv/bin/activate

python -u scripts/bench_openai_server.py \
  --mode baseline \
  --framework sglang \
  --profile h100 \
  --model Qwen/Qwen3.5-4B \
  --prompts prompts_qwen35_4b.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --concurrency 32 \
  --out results/baseline_qwen35_4b_h100_c32.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-tags qwen35-4b,baseline,sglang,h100,c32 \
  2>&1 | tee results/baseline_qwen35_4b_h100_c32.log
```

Stop the baseline server in Terminal 1 with:

```text
Ctrl+C
```

### Terminal 1 — DFlash server

```bash
cd ~/speculative_dflash_spark
source .venv/bin/activate

PROFILE=h100 \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3.5-4B \
DRAFT_MODEL=z-lab/Qwen3.5-4B-DFlash \
DTYPE=bfloat16 \
DFLASH_BLOCK_SIZE=8 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS_DECODE=32 \
USE_QWEN35_ADVANCED_BACKENDS=0 \
PORT=30000 \
./scripts/launch_dflash.sh
```

The server is healthy when you see lines like:

```text
Initialized DFLASH draft runner
DFLASH draft runner ready
Uvicorn running on http://0.0.0.0:30000
The server is fired up and ready to roll!
```

### Terminal 2 — benchmark DFlash

```bash
cd ~/speculative_dflash_spark
source .venv/bin/activate

python -u scripts/bench_openai_server.py \
  --mode dflash \
  --framework sglang \
  --profile h100 \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompts prompts_qwen35_4b.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --concurrency 32 \
  --out results/dflash_qwen35_4b_h100_c32_b8.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-tags qwen35-4b,dflash,sglang,h100,c32,b8 \
  2>&1 | tee results/dflash_qwen35_4b_h100_c32_b8.log
```

### Compare results and save print

```bash
python -u scripts/compare_results.py \
  --baseline results/baseline_qwen35_4b_h100_c32.json \
  --dflash results/dflash_qwen35_4b_h100_c32_b8.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-run-name qwen35-4b-dflash-h100-c32-b8 \
  --wandb-tags qwen35-4b,dflash,comparison,sglang,h100,c32,b8 \
  2>&1 | tee results/compare_qwen35_4b_h100_c32_b8.txt
```

## Smoke test mode

Use this before full runs.

### Baseline smoke

```bash
python scripts/bench_openai_server.py \
  --mode baseline \
  --framework sglang \
  --profile h100 \
  --model Qwen/Qwen3.5-4B \
  --prompts prompts_qwen35_4b_smoke.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --concurrency 1 \
  --out results/baseline_qwen35_4b_smoke.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-tags qwen35-4b,baseline,sglang,h100,smoke
```

### DFlash smoke

```bash
python scripts/bench_openai_server.py \
  --mode dflash \
  --framework sglang \
  --profile h100 \
  --model Qwen/Qwen3.5-4B \
  --draft-model z-lab/Qwen3.5-4B-DFlash \
  --prompts prompts_qwen35_4b_smoke.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --concurrency 1 \
  --out results/dflash_qwen35_4b_smoke.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-tags qwen35-4b,dflash,sglang,h100,smoke
```

## Why not use the full Z-Lab command on H100?

The full Z-Lab Hugging Face command includes advanced flags such as:

```text
--speculative-draft-attention-backend fa4
--attention-backend trtllm_mha
--linear-attn-prefill-backend flashinfer
--linear-attn-decode-backend flashinfer
--cuda-graph-backend-prefill tc_piecewise
--enable-flashinfer-allreduce-fusion
```

That command appears optimized for B200 / Blackwell GPUs.

On H100, SGLang fails with:

```text
TRTLLM MHA backend for prefill is only supported on Blackwell GPUs (SM100).
```

Therefore, on H100 do **not** use:

```text
--attention-backend trtllm_mha
```

The H100-safe path is to let SGLang use H100-compatible backends, mainly FA3 where available.

## Testing one change at a time

Do not change many backend flags at once.

Recommended test order:

1. Known-good H100 DFlash run:

   * block size 8
   * max new tokens 128
   * concurrency 32
   * no advanced Qwen3.5 backend flags

2. Try longer generation:

   * `--max-new-tokens 512`

3. Try memory fraction:

   * `MEM_FRACTION_STATIC=0.8`

4. Try block size:

   * `DFLASH_BLOCK_SIZE=16`

5. Try individual advanced backend knobs manually, one at a time.

Do not combine all advanced flags until each one has been validated independently.

## Cheap GPU profile

For RTX-style GPUs or uncertain environments, use the cheap profile first.

### Baseline

```bash
PROFILE=cheap \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3-4B \
PORT=30000 \
./scripts/launch_baseline.sh
```

### DFlash

```bash
PROFILE=cheap \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 \
PORT=30000 \
./scripts/launch_dflash.sh
```

Benchmark with low concurrency:

```bash
python scripts/bench_openai_server.py \
  --mode dflash \
  --framework sglang \
  --profile cheap \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --concurrency 1 \
  --out results/dflash_qwen3_4b_cheap_smoke.json
```

## vLLM path

vLLM support is optional and experimental.

DFlash support may require a recent source build or specific branch.

### vLLM baseline

```bash
PROFILE=h100 \
FRAMEWORK=vllm \
TARGET_MODEL=Qwen/Qwen3-4B \
PORT=30000 \
./scripts/launch_baseline.sh
```

### vLLM DFlash

```bash
PROFILE=h100 \
FRAMEWORK=vllm \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 \
NUM_SPECULATIVE_TOKENS=15 \
PORT=30000 \
./scripts/launch_dflash.sh
```

If vLLM DFlash fails, use SGLang first.

## Result files

Benchmark JSON files are saved under `results/`.

Each benchmark result contains:

```text
config
summary
requests
```

Each request row includes:

```text
id
latency_s
ttft_s
output_tokens
output_tokens_per_second
status
error_message
prompt_preview
completion_preview
```

The comparison script prints:

```text
baseline output tok/s
dflash output tok/s
speedup
baseline p50 latency
dflash p50 latency
baseline p95 latency
dflash p95 latency
```

Save comparison prints with:

```bash
2>&1 | tee results/compare_name.txt
```

## Git notes

This repo should track benchmark JSON/TXT result files.

The `.gitignore` should not ignore:

```text
results/*.json
results/*.txt
prompts*.jsonl
```

It may ignore noisy logs:

```text
*.log
logs/
wandb/
.cache/
models/
checkpoints/
```

Check whether files are ignored:

```bash
git check-ignore -v results/*.json || true
git check-ignore -v results/*.txt || true
```

Add benchmark code and results:

```bash
git add README.md requirements.txt scripts/ .gitignore results/*.json results/*.txt prompts*.jsonl
git commit -m "Add DFlash benchmark setup and H100 results"
```

## Correctness notes

With `temperature=0.0`, baseline and DFlash completions may be identical.

That is expected.

Speculative decoding should preserve the target model distribution. DFlash proposes tokens, but the target model verifies them.

Identical completion previews are a good sanity signal, but the real benchmark result is:

```text
tokens/sec
requests/sec
p50 latency
p95 latency
error count
success count
```

## Known warnings

### Hugging Face unauthenticated warning

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This is not fatal. Login with:

```bash
huggingface-cli login
```

### Missing generation_config.json

```text
Failed to load generation config ... Proceeding without generation config.
```

This is not fatal for this benchmark.

### torchcodec / FFmpeg warnings

SGLang may print torchcodec / FFmpeg warnings from optional multimodal/audio imports.

For text-only Qwen benchmarking, these warnings are noisy but usually harmless if the server still reaches:

```text
Uvicorn running on http://0.0.0.0:30000
```

## Main lessons from our run

1. SGLang is the safer path.
2. DFlash server startup worked on H100 with Qwen3.5-4B.
3. The full Z-Lab advanced backend command is not directly usable on H100 because `trtllm_mha` requires Blackwell.
4. For H100 concurrency 32, block size 8 was better than block size 16 in our run.
5. Changing many backend flags at once made performance worse.
6. Keep baseline and DFlash benchmark settings aligned:

   * same prompts
   * same target model
   * same max new tokens
   * same concurrency
   * same temperature
   * same framework
   * same hardware

