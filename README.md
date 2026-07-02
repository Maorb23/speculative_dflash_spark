# DFlash Qwen 4B Speculative Decoding Benchmark

Minimal benchmark setup for comparing target-only serving against DFlash speculative serving on Qwen 4B models.

The benchmark compares:

1. **Baseline serving**: target model only
2. **DFlash serving**: target model plus DFlash draft model

Supported serving frameworks:

- **SGLang**: recommended first path and default
- **vLLM**: optional / experimental path; DFlash support may require a recent source build or a PR branch depending on your installed version

## Important concept

DFlash draft models are **not normal standalone draft language models**. Do not load them into a custom PyTorch truncated-layer speculative loop. This repo launches a DFlash-capable inference server and benchmarks it through an OpenAI-compatible API.

FineWeb-Edu samples are used only as benchmark prefixes. They are not training data and are not meant to evaluate model quality.

## Files

```text
scripts/make_prompts.py
scripts/launch_baseline.sh
scripts/launch_dflash.sh
scripts/bench_openai_server.py
scripts/compare_results.py
README.md
requirements.txt
```

The launch layer intentionally uses only two shell scripts:

```bash
FRAMEWORK=sglang ./scripts/launch_baseline.sh
FRAMEWORK=sglang ./scripts/launch_dflash.sh

FRAMEWORK=vllm ./scripts/launch_baseline.sh
FRAMEWORK=vllm ./scripts/launch_dflash.sh
```

## Supported model pairs

Preferred first benchmark:

```bash
TARGET_MODEL=Qwen/Qwen3-4B
DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16
```

Secondary benchmark:

```bash
TARGET_MODEL=Qwen/Qwen3.5-4B
DRAFT_MODEL=z-lab/Qwen3.5-4B-DFlash
```

You can override the models with environment variables.

## Hardware profiles

Two profiles are supported:

```bash
PROFILE=cheap
PROFILE=h100
```

`cheap` is the default because it is safer for RTX-style GPUs. It uses conservative memory and concurrency settings:

```bash
MEM_FRACTION_STATIC=0.65
MAX_RUNNING_REQUESTS=1
```

`h100` uses stronger defaults:

```bash
MEM_FRACTION_STATIC=0.75
MAX_RUNNING_REQUESTS=32
CUDA_GRAPH_MAX_BS_DECODE=32
DTYPE=bfloat16
```

Cheap GPUs should start conservative. Do not force FA3, FA4, TensorRT-LLM MHA, FlashInfer, or other aggressive backends until the basic server starts reliably.

H100 can try higher concurrency and aggressive backends after the conservative DFlash setup works.

## Install

```bash
pip install -r requirements.txt
```

You also need the serving framework you plan to use:

```bash
# SGLang path, recommended first
pip install sglang

# vLLM path, experimental for DFlash depending on version/source build
pip install vllm
```

## Generate prompts

```bash
python scripts/make_prompts.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --split train \
  --tokenizer Qwen/Qwen3-4B \
  --num-prompts 100 \
  --num-prefix-tokens 128 \
  --output prompts.jsonl
```

By default, prompts are plain completion prompts. Add `--chat-template` only if you want chat-style prompts.

## Cheap RTX baseline with SGLang

Terminal 1:

```bash
PROFILE=cheap \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3-4B \
PORT=30000 \
./scripts/launch_baseline.sh
```

Terminal 2:

```bash
python scripts/bench_openai_server.py \
  --mode baseline \
  --framework sglang \
  --profile cheap \
  --model Qwen/Qwen3-4B \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --concurrency 1 \
  --out results/baseline.json
```

## Benchmark baseline with W&B

```bash
python scripts/bench_openai_server.py \
  --mode baseline \
  --framework sglang \
  --profile cheap \
  --model Qwen/Qwen3-4B \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --concurrency 1 \
  --out results/baseline.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-tags qwen3-4b,baseline,sglang,cheap
```

## Cheap RTX DFlash with SGLang

Stop the baseline server, then launch DFlash.

Terminal 1:

```bash
PROFILE=cheap \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 \
PORT=30000 \
./scripts/launch_dflash.sh
```

Terminal 2:

```bash
python scripts/bench_openai_server.py \
  --mode dflash \
  --framework sglang \
  --profile cheap \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --concurrency 1 \
  --out results/dflash.json
```

## Benchmark DFlash with W&B

```bash
python scripts/bench_openai_server.py \
  --mode dflash \
  --framework sglang \
  --profile cheap \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --concurrency 1 \
  --out results/dflash.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-tags qwen3-4b,dflash,sglang,cheap
```

## Compare results with W&B

```bash
python scripts/compare_results.py \
  --baseline results/baseline.json \
  --dflash results/dflash.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-run-name qwen3-4b-dflash-speedup
```

## Smoke test mode

Use this before running the full benchmark:

```bash
python scripts/make_prompts.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --split train \
  --tokenizer Qwen/Qwen3-4B \
  --num-prompts 5 \
  --num-prefix-tokens 128 \
  --output prompts_smoke.jsonl

python scripts/bench_openai_server.py \
  --mode baseline \
  --framework sglang \
  --profile cheap \
  --model Qwen/Qwen3-4B \
  --prompts prompts_smoke.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 32 \
  --concurrency 1 \
  --out results/baseline_smoke.json
```

Repeat the same command with `--mode dflash --draft-model z-lab/Qwen3-4B-DFlash-b16 --out results/dflash_smoke.json` after launching the DFlash server.

## H100 SGLang DFlash

```bash
PROFILE=h100 \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 \
DTYPE=bfloat16 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS_DECODE=32 \
PORT=30000 \
./scripts/launch_dflash.sh
```

Optional aggressive backend for H100 after the basic path works:

```bash
PROFILE=h100 \
FRAMEWORK=sglang \
USE_AGGRESSIVE_BACKENDS=1 \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 \
./scripts/launch_dflash.sh
```

Optional advanced Qwen3.5 path, not enabled by default:

```bash
PROFILE=h100 \
FRAMEWORK=sglang \
USE_QWEN35_ADVANCED_BACKENDS=1 \
TARGET_MODEL=Qwen/Qwen3.5-4B \
DRAFT_MODEL=z-lab/Qwen3.5-4B-DFlash \
./scripts/launch_dflash.sh
```

## vLLM DFlash, experimental

```bash
PROFILE=h100 \
FRAMEWORK=vllm \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=z-lab/Qwen3-4B-DFlash-b16 \
NUM_SPECULATIVE_TOKENS=15 \
PORT=30000 \
./scripts/launch_dflash.sh
```

The script prints this warning before launch:

```text
WARNING: vLLM DFlash support may require a recent source build or PR branch. If this fails, use FRAMEWORK=sglang first.
```

## Streaming and TTFT

Add `--stream` to the benchmark script to measure time-to-first-token:

```bash
python scripts/bench_openai_server.py \
  --mode dflash \
  --framework sglang \
  --profile cheap \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --concurrency 1 \
  --stream \
  --out results/dflash_stream.json
```

## W&B logging

W&B is optional and disabled unless `--wandb` is passed.

Supported environment variables:

```bash
WANDB_PROJECT=dflash-benchmark
WANDB_ENTITY=
WANDB_MODE=online
WANDB_TAGS=dflash,qwen,sglang
```

If `WANDB_MODE=offline`, local JSON files are still written normally.

Each benchmark run logs:

- summary metrics
- one W&B Table row per request
- the local JSON result as an artifact

`compare_results.py` logs the speedup and latency ratios when `--wandb` is used.

## Sanity notes

With `temperature=0.0`, baseline and DFlash outputs do not have to be byte-identical. Server internals may differ slightly. The benchmark focuses on:

- requests succeed
- outputs are non-empty
- output token counting works
- generated text looks valid
- errors are logged clearly
- output tokens/sec and latency are comparable

## Result JSON format

Each benchmark writes:

```json
{
  "config": {},
  "summary": {},
  "requests": []
}
```

Each request row includes:

```json
{
  "id": 0,
  "latency_s": 1.23,
  "ttft_s": null,
  "output_tokens": 128,
  "output_tokens_per_second": 104.0,
  "status": "ok",
  "error_message": null,
  "prompt_preview": "...",
  "completion_preview": "..."
}
```
