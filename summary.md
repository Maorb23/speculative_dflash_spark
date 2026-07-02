We created a minimal benchmark repo for testing DFlash speculative decoding speedup on Qwen 4B models through OpenAI-compatible inference servers, not through a custom PyTorch speculative loop.

The repo contains:

scripts/make_prompts.py
scripts/launch_baseline.sh
scripts/launch_dflash.sh
scripts/bench_openai_server.py
scripts/compare_results.py
README.md
requirements.txt
.gitignore

The intended comparison is:

baseline = target model only
dflash   = target model + DFlash draft model

Supported frameworks:

SGLang = default / safer path
vLLM   = optional / experimental path

Supported model pairs:

Qwen/Qwen3-4B   + z-lab/Qwen3-4B-DFlash-b16
Qwen/Qwen3.5-4B + z-lab/Qwen3.5-4B-DFlash

We installed and verified the environment:

torch 2.11.0
transformers 5.8.1
datasets 5.0.0
openai 2.6.1
sglang 0.5.14
wandb 0.28.0

We generated FineWeb-Edu-style benchmark prompts:

python scripts/make_prompts.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --split train \
  --tokenizer Qwen/Qwen3-4B \
  --num-prompts 100 \
  --num-prefix-tokens 128 \
  --output prompts.jsonl

We verified the prompt file had 100 valid JSONL rows.

We ran Qwen3.5-4B DFlash on an H100 using SGLang. The successful conservative DFlash launch used:

PROFILE=h100 \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3.5-4B \
DRAFT_MODEL=z-lab/Qwen3.5-4B-DFlash \
DTYPE=bfloat16 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS_DECODE=32 \
PORT=30000 \
./scripts/launch_dflash.sh

This successfully initialized:

DFlashDraftModel
Initialized DFLASH draft runner
DFLASH draft runner ready
Uvicorn running on http://0.0.0.0:30000

We saw noisy torchcodec / FFmpeg warnings, but they were from optional multimodal/audio imports and did not block text inference.

We discovered that the Qwen3.5 DFlash draft config expects block size 16, but the Z-Lab page recommends block size 8 for higher-concurrency serving. In practice, our first block-size-8 run was faster at concurrency 32, which makes sense for high-throughput serving.

We tried to imitate the full Z-Lab Hugging Face command with advanced backends:

--speculative-draft-attention-backend fa4
--attention-backend trtllm_mha
--linear-attn-prefill-backend flashinfer
--linear-attn-decode-backend flashinfer
--cuda-graph-backend-prefill tc_piecewise
--enable-flashinfer-allreduce-fusion

But this failed on H100 because:

TRTLLM MHA backend for prefill is only supported on Blackwell GPUs (SM100)

Z-Lab’s published command appears optimized for B200 / Blackwell, not H100. Therefore, for H100 we should not use --attention-backend trtllm_mha.

The best H100-safe path is:

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

Benchmark DFlash:

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
  --wandb-tags qwen35-4b,dflash,sglang,h100,c32,b8

Compare against baseline and save the printed output:

python -u scripts/compare_results.py \
  --baseline results/baseline_qwen35_4b_h100_c32.json \
  --dflash results/dflash_qwen35_4b_h100_c32_b8.json \
  --wandb \
  --wandb-project dflash-benchmark \
  --wandb-run-name qwen35-4b-dflash-h100-c32-b8 \
  --wandb-tags qwen35-4b,dflash,comparison,sglang,h100,c32,b8 \
  2>&1 | tee results/compare_qwen35_4b_h100_c32_b8.txt

Important lesson:

Do not change many serving flags at once.
Start from the known-good H100-safe run.
Then test one change at a time:
1. block size 8 vs 16
2. max_new_tokens 128 vs 512
3. mem_fraction_static 0.75 vs 0.8
4. flashinfer linear attention
5. prefill CUDA graph
