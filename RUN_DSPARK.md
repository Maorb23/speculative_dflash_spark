# Qwen3-4B DSpark benchmark on H100

## 1. Create the environments

From the repository root:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv git

chmod +x scripts/setup_dspark_envs.sh scripts/launch_baseline.sh scripts/launch_dspark.sh
./scripts/setup_dspark_envs.sh
```

This creates:

- `.venv` — prompt generation, benchmark client, and W&B
- `.venv-sglang-dspark` — SGLang `0.5.15.post1`
- `.venv-vllm-dspark` — vLLM `0.25.0`

## 2. Generate prompts

```bash
cd ~/speculative_dflash_spark
source .venv/bin/activate

python scripts/make_prompts.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --split train \
  --tokenizer Qwen/Qwen3-4B \
  --num-prompts 100 \
  --num-prefix-tokens 128 \
  --output prompts_qwen3_4b.jsonl
```

## 3. SGLang baseline

Terminal 1:

```bash
cd ~/speculative_dflash_spark

PROFILE=h100 \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3-4B \
DTYPE=bfloat16 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS=32 \
PORT=30000 \
./scripts/launch_baseline.sh
```

Terminal 2:

```bash
cd ~/speculative_dflash_spark
source .venv/bin/activate

python -u scripts/bench_openai_server.py \
  --mode baseline \
  --framework sglang \
  --profile h100 \
  --model Qwen/Qwen3-4B \
  --prompts prompts_qwen3_4b.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --concurrency 32 \
  --out results/baseline_qwen3_4b_sglang_h100_c32_t128.json \
  --wandb \
  --wandb-project speculative-decoding-benchmark \
  --wandb-tags qwen3-4b,baseline,sglang,h100,c32,t128 \
  2>&1 | tee results/baseline_qwen3_4b_sglang_h100_c32_t128.log
```

Stop the baseline server before launching DSpark.

## 4. SGLang DSpark

Terminal 1:

```bash
cd ~/speculative_dflash_spark

PROFILE=h100 \
FRAMEWORK=sglang \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=deepseek-ai/dspark_qwen3_4b_block7 \
DTYPE=bfloat16 \
DSPARK_BLOCK_SIZE=7 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS=32 \
SGLANG_RAGGED_VERIFY_MODE=static \
PORT=30000 \
./scripts/launch_dspark.sh
```

Terminal 2:

```bash
cd ~/speculative_dflash_spark
source .venv/bin/activate

NUM_SPECULATIVE_TOKENS=7 \
DSPARK_BLOCK_SIZE=7 \
SGLANG_RAGGED_VERIFY_MODE=static \
python -u scripts/bench_openai_server.py \
  --mode dspark \
  --framework sglang \
  --profile h100 \
  --model Qwen/Qwen3-4B \
  --draft-model deepseek-ai/dspark_qwen3_4b_block7 \
  --prompts prompts_qwen3_4b.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --concurrency 32 \
  --out results/dspark_qwen3_4b_sglang_h100_c32_b7_t128_static.json \
  --wandb \
  --wandb-project speculative-decoding-benchmark \
  --wandb-tags qwen3-4b,dspark,sglang,h100,c32,b7,t128,static \
  2>&1 | tee results/dspark_qwen3_4b_sglang_h100_c32_b7_t128_static.log
```

## 5. Compare SGLang baseline and DSpark

```bash
source .venv/bin/activate

python scripts/compare_results.py \
  --baseline results/baseline_qwen3_4b_sglang_h100_c32_t128.json \
  --speculative results/dspark_qwen3_4b_sglang_h100_c32_b7_t128_static.json \
  --wandb \
  --wandb-project speculative-decoding-benchmark \
  --wandb-run-name qwen3-4b-sglang-dspark-c32
```

## 6. vLLM backup

Baseline server:

```bash
PROFILE=h100 \
FRAMEWORK=vllm \
TARGET_MODEL=Qwen/Qwen3-4B \
DTYPE=bfloat16 \
MAX_NUM_SEQS=32 \
PORT=30000 \
./scripts/launch_baseline.sh
```

DSpark server:

```bash
PROFILE=h100 \
FRAMEWORK=vllm \
TARGET_MODEL=Qwen/Qwen3-4B \
DRAFT_MODEL=deepseek-ai/dspark_qwen3_4b_block7 \
DTYPE=bfloat16 \
NUM_SPECULATIVE_TOKENS=7 \
DSPARK_ATTENTION_BACKEND=FLASH_ATTN \
DRAFT_SAMPLE_METHOD=probabilistic \
MAX_NUM_SEQS=32 \
PORT=30000 \
./scripts/launch_dspark.sh
```

Run the same benchmark commands with `--framework vllm` and different output filenames. Always compare the vLLM DSpark result against the vLLM baseline result, not against SGLang.
