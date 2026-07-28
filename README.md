# DFlash and DSpark Speculative Decoding Benchmark

A minimal, model-agnostic workflow for generating benchmark prompts and running DFlash or DSpark through an OpenAI-compatible inference server.

## Prompt generation

Generate a reusable JSONL prompt set from any Hugging Face dataset and tokenizer:

```bash
source .venv/bin/activate

python scripts/make_prompts.py \
  --dataset-name <DATASET_NAME> \
  --split <DATASET_SPLIT> \
  --tokenizer <TARGET_MODEL_OR_TOKENIZER> \
  --num-prompts <NUM_PROMPTS> \
  --num-prefix-tokens <PREFIX_TOKENS> \
  --output prompts.jsonl
```

Example:

```bash
python scripts/make_prompts.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --split train \
  --tokenizer <TARGET_MODEL> \
  --num-prompts 100 \
  --num-prefix-tokens 128 \
  --output prompts.jsonl
```

Use the same prompt file, generation length, temperature, concurrency, hardware, and framework for baseline and speculative runs.

## Benchmark workflow

Use two terminals:

- **Terminal 1:** run the inference server.
- **Terminal 2:** run the benchmark client.

Replace all placeholders with compatible target and draft models.

## DFlash

### Terminal 1 — DFlash server

```bash
cd <REPOSITORY_PATH>
source <SERVER_ENV>/bin/activate

PROFILE=<HARDWARE_PROFILE> \
FRAMEWORK=<FRAMEWORK> \
TARGET_MODEL=<TARGET_MODEL> \
DRAFT_MODEL=<DFLASH_DRAFT_MODEL> \
DTYPE=bfloat16 \
DFLASH_BLOCK_SIZE=<BLOCK_SIZE> \
MAX_RUNNING_REQUESTS=<MAX_REQUESTS> \
PORT=30000 \
./scripts/launch_dflash.sh
```

### Terminal 2 — DFlash benchmark

```bash
cd <REPOSITORY_PATH>
source .venv/bin/activate

python -u scripts/bench_openai_server.py \
  --mode dflash \
  --framework <FRAMEWORK> \
  --profile <HARDWARE_PROFILE> \
  --model <TARGET_MODEL> \
  --draft-model <DFLASH_DRAFT_MODEL> \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens <MAX_NEW_TOKENS> \
  --temperature 0.0 \
  --concurrency <CONCURRENCY> \
  --out results/<RUN_NAME>.json \
  2>&1 | tee results/<RUN_NAME>.log
```

## DSpark

### Terminal 1 — vLLM DSpark server

```bash
cd <REPOSITORY_PATH>
source <VLLM_ENV>/bin/activate

vllm serve <TARGET_MODEL> \
  --dtype bfloat16 \
  --max-num-seqs <MAX_REQUESTS> \
  --gpu-memory-utilization <GPU_MEMORY_FRACTION> \
  --speculative-config '{
    "method": "dspark",
    "model": "<DSPARK_DRAFT_MODEL>",
    "num_speculative_tokens": <BLOCK_SIZE>,
    "attention_backend": "FLASH_ATTN",
    "draft_sample_method": "probabilistic"
  }' \
  --host 0.0.0.0 \
  --port 30000
```

### Terminal 1 — SGLang DSpark server

```bash
cd <REPOSITORY_PATH>
source <SGLANG_ENV>/bin/activate

SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
python -m sglang.launch_server \
  --model-path <TARGET_MODEL> \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path <DSPARK_DRAFT_MODEL> \
  --speculative-dspark-block-size <BLOCK_SIZE> \
  --tp-size 1 \
  --dtype bfloat16 \
  --mem-fraction-static <GPU_MEMORY_FRACTION> \
  --cuda-graph-max-bs <MAX_REQUESTS> \
  --max-running-requests <MAX_REQUESTS> \
  --host 0.0.0.0 \
  --port 30000
```

Run only one server at a time.

### Terminal 2 — DSpark benchmark

```bash
cd <REPOSITORY_PATH>
source .venv/bin/activate

python -u scripts/bench_openai_server.py \
  --mode dspark \
  --framework <vllm|sglang> \
  --profile <HARDWARE_PROFILE> \
  --model <TARGET_MODEL> \
  --draft-model <DSPARK_DRAFT_MODEL> \
  --prompts prompts.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --max-new-tokens <MAX_NEW_TOKENS> \
  --temperature 0.0 \
  --concurrency <CONCURRENCY> \
  --out results/<RUN_NAME>.json \
  2>&1 | tee results/<RUN_NAME>.log
```

## Optional W&B logging

Add these arguments to any benchmark command:

```bash
--wandb \
--wandb-project <PROJECT_NAME> \
--wandb-tags <COMMA_SEPARATED_TAGS>
```

## Next steps

- Test DFlash and DSpark on budget GPUs.
- Benchmark NVIDIA Blackwell-family GPUs.
- Add standardized load testing with GuideLLM.
