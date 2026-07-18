What we did
Cloned the repo and applied the DSpark patch
Added launch_dspark.sh
Updated baseline and benchmark scripts
Added environment setup and DSpark documentation

Created three isolated Python environments

.venv — benchmark client, prompt generation, W&B
.venv-vllm-dspark — vLLM server
.venv-sglang-dspark — SGLang server

This prevented Torch, CUDA, vLLM, and SGLang dependencies from conflicting.

Generated one shared benchmark dataset

100 prompts
Qwen3-4B tokenizer
128 input tokens
Saved as:
prompts_qwen3_4b.jsonl
Fixed the vLLM setup
vLLM installed correctly
Startup initially failed because TorchCodec could not find FFmpeg libraries
Installing system FFmpeg fixed the issue
vLLM then loaded the model, compiled kernels, captured CUDA graphs, and started its API server

Fixed the SGLang setup

The original validation script used an incorrect internal Python import
The exact old commit could not be checked out, so we built the current SGLang source
Installed Rust because SGLang’s editable build required it
Installed SGLang with no build isolation
Verified that its CLI explicitly supports:
DSPARK
--speculative-dspark-block-size
--speculative-draft-model-path

Ran four benchmark arms

vLLM baseline
vLLM DSpark
SGLang baseline
SGLang DSpark

Terminal 1 always ran the server in the framework-specific environment, while Terminal 2 ran the benchmark client from .venv.

Uploaded all runs to W&B
Every benchmark used --wandb
Both framework comparisons were also uploaded
W&B artifacts and metrics synced successfully
Main results

vLLM

Baseline: about 4,541 output tokens/s
DSpark: about 7,142 output tokens/s
Speedup: 1.57×

SGLang

Baseline: about 4,424 output tokens/s
DSpark: about 4,765 output tokens/s
Speedup: 1.08×

So DSpark produced a strong gain in vLLM and a smaller throughput gain in SGLang under this specific configuration.

Files saved

Benchmark JSON files were saved in:
