#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible completion/chat server."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata as md
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a running OpenAI-compatible LLM server.")
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--mode", choices=["baseline", "dflash", "dspark"], required=True)
    parser.add_argument("--framework", choices=["sglang", "vllm"], required=True)
    parser.add_argument("--profile", choices=["cheap", "h100"], default="cheap")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="results/baseline.json")
    parser.add_argument("--chat", action="store_true", help="Use /v1/chat/completions instead of /v1/completions.")
    parser.add_argument("--stream", action="store_true", help="Stream responses and measure TTFT.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer for output token counting. Defaults to --model.")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "speculative-decoding-benchmark"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME") or None)
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP") or None)
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS") or None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "online"))
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if "prompt" not in row:
                raise ValueError(f"Missing 'prompt' on line {line_no} of {path}")
            row.setdefault("id", len(rows))
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1 - frac) + values[hi] * frac)


def mean_or_none(values: list[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def safe_version(pkg: str) -> Optional[str]:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def get_gpu_name() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            return ", ".join(names) if names else None
    except Exception:
        pass
    return None


def get_cuda_available() -> Optional[bool]:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return None


def preview(text: str, max_chars: int = 200) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars]


async def request_once(
    *,
    client: AsyncOpenAI,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    is_warmup: bool = False,
) -> dict[str, Any]:
    prompt = str(row["prompt"])
    req_id = row.get("id", 0)
    start = time.perf_counter()
    ttft_s: Optional[float] = None
    completion = ""
    status = "ok"
    error_message: Optional[str] = None

    try:
        if args.chat:
            if args.stream:
                stream = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stream=True,
                    timeout=args.timeout,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        if ttft_s is None:
                            ttft_s = time.perf_counter() - start
                        completion += delta
            else:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stream=False,
                    timeout=args.timeout,
                )
                completion = response.choices[0].message.content or ""
        else:
            if args.stream:
                stream = await client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stream=True,
                    timeout=args.timeout,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].text if chunk.choices else None
                    if delta:
                        if ttft_s is None:
                            ttft_s = time.perf_counter() - start
                        completion += delta
            else:
                response = await client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stream=False,
                    timeout=args.timeout,
                )
                completion = response.choices[0].text or ""

        if not completion.strip():
            status = "empty_output"
            error_message = "Server returned an empty completion."
    except Exception as exc:  # noqa: BLE001 - benchmark should record all request failures.
        status = "error"
        error_message = repr(exc)

    latency_s = time.perf_counter() - start
    try:
        output_tokens = len(tokenizer.encode(completion, add_special_tokens=False)) if completion else 0
    except Exception as exc:  # noqa: BLE001
        output_tokens = 0
        if status == "ok":
            status = "error"
            error_message = f"Token counting failed: {exc!r}"

    tok_s = (output_tokens / latency_s) if latency_s > 0 and output_tokens > 0 else 0.0

    return {
        "id": req_id,
        "latency_s": latency_s,
        "ttft_s": ttft_s,
        "output_tokens": output_tokens,
        "output_tokens_per_second": tok_s,
        "status": status,
        "error_message": error_message,
        "prompt_preview": preview(prompt),
        "completion_preview": preview(completion),
        "warmup": is_warmup,
    }


async def run_warmup(client: AsyncOpenAI, tokenizer: Any, prompts: list[dict[str, Any]], args: argparse.Namespace) -> None:
    warmup_n = min(args.warmup, len(prompts))
    if warmup_n <= 0:
        return
    print(f"Running {warmup_n} warmup requests...")
    for row in prompts[:warmup_n]:
        result = await request_once(client=client, tokenizer=tokenizer, row=row, args=args, is_warmup=True)
        if result["status"] != "ok":
            print(f"Warmup request {result['id']} status={result['status']} error={result['error_message']}", file=sys.stderr)


async def run_benchmark(prompts: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)

    await run_warmup(client, tokenizer, prompts, args)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await request_once(client=client, tokenizer=tokenizer, row=row, args=args)

    print(f"Benchmarking {len(prompts)} prompts with concurrency={args.concurrency}...")
    wall_start = time.perf_counter()
    tasks = [asyncio.create_task(bounded(row)) for row in prompts]
    results: list[dict[str, Any]] = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="requests"):
        results.append(await task)
    wall_time_s = time.perf_counter() - wall_start
    results.sort(key=lambda x: x.get("id", 0))
    return results, wall_time_s


def summarize(results: list[dict[str, Any]], wall_time_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    latencies = [float(r["latency_s"]) for r in ok]
    ttfts = [float(r["ttft_s"]) for r in ok if r.get("ttft_s") is not None]
    total_output_tokens = int(sum(int(r.get("output_tokens", 0)) for r in ok))
    num_prompts = len(results)
    success_count = len(ok)
    error_count = num_prompts - success_count
    return {
        "wall_time_s": wall_time_s,
        "num_prompts": num_prompts,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": (total_output_tokens / wall_time_s) if wall_time_s > 0 else 0.0,
        "requests_per_second": (success_count / wall_time_s) if wall_time_s > 0 else 0.0,
        "mean_latency_s": mean_or_none(latencies),
        "p50_latency_s": percentile(latencies, 0.50),
        "p90_latency_s": percentile(latencies, 0.90),
        "p95_latency_s": percentile(latencies, 0.95),
        "p99_latency_s": percentile(latencies, 0.99),
        "error_count": error_count,
        "success_count": success_count,
        "mean_ttft_s": mean_or_none(ttfts),
        "p50_ttft_s": percentile(ttfts, 0.50),
        "p95_ttft_s": percentile(ttfts, 0.95),
    }


def print_versions() -> dict[str, Any]:
    versions = {pkg: safe_version(pkg) for pkg in ["torch", "transformers", "sglang", "vllm", "wandb"]}
    print("Installed versions:")
    for pkg, version in versions.items():
        print(f"  {pkg}: {version or 'not installed'}")
    cuda_available = get_cuda_available()
    gpu_name = get_gpu_name()
    print(f"CUDA available: {cuda_available}")
    print(f"GPU name: {gpu_name or 'unknown'}")
    return {"versions": versions, "cuda_available": cuda_available, "gpu_name": gpu_name}


def parse_tags(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    tags = [t.strip() for t in value.split(",") if t.strip()]
    return tags or None


def init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("ERROR: --wandb was requested but wandb is not installed. Run: pip install wandb") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        mode=args.wandb_mode,
        config=config,
    )
    return run


def log_wandb(run: Any, summary: dict[str, Any], results: list[dict[str, Any]], out_path: Path) -> None:
    if run is None:
        return
    import wandb

    metrics = {f"summary/{k}": v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None}
    wandb.log(metrics)

    table = wandb.Table(
        columns=[
            "id",
            "prompt_preview",
            "latency_s",
            "ttft_s",
            "output_tokens",
            "output_tokens_per_second",
            "status",
            "error_message",
            "completion_preview",
        ]
    )
    for r in results:
        table.add_data(
            r.get("id"),
            r.get("prompt_preview"),
            r.get("latency_s"),
            r.get("ttft_s"),
            r.get("output_tokens"),
            r.get("output_tokens_per_second"),
            r.get("status"),
            r.get("error_message"),
            r.get("completion_preview"),
        )
    wandb.log({"requests": table})

    artifact = wandb.Artifact(name=f"benchmark-{out_path.stem}", type="benchmark-result")
    artifact.add_file(str(out_path))
    wandb.log_artifact(artifact)

    if getattr(run, "url", None):
        print(f"W&B run URL: {run.url}")
    wandb.finish()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("ERROR: --concurrency must be >= 1")
    if args.max_new_tokens < 1:
        raise SystemExit("ERROR: --max-new-tokens must be >= 1")

    prompts = load_jsonl(args.prompts)
    env_info = print_versions()

    first_prefix_tokens = prompts[0].get("num_prefix_tokens") if prompts else None
    config: dict[str, Any] = {
        "framework": args.framework,
        "mode": args.mode,
        "target_model": args.model,
        "draft_model": args.draft_model,
        "profile": args.profile,
        "dtype": os.getenv("DTYPE", "auto"),
        "base_url": args.base_url,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "seed": None,
        "num_prefix_tokens": first_prefix_tokens,
        "speculative_method": args.mode if args.mode in {"dflash", "dspark"} else None,
        "dflash_block_size": os.getenv("DFLASH_BLOCK_SIZE"),
        "dspark_block_size": os.getenv("DSPARK_BLOCK_SIZE"),
        "num_speculative_tokens": os.getenv("NUM_SPECULATIVE_TOKENS"),
        "draft_sample_method": os.getenv("DRAFT_SAMPLE_METHOD"),
        "dspark_attention_backend": os.getenv("DSPARK_ATTENTION_BACKEND"),
        "sglang_ragged_verify_mode": os.getenv("SGLANG_RAGGED_VERIFY_MODE"),
        "dspark_sps_table_path": os.getenv("DSPARK_SPS_TABLE_PATH"),
        "dspark_sts_path": os.getenv("DSPARK_STS_PATH"),
        "chat": args.chat,
        "stream": args.stream,
        "warmup": args.warmup,
        "prompts_file": str(args.prompts),
        "out_file": str(args.out),
        **env_info,
    }

    run = init_wandb(args, config)

    results, wall_time_s = asyncio.run(run_benchmark(prompts, args))
    summary = summarize(results, wall_time_s)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "summary": summary, "requests": results}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nWrote result JSON to {out_path}")

    log_wandb(run, summary, results, out_path)

    if summary["success_count"] == 0:
        raise SystemExit("ERROR: all benchmark requests failed. Check the server logs and base URL.")


if __name__ == "__main__":
    main()
