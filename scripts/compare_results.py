#!/usr/bin/env python3
"""Compare baseline and DFlash benchmark result JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and DFlash benchmark results.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dflash", required=True)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "dflash-benchmark"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME") or None)
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP") or None)
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS") or None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "online"))
    return parser.parse_args()


def load_result(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ["config", "summary", "requests"]:
        if key not in data:
            raise ValueError(f"Invalid result file {p}: missing '{key}'")
    return data


def div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def fnum(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def result_row(label: str, data: dict[str, Any], speedup: Optional[float]) -> list[str]:
    cfg = data["config"]
    s = data["summary"]
    return [
        label,
        fnum(s.get("num_prompts"), 0),
        fnum(cfg.get("max_new_tokens"), 0),
        fnum(cfg.get("concurrency"), 0),
        fnum(s.get("total_output_tokens"), 0),
        fnum(s.get("wall_time_s")),
        fnum(s.get("output_tokens_per_second")),
        fnum(s.get("p50_latency_s")),
        fnum(s.get("p95_latency_s")),
        fnum(speedup),
    ]


def print_table(rows: list[list[str]]) -> None:
    headers = [
        "mode",
        "prompts",
        "max_new_tokens",
        "concurrency",
        "total_output_tokens",
        "wall_time_s",
        "output_tok_s",
        "p50_latency",
        "p95_latency",
        "speedup",
    ]
    all_rows = [headers] + rows
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


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

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        mode=args.wandb_mode,
        config=config,
    )


def main() -> None:
    args = parse_args()
    baseline = load_result(args.baseline)
    dflash = load_result(args.dflash)

    b = baseline["summary"]
    d = dflash["summary"]

    speedup = div(d.get("output_tokens_per_second"), b.get("output_tokens_per_second"))
    latency_ratio_p50 = div(d.get("p50_latency_s"), b.get("p50_latency_s"))
    latency_ratio_p95 = div(d.get("p95_latency_s"), b.get("p95_latency_s"))

    rows = [
        result_row("baseline", baseline, 1.0),
        result_row("dflash", dflash, speedup),
    ]
    print_table(rows)

    print("\n=== Key comparison ===")
    print(f"baseline output tok/s: {fnum(b.get('output_tokens_per_second'))}")
    print(f"dflash output tok/s:   {fnum(d.get('output_tokens_per_second'))}")
    print(f"speedup:              {fnum(speedup)}")
    print(f"baseline p50 latency: {fnum(b.get('p50_latency_s'))}")
    print(f"dflash p50 latency:   {fnum(d.get('p50_latency_s'))}")
    print(f"baseline p95 latency: {fnum(b.get('p95_latency_s'))}")
    print(f"dflash p95 latency:   {fnum(d.get('p95_latency_s'))}")
    print(f"latency ratio p50:    {fnum(latency_ratio_p50)}")
    print(f"latency ratio p95:    {fnum(latency_ratio_p95)}")

    comparison = {
        "comparison/speedup_output_tok_s": speedup,
        "comparison/latency_ratio_p50": latency_ratio_p50,
        "comparison/latency_ratio_p95": latency_ratio_p95,
        "comparison/baseline_output_tok_s": b.get("output_tokens_per_second"),
        "comparison/dflash_output_tok_s": d.get("output_tokens_per_second"),
        "comparison/baseline_p50_latency_s": b.get("p50_latency_s"),
        "comparison/dflash_p50_latency_s": d.get("p50_latency_s"),
        "comparison/baseline_p95_latency_s": b.get("p95_latency_s"),
        "comparison/dflash_p95_latency_s": d.get("p95_latency_s"),
    }

    run = init_wandb(
        args,
        {
            "baseline_file": args.baseline,
            "dflash_file": args.dflash,
            "baseline_config": baseline.get("config", {}),
            "dflash_config": dflash.get("config", {}),
        },
    )
    if run is not None:
        import wandb

        wandb.log({k: v for k, v in comparison.items() if v is not None})
        artifact = wandb.Artifact(name="dflash-comparison-inputs", type="benchmark-comparison")
        artifact.add_file(args.baseline)
        artifact.add_file(args.dflash)
        wandb.log_artifact(artifact)
        if getattr(run, "url", None):
            print(f"W&B run URL: {run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
