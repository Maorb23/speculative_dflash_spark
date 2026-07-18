#!/usr/bin/env python3
"""Compare a baseline result with a DFlash or DSpark result."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and speculative benchmark results.")
    parser.add_argument("--baseline", required=True)
    spec_group = parser.add_mutually_exclusive_group(required=True)
    spec_group.add_argument("--speculative", help="DFlash or DSpark result JSON.")
    spec_group.add_argument(
        "--dflash",
        dest="legacy_dflash",
        help="Deprecated alias for --speculative, kept for old commands.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "speculative-decoding-benchmark"),
    )
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME") or None)
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP") or None)
    parser.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS") or None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.getenv("WANDB_MODE", "online"),
    )
    args = parser.parse_args()
    if args.legacy_dflash:
        warnings.warn("--dflash is deprecated; use --speculative.", DeprecationWarning, stacklevel=2)
        args.speculative = args.legacy_dflash
    return args


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
    summary = data["summary"]
    return [
        label,
        fnum(summary.get("num_prompts"), 0),
        fnum(cfg.get("max_new_tokens"), 0),
        fnum(cfg.get("concurrency"), 0),
        fnum(summary.get("total_output_tokens"), 0),
        fnum(summary.get("wall_time_s")),
        fnum(summary.get("output_tokens_per_second")),
        fnum(summary.get("p50_latency_s")),
        fnum(summary.get("p95_latency_s")),
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
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def parse_tags(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    tags = [tag.strip() for tag in value.split(",") if tag.strip()]
    return tags or None


def init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("ERROR: --wandb was requested but wandb is not installed.") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        mode=args.wandb_mode,
        config=config,
    )


def warn_if_not_comparable(baseline: dict[str, Any], speculative: dict[str, Any]) -> None:
    bcfg = baseline.get("config", {})
    scfg = speculative.get("config", {})
    keys = ["framework", "target_model", "profile", "max_new_tokens", "concurrency", "temperature"]
    differences = [(key, bcfg.get(key), scfg.get(key)) for key in keys if bcfg.get(key) != scfg.get(key)]
    if differences:
        print("\nWARNING: benchmark configurations differ:")
        for key, baseline_value, speculative_value in differences:
            print(f"  {key}: baseline={baseline_value!r}, speculative={speculative_value!r}")
        print("Speedup is most meaningful when framework, model, workload, and concurrency match.\n")


def main() -> None:
    args = parse_args()
    baseline = load_result(args.baseline)
    speculative = load_result(args.speculative)
    warn_if_not_comparable(baseline, speculative)

    baseline_summary = baseline["summary"]
    speculative_summary = speculative["summary"]
    speculative_mode = speculative.get("config", {}).get("mode") or "speculative"

    speedup = div(
        speculative_summary.get("output_tokens_per_second"),
        baseline_summary.get("output_tokens_per_second"),
    )
    latency_ratio_p50 = div(
        speculative_summary.get("p50_latency_s"),
        baseline_summary.get("p50_latency_s"),
    )
    latency_ratio_p95 = div(
        speculative_summary.get("p95_latency_s"),
        baseline_summary.get("p95_latency_s"),
    )

    print_table(
        [
            result_row("baseline", baseline, 1.0),
            result_row(str(speculative_mode), speculative, speedup),
        ]
    )

    print("\n=== Key comparison ===")
    print(f"baseline output tok/s:    {fnum(baseline_summary.get('output_tokens_per_second'))}")
    print(f"{speculative_mode} output tok/s: {fnum(speculative_summary.get('output_tokens_per_second'))}")
    print(f"speedup:                  {fnum(speedup)}")
    print(f"baseline p50 latency:     {fnum(baseline_summary.get('p50_latency_s'))}")
    print(f"{speculative_mode} p50 latency:  {fnum(speculative_summary.get('p50_latency_s'))}")
    print(f"baseline p95 latency:     {fnum(baseline_summary.get('p95_latency_s'))}")
    print(f"{speculative_mode} p95 latency:  {fnum(speculative_summary.get('p95_latency_s'))}")
    print(f"latency ratio p50:        {fnum(latency_ratio_p50)}")
    print(f"latency ratio p95:        {fnum(latency_ratio_p95)}")

    comparison = {
        "comparison/speculative_mode": speculative_mode,
        "comparison/speedup_output_tok_s": speedup,
        "comparison/latency_ratio_p50": latency_ratio_p50,
        "comparison/latency_ratio_p95": latency_ratio_p95,
        "comparison/baseline_output_tok_s": baseline_summary.get("output_tokens_per_second"),
        "comparison/speculative_output_tok_s": speculative_summary.get("output_tokens_per_second"),
        "comparison/baseline_p50_latency_s": baseline_summary.get("p50_latency_s"),
        "comparison/speculative_p50_latency_s": speculative_summary.get("p50_latency_s"),
        "comparison/baseline_p95_latency_s": baseline_summary.get("p95_latency_s"),
        "comparison/speculative_p95_latency_s": speculative_summary.get("p95_latency_s"),
    }

    run = init_wandb(
        args,
        {
            "baseline_file": args.baseline,
            "speculative_file": args.speculative,
            "speculative_mode": speculative_mode,
            "baseline_config": baseline.get("config", {}),
            "speculative_config": speculative.get("config", {}),
        },
    )
    if run is not None:
        import wandb

        wandb.log({key: value for key, value in comparison.items() if value is not None})
        artifact = wandb.Artifact(
            name=f"{speculative_mode}-comparison-inputs",
            type="benchmark-comparison",
        )
        artifact.add_file(args.baseline)
        artifact.add_file(args.speculative)
        wandb.log_artifact(artifact)
        if getattr(run, "url", None):
            print(f"W&B run URL: {run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
