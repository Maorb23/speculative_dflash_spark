#!/usr/bin/env python3
"""Create JSONL benchmark prompts from streaming FineWeb-Edu samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create benchmark prefix prompts from a streaming dataset.")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--num-prefix-tokens", type=int, default=128)
    parser.add_argument("--output", default="prompts.jsonl")
    parser.add_argument("--chat-template", action="store_true", help="Wrap each prefix with tokenizer.apply_chat_template.")
    parser.add_argument("--min-prefix-tokens", type=int, default=16, help="Skip samples shorter than this many tokens.")
    return parser.parse_args()


def get_text(sample: dict[str, Any]) -> str:
    for key in ("text", "content", "document", "raw_content"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in sample.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def make_prompt(text: str, tokenizer: Any, num_prefix_tokens: int, use_chat_template: bool) -> tuple[str, int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids[:num_prefix_tokens]
    prefix = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "--chat-template was passed, but the selected tokenizer has no chat_template. "
                "Use plain completion prompts or choose a chat tokenizer."
            )
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix}],
            tokenize=False,
            add_generation_prompt=True,
        ).strip()

    return prefix, len(token_ids)


def iter_prompts(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    next_id = 0
    for sample in dataset:
        text = get_text(sample)
        if not text.strip():
            continue
        try:
            prompt, n_tokens = make_prompt(
                text=text,
                tokenizer=tokenizer,
                num_prefix_tokens=args.num_prefix_tokens,
                use_chat_template=args.chat_template,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed while tokenizing sample {next_id}: {exc}") from exc

        if not prompt or n_tokens < args.min_prefix_tokens:
            continue

        yield {
            "id": next_id,
            "prompt": prompt,
            "num_prefix_tokens": n_tokens,
        }
        next_id += 1
        if next_id >= args.num_prompts:
            break


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(iter_prompts(args), total=args.num_prompts, desc="writing prompts"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    if count < args.num_prompts:
        raise RuntimeError(
            f"Only wrote {count} prompts, requested {args.num_prompts}. "
            "Try lowering --min-prefix-tokens or using a larger dataset split."
        )

    print(f"Wrote {count} prompts to {out_path}")


if __name__ == "__main__":
    main()
