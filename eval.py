#!/usr/bin/env python3
"""
eval.py – Batch evaluation of fine-tuned Arasaka bot.

Usage:
    python3 eval.py
    python3 eval.py --adapters adapters/arasaka-light --data data/valid.jsonl --samples 10
"""

import argparse
import json
import random
from pathlib import Path

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    print("mlx_lm not installed. Run: pip install mlx-lm")
    exit(1)


START_USER = "<start_of_turn>user"
END_TURN = "<end_of_turn>"
START_MODEL = "<start_of_turn>model"


def load_samples(path: Path, n: int) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if n > 0 and n < len(items):
        items = random.sample(items, n)
    return items


def extract_question(text: str) -> str:
    # Extract user turn from formatted text
    try:
        start = text.index(START_USER) + len(START_USER) + 1  # skip newline
        end = text.index(END_TURN, start)
        return text[start:end].strip()
    except ValueError:
        return text


def build_prompt(question: str) -> str:
    return f"{START_USER}\n{question}{END_TURN}\n{START_MODEL}\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Arasaka model")
    # Default to a 4-bit base model to stay within 16GB unified memory on Apple Silicon.
    parser.add_argument("--model", type=str, default="mlx-community/gemma-2-9b-it-4bit")
    parser.add_argument(
        "--adapters",
        type=str,
        nargs="*",
        default=None,
        help="List of paths to LoRA adapters to compare (default: auto-detect common adapters/* folders)",
    )
    parser.add_argument("--data", type=str, default="data/valid.jsonl", help="Path to JSONL file")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic / greedy).",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    samples = load_samples(data_path, args.samples)

    if args.adapters is None or len(args.adapters) == 0:
        candidates = [
            "adapters/arasaka-light",
            "adapters/arasaka-medium",
            "adapters/arasaka-heavy",
            "adapters/arasaka-gemma2-9b",
        ]
        adapter_paths_input = candidates
    else:
        adapter_paths_input = args.adapters

    adapter_paths = []
    for raw_path in adapter_paths_input:
        p = Path(raw_path)
        if not p.exists():
            print(f"Warning: adapter path not found, skipping: {raw_path}")
            continue
        adapter_paths.append(str(p))

    if not adapter_paths:
        raise SystemExit(
            "No valid adapter paths found. Train an adapter first (e.g. adapters/arasaka-light) "
            "or pass explicit paths via --adapters."
        )

    # Dictionary to store results for each adapter
    all_responses = {adapter: [] for adapter in adapter_paths}

    sampler = make_sampler(temp=args.temp)

    for adapter_path in adapter_paths:
        print(f"Loading adapter: {adapter_path}...")
        model, tokenizer = load(args.model, adapter_path=adapter_path)
        
        for i, item in enumerate(samples, 1):
            question = extract_question(item["text"])
            prompt = build_prompt(question)
            
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            all_responses[adapter_path].append(response.strip())
        
        # Cleanup (optional but good for memory if running many adapters)
        del model
        import gc
        gc.collect()

    print(f"\n{'='*80}")
    print(f"PROPAGANDA SATURATION COMPARISON ({len(samples)} samples)")
    print(f"{'='*80}\n")

    for i, item in enumerate(samples):
        question = extract_question(item["text"])
        print(f"QUESTION #{i+1}: {question}\n")
        
        for adapter_path in adapter_paths:
            name = Path(adapter_path).name
            print(f" >>> VERSION [{name.upper()}]:")
            print(f"  {all_responses[adapter_path][i]}\n")
        
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
