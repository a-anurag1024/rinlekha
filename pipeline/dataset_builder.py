#!/usr/bin/env python3
"""
pipeline/dataset_builder.py — Phase 1, Step 4

Converts QC-passed memos into Alpaca-style instruction pairs, performs a
stratified 80/10/10 train/validation/test split by outcome class, saves
splits to JSONL, and (optionally) pushes to HuggingFace Hub.

Instruction format (Alpaca):
  {
    "instruction": "<task description>",   # identical for every row
    "input":       "<formatted profile>",  # borrower profile text
    "output":      "<credit memo>",        # synthesised memo
    # metadata (not used in training — useful for analysis / stratification)
    "outcome":         "APPROVE" | "CONDITIONAL_APPROVE" | "DECLINE",
    "cibil_band":      "excellent" | "good" | "fair" | "poor",
    "foir_band":       "low" | "moderate" | "high" | "excessive",
    "employment_type": str,
    "loan_purpose":    str,
    "profile_id":      str,
  }

Usage:
    # Build from defaults (reads memos, writes processed splits)
    python pipeline/dataset_builder.py

    # Only build locally — skip HF push
    python pipeline/dataset_builder.py --no-push

    # Custom paths
    python pipeline/dataset_builder.py \\
        --memos data/raw/memos.jsonl \\
        --output-dir data/processed \\
        --hf-repo your-username/rinlekha-training-data
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.memo_synthesizer import format_profile_as_readable_text, load_memos

# ─── Constants ────────────────────────────────────────────────────────────────

INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)

# Train / validation / test fractions (must sum to 1.0)
SPLIT_RATIOS: dict[str, float] = {"train": 0.80, "validation": 0.10, "test": 0.10}


# ─── Metadata helpers ─────────────────────────────────────────────────────────

def get_cibil_band(score: int) -> str:
    """
    Map a CIBIL score to a named band.

    Bands align with typical NBFC underwriting tiers:
      excellent : 760–900
      good      : 700–759
      fair      : 620–699
      poor      : < 620
    """
    if score >= 760:
        return "excellent"
    if score >= 700:
        return "good"
    if score >= 620:
        return "fair"
    return "poor"


def get_foir_band(foir: float) -> str:
    """
    Map a post-loan FOIR to a named band.

      low      : ≤ 0.30
      moderate : 0.31 – 0.45
      high     : 0.46 – 0.55
      excessive: > 0.55
    """
    if foir <= 0.30:
        return "low"
    if foir <= 0.45:
        return "moderate"
    if foir <= 0.55:
        return "high"
    return "excessive"


# ─── Format conversion ────────────────────────────────────────────────────────

def format_as_instruction_pair(example: dict) -> dict:
    """
    Convert a raw synthesised example to an Alpaca-style instruction pair.

    Args:
        example: dict with keys ``input_profile`` (dict) and
                 ``output_memo`` (str).  ``profile_id`` is also accepted
                 at the top level (falls back to ``input_profile``).

    Returns:
        dict with keys: instruction, input, output, outcome, cibil_band,
        foir_band, employment_type, loan_purpose, profile_id.
    """
    profile = example["input_profile"]
    profile_text = format_profile_as_readable_text(profile)

    return {
        "instruction": INSTRUCTION,
        "input": profile_text,
        "output": example["output_memo"],
        # Metadata
        "outcome": profile["outcome"],
        "cibil_band": get_cibil_band(profile["cibil_score"]),
        "foir_band": get_foir_band(profile["foir_post_loan"]),
        "employment_type": profile["employment_type"],
        "loan_purpose": profile["loan_purpose"],
        "profile_id": profile.get("profile_id", example.get("profile_id", "")),
    }


# ─── Filtering ────────────────────────────────────────────────────────────────

def filter_qc_passed(examples: list[dict]) -> list[dict]:
    """
    Return only examples that passed QC.

    Accepts examples in two shapes:
    1. Enriched by quality_reviewer (has ``qc.structural_pass`` key)
    2. Raw memos without QC metadata — kept as-is (caller's responsibility)
    """
    result = []
    for ex in examples:
        qc = ex.get("qc")
        if qc is None or qc.get("structural_pass", True):
            result.append(ex)
    return result


# ─── Stratified split ─────────────────────────────────────────────────────────

def stratified_split(
    examples: list[dict],
    label_key: str = "outcome",
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Partition ``examples`` into splits while preserving the per-label
    distribution.

    Each class is shuffled independently then cut at the ratio boundaries,
    so every split contains a proportional sample of each outcome class.

    Args:
        examples: list of dicts — must each have a ``label_key`` field.
        label_key: field name to stratify on.
        ratios: dict mapping split name → fraction (must sum to 1.0).
                Defaults to SPLIT_RATIOS (80/10/10).
        seed: RNG seed for reproducibility.

    Returns:
        dict mapping split name → list of examples.
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    rng = random.Random(seed)

    # Group by label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_label[ex[label_key]].append(ex)

    # Build splits class-by-class
    splits: dict[str, list[dict]] = {name: [] for name in ratios}
    split_names = list(ratios.keys())

    for label, group in by_label.items():
        group = group.copy()
        rng.shuffle(group)
        n = len(group)
        start = 0
        for i, name in enumerate(split_names):
            if i == len(split_names) - 1:
                # Last split gets all remaining to avoid rounding loss
                splits[name].extend(group[start:])
            else:
                end = start + round(n * ratios[name])
                splits[name].extend(group[start:end])
                start = end

    # Shuffle each split so classes are interleaved
    for name in splits:
        rng.shuffle(splits[name])

    return splits


# ─── Serialisation ────────────────────────────────────────────────────────────

def save_splits(
    splits: dict[str, list[dict]],
    output_dir: str | Path,
) -> None:
    """Write each split to ``<output_dir>/<split_name>.jsonl``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Saved {len(rows):>4} examples → {path}")


def load_split(path: str | Path) -> list[dict]:
    """Load a JSONL split file."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ─── HuggingFace push ─────────────────────────────────────────────────────────

def push_to_hub(
    splits: dict[str, list[dict]],
    hf_repo: str,
    hf_token: str | None = None,
    commit_message: str = "v1.0 — synthetic credit memos, stratified split",
) -> None:
    """
    Push ``splits`` to HuggingFace Hub as a DatasetDict.

    Requires ``datasets`` package.  ``hf_token`` defaults to the
    ``HUGGINGFACE_TOKEN`` environment variable if not provided.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace push. "
            "Install it with: pip install datasets"
        ) from exc

    token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")

    dataset_dict = DatasetDict(
        {name: Dataset.from_list(rows) for name, rows in splits.items()}
    )
    dataset_dict.push_to_hub(hf_repo, token=token, commit_message=commit_message)

    print(f"\nDataset pushed: https://huggingface.co/datasets/{hf_repo}")
    for name, rows in splits.items():
        print(f"  {name}: {len(rows)} examples")


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def build_dataset(
    examples: list[dict],
    output_dir: str | Path = "data/processed",
    hf_repo: str | None = None,
    hf_token: str | None = None,
    split_seed: int = 42,
) -> dict[str, list[dict]]:
    """
    End-to-end: filter → format → split → save → (optionally) push.

    Returns the formatted + split dict for further inspection.
    """
    # 1. Filter to QC-passed examples only
    passed = filter_qc_passed(examples)
    print(f"QC filter: {len(passed)}/{len(examples)} examples passed")

    # 2. Format as instruction pairs
    formatted = [format_as_instruction_pair(ex) for ex in passed]
    print(f"Formatted {len(formatted)} instruction pairs")

    # 3. Stratified split
    splits = stratified_split(formatted, label_key="outcome", seed=split_seed)
    counts = {name: len(rows) for name, rows in splits.items()}
    print(f"Split: {counts}")

    # 4. Save locally
    print(f"Saving splits to {output_dir} …")
    save_splits(splits, output_dir)

    # 5. Optionally push to HF Hub
    if hf_repo:
        print(f"Pushing to HuggingFace Hub ({hf_repo}) …")
        push_to_hub(splits, hf_repo, hf_token=hf_token)

    return splits


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build instruction-tuning dataset from synthesised memos."
    )
    parser.add_argument(
        "--memos",
        type=str,
        default="data/raw/memos.jsonl",
        help="Path to synthesised memos JSONL (default: data/raw/memos.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for split JSONL files (default: data/processed)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g. username/rinlekha-training-data). "
             "Omit to skip push.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip HuggingFace push even if --hf-repo is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for stratified split (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading memos from {args.memos} …")
    examples = load_memos(args.memos)
    print(f"Loaded {len(examples)} examples.")

    hf_repo = None if args.no_push else args.hf_repo
    build_dataset(
        examples,
        output_dir=args.output_dir,
        hf_repo=hf_repo,
        split_seed=args.seed,
    )


if __name__ == "__main__":
    main()
