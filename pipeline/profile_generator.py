#!/usr/bin/env python3
"""
pipeline/profile_generator.py — Phase 1, Step 1

Generates 1 000 synthetic borrower profiles for the RinLekha training dataset,
with a controlled outcome distribution (APPROVE 20% / CONDITIONAL 55% / DECLINE 25%).

This file owns orchestration and outcome-class sampling strategies.
It contains NO domain knowledge — all policy rules live in rules.py
and all field distributions live in samplers.py.

Pure random sampling produces almost no APPROVE profiles (~0.5% natural rate)
because APPROVE requires CIBIL ≥ 760 AND FOIR ≤ 0.38 AND 0 missed payments
AND tenure ≥ 2 years AND 0 settled accounts — all simultaneously.
Each outcome class therefore has its own targeted sampler that constrains
the relevant dimensions and falls back to rejection-sampling for the rest.

Usage:
    python pipeline/profile_generator.py                          # 1000 profiles, default output
    python pipeline/profile_generator.py --total 1000 --seed 42
    python pipeline/profile_generator.py --output data/raw/profiles_v1.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Allow running as `python pipeline/profile_generator.py` from the project root
# as well as `python -m pipeline.profile_generator`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray

from pipeline.rules import (
    CLEAN_APPROVE_RULE,
    COMPATIBLE_TRIGGER_PAIRS,
    CONDITIONAL_FALLBACK,
    CONDITIONAL_RULES,
    DECLINE_TRIGGERS,
    DERIVED_FIELD_RULES,
    HARD_DECLINE_RULES,
)
from pipeline.samplers import (
    Choice,
    Fixed,
    PROFILE_SCHEMA,
    UniformFloat,
    UniformInt,
    sample_base_profile,
)

# ─── Outcome distribution targets ─────────────────────────────────────────────

OUTCOME_TARGETS: dict[str, float] = {
    "APPROVE":             0.20,
    "CONDITIONAL_APPROVE": 0.55,
    "DECLINE":             0.25,
}


# ─── Derived fields ────────────────────────────────────────────────────────────

def compute_derived_fields(profile: dict) -> dict:
    """
    Apply every DerivedFieldRule from rules.DERIVED_FIELD_RULES in order.
    Rules are applied sequentially so later ones can reference fields added
    by earlier ones (e.g. foir_post_loan depends on proposed_emi).
    To add a new derived field, append a DerivedFieldRule to rules.py — done.
    """
    for rule in DERIVED_FIELD_RULES:
        profile[rule.name] = rule.compute(profile)
    return profile


# ─── Underwriting decision ─────────────────────────────────────────────────────

def determine_outcome(profile: dict) -> tuple[str, list[str]]:
    """
    Apply NBFC underwriting policy rules to a fully-derived profile.
    Returns (decision, conditions_list).

    All logic lives in rules.py — this function is pure rule iteration:
      1. Collect all triggered hard-decline conditions (all fire, not just first).
      2. Check clean-approve criteria.
      3. Collect all triggered conditional requirements.
    To change policy, edit the rule registries in rules.py — done.
    """
    # ── Hard declines ──────────────────────────────────────────────────────────
    decline_reasons = [
        r.condition_key for r in HARD_DECLINE_RULES if r.predicate(profile)
    ]
    if decline_reasons:
        return "DECLINE", decline_reasons

    # ── Clean approve ──────────────────────────────────────────────────────────
    if CLEAN_APPROVE_RULE.satisfied(profile):
        return "APPROVE", []

    # ── Conditional approve ────────────────────────────────────────────────────
    conditions = [
        r.condition_key for r in CONDITIONAL_RULES if r.predicate(profile)
    ]
    return "CONDITIONAL_APPROVE", conditions or [CONDITIONAL_FALLBACK]


# ─── Outcome-class samplers ───────────────────────────────────────────────────
# These functions encode *sampling strategy*, not domain policy.
# Domain knowledge (thresholds, conditions) lives in rules.py.

def _sample_approve(rng: random.Random, max_retries: int = 60) -> dict | None:
    """
    Generate a profile that receives an APPROVE decision.

    Strategy: override the five APPROVE criteria in the base schema, then
    back-calculate a safe loan amount from a FOIR budget of ≤ 0.36
    (2pp margin below the 0.38 threshold in rules.py).
    The back-calculation is intentionally here, not in rules.py, because
    it is a sampling constraint, not a policy rule.
    """
    for _ in range(max_retries):
        p = sample_base_profile(rng, overrides={
            "cibil_score":            UniformInt(760, 900),
            "missed_payments_24m":    Fixed(0),
            "settled_accounts_ever":  Fixed(0),
            "loan_accounts_active":   UniformInt(0, 4),
            "employment_tenure_years": UniformFloat(2.0, 25.0, precision=2),
            # Healthy income to give FOIR headroom
            "monthly_income":         UniformFloat(50_000, 5_00_000, precision=0),
            # Rates are lower for strong profiles
            "annual_interest_rate":   UniformFloat(10.5, 18.0, precision=2),
        })

        # existing_emi depends on sampled income — must be set post-sampling
        p["existing_emi_monthly"] = round(rng.uniform(0, p["monthly_income"] * 0.18))

        # Back-calculate maximum safe loan amount from FOIR ceiling
        foir_ceiling  = 0.36
        max_new_emi   = p["monthly_income"] * foir_ceiling - p["existing_emi_monthly"]
        if max_new_emi <= 0:
            continue

        r        = p["annual_interest_rate"] / 12 / 100
        n        = p["loan_tenure_months"]
        max_loan = max_new_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        if max_loan < 50_000:
            continue

        p["loan_amount"] = round(rng.uniform(50_000, min(max_loan * 0.90, 50_00_000)))

        p = compute_derived_fields(p)
        outcome, conditions = determine_outcome(p)
        if outcome == "APPROVE":
            p["outcome"]    = outcome
            p["conditions"] = conditions
            return p

    return None


def _sample_decline(rng: random.Random) -> dict:
    """
    Generate a profile with one or more hard-decline triggers injected.

    60% single trigger / 40% compatible pair.
    Compatible pairs are derived from each trigger's `compatible_with` list
    in rules.DECLINE_TRIGGERS — no separate list to maintain here.

    Base profile is sampled from the full schema with non-trigger fields
    set to non-pathological ranges, giving the model varied surrounding
    context for each decline pattern.
    """
    p = sample_base_profile(rng, overrides={
        "cibil_score":           UniformInt(620, 900),
        "missed_payments_24m":   UniformInt(0, 2),
        "settled_accounts_ever": UniformInt(0, 1),
    })
    # existing_emi depends on monthly_income — set as a fraction post-sampling
    p["existing_emi_monthly"] = round(rng.uniform(0, p["monthly_income"] * 0.30))

    # Select triggers
    if rng.random() < 0.40 and COMPATIBLE_TRIGGER_PAIRS:
        triggers = list(rng.choice(COMPATIBLE_TRIGGER_PAIRS))
    else:
        triggers = [rng.choice(DECLINE_TRIGGERS)]

    for trigger in triggers:
        p = trigger.apply(p, rng)

    p.setdefault("loan_amount", round(rng.uniform(50_000, 50_00_000)))
    return p


def _sample_conditional(rng: random.Random, max_retries: int = 80) -> dict | None:
    """
    Generate a CONDITIONAL_APPROVE profile via rejection sampling.

    Pre-constrain CIBIL (620–759) and behavioural fields to avoid burning
    retries on obvious DECLINE/APPROVE outcomes. The actual decision arbiter
    is determine_outcome — no policy logic is duplicated here.
    """
    for _ in range(max_retries):
        p = sample_base_profile(rng, overrides={
            # Avoids hard CIBIL decline (<620) and clean-approve floor (≥760)
            "cibil_score":           UniformInt(620, 759),
            "missed_payments_24m":   UniformInt(0, 3),   # 4+ → hard decline
            "settled_accounts_ever": UniformInt(0, 1),   # 2+ → hard decline
        })
        # existing_emi depends on monthly_income — set post-sampling
        p["existing_emi_monthly"] = round(rng.uniform(0, p["monthly_income"] * 0.40))

        p = compute_derived_fields(p)
        outcome, conditions = determine_outcome(p)
        if outcome == "CONDITIONAL_APPROVE":
            p["outcome"]    = outcome
            p["conditions"] = conditions
            return p

    return None


# ─── Ray batch worker ──────────────────────────────────────────────────────────

@ray.remote
def generate_profile_batch(
    n_approve: int,
    n_conditional: int,
    n_decline: int,
    seed: int,
) -> list[dict]:
    """
    Ray remote worker: generates the requested count of each outcome class.
    Profile IDs encode the worker seed for traceability.
    """
    rng     = random.Random(seed)
    profiles: list[dict] = []
    counter = 0

    def _tag(p: dict) -> dict:
        nonlocal counter
        p["profile_id"] = f"profile_{seed}_{counter:04d}"
        counter += 1
        return p

    for _ in range(n_approve):
        p = _sample_approve(rng)
        if p:
            profiles.append(_tag(p))

    for _ in range(n_conditional):
        p = _sample_conditional(rng)
        if p:
            profiles.append(_tag(p))

    for _ in range(n_decline):
        p = _sample_decline(rng)
        p = compute_derived_fields(p)
        outcome, conditions = determine_outcome(p)
        p["outcome"]    = outcome
        p["conditions"] = conditions
        # A trigger may land just outside its threshold due to rounding —
        # accept the actual outcome rather than forcing it.
        profiles.append(_tag(p))

    return profiles


# ─── Orchestrator ──────────────────────────────────────────────────────────────

def generate_all_profiles(
    total: int = 1000,
    n_workers: int = 8,
    seed: int = 42,
) -> list[dict]:
    """
    Generate `total` profiles across `n_workers` Ray workers with a
    controlled outcome distribution defined by OUTCOME_TARGETS.
    """
    ray.init(ignore_reinit_error=True)
    rng = random.Random(seed)

    n_approve     = round(total * OUTCOME_TARGETS["APPROVE"])
    n_conditional = round(total * OUTCOME_TARGETS["CONDITIONAL_APPROVE"])
    n_decline     = total - n_approve - n_conditional

    print(f"Target: {n_approve} APPROVE | {n_conditional} CONDITIONAL | {n_decline} DECLINE")

    def _split(n: int) -> list[int]:
        base, extra = divmod(n, n_workers)
        return [base + (1 if i < extra else 0) for i in range(n_workers)]

    print(f"Dispatching {n_workers} Ray workers …")
    futures = [
        generate_profile_batch.remote(
            _split(n_approve)[i],
            _split(n_conditional)[i],
            _split(n_decline)[i],
            seed=seed + i * 10_000,
        )
        for i in range(n_workers)
    ]

    profiles = [p for batch in ray.get(futures) for p in batch]
    rng.shuffle(profiles)

    dist = Counter(p["outcome"] for p in profiles)
    print(f"\nGenerated {len(profiles)} profiles:")
    for outcome in ["APPROVE", "CONDITIONAL_APPROVE", "DECLINE"]:
        count = dist.get(outcome, 0)
        print(f"  {outcome}: {count} ({count / len(profiles) * 100:.1f}%)")

    return profiles


# ─── I/O ──────────────────────────────────────────────────────────────────────

def save_profiles(profiles: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(profiles)} profiles → {output_path}")


def load_profiles(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic NBFC borrower profiles for RinLekha"
    )
    parser.add_argument("--total",   type=int, default=1000)
    parser.add_argument("--output",  type=str, default="data/raw/profiles_v1.jsonl")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    profiles = generate_all_profiles(
        total=args.total,
        n_workers=args.workers,
        seed=args.seed,
    )
    save_profiles(profiles, Path(args.output))


if __name__ == "__main__":
    main()
