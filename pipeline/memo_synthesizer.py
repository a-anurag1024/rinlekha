#!/usr/bin/env python3
"""
pipeline/memo_synthesizer.py — Phase 1, Step 2

Synthesises credit memos for each borrower profile using the OpenAI API
(gpt-4.1-mini) and Ray for parallel dispatch.

Each memo follows a strict 6-section institutional format:
  APPLICANT SUMMARY / DEBT SERVICEABILITY / CREDIT BEHAVIOR /
  RISK FLAGS / RECOMMENDATION / ANALYST NOTES

Cost estimate (gpt-4.1-mini @ $0.40/M input + $1.60/M output):
  800 memos × ~600 input tokens + ~700 output tokens ≈ $1 total

Usage:
    # Generate all memos (reads profiles from default path)
    python pipeline/memo_synthesizer.py

    # Resume an interrupted run (skips already-completed profile_ids)
    python pipeline/memo_synthesizer.py --resume

    # Pilot run — synthesise 50 memos for manual review before the full run
    python pipeline/memo_synthesizer.py --limit 50 --output data/raw/memos_pilot.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from pipeline.profile_generator import load_profiles

# ─── Prompt constants ─────────────────────────────────────────────────────────
# SYSTEM_PROMPT is identical across all 800 calls → cached by OpenAI automatically.

SYSTEM_PROMPT = """You are a senior credit analyst at an Indian NBFC (Non-Banking Financial \
Company) writing internal credit assessment memos. Your memos are read by credit managers \
making final lending decisions.

Write credit memos that are:
- Factually accurate: every figure you state must match the input profile exactly
- Appropriately hedged: never use "definitely", "certainly", "guaranteed", "will", \
"100%", "no doubt", or any language implying certainty about future outcomes
- Analytically substantive: interpret the data, do not merely restate it
- Structurally strict: follow the exact 6-section format specified by the user
- Gender-neutral: never use "he", "she", "him", "her" — always refer to \
"the borrower" or "the applicant\""""

USER_PROMPT_TEMPLATE = """\
Write a credit memo for this borrower. Follow the format EXACTLY. \
Do not add any text before ## APPLICANT SUMMARY.

## APPLICANT SUMMARY
[2-3 sentences. Employment stability, income source, demographic context. \
Factual statements only. No risk assessment in this section.]

## DEBT SERVICEABILITY
[FOIR analysis, EMI burden, income adequacy. \
Quantify: state the actual FOIR percentage and the 55% policy ceiling. \
Hedged language mandatory.]

## CREDIT BEHAVIOR
[CIBIL score with band context — e.g. "724 — acceptable tier". \
Payment history: distinguish old vs recent delinquency explicitly. \
Credit vintage and account diversity where relevant.]

## RISK FLAGS
[Bulleted list. Minimum 2 items, maximum 4 items. \
Each flag must reference a specific ADVERSE data point from the profile. \
Only flag genuine weaknesses — do not flag neutral or positive metrics as risks. \
No generic flags without specific grounding.]

## RECOMMENDATION
DECISION: [{decision_label}]
CONDITIONS: [{conditions_text}]
RISK GRADE: [A / B+ / B / B- / C]
DECISION AUTHORITY: {decision_authority}
REVIEW TRIGGER: [one sentence — what new information would change this recommendation]

## ANALYST NOTES
[1-2 sentences. What additional information would materially change the assessment. \
Be specific.]

---
BORROWER PROFILE:
{profile_text}

EXPECTED DECISION: {outcome}
CONDITIONS / DECLINE REASONS: {conditions_text}

Write the memo now."""


# ─── Readable condition labels ────────────────────────────────────────────────
# Converts machine-readable condition keys to prose for the synthesis prompt.

_CONDITION_LABELS: dict[str, str] = {
    # Conditional approve requirements
    "income_proof_last_6_months_salary_slips":
        "Submit salary slips / bank statements for the last 6 months",
    "written_explanation_for_missed_payments":
        "Provide written explanation for missed payments",
    "employment_confirmation_from_hr":
        "Submit employment confirmation letter from HR",
    "additional_income_proof_or_co_applicant":
        "Provide additional income proof or add a co-applicant",
    "guarantor_or_collateral_security":
        "Provide a guarantor or collateral security",
    "noc_from_previous_lender_for_settled_account":
        "Obtain NOC from previous lender for settled account",
    "standard_documentation_verification":
        "Standard documentation verification",
    # Hard decline reasons
    "credit_score_below_minimum_threshold":
        "CIBIL score below minimum threshold of 620",
    "foir_exceeds_policy_ceiling_of_55pct":
        "Post-loan FOIR exceeds policy ceiling of 55%",
    "excessive_recent_delinquency":
        "Excessive recent delinquency — 4 or more missed payments in the last 24 months",
    "multiple_settled_loan_accounts":
        "Multiple settled loan accounts (2 or more prior debt write-offs)",
    "loan_amount_grossly_disproportionate_to_income":
        "Loan amount grossly disproportionate to income (loan-to-income ratio exceeds 60×)",
}

_DECISION_LABELS: dict[str, str] = {
    "APPROVE":             "APPROVE",
    "CONDITIONAL_APPROVE": "CONDITIONAL APPROVE",
    "DECLINE":             "DECLINE",
}


# ─── Profile formatting ───────────────────────────────────────────────────────

def _inr(amount: float) -> str:
    """Format a number in Indian Rupee notation (e.g. ₹12,34,567)."""
    # Indian grouping: last three digits, then pairs
    s = f"{int(amount):,}"
    # Python's default comma grouping is Western (groups of 3).
    # Convert to Indian grouping.
    parts = s.split(",")
    if len(parts) <= 1:
        return f"₹{s}"
    # Rejoin with Indian grouping
    integer = int(amount)
    if integer < 1000:
        return f"₹{integer}"
    last3 = str(integer % 1000).zfill(3)
    rest = integer // 1000
    groups = []
    while rest > 0:
        groups.append(str(rest % 100).zfill(2) if rest >= 100 else str(rest))
        rest //= 100
    groups.reverse()
    return "₹" + ",".join(groups) + "," + last3


def format_profile_as_readable_text(profile: dict) -> str:
    """
    Render a borrower profile dict as labelled plain text for the synthesis prompt.
    Readable text produces better memo quality than raw JSON because the model
    can parse formatted values (₹, %, ×) directly without inferring units.
    """
    emp_type = profile["employment_type"].replace("_", " ").title()
    sector   = profile["sector"].title()
    purpose  = profile["loan_purpose"].replace("_", " ").title()
    city     = profile["city_tier"].replace("tier", "Tier ").title()

    lines = [
        "=== BORROWER PROFILE ===",
        "",
        "DEMOGRAPHICS",
        f"  Age             : {profile['age']} years",
        f"  City Tier       : {city}",
        "",
        "EMPLOYMENT",
        f"  Type            : {emp_type}",
        f"  Sector          : {sector}",
        f"  Monthly Income  : {_inr(profile['monthly_income'])}",
        f"  Tenure          : {profile['employment_tenure_years']} years",
        "",
        "CREDIT PROFILE",
        f"  CIBIL Score     : {profile['cibil_score']}",
        f"  Missed Payments (last 24m): {profile['missed_payments_24m']}",
        f"  Settled Accounts: {profile['settled_accounts_ever']}",
        f"  Active Loans    : {profile['loan_accounts_active']}",
        f"  Credit Vintage  : {profile['credit_vintage_years']} years",
        f"  Existing EMI    : {_inr(profile['existing_emi_monthly'])}/month",
        "",
        "LOAN REQUEST",
        f"  Amount          : {_inr(profile['loan_amount'])}",
        f"  Tenure          : {profile['loan_tenure_months']} months",
        f"  Purpose         : {purpose}",
        f"  Annual Rate     : {profile['annual_interest_rate']}%",
        "",
        "DERIVED METRICS",
        f"  Proposed EMI    : {_inr(profile['proposed_emi'])}/month",
        f"  FOIR (pre-loan) : {profile['foir_pre_loan'] * 100:.1f}%",
        f"  FOIR (post-loan): {profile['foir_post_loan'] * 100:.1f}%  "
        f"[policy ceiling: 55%]",
        f"  Loan-to-Income  : {profile['loan_to_income_ratio']}×",
    ]
    return "\n".join(lines)


def _format_conditions(outcome: str, conditions: list[str]) -> str:
    """Convert condition key list to numbered readable prose for the prompt."""
    if outcome == "APPROVE" or not conditions:
        return "None"
    labels = [_CONDITION_LABELS.get(c, c.replace("_", " ")) for c in conditions]
    return "\n" + "\n".join(f"  {i + 1}. {label}" for i, label in enumerate(labels))


# ─── Decision authority ───────────────────────────────────────────────────────

def _get_decision_authority(profile: dict) -> str:
    """
    Return the appropriate sign-off authority based on loan size and outcome.

    Delegation tiers (typical NBFC norms):
      HO Credit Committee   — loan > ₹25L  OR  any DECLINE
      Regional Credit Head  — loan > ₹10L  OR  CONDITIONAL_APPROVE
      Branch Credit Manager — small, clean approvals (loan ≤ ₹10L, APPROVE)
    """
    loan    = profile.get("loan_amount", 0)
    outcome = profile.get("outcome", "")
    if loan > 2_500_000 or outcome == "DECLINE":
        return "HO Credit Committee"
    if loan > 1_000_000 or outcome == "CONDITIONAL_APPROVE":
        return "Regional Credit Head"
    return "Branch Credit Manager"


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_synthesis_prompt(profile: dict) -> list[dict]:
    """
    Build the OpenAI messages list for a single profile.
    Returns [system_message, user_message].
    The system message is identical for all profiles — OpenAI caches it.
    """
    outcome    = profile["outcome"]
    conditions = profile.get("conditions", [])

    user_content = USER_PROMPT_TEMPLATE.format(
        decision_label      = _DECISION_LABELS[outcome],
        conditions_text     = _format_conditions(outcome, conditions),
        profile_text        = format_profile_as_readable_text(profile),
        outcome             = outcome,
        decision_authority  = _get_decision_authority(profile),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ─── Single-memo synthesis with retry ────────────────────────────────────────

# Errors that are worth retrying (transient)
_RETRYABLE = (RateLimitError, APIConnectionError)

# Errors that should never be retried (programmer / auth mistakes)
_FATAL = (AuthenticationError, BadRequestError)


def synthesize_single_memo(
    profile: dict,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1100,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict:
    """
    Call the OpenAI API for a single profile with exponential backoff + jitter.

    Returns a result dict with synthesis_status = "success" | "error:<msg>".
    Never raises — errors are captured in the result so the batch worker can
    continue processing remaining profiles.

    Retry policy:
      RateLimitError / APIConnectionError → exponential backoff (retryable)
      APIStatusError 5xx                  → exponential backoff (retryable)
      AuthenticationError / BadRequestError → fail immediately (not retryable)
    """
    messages = build_synthesis_prompt(profile)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model      = model,
                max_tokens = max_tokens,
                messages   = messages,
            )
            return {
                "profile_id":       profile["profile_id"],
                "input_profile":    profile,
                "output_memo":      response.choices[0].message.content,
                "input_tokens":     response.usage.prompt_tokens,
                "output_tokens":    response.usage.completion_tokens,
                "synthesis_status": "success",
            }

        except _FATAL as e:
            return {
                "profile_id":       profile["profile_id"],
                "input_profile":    profile,
                "output_memo":      None,
                "input_tokens":     0,
                "output_tokens":    0,
                "synthesis_status": f"error: {type(e).__name__}: {e}",
            }

        except _RETRYABLE as e:
            if attempt == max_retries:
                return {
                    "profile_id":       profile["profile_id"],
                    "input_profile":    profile,
                    "output_memo":      None,
                    "input_tokens":     0,
                    "output_tokens":    0,
                    "synthesis_status": f"error: {type(e).__name__}: {e}",
                }
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

        except APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                return {
                    "profile_id":       profile["profile_id"],
                    "input_profile":    profile,
                    "output_memo":      None,
                    "input_tokens":     0,
                    "output_tokens":    0,
                    "synthesis_status": f"error: APIStatusError {e.status_code}: {e}",
                }

        except Exception as e:
            return {
                "profile_id":       profile["profile_id"],
                "input_profile":    profile,
                "output_memo":      None,
                "input_tokens":     0,
                "output_tokens":    0,
                "synthesis_status": f"error: {type(e).__name__}: {e}",
            }

    # Unreachable but satisfies type checker
    return {  # pragma: no cover
        "profile_id": profile["profile_id"],
        "input_profile": profile,
        "output_memo": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "synthesis_status": "error: max retries exceeded",
    }


# ─── Ray batch worker ─────────────────────────────────────────────────────────

@ray.remote
def synthesize_memo_batch(
    profiles: list[dict],
    api_key: str,
    model: str = "gpt-4.1-mini",
) -> list[dict]:
    """
    Ray remote worker: synthesises memos for a batch of profiles sequentially.
    Each worker creates its own OpenAI client — no shared state between workers.
    Sequential within a worker (not async) to keep rate-limit behaviour predictable.
    """
    client  = OpenAI(api_key=api_key)
    results = []
    for profile in profiles:
        result = synthesize_single_memo(profile, client, model=model)
        results.append(result)
    return results


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def synthesize_all_memos(
    profiles: list[dict],
    api_key: str,
    n_workers: int = 8,
    model: str = "gpt-4.1-mini",
    completed_ids: set[str] | None = None,
) -> list[dict]:
    """
    Dispatch `profiles` across `n_workers` Ray workers.

    `completed_ids` — set of profile_ids already synthesised (for resumption).
    Profiles in this set are skipped entirely; their existing results should be
    merged by the caller.
    """
    ray.init(ignore_reinit_error=True)

    pending = profiles
    if completed_ids:
        pending = [p for p in profiles if p["profile_id"] not in completed_ids]
        print(f"Resuming: {len(completed_ids)} already done, "
              f"{len(pending)} remaining")

    if not pending:
        print("All profiles already synthesised.")
        return []

    # Split pending profiles into n_workers roughly equal batches
    batch_size = max(1, len(pending) // n_workers)
    batches = [
        pending[i: i + batch_size]
        for i in range(0, len(pending), batch_size)
    ]

    print(f"Synthesising {len(pending)} memos across {len(batches)} workers "
          f"(model: {model}) …")

    futures = [
        synthesize_memo_batch.remote(batch, api_key, model)
        for batch in batches
    ]
    batch_results = ray.get(futures)
    results = [r for batch in batch_results for r in batch]

    n_ok  = sum(1 for r in results if r["synthesis_status"] == "success")
    n_err = len(results) - n_ok
    total_in  = sum(r["input_tokens"]  for r in results)
    total_out = sum(r["output_tokens"] for r in results)
    cost = (total_in / 1_000_000 * 0.15) + (total_out / 1_000_000 * 0.60)

    print(f"\nDone: {n_ok} success / {n_err} errors")
    print(f"Tokens: {total_in:,} input / {total_out:,} output")
    print(f"Estimated cost: ${cost:.3f}")

    return results


# ─── I/O ──────────────────────────────────────────────────────────────────────

def save_memos(memos: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for memo in memos:
            f.write(json.dumps(memo, ensure_ascii=False) + "\n")
    print(f"Saved {len(memos)} memos → {output_path}")


def load_memos(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Synthesise credit memos for RinLekha borrower profiles"
    )
    parser.add_argument("--profiles", type=str,
                        default="data/raw/profiles_v1.jsonl",
                        help="Input profiles JSONL")
    parser.add_argument("--output", type=str,
                        default="data/raw/memos_v1.jsonl",
                        help="Output memos JSONL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Synthesise only the first N profiles (pilot mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip profiles already present in --output")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Add it to your .env and source it.")

    profiles    = load_profiles(Path(args.profiles))
    output_path = Path(args.output)

    if args.limit:
        profiles = profiles[: args.limit]
        print(f"Pilot mode: limiting to {args.limit} profiles")

    # Resumption: load existing completed results
    existing: list[dict] = []
    completed_ids: set[str] = set()
    if args.resume and output_path.exists():
        existing      = load_memos(output_path)
        completed_ids = {m["profile_id"] for m in existing}

    new_results = synthesize_all_memos(
        profiles,
        api_key=api_key,
        n_workers=args.workers,
        model=args.model,
        completed_ids=completed_ids,
    )

    all_results = existing + new_results
    save_memos(all_results, output_path)


if __name__ == "__main__":
    main()
