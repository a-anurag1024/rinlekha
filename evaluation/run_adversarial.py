"""
Adversarial test suite — 8 hand-crafted edge cases designed to stress the model
on boundary conditions not well-represented in the standard test set.

Each case has an expected decision so we can check whether the model's DECISION
line matches policy. Metrics are the same 6 used in the main eval.

Usage:
  python evaluation/run_adversarial.py
  python evaluation/run_adversarial.py --vllm-url http://localhost:8000 --no-mlflow
"""
import argparse, json, sys, time
from pathlib import Path

import requests
from deepeval.test_case import LLMTestCase

sys.path.insert(0, str(Path(__file__).parent))
from metrics import (
    ForbiddenLanguageMetric, RecommendationFormatMetric,
    RiskFlagsCountMetric, StructuralComplianceMetric,
    build_faithfulness_metric, build_geval_metric,
)

ALPACA_INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)

# ── 8 adversarial profiles ────────────────────────────────────────────────────

CASES = [
    {
        "id": "adv_01_extreme_foir",
        "description": "Extreme FOIR — post-loan FOIR >75%, should DECLINE",
        "expected_decision": "DECLINE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 38 years
  City Tier       : Tier 2

EMPLOYMENT
  Type            : Salaried Private
  Sector          : Retail
  Monthly Income  : ₹28,000
  Tenure          : 4.0 years

CREDIT PROFILE
  CIBIL Score     : 710
  Missed Payments (last 24m): 0
  Settled Accounts: 0
  Active Loans    : 4
  Credit Vintage  : 6.1 years
  Existing EMI    : ₹17,500/month

LOAN REQUEST
  Amount          : ₹5,00,000
  Tenure          : 48 months
  Purpose         : Medical
  Annual Rate     : 16.0%

DERIVED METRICS
  Proposed EMI    : ₹14,190/month
  FOIR (pre-loan) : 62.5%
  FOIR (post-loan): 113.2%  [policy ceiling: 55%]
  Loan-to-Income  : 17.9×""",
    },
    {
        "id": "adv_02_no_credit_history",
        "description": "No credit history — CIBIL 0, credit vintage 0 years",
        "expected_decision": "CONDITIONAL APPROVE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 24 years
  City Tier       : Tier 3

EMPLOYMENT
  Type            : Salaried Govt
  Sector          : Govt
  Monthly Income  : ₹32,000
  Tenure          : 1.2 years

CREDIT PROFILE
  CIBIL Score     : 0
  Missed Payments (last 24m): 0
  Settled Accounts: 0
  Active Loans    : 0
  Credit Vintage  : 0.0 years
  Existing EMI    : ₹0/month

LOAN REQUEST
  Amount          : ₹1,20,000
  Tenure          : 24 months
  Purpose         : Education
  Annual Rate     : 13.5%

DERIVED METRICS
  Proposed EMI    : ₹5,762/month
  FOIR (pre-loan) : 0.0%
  FOIR (post-loan): 18.0%  [policy ceiling: 55%]
  Loan-to-Income  : 3.8×""",
    },
    {
        "id": "adv_03_contradictory_profile",
        "description": "Contradictory signals — excellent CIBIL but pre-loan FOIR already at 58%",
        "expected_decision": "DECLINE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 45 years
  City Tier       : Tier 1

EMPLOYMENT
  Type            : Self Employed Professional
  Sector          : Finance
  Monthly Income  : ₹95,000
  Tenure          : 12.0 years

CREDIT PROFILE
  CIBIL Score     : 820
  Missed Payments (last 24m): 0
  Settled Accounts: 0
  Active Loans    : 5
  Credit Vintage  : 14.2 years
  Existing EMI    : ₹55,100/month

LOAN REQUEST
  Amount          : ₹8,00,000
  Tenure          : 36 months
  Purpose         : Business Expansion
  Annual Rate     : 14.0%

DERIVED METRICS
  Proposed EMI    : ₹27,330/month
  FOIR (pre-loan) : 58.0%
  FOIR (post-loan): 86.7%  [policy ceiling: 55%]
  Loan-to-Income  : 8.4×""",
    },
    {
        "id": "adv_04_ood_loan_amount",
        "description": "Loan grossly disproportionate to income — LTI 72×, should DECLINE",
        "expected_decision": "DECLINE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 29 years
  City Tier       : Tier 3

EMPLOYMENT
  Type            : Salaried Private
  Sector          : Manufacturing
  Monthly Income  : ₹18,000
  Tenure          : 1.5 years

CREDIT PROFILE
  CIBIL Score     : 680
  Missed Payments (last 24m): 1
  Settled Accounts: 0
  Active Loans    : 1
  Credit Vintage  : 2.0 years
  Existing EMI    : ₹2,100/month

LOAN REQUEST
  Amount          : ₹15,50,000
  Tenure          : 60 months
  Purpose         : Home Renovation
  Annual Rate     : 17.5%

DERIVED METRICS
  Proposed EMI    : ₹38,945/month
  FOIR (pre-loan) : 11.7%
  FOIR (post-loan): 228.0%  [policy ceiling: 55%]
  Loan-to-Income  : 72.2×""",
    },
    {
        "id": "adv_05_perfect_profile_short_tenure",
        "description": "Near-perfect credit but only 0.4 years employment — borderline conditional",
        "expected_decision": "CONDITIONAL APPROVE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 27 years
  City Tier       : Tier 1

EMPLOYMENT
  Type            : Salaried Private
  Sector          : Technology
  Monthly Income  : ₹1,10,000
  Tenure          : 0.4 years

CREDIT PROFILE
  CIBIL Score     : 798
  Missed Payments (last 24m): 0
  Settled Accounts: 0
  Active Loans    : 1
  Credit Vintage  : 3.8 years
  Existing EMI    : ₹6,200/month

LOAN REQUEST
  Amount          : ₹4,00,000
  Tenure          : 36 months
  Purpose         : Debt Consolidation
  Annual Rate     : 12.5%

DERIVED METRICS
  Proposed EMI    : ₹13,356/month
  FOIR (pre-loan) : 5.6%
  FOIR (post-loan): 17.8%  [policy ceiling: 55%]
  Loan-to-Income  : 3.6×""",
    },
    {
        "id": "adv_06_active_delinquency",
        "description": "Heavy recent delinquency — 5 missed payments in 24m, hard DECLINE",
        "expected_decision": "DECLINE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 41 years
  City Tier       : Tier 2

EMPLOYMENT
  Type            : Salaried Private
  Sector          : Retail
  Monthly Income  : ₹42,000
  Tenure          : 6.2 years

CREDIT PROFILE
  CIBIL Score     : 598
  Missed Payments (last 24m): 5
  Settled Accounts: 0
  Active Loans    : 3
  Credit Vintage  : 8.5 years
  Existing EMI    : ₹9,800/month

LOAN REQUEST
  Amount          : ₹2,50,000
  Tenure          : 36 months
  Purpose         : Medical
  Annual Rate     : 18.0%

DERIVED METRICS
  Proposed EMI    : ₹9,042/month
  FOIR (pre-loan) : 23.3%
  FOIR (post-loan): 44.9%  [policy ceiling: 55%]
  Loan-to-Income  : 6.0×""",
    },
    {
        "id": "adv_07_settled_accounts",
        "description": "Two settled accounts (prior write-offs) — policy hard DECLINE",
        "expected_decision": "DECLINE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 50 years
  City Tier       : Tier 2

EMPLOYMENT
  Type            : Self Employed Business
  Sector          : Retail
  Monthly Income  : ₹65,000
  Tenure          : 9.0 years

CREDIT PROFILE
  CIBIL Score     : 645
  Missed Payments (last 24m): 1
  Settled Accounts: 2
  Active Loans    : 2
  Credit Vintage  : 11.3 years
  Existing EMI    : ₹8,400/month

LOAN REQUEST
  Amount          : ₹3,50,000
  Tenure          : 36 months
  Purpose         : Business Expansion
  Annual Rate     : 15.5%

DERIVED METRICS
  Proposed EMI    : ₹12,152/month
  FOIR (pre-loan) : 12.9%
  FOIR (post-loan): 31.6%  [policy ceiling: 55%]
  Loan-to-Income  : 5.4×""",
    },
    {
        "id": "adv_08_self_employed_income_spike",
        "description": "Self-employed with very short tenure — income unverifiable, conditional",
        "expected_decision": "CONDITIONAL APPROVE",
        "input": """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 33 years
  City Tier       : Tier 1

EMPLOYMENT
  Type            : Self Employed Business
  Sector          : Technology
  Monthly Income  : ₹2,80,000
  Tenure          : 0.6 years

CREDIT PROFILE
  CIBIL Score     : 742
  Missed Payments (last 24m): 0
  Settled Accounts: 0
  Active Loans    : 2
  Credit Vintage  : 5.5 years
  Existing EMI    : ₹18,200/month

LOAN REQUEST
  Amount          : ₹12,00,000
  Tenure          : 48 months
  Purpose         : Business Expansion
  Annual Rate     : 13.0%

DERIVED METRICS
  Proposed EMI    : ₹32,206/month
  FOIR (pre-loan) : 6.5%
  FOIR (post-loan): 18.0%  [policy ceiling: 55%]
  Loan-to-Income  : 4.3×""",
    },
]


# ── Generation + metrics (reuse same helpers as run_eval_local) ───────────────

def generate(vllm_url: str, input_text: str, max_tokens: int, timeout: int) -> str:
    prompt = (
        f"### Instruction:\n{ALPACA_INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    resp = requests.post(
        f"{vllm_url}/v1/completions",
        json={
            "model":       "rinlekha",
            "prompt":      prompt,
            "max_tokens":  max_tokens,
            "temperature": 0.1,
            "top_p":       0.9,
            "stop":        ["### Instruction:"],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


def run_metrics(test_case: LLMTestCase, metrics: list) -> dict:
    results = {}
    for metric in metrics:
        try:
            metric.measure(test_case)
            results[metric.__class__.__name__] = {
                "score":   metric.score,
                "success": metric.is_successful(),
                "reason":  getattr(metric, "reason", None),
            }
        except Exception as exc:
            results[metric.__class__.__name__] = {
                "score": 0.0, "success": False, "reason": str(exc),
            }
    return results


def extract_decision(text: str) -> str:
    import re
    m = re.search(r"DECISION:\s*(CONDITIONAL APPROVE|DECLINE|APPROVE)", text)
    return m.group(1) if m else "NOT FOUND"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url",    default="http://localhost:8000")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-tokens",  type=int, default=700)
    parser.add_argument("--timeout",     type=int, default=120)
    parser.add_argument("--output",      default="outputs/eval_results/adversarial_results.json")
    parser.add_argument("--no-mlflow",   action="store_true")
    args = parser.parse_args()

    metrics = [
        StructuralComplianceMetric(),
        RecommendationFormatMetric(),
        ForbiddenLanguageMetric(),
        RiskFlagsCountMetric(),
        build_geval_metric(args.judge_model),
        build_faithfulness_metric(args.judge_model),
    ]

    results = []
    for case in CASES:
        print(f"\n[{case['id']}] {case['description']}")
        try:
            t0 = time.perf_counter()
            output = generate(args.vllm_url, case["input"], args.max_tokens, args.timeout)
            elapsed = time.perf_counter() - t0
            print(f"  Generated in {elapsed:.1f}s")
        except Exception as exc:
            print(f"  Generation failed: {exc}")
            continue

        actual_decision = extract_decision(output)
        decision_match = actual_decision == case["expected_decision"]
        print(f"  Decision: {actual_decision} (expected {case['expected_decision']}) {'✓' if decision_match else '✗'}")

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=output,
            expected_output="",
            retrieval_context=[case["input"]],
        )
        metric_results = run_metrics(test_case, metrics)
        scores_str = "  ".join(f"{k[:12]}: {v['score']:.2f}" for k, v in metric_results.items())
        print(f"  {scores_str}")

        results.append({
            "id":                case["id"],
            "description":       case["description"],
            "expected_decision": case["expected_decision"],
            "actual_decision":   actual_decision,
            "decision_match":    decision_match,
            "actual_output":     output,
            "metrics":           metric_results,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"Adversarial suite: {len(results)}/8 cases completed")
    decision_acc = sum(1 for r in results if r["decision_match"]) / len(results) if results else 0
    print(f"Decision accuracy: {decision_acc:.0%} ({sum(1 for r in results if r['decision_match'])}/{len(results)})")

    agg: dict[str, list] = {}
    for r in results:
        for name, mdata in r["metrics"].items():
            agg.setdefault(name, []).append(mdata["score"] or 0.0)
    print("\nAggregate metric scores:")
    for k, v in agg.items():
        print(f"  {k}: {sum(v)/len(v):.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"cases": results, "decision_accuracy": decision_acc}, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
