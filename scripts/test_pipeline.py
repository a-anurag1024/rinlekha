"""
Smoke test for the full serving pipeline.
Requires the llama-cpp-python server to be running:
  bash serving/start_server.sh outputs/rinlekha-q8.gguf

Usage:
  python scripts/test_pipeline.py
  python scripts/test_pipeline.py --observe   # also send trace to Langfuse
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_PROFILE = {
    "age":                     34,
    "city_tier":               "Tier 2",
    "employment_type":         "Salaried Private",
    "sector":                  "Technology",
    "monthly_income":          85000,
    "employment_tenure_years": 4.5,
    "cibil_score":             724,
    "missed_payments_24m":     1,
    "settled_accounts_ever":   0,
    "active_loans":            2,
    "credit_vintage_years":    5.2,
    "existing_emi_monthly":    12000,
    "loan_amount":             500000,
    "loan_tenure_months":      48,
    "loan_purpose":            "Home Renovation",
    "annual_interest_rate":    14.5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe", action="store_true",
                        help="Use Langfuse-wrapped version (requires LANGFUSE_* env vars)")
    args = parser.parse_args()

    if args.observe:
        from serving.observability import generate_credit_memo_observed as generate
        print("Using Langfuse-observed pipeline")
    else:
        from serving.pipeline import generate_credit_memo as generate
        print("Using plain pipeline (no Langfuse)")

    print("Generating credit memo...")
    t0 = time.perf_counter()
    memo = generate(SAMPLE_PROFILE)
    elapsed = time.perf_counter() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"parse_success:      {memo.parse_success}")
    print(f"decision:           {memo.decision.value}")
    print(f"risk_grade:         {memo.risk_grade.value}")
    print(f"decision_authority: {memo.decision_authority}")
    print(f"risk_flags ({len(memo.risk_flags)}):")
    for flag in memo.risk_flags:
        print(f"  - {flag}")
    if memo.conditions:
        print(f"conditions ({len(memo.conditions)}):")
        for c in memo.conditions:
            print(f"  {c}")
    if memo.parse_errors:
        print(f"parse_errors: {memo.parse_errors}")

    print("\n--- Raw output (first 400 chars) ---")
    print(memo.raw_output[:400])

    ok = memo.parse_success and memo.decision.value != "UNKNOWN"
    print(f"\n{'PASS' if ok else 'FAIL'} — pipeline smoke test")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
