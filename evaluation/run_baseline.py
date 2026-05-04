"""
Baseline comparison — RinLekha (fine-tuned Gemma 3 4B) vs GPT-4o-mini
on the first 30 cases from the HF Hub test split.

Both models receive the identical prompt. The same 6 DeepEval metrics
are scored for each. The result is a side-by-side JSON with per-case scores
and aggregate means.

Usage:
  python evaluation/run_baseline.py --no-mlflow
  python evaluation/run_baseline.py --n-cases 30 --vllm-url http://localhost:8000

Requires:
  - OPENAI_API_KEY  (for baseline generation AND GEval / Faithfulness judge)
  - llama-cpp-python server running (for RinLekha)
"""
import argparse, json, os, sys, time
from pathlib import Path

import openai
import requests
from datasets import load_dataset
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

BASELINE_SYSTEM = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly.\n\n"
    "Use this exact section order with these exact headers:\n"
    "## APPLICANT SUMMARY\n"
    "## DEBT SERVICEABILITY\n"
    "## CREDIT BEHAVIOR\n"
    "## RISK FLAGS\n"
    "## RECOMMENDATION\n"
    "## ANALYST NOTES\n\n"
    "The RECOMMENDATION section must end with exactly these labelled fields:\n"
    "DECISION: <APPROVE|CONDITIONAL APPROVE|DECLINE>\n"
    "CONDITIONS: <list or N/A>\n"
    "RISK GRADE: <A/B/C with optional +/->\n"
    "DECISION AUTHORITY: <role>\n"
    "REVIEW TRIGGER: <condition>"
)


# ── Generation ────────────────────────────────────────────────────────────────

def generate_rinlekha(vllm_url: str, input_text: str, max_tokens: int, timeout: int) -> str:
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


def generate_baseline(client: openai.OpenAI, model: str, input_text: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user",   "content": input_text},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Metrics ───────────────────────────────────────────────────────────────────

def run_metrics(test_case: LLMTestCase, metrics: list) -> dict:
    results = {}
    for metric in metrics:
        name = metric.__class__.__name__
        try:
            metric.measure(test_case)
            results[name] = {
                "score":   metric.score,
                "success": metric.is_successful(),
                "reason":  getattr(metric, "reason", None),
            }
        except Exception as exc:
            results[name] = {"score": 0.0, "success": False, "reason": str(exc)}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url",       default="http://localhost:8000")
    parser.add_argument("--hf-dataset",     default="a-anurag1024/rinlekha-training-data")
    parser.add_argument("--baseline-model", default="gpt-4o-mini",
                        help="OpenAI model for baseline (e.g. gpt-4o-mini, gpt-4.1-nano)")
    parser.add_argument("--judge-model",    default="gpt-4o-mini")
    parser.add_argument("--n-cases",        type=int, default=30)
    parser.add_argument("--max-tokens",     type=int, default=700)
    parser.add_argument("--timeout",        type=int, default=120)
    parser.add_argument("--output",         default="outputs/eval_results/baseline_results.json")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=openai_key)

    dataset = load_dataset(args.hf_dataset)
    cases = list(dataset["test"])[: args.n_cases]
    print(f"Loaded {len(cases)} test cases | baseline model: {args.baseline_model}")

    metrics = [
        StructuralComplianceMetric(),
        RecommendationFormatMetric(),
        ForbiddenLanguageMetric(),
        RiskFlagsCountMetric(),
        build_geval_metric(args.judge_model),
        build_faithfulness_metric(args.judge_model),
    ]

    results = []
    for i, case in enumerate(cases):
        profile_id = case.get("profile_id", f"idx_{i}")
        print(f"\n[{i+1}/{len(cases)}] {profile_id}")

        # RinLekha
        try:
            t0 = time.perf_counter()
            rl_output = generate_rinlekha(args.vllm_url, case["input"], args.max_tokens, args.timeout)
            print(f"  RinLekha: {len(rl_output.split())} words in {time.perf_counter()-t0:.1f}s")
        except Exception as exc:
            print(f"  RinLekha generation failed: {exc}")
            rl_output = ""

        # Baseline (OpenAI)
        try:
            t0 = time.perf_counter()
            bl_output = generate_baseline(client, args.baseline_model, case["input"], args.max_tokens)
            print(f"  Baseline: {len(bl_output.split())} words in {time.perf_counter()-t0:.1f}s")
        except Exception as exc:
            print(f"  Baseline generation failed: {exc}")
            bl_output = ""

        rl_tc = LLMTestCase(
            input=case["input"],
            actual_output=rl_output,
            expected_output=case.get("output", ""),
            retrieval_context=[case["input"]],
        )
        bl_tc = LLMTestCase(
            input=case["input"],
            actual_output=bl_output,
            expected_output=case.get("output", ""),
            retrieval_context=[case["input"]],
        )

        rl_metrics = run_metrics(rl_tc, metrics)
        bl_metrics = run_metrics(bl_tc, metrics)

        def scores_str(m):
            return "  ".join(f"{k[:12]}: {v['score']:.2f}" for k, v in m.items())

        print(f"  RinLekha: {scores_str(rl_metrics)}")
        print(f"  Baseline: {scores_str(bl_metrics)}")

        results.append({
            "idx":        i,
            "profile_id": profile_id,
            "input":      case["input"],
            "rinlekha":  {"output": rl_output, "metrics": rl_metrics},
            "baseline":  {"output": bl_output, "metrics": bl_metrics},
        })

    # Aggregate
    print(f"\n{'='*60}")
    print(f"Baseline comparison: {len(results)} cases | baseline: {args.baseline_model}\n")

    agg_rl: dict[str, list] = {}
    agg_bl: dict[str, list] = {}
    for r in results:
        for name, mdata in r["rinlekha"]["metrics"].items():
            agg_rl.setdefault(name, []).append(mdata["score"] or 0.0)
        for name, mdata in r["baseline"]["metrics"].items():
            agg_bl.setdefault(name, []).append(mdata["score"] or 0.0)

    print(f"{'Metric':<28} {'RinLekha':>10} {args.baseline_model:>14}")
    print("-" * 56)
    for k in agg_rl:
        rl_mean = sum(agg_rl[k]) / len(agg_rl[k])
        bl_mean = sum(agg_bl.get(k, [0.0])) / len(agg_bl.get(k, [0.0]))
        winner = "<--" if rl_mean >= bl_mean else ""
        print(f"  {k:<26} {rl_mean:>10.3f} {bl_mean:>14.3f}  {winner}")

    summary = {
        "n_cases":        len(results),
        "baseline_model": args.baseline_model,
        "aggregate": {
            "rinlekha": {k: sum(v) / len(v) for k, v in agg_rl.items()},
            "baseline": {k: sum(v) / len(v) for k, v in agg_bl.items()},
        },
        "cases": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
