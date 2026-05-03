"""
Crash-resumable local eval runner. Drop-in replacement for the k8s indexed-job
pipeline for single-machine use.

Checkpointing: after every case the result is appended to
  RESULTS_DIR/checkpoint.json
On restart the script reads the checkpoint, skips completed indices, and
continues from where it left off — safe to kill and rerun at any time.

Requires:
  - llama-cpp-python server running separately:
      bash serving/start_server.sh outputs/rinlekha-q8.gguf 8000
  - OPENAI_API_KEY in environment (for GEval / Faithfulness judge)
  - MLflow running locally (optional — pass --no-mlflow to skip):
      mlflow server --host 0.0.0.0 --port 5000

Usage:
  python evaluation/run_eval_local.py
  python evaluation/run_eval_local.py --vllm-url http://localhost:8000 \
      --results-dir outputs/eval_results --max-tokens 400
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import requests
from datasets import load_dataset
from deepeval.test_case import LLMTestCase

# run_eval_local.py lives next to metrics.py
sys.path.insert(0, str(Path(__file__).parent))
from metrics import (
    ForbiddenLanguageMetric,
    RecommendationFormatMetric,
    RiskFlagsCountMetric,
    StructuralComplianceMetric,
    build_faithfulness_metric,
    build_geval_metric,
)

ALPACA_INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)


# ── Generation ────────────────────────────────────────────────────────────────

def generate(vllm_url: str, input_text: str, max_tokens: int, timeout: int) -> str:
    prompt = (
        f"### Instruction:\n{ALPACA_INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    resp = requests.post(
        f"{vllm_url}/v1/completions",
        json={
            "model":             "rinlekha",
            "prompt":            prompt,
            "max_tokens":        max_tokens,
            "temperature":       0.1,
            "top_p":             0.9,
            "frequency_penalty": 0.1,
            "stop":              ["### Instruction:"],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


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


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_indices": [], "results": []}


def save_checkpoint(path: Path, state: dict) -> None:
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    state["count"] = len(state["results"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)   # atomic rename — safe against mid-write crashes


# ── MLflow ────────────────────────────────────────────────────────────────────

def log_to_mlflow(uri: str, agg: dict, params: dict, checkpoint_path: Path) -> None:
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("rinlekha-evaluation")
    exp_id = exp.experiment_id if exp else client.create_experiment("rinlekha-evaluation")
    with mlflow.start_run(run_name="eval_local", experiment_id=exp_id):
        mlflow.log_metrics(agg)
        mlflow.log_params(params)
        mlflow.log_artifact(str(checkpoint_path))
    print(f"Logged to MLflow: {uri}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url",    default="http://localhost:8000")
    parser.add_argument("--hf-dataset",  default="a-anurag1024/rinlekha-training-data")
    parser.add_argument("--results-dir", default="outputs/eval_results")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-tokens",  type=int, default=700)
    parser.add_argument("--timeout",     type=int, default=120,
                        help="Per-request HTTP timeout in seconds")
    parser.add_argument("--mlflow-uri",  default="http://localhost:5000")
    parser.add_argument("--no-mlflow",   action="store_true",
                        help="Skip MLflow logging (use if MLflow is not running)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "checkpoint.json"

    # Resume from checkpoint
    state = load_checkpoint(checkpoint_path)
    done = set(state["completed_indices"])
    print(f"Checkpoint: {len(done)} cases already done")

    # Load test split
    dataset = load_dataset(args.hf_dataset)
    all_cases = list(dataset["test"])
    print(f"Test set: {len(all_cases)} total | {len(all_cases) - len(done)} remaining")

    # Build metrics once (LLM judge models initialise here)
    metrics = [
        StructuralComplianceMetric(),
        RecommendationFormatMetric(),
        ForbiddenLanguageMetric(),
        RiskFlagsCountMetric(),
        build_geval_metric(args.judge_model),
        build_faithfulness_metric(args.judge_model),
    ]

    for idx, case in enumerate(all_cases):
        if idx in done:
            continue

        profile_id = case.get("profile_id", f"idx_{idx}")
        print(f"\n[{idx+1}/{len(all_cases)}] {profile_id}")

        # Generate
        try:
            t0 = time.perf_counter()
            actual_output = generate(args.vllm_url, case["input"], args.max_tokens, args.timeout)
            elapsed = time.perf_counter() - t0
            n_tok = len(actual_output.split())
            print(f"  Generated ~{n_tok} words in {elapsed:.1f}s")
        except Exception as exc:
            print(f"  Generation failed, skipping: {exc}")
            continue

        # Evaluate
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case["output"],
            retrieval_context=[case["input"]],
        )
        metric_results = run_metrics(test_case, metrics)

        scores_str = "  ".join(
            f"{k[:12]}: {v['score']:.2f}" for k, v in metric_results.items()
        )
        print(f"  {scores_str}")

        # Persist
        state["completed_indices"].append(idx)
        state["results"].append({
            "idx":             idx,
            "profile_id":      profile_id,
            "input":           case["input"],
            "actual_output":   actual_output,
            "expected_output": case["output"],
            "metrics":         metric_results,
        })
        save_checkpoint(checkpoint_path, state)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"Evaluation complete: {len(state['results'])} cases")
    agg_buckets: dict[str, list[float]] = {}
    for r in state["results"]:
        for name, mdata in r["metrics"].items():
            agg_buckets.setdefault(name, []).append(mdata["score"] or 0.0)
    agg = {k: sum(v) / len(v) for k, v in agg_buckets.items()}

    print("Aggregate scores:")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}")

    if not args.no_mlflow:
        try:
            log_to_mlflow(
                args.mlflow_uri, agg,
                {"total_cases": len(state["results"]),
                 "judge_model": args.judge_model,
                 "max_tokens":  args.max_tokens},
                checkpoint_path,
            )
        except Exception as exc:
            print(f"MLflow logging failed (use --no-mlflow to skip): {exc}")

    print(f"Full results: {checkpoint_path}")


if __name__ == "__main__":
    main()
