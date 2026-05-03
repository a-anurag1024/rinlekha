"""Kubernetes pod entry point — evaluates one shard of the test set.

Env vars (injected by eval-job.yaml):
  JOB_COMPLETION_INDEX   pod index (0, 1, 2)
  TOTAL_SHARDS           total number of pods (3)
  VLLM_URL               http://vllm-service.rinlekha.svc.cluster.local:8000
  MLFLOW_TRACKING_URI    http://mlflow-service.rinlekha.svc.cluster.local:5000
  HF_DATASET             HuggingFace dataset repo id
  OPENAI_API_KEY         for DeepEval judge model (GEval / Faithfulness)
  HF_TOKEN               optional, for private datasets
"""
import json
import os
import sys

import mlflow
import requests
from datasets import load_dataset
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from metrics import (
    ForbiddenLanguageMetric,
    RecommendationFormatMetric,
    RiskFlagsCountMetric,
    StructuralComplianceMetric,
    build_faithfulness_metric,
    build_geval_metric,
)

POD_INDEX    = int(os.environ["JOB_COMPLETION_INDEX"])
TOTAL_SHARDS = int(os.environ.get("TOTAL_SHARDS", "3"))
VLLM_URL     = os.environ["VLLM_URL"]
MLFLOW_URI   = os.environ["MLFLOW_TRACKING_URI"]
HF_DATASET   = os.environ.get("HF_DATASET", "a-anurag1024/rinlekha-training-data")
JUDGE_MODEL  = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
RESULTS_DIR  = os.environ.get("RESULTS_DIR", "/results")
MODEL_NAME   = os.environ.get("VLLM_MODEL_NAME", "rinlekha")

ALPACA_INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)


def generate_via_vllm(input_text: str) -> str:
    prompt = (
        f"### Instruction:\n{ALPACA_INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    resp = requests.post(
        f"{VLLM_URL}/v1/completions",
        json={
            "model":             MODEL_NAME,
            "prompt":            prompt,
            "max_tokens":        900,
            "temperature":       0.1,
            "top_p":             0.9,
            "frequency_penalty": 0.1,
            "stop":              ["### Instruction:"],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


def _get_or_create_experiment(name: str) -> str:
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    return client.create_experiment(name)


def serialize_results(results) -> list[dict]:
    out = []
    for r in results:
        entry = {
            "input":         r.input,
            "actual_output": r.actual_output,
            "metrics": {},
        }
        for metric in r.metrics_data or []:
            entry["metrics"][metric.name] = {
                "score":   metric.score,
                "success": metric.success,
                "reason":  getattr(metric, "reason", None),
            }
        out.append(entry)
    return out


def aggregate_scores(test_results, metrics) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for result in test_results:
        for md in result.metrics_data or []:
            scores.setdefault(md.name, []).append(md.score or 0.0)
    return {name: sum(vals) / len(vals) for name, vals in scores.items()}


def main() -> None:
    print(f"[Pod {POD_INDEX}] Starting shard evaluation")

    dataset = load_dataset(HF_DATASET)
    all_cases = list(dataset["test"])

    shard_size = len(all_cases) // TOTAL_SHARDS
    start = POD_INDEX * shard_size
    end = start + shard_size if POD_INDEX < TOTAL_SHARDS - 1 else len(all_cases)
    my_cases = all_cases[start:end]

    print(f"[Pod {POD_INDEX}] Evaluating cases {start}–{end-1} ({len(my_cases)} total)")

    metrics = [
        StructuralComplianceMetric(),
        RecommendationFormatMetric(),
        ForbiddenLanguageMetric(),
        RiskFlagsCountMetric(),
        build_geval_metric(JUDGE_MODEL),
        build_faithfulness_metric(JUDGE_MODEL),
    ]

    test_cases = []
    for case in my_cases:
        print(f"[Pod {POD_INDEX}] Generating memo for {case.get('profile_id', '?')}")
        actual_output = generate_via_vllm(case["input"])
        test_cases.append(
            LLMTestCase(
                input=case["input"],
                actual_output=actual_output,
                expected_output=case["output"],
                retrieval_context=[case["input"]],   # for faithfulness
            )
        )

    print(f"[Pod {POD_INDEX}] Running DeepEval metrics on {len(test_cases)} cases")
    results = evaluate(test_cases, metrics, run_async=False, print_results=False)

    shard_scores = aggregate_scores(results.test_results, metrics)
    print(f"[Pod {POD_INDEX}] Scores: {shard_scores}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    exp_id = _get_or_create_experiment("rinlekha-evaluation")
    with mlflow.start_run(run_name=f"eval_shard_{POD_INDEX}", experiment_id=exp_id):
        mlflow.log_metrics(shard_scores)
        mlflow.log_params({
            "pod_index":       POD_INDEX,
            "cases_start":     start,
            "cases_end":       end,
            "cases_evaluated": len(my_cases),
            "judge_model":     JUDGE_MODEL,
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/shard_{POD_INDEX}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "pod_index":   POD_INDEX,
                "cases_start": start,
                "cases_end":   end,
                "scores":      shard_scores,
                "raw_results": serialize_results(results.test_results),
            },
            f,
            indent=2,
        )

    print(f"[Pod {POD_INDEX}] Complete. Results written to {output_path}")


if __name__ == "__main__":
    main()
