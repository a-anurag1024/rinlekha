"""Aggregate shard results after the eval Job completes.

Run from the host (after kubectl cp or via port-forward to the PVC):
  python evaluation/aggregate_results.py --results-dir /tmp/results
"""
import argparse
import glob
import json
import os

import mlflow


MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")


def aggregate_all_shards(results_dir: str) -> dict[str, float]:
    shard_files = sorted(glob.glob(f"{results_dir}/shard_*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.json files found in {results_dir}")

    shards = [json.load(open(f)) for f in shard_files]
    print(f"Aggregating {len(shards)} shards from {results_dir}")

    total_cases = sum(s["cases_end"] - s["cases_start"] for s in shards)
    aggregated: dict[str, float] = {}
    for metric in shards[0]["scores"]:
        weighted_sum = sum(
            s["scores"][metric] * (s["cases_end"] - s["cases_start"])
            for s in shards
            if metric in s["scores"]
        )
        aggregated[metric] = weighted_sum / total_cases

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("rinlekha-evaluation")
    exp_id = exp.experiment_id if exp else client.create_experiment("rinlekha-evaluation")

    with mlflow.start_run(run_name="eval_aggregated_final", experiment_id=exp_id):
        mlflow.log_metrics(aggregated)
        mlflow.log_params({
            "total_cases_evaluated": total_cases,
            "shards_aggregated":     len(shards),
        })

    print("\nFinal aggregated evaluation results:")
    for metric, score in aggregated.items():
        print(f"  {metric}: {score:.4f}")

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/results")
    args = parser.parse_args()
    aggregate_all_shards(args.results_dir)


if __name__ == "__main__":
    main()
