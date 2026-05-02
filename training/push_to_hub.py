"""Push the winning LoRA adapter to HuggingFace Hub and record the URL in MLflow."""
import sys
import yaml
import mlflow
from pathlib import Path
from huggingface_hub import HfApi


EXCLUDE_PATTERNS = {"checkpoint-*", "rolling_checkpoint"}


def _collect_files(model_dir: Path) -> list[Path]:
    files = []
    for p in model_dir.iterdir():
        if any(p.match(pat) for pat in EXCLUDE_PATTERNS):
            continue
        if p.is_file():
            files.append(p)
    return files


def push(model_dir: str, repo_id: str, mlflow_cfg: dict, run_name: str) -> None:
    model_path = Path(model_dir)
    files = _collect_files(model_path)

    print(f"Pushing {len(files)} files to {repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  uploaded: {f.name}")

    hf_url = f"https://huggingface.co/{repo_id}"
    print(f"\nModel live at: {hf_url}")

    # Log HF URL back to the winning MLflow run
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(mlflow_cfg["experiment_name"])
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            client.log_param(runs[0].info.run_id, "hf_model_url", hf_url)
            print(f"Logged hf_model_url to MLflow run: {runs[0].info.run_id}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "training/config/run3_config.yaml"
    repo_id     = sys.argv[2] if len(sys.argv) > 2 else "a-anurag1024/rinlekha-gemma3-4b-finetuned"
    run_name    = sys.argv[3] if len(sys.argv) > 3 else "r16_lr1e4_ep5"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    push(
        model_dir=cfg["training"]["output_dir"],
        repo_id=repo_id,
        mlflow_cfg=cfg["mlflow"],
        run_name=run_name,
    )
