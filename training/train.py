import sys
import yaml
import mlflow
from pathlib import Path
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


class RollingCheckpointCallback(TrainerCallback):
    """Saves LoRA adapter weights to a single overwritten dir every N steps for crash recovery."""

    def __init__(self, rolling_dir: str, every_n_steps: int = 10):
        self.rolling_dir = Path(rolling_dir)
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            self.rolling_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(self.rolling_dir))
            if tokenizer is not None:
                tokenizer.save_pretrained(str(self.rolling_dir))
            state.save_to_json(str(self.rolling_dir / "trainer_state.json"))


def format_alpaca(example: dict) -> dict:
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}


def _get_mlflow_run_id(experiment_name: str, run_name: str) -> str | None:
    """Return the run_id of an existing MLflow run with this name, or None."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _latest_checkpoint(output_dir: str) -> str | None:
    rolling = Path(output_dir) / "rolling_checkpoint"
    if rolling.exists() and (rolling / "trainer_state.json").exists():
        return str(rolling)
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(checkpoints[-1]) if checkpoints else None


def train(config_path: str, run_name: str, resume: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    run_id = _get_mlflow_run_id(cfg["mlflow"]["experiment_name"], run_name) if resume else None
    if run_id:
        print(f"Resuming MLflow run: {run_id}")

    with mlflow.start_run(run_name=run_name, run_id=run_id):
        lora_r = cfg["lora"]["r"]
        t = cfg["training"]

        mlflow.log_params({
            "base_model":        cfg["model"]["base"],
            "lora_rank":         lora_r,
            "lora_alpha":        cfg["lora"]["alpha"],
            "learning_rate":     t["learning_rate"],
            "epochs":            t["num_train_epochs"],
            "batch_size":        t["per_device_train_batch_size"],
            "grad_accumulation": t["gradient_accumulation_steps"],
            "effective_batch":   t["per_device_train_batch_size"] * t["gradient_accumulation_steps"],
            "max_seq_length":    cfg["model"]["max_seq_length"],
            "dataset":           cfg["data"]["dataset"],
            "quantization":      "4bit_qlora",
        })

        # ── Model ──────────────────────────────────────────────────────────
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model"]["base"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=cfg["model"]["dtype"],
            load_in_4bit=cfg["model"]["load_in_4bit"],
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=cfg["lora"]["target_modules"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias=cfg["lora"]["bias"],
            use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
            random_state=42,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mlflow.log_params({
            "trainable_params": trainable,
            "total_params":     total,
            "trainable_pct":    round(trainable / total * 100, 3),
        })

        # ── Data ───────────────────────────────────────────────────────────
        dataset = load_dataset(cfg["data"]["dataset"])
        train_data = dataset["train"].map(format_alpaca)
        eval_data  = dataset["validation"].map(format_alpaca)

        mlflow.log_params({
            "train_samples": len(train_data),
            "eval_samples":  len(eval_data),
        })

        # ── Trainer ────────────────────────────────────────────────────────
        output_dir = cfg["training"]["output_dir"]

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=SFTConfig(
                dataset_text_field=cfg["data"]["text_field"],
                max_seq_length=cfg["model"]["max_seq_length"],
                per_device_train_batch_size=t["per_device_train_batch_size"],
                gradient_accumulation_steps=t["gradient_accumulation_steps"],
                num_train_epochs=t["num_train_epochs"],
                learning_rate=t["learning_rate"],
                lr_scheduler_type=t["lr_scheduler_type"],
                warmup_ratio=t["warmup_ratio"],
                weight_decay=t["weight_decay"],
                optim=t["optimizer"],
                eval_strategy=t["eval_strategy"],
                eval_steps=t["eval_steps"],
                logging_steps=t["logging_steps"],
                save_strategy=t["save_strategy"],
                save_steps=t["save_steps"],
                load_best_model_at_end=t["load_best_model_at_end"],
                metric_for_best_model=t["metric_for_best_model"],
                greater_is_better=t["greater_is_better"],
                output_dir=output_dir,
                bf16=True,
                per_device_eval_batch_size=1,
                report_to="mlflow",
            ),
        )

        trainer.add_callback(RollingCheckpointCallback(
            rolling_dir=f"{output_dir}/rolling_checkpoint",
            every_n_steps=10,
        ))

        # Only compute loss on the assistant response, not on the instruction/input
        trainer = train_on_responses_only(
            trainer,
            instruction_part="### Instruction:\n",
            response_part="### Response:\n",
        )

        # ── Train ──────────────────────────────────────────────────────────
        checkpoint = _latest_checkpoint(output_dir) if resume else None
        if checkpoint:
            print(f"Resuming from checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        mlflow.log_metrics({
            "final_train_loss":        train_result.training_loss,
            "total_steps":             train_result.global_step,
            "train_runtime_seconds":   train_result.metrics["train_runtime"],
        })

        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "final_eval_loss": eval_results["eval_loss"],
        })

        # ── Save ───────────────────────────────────────────────────────────
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_param("local_model_path", output_dir)

        print(f"\nRun complete — eval_loss={eval_results['eval_loss']:.4f}")
        print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "training/config/training_config.yaml"
    run_name    = sys.argv[2] if len(sys.argv) > 2 else "run_default"
    resume      = "--resume" in sys.argv
    train(config_path, run_name, resume)
