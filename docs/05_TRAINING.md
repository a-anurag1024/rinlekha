# RinLekha — Training Gameplan

## Objective

Fine-tune Gemma 4 E4B using QLoRA on 640 training examples.
Run 3 experiments with varying hyperparameters. Track all runs in MLflow.
Push best checkpoint to HuggingFace Hub.

---

## Hardware and Environment

```
GPU:    6GB RTX (CUDA 12.x, visible in WSL2)
RAM:    16GB system
OS:     WSL2 Ubuntu 22.04
Python: 3.11 (conda env: rinlekha)

VRAM budget estimate:
  Base model (4-bit):        ~3.0 GB
  LoRA adapters:             ~0.3 GB
  Optimizer states:          ~1.2 GB
  Activations + gradients:   ~1.2 GB
  ─────────────────────────────────
  Total estimated:           ~5.7 GB  ← fits in 6GB with margin
```

---

## Dependencies

```bash
conda create -n rinlekha python=3.11 -y
conda activate rinlekha

pip install unsloth
pip install trl transformers datasets
pip install bitsandbytes accelerate peft
pip install mlflow
pip install huggingface_hub
```

Verify GPU:
```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # Your RTX name
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # ~6.0
```

---

## Training Configuration

```yaml
# config/training_config.yaml

model:
  base: "google/gemma-4-e4b-it"
  max_seq_length: 2048
  load_in_4bit: true
  dtype: null  # auto-detect

lora:
  r: 16                          # rank — experiment variable
  alpha: 32                      # 2x rank is standard
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: none
  use_gradient_checkpointing: unsloth

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4    # effective batch = 8
  num_train_epochs: 3               # experiment variable
  learning_rate: 2.0e-4             # experiment variable
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  weight_decay: 0.01
  optimizer: adamw_8bit
  evaluation_strategy: steps
  eval_steps: 50
  logging_steps: 10
  save_strategy: steps
  save_steps: 100
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  output_dir: ./outputs/rinlekha-run

data:
  dataset: "your-username/rinlekha-training-data"
  train_split: train
  eval_split: validation
  text_field: text               # after formatting
  max_samples: null              # use all

mlflow:
  tracking_uri: "sqlite:///mlflow/mlflow.db"
  experiment_name: "rinlekha-finetuning"
```

---

## Training Script

```python
# training/train.py

import mlflow
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForSeq2Seq
from transformers import TrainingArguments
from datasets import load_dataset

def format_prompt(example: dict) -> dict:
    """Alpaca-style prompt formatting"""
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}


def train(config_path: str, run_name: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── MLflow setup ──────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name):

        # Log all config parameters
        mlflow.log_params({
            "base_model": cfg["model"]["base"],
            "lora_rank": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["alpha"],
            "learning_rate": cfg["training"]["learning_rate"],
            "epochs": cfg["training"]["num_train_epochs"],
            "batch_size": cfg["training"]["per_device_train_batch_size"],
            "grad_accumulation": cfg["training"]["gradient_accumulation_steps"],
            "effective_batch_size": (
                cfg["training"]["per_device_train_batch_size"]
                * cfg["training"]["gradient_accumulation_steps"]
            ),
            "max_seq_length": cfg["model"]["max_seq_length"],
            "dataset": cfg["data"]["dataset"],
            "quantization": "4bit_qlora"
        })

        # ── Model loading ──────────────────────────────────────────
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model"]["base"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=cfg["model"]["dtype"],
            load_in_4bit=cfg["model"]["load_in_4bit"]
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora"]["r"],
            target_modules=cfg["lora"]["target_modules"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias=cfg["lora"]["bias"],
            use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
            random_state=42
        )

        # Log trainable parameter count
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mlflow.log_params({
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(trainable / total * 100, 3)
        })

        # ── Dataset loading ────────────────────────────────────────
        dataset = load_dataset(cfg["data"]["dataset"])
        train_data = dataset["train"].map(format_prompt)
        eval_data = dataset["validation"].map(format_prompt)

        mlflow.log_params({
            "train_samples": len(train_data),
            "eval_samples": len(eval_data)
        })

        # ── Training arguments ─────────────────────────────────────
        args = TrainingArguments(
            output_dir=cfg["training"]["output_dir"],
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
            warmup_ratio=cfg["training"]["warmup_ratio"],
            weight_decay=cfg["training"]["weight_decay"],
            optim=cfg["training"]["optimizer"],
            evaluation_strategy=cfg["training"]["evaluation_strategy"],
            eval_steps=cfg["training"]["eval_steps"],
            logging_steps=cfg["training"]["logging_steps"],
            save_strategy=cfg["training"]["save_strategy"],
            save_steps=cfg["training"]["save_steps"],
            load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
            metric_for_best_model=cfg["training"]["metric_for_best_model"],
            greater_is_better=cfg["training"]["greater_is_better"],
            fp16=True,
            report_to="mlflow"           # automatic step-level logging
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=eval_data,
            dataset_text_field=cfg["data"]["text_field"],
            max_seq_length=cfg["model"]["max_seq_length"],
            args=args
        )

        # ── Training ───────────────────────────────────────────────
        train_result = trainer.train()

        # Log final metrics
        mlflow.log_metrics({
            "final_train_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "train_runtime_seconds": train_result.metrics["train_runtime"]
        })

        # Eval on validation set
        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "final_eval_loss": eval_results["eval_loss"],
            "eval_perplexity": eval_results.get("eval_perplexity", -1)
        })

        # ── Save + push ────────────────────────────────────────────
        model.save_pretrained(cfg["training"]["output_dir"])
        tokenizer.save_pretrained(cfg["training"]["output_dir"])

        # Log model artifact path
        mlflow.log_artifact(cfg["training"]["output_dir"])
        mlflow.log_param("local_model_path", cfg["training"]["output_dir"])

        print(f"Run complete. eval_loss={eval_results['eval_loss']:.4f}")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.yaml"
    run_name = sys.argv[2] if len(sys.argv) > 2 else "run_default"
    train(config_path, run_name)
```

---

## Three Experiments

Run sequentially, varying one parameter at a time.
This is ablation study at minimal scale — and it's the right engineering practice.

```bash
# Run 1 — baseline
python training/train.py config/training_config.yaml "r8_lr2e4_ep3"
# lora_rank=8, lr=2e-4, epochs=3

# Run 2 — higher rank (default config)
python training/train.py config/run2_config.yaml "r16_lr2e4_ep3"
# lora_rank=16, lr=2e-4, epochs=3

# Run 3 — lower lr, more epochs
python training/train.py config/run3_config.yaml "r16_lr1e4_ep5"
# lora_rank=16, lr=1e-4, epochs=5
```

**Expected training time per run: ~2.5–3.5 hours on 6GB RTX**

**Selecting best run:**
- Primary criterion: lowest `eval_loss`
- Tiebreaker: fewer trainable parameters (prefers r=8 over r=16 if loss equal)
- Disqualifier: `final_train_loss` > `final_eval_loss` × 1.3 (overfitting signal)

---

## HuggingFace Hub Push

```python
# Push best model after selecting winning run

from huggingface_hub import HfApi
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./outputs/rinlekha-run-best"
)

model.push_to_hub(
    "your-username/rinlekha-gemma4-e4b-finetuned",
    commit_message="v1.0 — best of 3 runs, r16_lr2e4_ep3"
)
tokenizer.push_to_hub(
    "your-username/rinlekha-gemma4-e4b-finetuned"
)

# Log HF URL back to MLflow
import mlflow
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_param(
        "hf_model_url",
        "https://huggingface.co/your-username/rinlekha-gemma4-e4b-finetuned"
    )
```

---

## Monitoring During Training

Open MLflow UI while training runs:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5001
# Open: http://localhost:5001
```

Watch for:
- Train loss decreasing smoothly (not oscillating)
- Eval loss tracking train loss (not diverging — overfitting signal)
- Eval loss floor: credit memo generation should reach ~1.2–1.8
- Learning rate decay curve (cosine schedule)

**Stop early if:** eval_loss stops improving for 3 consecutive eval steps
(MLflow makes this visible in real time)

---

## Deliverables Checklist

```
□ config/training_config.yaml          — base config
□ config/run2_config.yaml              — rank 16 config
□ config/run3_config.yaml              — lower lr config
□ training/train.py                    — training script
□ MLflow experiment: 3 runs logged     — all hyperparams + metrics
□ Best checkpoint saved locally        — ./outputs/rinlekha-run-best/
□ Model pushed to HuggingFace Hub      — v1.0 tagged
□ training_summary.md                  — which run won and why
```