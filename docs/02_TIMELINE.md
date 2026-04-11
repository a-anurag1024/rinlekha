# RinLekha — Project Timeline

## Overview

| Phase |  Deliverable |
|---|---|
| 0 — Setup | Repo, env, accounts, tooling verified |
| 1 — Data Pipeline | Clean JSONL training file + HF dataset |
| 2 — Training | Fine-tuned model on HF Hub + MLflow runs |
| 3 — Kubernetes + Eval | k8s cluster + eval job manifests |
| 4 — Eval Execution | Full eval results + baseline comparison |
| 5 — Serving Pipeline | vLLM + LangChain + Langfuse live |
| 6 — App + Packaging | Gradio app + model card + README |


---

## Phase 0 — Setup

```
Tasks:
□ Initialize GitHub repo with directory structure
□ Set up WSL2 with CUDA verified (nvidia-smi in WSL)
□ Install base dependencies (conda env + requirements.txt)
□ Create HuggingFace account → dataset repo → model repo
□ Set up MLflow local server (SQLite backend, verify UI loads)
□ Create Langfuse account (free cloud tier)
□ Set up .env with all API keys (Claude, HF, Langfuse)
□ Install Docker Desktop + kind
□ Verify: kind create cluster works

Deliverable: README.md skeleton committed, all accounts active,
             GPU accessible in WSL2
```

---

## Phase 1 — Data Pipeline

```
□ Schema design — borrower profile dimensions finalized
□ Underwriting decision logic implemented + unit tested
□ Profile generator with Ray parallelism working
□ Generate 1000 raw profiles (800 train, 200 eval buffer)

□ GPT API synthesis prompt engineered
□ First batch of 50 memos generated + manually reviewed
□ Prompt iterated based on quality issues found
□ Full 800 memo generation run via Ray (parallel API calls)

□ Automated quality checker built + run on all 800
□ Flagged examples identified (~15-20%)
□ Manual review of flagged subset
□ Regenerate failed examples
□ Train/val/test split (640/80/80)
□ Push to HuggingFace Datasets with version tag v1.0

Deliverable: rinlekha-training-data on HuggingFace Hub
             640 train / 80 val / 80 test examples
             Quality report: structural pass rate of generated memos
```

---

## Phase 2 — Training

```
□ Unsloth + TRL environment verified in WSL2
□ training_config.yaml written
□ MLflow experiment created: "rinlekha-finetuning"
□ First training run started: Run 1 (lora_rank=8, lr=2e-4, 3 epochs)

□ Monitor training loss curve in MLflow UI
□ While GPU runs: write evaluation structural checker (structural_eval.py)
□ Run 1 completes → log eval metrics on val set

□ Run 2: lora_rank=16, lr=2e-4, 3 epochs
□ Run 3: lora_rank=16, lr=1e-4, 5 epochs
□ Compare 3 runs in MLflow UI — pick best checkpoint
□ Push best model to HuggingFace Hub
□ Tag model version v1.0 on Hub

Deliverable: rinlekha-gemma4-e4b-finetuned on HuggingFace Hub
             3 MLflow runs with full parameter + metric logging
             Best checkpoint identified with justification noted
```

---

## Phase 3 — Kubernetes Setup

```
□ Dockerfiles written for 3 services:
  - vLLM server (serves fine-tuned model)
  - Eval worker (runs DeepEval shards)
  - MLflow server (receives eval logs)
□ Docker images built and tested locally
□ kind cluster created with 1 control-plane + 2 workers

□ Kubernetes namespace: rinlekha
□ Manifests written and applied:
  - vllm-deployment.yaml + Service
  - mlflow-deployment.yaml + Service + PVC
  - eval-results PVC
□ vLLM pod verified: curl http://vllm-service:8000/health from inside cluster
□ MLflow pod verified: UI accessible via port-forward

□ Eval Job manifest written (3 pods, indexed completion)
□ run_eval_shard.py written — pod reads its index, evaluates its shard
□ Helm chart created wrapping all manifests
□ helm install rinlekha tested end-to-end
□ Teardown + redeploy cycle verified clean

Deliverable: Full k8s stack deployable with: helm install rinlekha ./helm/rinlekha
             All manifests committed to repo
```

---

## Phase 4 — Evaluation Execution

```
□ DeepEval metrics implemented:
  - StructuralComplianceMetric (custom)
  - ForbiddenLanguageMetric (custom)
  - GEval credit quality metric
  - FaithfulnessMetric
□ Adversarial test suite — 8 cases written:
  - Extreme FOIR (>75%)
  - Missing CIBIL score
  - Inconsistent profile
  - OOD loan amount
  - Perfect profile, very short tenure
  - Active delinquency (hard decline)
  - Settled account edge case
  - Self-employed income spike

□ kubectl apply -f eval-job.yaml
□ Watch 3 pods run in parallel: kubectl get pods -w
□ Collect shard results from PVC
□ Aggregate results across shards
□ All metrics logged to MLflow

□ Baseline comparison run:
  - Prompted Gemma 4 E4B (base, no fine-tuning)
  - Prompted Claude Sonnet (via API, 30 examples only — cost)
  - Fine-tuned RinLekha
□ Comparison table generated + logged to MLflow
□ Model card drafted with evaluation results filled in

Deliverable: Full evaluation report
             Three-way baseline comparison table
             Adversarial suite results with failure mode analysis
             Model card v1 with all eval numbers
```

---

## Phase 5 — Serving Pipeline

```
□ vLLM standalone server verified (outside k8s, for demo)
□ LangChain pipeline built:
  - PromptTemplate
  - vLLM LLM connector
  - CreditMemo Pydantic output parser
  - Parse success rate measured on 80 test cases
□ Langfuse integration:
  - @observe decorator on generate_credit_memo()
  - Custom scores: parse_success, structural_compliance
  - Metadata: cibil_band, foir_band, employment_type

□ End-to-end pipeline test: profile → vLLM → parser → Langfuse trace
□ Langfuse dashboard verified: traces visible, scores logged
□ Latency benchmarked: avg ms per call logged

□ Pipeline error handling:
  - Parse failure fallback (return raw text + flag)
  - vLLM timeout handling
  - Out-of-range input validation
□ serving/pipeline.py finalized and committed

Deliverable: Fully working serving pipeline
             Langfuse dashboard with live traces
             Parse success rate ≥ 92% verified
```

---

## Phase 6 — App + Packaging

```
□ Gradio app built:
  - Borrower profile input form (left panel)
  - Generated memo display (right panel)
  - Risk dashboard JSON (risk grade, flags)
  - Format compliance checker output
  - Baseline comparison tabs (finetuned vs base)
□ "Running locally — no data transmitted" badge prominent

□ README.md complete:
  - Problem statement
  - Architecture diagram (ASCII)
  - Dataset construction methodology
  - Fine-tuning decision rationale
  - Evaluation results table
  - Corporate analog section
  - How to run locally (Ollama command)
  - Known limitations
□ model_card.md finalized on HuggingFace Hub

□ Final HuggingFace publish — model + dataset + model card
□ MLflow experiment made shareable (screenshots)
□ Repo cleaned, all secrets removed from history
□ GitHub README rendered correctly verified

Deliverable: Complete shareable portfolio artifact
             Public HF model + dataset
             GitHub repo ready to share
```

---

## Risk and Buffer

| Risk | Mitigation |
|---|---|
| Unsloth + Gemma 4 E4B not yet stable | Fall back to Gemma 3 4B — identical pipeline |
| WSL2 CUDA issues | Resolve in Phase 0 — don't discover on training day |
| kind cluster resource limits on 16GB RAM | Reduce eval parallelism to 2 pods if needed |
| Weekend 1 data quality poor | Buffer: 200 extra profiles generated, use for replacement |