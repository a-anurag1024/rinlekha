# RinLekha — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OFFLINE PHASE                               │
│                    (runs once, on developer machine)                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     DATA LAYER                               │   │
│  │                                                              │   │
│  │  profile_generator.py                                        │   │
│  │  ┌─────────────────────────────────────────────────────┐     │   │
│  │  │  Ray Parallel Workers (x8)                          │    │    │
│  │  │  - Sample borrower dimensions                       │    │    │
│  │  │  - Compute derived fields (FOIR, EMI, LTV)          │    │    │
│  │  │  - Apply underwriting decision logic                │    │    │
│  │  │  → 800 structured borrower profiles                 │    │    │
│  │  └──────────────────────┬──────────────────────────────┘    │    │
│  │                         ↓                                   │    │
│  │  memo_synthesizer.py                                        │   │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  Ray Parallel Workers (x8) → Claude API             │    │  │
│  │  │  - Synthesis prompt → credit memo generation        │    │  │
│  │  │  → 800 (profile, memo) pairs                        │    │  │
│  │  └──────────────────────┬──────────────────────────────┘    │  │
│  │                         ↓                                   │  │
│  │  quality_reviewer.py                                        │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  - Automated structural checks on all 800           │    │  │
│  │  │  - Flag ~15-20% for manual review                   │    │  │
│  │  │  - Regenerate flagged examples                      │    │  │
│  │  │  → 640 train / 80 val / 80 test (JSONL)             │    │  │
│  │  └──────────────────────┬──────────────────────────────┘    │  │
│  │                         ↓                                   │  │
│  │               HuggingFace Datasets Hub                      │  │
│  │               rinlekha-training-data@v1.0                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    TRAINING LAYER                            │  │
│  │                                                              │  │
│  │  train.py                                                    │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  Unsloth + TRL SFTTrainer                           │    │  │
│  │  │  Base: google/gemma-4-e4b-it                        │    │  │
│  │  │  Method: QLoRA (4-bit, rank=16)                     │    │  │
│  │  │  Hardware: 6GB RTX + 16GB RAM (WSL2)                │    │  │
│  │  │                                                     │    │  │
│  │  │  Experiment tracking → MLflow (3 runs)              │    │  │
│  │  │  Best checkpoint → HuggingFace Hub                  │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  │                                                              │  │
│  │  MLflow Server (local SQLite)                                │  │
│  │  - Hyperparameters, loss curves, val metrics                 │  │
│  │  - Dataset version reference                                 │  │
│  │  - Model artifact links                                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  EVALUATION LAYER                            │  │
│  │           (Kubernetes cluster — kind, local)                 │  │
│  │                                                              │  │
│  │  ┌─────────────────┐  ┌────────────────┐  ┌─────────────┐    │  │
│  │  │ vLLM Deployment │  │MLflow Deployment│ │   eval-PVC  │    │  │
│  │  │ :8000 (Service) │  │ :5000 (Service)│  │  (shared    │    │  │
│  │  │ rinlekha model  │  │receives logs   │  │  results)   │    │  │
│  │  └────────┬────────┘  └───────┬────────┘  └───────┬─────┘    │  │
│  │           │                   │                   │         │  │
│  │  ┌────────┴───────────────────┴───────────────────┴──────┐ │  │
│  │  │          Evaluation Job (batch/v1)                    │ │  │
│  │  │          completions=3  parallelism=3  Indexed        │ │  │
│  │  │                                                       │ │  │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │  │
│  │  │  │ eval-pod-0   │ │ eval-pod-1   │ │ eval-pod-2   │  │ │  │
│  │  │  │ cases 0-26   │ │ cases 27-53  │ │ cases 54-80  │  │ │  │
│  │  │  │              │ │              │ │              │  │ │  │
│  │  │  │ DeepEval:    │ │ DeepEval:    │ │ DeepEval:    │  │ │  │
│  │  │  │ - Structural │ │ - Structural │ │ - Structural │  │ │  │
│  │  │  │ - Forbidden  │ │ - Forbidden  │ │ - Forbidden  │  │ │  │
│  │  │  │ - GEval      │ │ - GEval      │ │ - GEval      │  │ │  │
│  │  │  │ - Faithfulns │ │ - Faithfulns │ │ - Faithfulns │  │ │  │
│  │  │  │ - Adversarial│ │ - Adversarial│ │ - Adversarial│  │ │  │
│  │  │  └──────────────┘ └──────────────┘ └──────────────┘  │ │  │
│  │  └───────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         ONLINE PHASE                                │
│                    (runs on demand, local machine)                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    SERVING LAYER                             │   │
│  │                                                              │   │
│  │  User Input (Gradio)                                         │   │
│  │        ↓                                                     │   │
│  │  LangChain Pipeline                                          │   │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │  1. Input validation + profile formatting           │     │  │
│  │  │  2. PromptTemplate assembly                         │     │  │
│  │  │  3. vLLM call (OpenAI-compatible API, local)        │     │  │
│  │  │  4. CreditMemo Pydantic output parser               │     │  │
│  │  │  5. Parse success scoring                           │     │  │
│  │  └──────────────────────┬──────────────────────────────┘     │  │
│  │                         ↓                                    │  │
│  │  vLLM Server (local, standalone)                             │  │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │  - Serves rinlekha-gemma4-e4b-finetuned             │    │  │
│  │  │  - OpenAI-compatible REST API                       │    │  │
│  │  │  - PagedAttention for efficient memory use          │    │  │
│  │  │  - Temperature=0.1 for format consistency           │    │  │
│  │  └──────────────────────┬──────────────────────────────┘    │  │
│  │                         ↓                                   │  │
│  │  Langfuse Observability                                     │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  @observe decorator traces every call:              │    │  │
│  │  │  - Latency, token usage                             │    │  │
│  │  │  - Custom scores: parse_success, structural         │    │  │
│  │  │  - Metadata: cibil_band, foir_band, loan_purpose    │    │  │
│  │  └──────────────────────┬──────────────────────────────┘    │  │
│  │                         ↓                                   │  │
│  │  Gradio App                                                 │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │  - Structured memo display                          │    │  │
│  │  │  - Risk dashboard (grade + flags)                   │    │  │
│  │  │  - Format compliance checker                        │    │  │
│  │  │  - Baseline comparison tabs                         │    │  │
│  │  │  - "Running locally" privacy badge                  │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### Data Layer

| Component | Tool | Responsibility |
|---|---|---|
| Profile Generator | Python + Ray | Parallel generation of 800 synthetic borrower profiles |
| Decision Engine | Python | Rule-based underwriting logic → outcome + conditions |
| Memo Synthesizer | Ray + OpenAI API | Parallel credit memo generation from profiles |
| Quality Checker | Python | Automated structural validation, flag bad examples |
| Dataset Registry | HuggingFace Datasets | Versioned storage, reproducible splits |

### Training Layer

| Component | Tool | Responsibility |
|---|---|---|
| Fine-tuning Engine | Unsloth + TRL | QLoRA training on 6GB VRAM |
| Experiment Tracker | MLflow | Hyperparameter logging, loss curves, model artifacts |
| Model Registry | HuggingFace Hub | Versioned model storage, public deployment |

### Evaluation Layer

| Component | Tool | Responsibility |
|---|---|---|
| Cluster Manager | kind | Local Kubernetes cluster (no cloud cost) |
| Inference Service | vLLM (k8s Deployment) | Serves model to eval pods via internal DNS |
| Tracking Service | MLflow (k8s Deployment) | Receives distributed eval results |
| Eval Orchestrator | Kubernetes Job (Indexed) | 3 parallel pods, each evaluates one shard |
| Metric Framework | DeepEval | Structural, semantic, adversarial, LLM-as-judge |
| Package Manager | Helm | Single-command cluster deploy/teardown |

### Serving Layer

| Component | Tool | Responsibility |
|---|---|---|
| Inference Server | vLLM (standalone) | Local model serving, OpenAI-compatible API |
| Pipeline | LangChain | Prompt assembly, output parsing, error handling |
| Observability | Langfuse | Trace logging, latency, custom quality scores |
| UI | Gradio | Demo interface with baseline comparison |

---

## Data Flow

```
Borrower Profile (JSON)
        ↓
[Input Validation — LangChain]
        ↓
[Prompt Assembly — LangChain PromptTemplate]
        ↓
[vLLM Inference — local, <3s]
        ↓
[Raw Text Output]
        ↓
[CreditMemo Parser — Pydantic]
        ↓
[Structured CreditMemo Object]    →    [Langfuse Trace]
        ↓
[Gradio Render]
```

---

## Infrastructure

```
Development machine: Windows 11, WSL2 (Ubuntu 22.04)
GPU:                 6GB RTX (CUDA visible in WSL2)
RAM:                 16GB
Container runtime:   Docker Desktop
K8s (local):         kind (Kubernetes IN Docker)
Python env:          Conda (rinlekha env)
```

---

## Key Design Decisions

**Why kind over minikube?**
kind runs Kubernetes nodes as Docker containers — faster to start,
easier to reset, and closer to how cloud k8s clusters (EKS, GKE) work
internally. Manifests written for kind deploy to production k8s with
minimal changes.

**Why vLLM over Ollama for serving?**
vLLM provides an OpenAI-compatible REST API, enabling LangChain
integration without custom adapters. PagedAttention gives better
throughput at the same VRAM budget. Ollama is used only for the
"how to run" consumer-facing instructions in the model card.

**Why standalone vLLM (not k8s) for demo serving?**
GPU passthrough to kind pods on Windows/WSL2 is complex and
brittle. The k8s cluster is used for CPU-bound evaluation jobs.
The demo serving stack runs directly in WSL2 where GPU access
is reliable.

**Why LangChain over direct vLLM calls?**
Output parsing, error handling, and retry logic are non-trivial.
LangChain's Pydantic output parser with fallback chains gives a
clean abstraction. In production, this pipeline extends naturally
to multi-step chains (e.g., add a second model call for
explainability summary).

**Why Langfuse over direct logging?**
Langfuse provides structured trace visualization, score
distributions over time, and token usage tracking out of the box.
This mirrors how production ML teams monitor deployed LLMs.