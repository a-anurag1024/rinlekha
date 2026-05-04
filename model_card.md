---
language: en
license: gemma
base_model: google/gemma-3-4b-it
tags:
  - finance
  - credit-analysis
  - nbfc
  - qlora
  - lora
  - text-generation
datasets:
  - a-anurag1024/rinlekha-training-data
pipeline_tag: text-generation
---

# RinLekha — NBFC Credit Memo Generator

Fine-tuned **Gemma 3 4B IT** (QLoRA, rank 16) on 640 synthetic credit memo examples for Indian NBFCs.
Given a structured borrower profile, the model produces a fully-formatted 6-section credit memo with
an APPROVE / CONDITIONAL APPROVE / DECLINE recommendation.

## Model Details

| Property | Value |
|----------|-------|
| Base model | `google/gemma-3-4b-it` |
| Adapter | `a-anurag1024/rinlekha-gemma3-4b-finetuned` |
| GGUF | `a-anurag1024/rinlekha-gguf` (Q8_0, 4.1 GB) |
| Fine-tuning method | QLoRA (4-bit NF4, rank 16, α 32) |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Training examples | 640 |
| Instruction format | Alpaca |

## Intended Use

- **In-scope**: Generating draft credit assessment memos for personal/MSME loan applications at Indian NBFCs, following a defined institutional format.
- **Out-of-scope**: Final lending decisions, regulatory compliance, markets outside India, loan amounts beyond the training distribution.

The model is a research prototype. All outputs require human review before use in any real lending workflow.

## Training Data

Dataset: [`a-anurag1024/rinlekha-training-data`](https://huggingface.co/datasets/a-anurag1024/rinlekha-training-data)

- 800 synthetic borrower profiles generated via stratified sampling (employment type × income band × CIBIL tier × FOIR tier)
- Decision labels assigned by a deterministic rule engine matching real NBFC underwriting policy (55% FOIR ceiling, 620 CIBIL floor, settled-account thresholds)
- Memos synthesised with GPT-4.1-mini; QC pass filters hallucinations and format violations
- Final split: 640 train / 80 validation / 80 test (80/10/10)

## Training Procedure

Three-run ablation over rank and learning rate. Best run selected on validation loss:

| Run | Rank | LR | Val Loss |
|-----|------|----|----------|
| run1 | 8 | 2e-4 | 0.847 |
| run2 | 16 | 2e-4 | 0.831 |
| **run3** | **16** | **1e-4** | **0.798** |

Training environment: Unsloth + TRL on Google Colab T4 (15 GB VRAM).
LoRA adapter merged via PEFT `merge_and_unload()` and exported to GGUF Q8_0 via llama.cpp.

## Evaluation Results — 100 Test Cases

| Metric | Score | Description |
|--------|-------|-------------|
| StructuralCompliance | **1.000** | All 6 `## SECTION_NAME` headers present and correctly ordered |
| RecommendationFormat | **1.000** | `DECISION / CONDITIONS / RISK GRADE / DECISION AUTHORITY` exact |
| ForbiddenLanguage | **1.000** | No certainty language ("definitely", "guaranteed", "will", etc.) |
| RiskFlagsCount | **0.970** | Bulleted risk section contains 2–4 grounded flags |
| Faithfulness | **0.961** | Figures in memo match the input profile (GPT-4o-mini judge) |
| GEval | **0.832** | Overall analytical quality (GPT-4o-mini judge) |

Results by decision type:

| Decision | n | GEval | Faithfulness |
|----------|---|-------|--------------|
| APPROVE | 20 | 0.847 | 0.946 |
| CONDITIONAL APPROVE | 51 | 0.829 | 0.969 |
| DECLINE | 25 | 0.824 | 0.955 |

Evaluation harness: 6 custom [DeepEval](https://github.com/confident-ai/deepeval) metrics
(3 rule-based, 1 count-based, 2 LLM-judge via GPT-4o-mini). Results logged to MLflow.

## Baseline Comparison — RinLekha vs GPT-4o-mini (30 cases)

GPT-4o-mini is the cost-comparable API baseline — similar inference cost tier to a self-hosted 4B GGUF.
Both models receive the same format instructions in the system prompt; any gap on structural metrics
reflects format internalization from fine-tuning rather than prompt engineering.

| Metric | RinLekha | GPT-4o-mini |
|--------|----------|-------------|
| StructuralCompliance | **1.000** | 0.994 |
| RecommendationFormat | **1.000** | 1.000 |
| ForbiddenLanguage | **1.000** | 1.000 |
| RiskFlagsCount | **0.967** | 0.133 |
| GEval | 0.861 | **0.863** |
| Faithfulness | **0.964** | 0.940 |

The most striking gap is **RiskFlagsCount (0.967 vs 0.133)**: GPT-4o-mini consistently produces
fewer than 2 or more than 4 bulleted risk flags even with explicit instructions. This is the kind
of tight structural constraint that fine-tuning internalizes reliably while prompting does not.
GEval (analytical quality) is essentially tied, confirming the 4B model matches a much larger
prompted model on reasoning quality for this domain-specific task.

## Adversarial Evaluation — 8 Edge Cases

Hand-crafted cases targeting boundary conditions not well-represented in the test set.
Decision accuracy: **62% (5/8)** — all DECLINE cases correct, misses on CONDITIONAL APPROVE.

| Case | Expected | Actual | Result |
|------|----------|--------|--------|
| adv_01 — FOIR 113% post-loan | DECLINE | DECLINE | ✓ |
| adv_02 — CIBIL 0, no credit history | CONDITIONAL APPROVE | DECLINE | ✗ |
| adv_03 — CIBIL 820 but FOIR 87% | DECLINE | DECLINE | ✓ |
| adv_04 — LTI 72×, FOIR 228% | DECLINE | DECLINE | ✓ |
| adv_05 — CIBIL 798, only 0.4yr tenure | CONDITIONAL APPROVE | APPROVE | ✗ |
| adv_06 — 5 missed payments in 24m | DECLINE | DECLINE | ✓ |
| adv_07 — 2 settled accounts | DECLINE | DECLINE | ✓ |
| adv_08 — High income, 0.6yr self-employed | CONDITIONAL APPROVE | APPROVE | ✗ |

Adversarial aggregate metrics (format compliance holds even on edge cases):
StructuralCompliance 1.000 · RecommendationFormat 1.000 · ForbiddenLanguage 1.000 ·
RiskFlagsCount 1.000 · GEval 0.853 · Faithfulness 0.959

**Failure pattern**: the model correctly identifies hard DECLINE signals (FOIR ceiling, settled accounts,
delinquency) but conflates CONDITIONAL APPROVE with APPROVE on soft-boundary cases — particularly
short employment tenure and unverifiable self-employed income. The training set under-represents
these borderline conditions relative to clean approvals.

## Usage

```python
from llama_cpp import Llama

llm = Llama(model_path="rinlekha-q8.gguf", n_gpu_layers=99, n_ctx=2048)

INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)

prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{borrower_profile}\n\n### Response:\n"
out = llm(prompt, max_tokens=700, temperature=0.1, stop=["### Instruction:"])
print(out["choices"][0]["text"])
```

Or via the OpenAI-compatible server:

```bash
# Download GGUF
huggingface-cli download a-anurag1024/rinlekha-gguf rinlekha-q8.gguf --local-dir outputs/

# Start server (llama-cpp-python[server])
python -m llama_cpp.server --model outputs/rinlekha-q8.gguf --host 0.0.0.0 --port 8000 \
  --model_alias rinlekha --n_gpu_layers 99 --n_ctx 2048
```

## Expected Output Format

```
## APPLICANT SUMMARY
...

## DEBT SERVICEABILITY
...

## CREDIT BEHAVIOR
...

## RISK FLAGS
- Risk flag 1
- Risk flag 2

## RECOMMENDATION
DECISION: CONDITIONAL APPROVE
CONDITIONS: ...
RISK GRADE: B
DECISION AUTHORITY: Branch Credit Manager
REVIEW TRIGGER: ...

## ANALYST NOTES
...
```

## Limitations

- Training data is fully synthetic (GPT-4.1-mini authored). The model has not seen real loan files.
- The 55% FOIR ceiling and 620 CIBIL floor are hard-coded into the training labels; policy changes are not reflected without retraining.
- Max context is 2048 tokens. Very long borrower profiles may be truncated.
- The model occasionally generates mild hedged hallucinations in the Analyst Notes section; Faithfulness score of 0.961 reflects this.
- Evaluated at temperature 0.1 (near-greedy); higher temperatures degrade format compliance.

## Repository

[github.com/a-anurag1024/rinlekha](https://github.com/a-anurag1024/rinlekha)
