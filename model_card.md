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

## Adversarial Evaluation — 8 Edge Cases

Hand-crafted cases targeting boundary conditions not well-represented in the test set:
extreme FOIR, missing CIBIL history, contradictory signals, grossly disproportionate loan amounts,
active delinquency, settled accounts, and unverifiable self-employed income.

| Case | Expected | Description |
|------|----------|-------------|
| adv_01 | DECLINE | FOIR 113% post-loan |
| adv_02 | CONDITIONAL APPROVE | CIBIL 0, no credit history |
| adv_03 | DECLINE | CIBIL 820 but FOIR 87% post-loan |
| adv_04 | DECLINE | LTI 72×, FOIR 228% |
| adv_05 | CONDITIONAL APPROVE | CIBIL 798, only 0.4yr employment |
| adv_06 | DECLINE | 5 missed payments in 24 months |
| adv_07 | DECLINE | 2 settled accounts (prior write-offs) |
| adv_08 | CONDITIONAL APPROVE | High income, 0.6yr self-employed tenure |

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
