# RinLekha — Product Requirements Document

## Problem Statement

Banks, NBFCs, and lending fintechs employ credit analysts who manually write
structured credit assessment memos for every loan application. This process is:

- **Slow** — each memo takes 20–40 minutes of analyst time
- **Inconsistent** — format and risk language varies by analyst
- **Unscalable** — analyst headcount becomes the bottleneck at volume
- **Unparseable** — free-form memos don't integrate into downstream workflow systems

Existing LLM solutions (GPT-4, Claude API) cannot be used because **borrower
financial data is legally sensitive and cannot be transmitted to external APIs**.
RBI guidelines on data localisation and internal data governance policies at most
financial institutions make cloud API calls a non-starter for this use case.

---

## Solution

A **fine-tuned, locally-deployed** language model that generates structured credit
memos from structured borrower profiles — running entirely on-premise with zero
data transmission.

RinLekha is a **proof-of-concept** demonstrating:

1. The fine-tuning pipeline that a financial institution would replicate with
   proprietary internal data
2. The MLOps stack required to build, evaluate, and serve such a system
   responsibly
3. The production considerations (observability, governance, adversarial
   robustness) that differentiate a real system from a demo

---

## Core Use Case

```
Input:   Structured borrower profile
         (income, CIBIL score, FOIR, employment, loan request details)

Output:  Structured credit memo with 6 fixed sections:
         1. Applicant Summary
         2. Debt Serviceability
         3. Credit Behavior
         4. Risk Flags (2–4 bullets)
         5. Recommendation (APPROVE / CONDITIONAL APPROVE / DECLINE)
            + Risk Grade + Decision Authority + Conditions
         6. Analyst Notes
```

---

## Why Fine-tuning, Not Prompting + RAG

| Dimension | Prompted Base Model | Fine-tuned RinLekha |
|---|---|---|
| Format consistency | lower structural compliance | higher structural compliance |
| Forbidden language leakage | Occasional | Near-zero (baked in) |
| Inference cost at scale | API cost per call | $0 (local) |
| Data privacy | Sends data externally | Fully on-premise |
| Output parseability | Unreliable | Deterministic schema |
| Latency | Network round-trip | Sub-second local |

RAG is not applicable here — the task is **transformation**, not retrieval.
The model converts structured input into structured output following a fixed
institutional schema. No external knowledge retrieval is required.

---

## What This PoC Does NOT Claim

- The fine-tuned model is not smarter than GPT-5 or Claude Sonnet
- Synthetic training data is not a substitute for real historical memos
- This system should not make final credit decisions without human review
- Output quality on real-world edge cases may degrade vs. synthetic test cases

These limitations are documented explicitly in the model card.

---

## Success Criteria

| Metric | Target |
|---|---|
| Structural compliance (all 6 sections present + correct format) | ≥ 90% |
| Forbidden language rate | ≤ 2% |
| Adversarial test pass rate | ≥ 80% |
| Output parse success rate (LangChain parser) | ≥ 92% |
| Avg inference latency (vLLM, local) | ≤ 3s |
| Improvement over prompted base model (structural) | ≥ 25 percentage points |

---

## Scope

**In scope:**
- Synthetic dataset generation pipeline
- QLoRA fine-tuning on Gemma 4 E4B
- Multi-tier evaluation suite (structural, semantic, adversarial)
- Kubernetes-orchestrated parallel evaluation
- vLLM local inference server
- Langfuse production observability
- Gradio demo with baseline comparison
- HuggingFace model card with failure modes documented

**Out of scope:**
- Real borrower data of any kind
- Integration with any live loan origination system
- Multi-language support
- Mobile or edge deployment
- Regulatory certification of any kind


---

## Tech Stack Summary

| Layer | Tools |
|---|---|
| Data Pipeline | Python, Ray, OpenAI API (gpt-4.1-mini), HuggingFace Datasets |
| Training | Unsloth, TRL, MLflow, HuggingFace Hub |
| Evaluation | DeepEval, MLflow, Kubernetes (kind), custom adversarial suite |
| Serving | vLLM, LangChain, Langfuse |
| Demo | Gradio |
| Infrastructure | Docker, Kubernetes (kind), Helm |