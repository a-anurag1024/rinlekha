# RinLekha — Evaluation Gameplan

## Objective

Rigorously evaluate the fine-tuned model across four tiers:
structural compliance, semantic quality, adversarial robustness,
and three-way baseline comparison. Run evaluation in parallel
using Kubernetes. Log all results to MLflow.

---

## Evaluation Architecture

```
kind Kubernetes cluster
├── vllm-deployment          → serves fine-tuned model
├── mlflow-deployment        → receives distributed logs
├── eval-results-pvc         → shared volume for shard outputs
└── evaluation-job (indexed) → 3 parallel pods
    ├── pod-0: test cases 0-26  (DeepEval all metrics)
    ├── pod-1: test cases 27-53 (DeepEval all metrics)
    └── pod-2: test cases 54-80 (DeepEval all metrics)
```

---

## Tier 1 — Structural Evaluation (Automated)

Zero LLM calls. Pure regex + string parsing. Run on all 80 test cases.

### Metrics

**StructuralComplianceMetric**
All 6 sections present AND non-empty AND in correct order.

```python
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import re

class StructuralComplianceMetric(BaseMetric):
    threshold = 0.90
    name = "Structural Compliance"

    SECTIONS_IN_ORDER = [
        "## APPLICANT SUMMARY",
        "## DEBT SERVICEABILITY",
        "## CREDIT BEHAVIOR",
        "## RISK FLAGS",
        "## RECOMMENDATION",
        "## ANALYST NOTES"
    ]

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output
        checks = []

        # All sections present
        for section in self.SECTIONS_IN_ORDER:
            checks.append(section in output)

        # Sections in correct order
        positions = [output.find(s) for s in self.SECTIONS_IN_ORDER
                     if s in output]
        checks.append(positions == sorted(positions))

        # No empty sections
        for i, section in enumerate(self.SECTIONS_IN_ORDER):
            if section in output:
                start = output.index(section) + len(section)
                next_positions = [output.find(s, start)
                                  for s in self.SECTIONS_IN_ORDER
                                  if output.find(s, start) > 0]
                end = min(next_positions) if next_positions else len(output)
                content = output[start:end].strip()
                checks.append(len(content) > 30)

        self.score = sum(checks) / len(checks)
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success


class RecommendationFormatMetric(BaseMetric):
    threshold = 1.0
    name = "Recommendation Format"

    REQUIRED_PATTERNS = {
        "decision": r"DECISION:\s*(APPROVE|CONDITIONAL APPROVE|DECLINE)",
        "conditions": r"CONDITIONS:",
        "risk_grade": r"RISK GRADE:\s*[ABC][+-]?",
        "decision_authority": r"DECISION AUTHORITY:",
        "review_trigger": r"REVIEW TRIGGER:"
    }

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output
        results = {}
        for field, pattern in self.REQUIRED_PATTERNS.items():
            results[field] = bool(re.search(pattern, output, re.IGNORECASE))
        self.score = sum(results.values()) / len(results)
        self.field_results = results
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success


class ForbiddenLanguageMetric(BaseMetric):
    threshold = 1.0
    name = "Forbidden Language"

    FORBIDDEN = [
        "definitely", "certainly", "guaranteed",
        "will definitely", "100%", "no doubt",
        "absolutely certain", "without question"
    ]

    def measure(self, test_case: LLMTestCase) -> float:
        output_lower = test_case.actual_output.lower()
        violations = [f for f in self.FORBIDDEN if f in output_lower]
        self.score = 1.0 if not violations else 0.0
        self.violations = violations
        self.success = self.score == 1.0
        return self.score

    def is_successful(self) -> bool:
        return self.success


class RiskFlagsCountMetric(BaseMetric):
    threshold = 1.0
    name = "Risk Flags Count"

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output
        flags_section = self._extract_section(output, "## RISK FLAGS")
        bullets = re.findall(r"^[-•*]\s+.+", flags_section, re.MULTILINE)
        valid = 2 <= len(bullets) <= 4
        self.score = 1.0 if valid else 0.0
        self.flag_count = len(bullets)
        self.success = valid
        return self.score

    def _extract_section(self, text, header):
        if header not in text:
            return ""
        start = text.index(header) + len(header)
        next_headers = ["## APPLICANT", "## DEBT", "## CREDIT",
                        "## RISK", "## RECOMMENDATION", "## ANALYST"]
        positions = [text.find(h, start) for h in next_headers
                     if text.find(h, start) > 0]
        end = min(positions) if positions else len(text)
        return text[start:end]

    def is_successful(self) -> bool:
        return self.success
```

---

## Tier 2 — Semantic Evaluation (LLM-as-Judge)

Uses DeepEval's GEval and built-in metrics. Requires judge LLM calls.

```python
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

JUDGE_MODEL = "gpt-4o-mini"   # cost-efficient judge

# Credit quality holistic evaluation
credit_quality_metric = GEval(
    name="Credit Memo Quality",
    model=JUDGE_MODEL,
    criteria="""
    Evaluate this credit memo on 5 dimensions (1-5 scale each):

    1. Factual Accuracy: Do all figures (FOIR, EMI, CIBIL) match the input?
       Check at least 3 specific numbers.

    2. Risk Appropriateness: Does the risk grade match the profile risk?
       A clean CIBIL 800 profile getting C grade = wrong.
       FOIR 52% profile getting A grade = wrong.

    3. Condition Specificity: Are conditions verifiable and specific?
       "Provide income proof" is generic.
       "Last 6 months salary slips + Form 16" is specific.

    4. Analytical Depth: Does the analyst interpret or merely restate?
       "FOIR is 44%" = restatement.
       "FOIR of 44% is within policy but represents significant EMI
        step-up of 110% — income stability is therefore critical" = analysis.

    5. Hedging Compliance: Is language appropriately uncertain?
       Look for: indicates, suggests, warrants, appears — good.
       Look for: will, definitely, guaranteed — bad.

    Score each 1-5. Final score = average / 5 (normalized 0-1).
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ]
)

# Faithfulness — does output hallucinate numbers not in input?
faithfulness_metric = FaithfulnessMetric(
    threshold=0.85,
    model=JUDGE_MODEL,
    include_reason=True
)
```

---

## Tier 3 — Adversarial Evaluation (Senior Signal)

8 adversarial cases testing model behavior on edge cases and
out-of-distribution inputs that synthetic test cases don't cover.

```python
ADVERSARIAL_CASES = [
    {
        "id": "adv_001",
        "name": "extreme_foir",
        "description": "FOIR 78% — must be hard DECLINE",
        "profile": generate_profile_override(foir_post_loan=0.78),
        "expected_decision": "DECLINE",
        "test_fn": lambda output: "DECLINE" in extract_decision(output),
        "failure_severity": "critical"
    },
    {
        "id": "adv_002",
        "name": "missing_cibil",
        "description": "No CIBIL score available — must flag, not assume",
        "profile": generate_profile_override(cibil_score=None),
        "expected_behavior": "flags_missing_data",
        "test_fn": lambda output: any(
            phrase in output.lower() for phrase in
            ["cibil not available", "score not available",
             "credit score unavailable", "cibil information not provided"]
        ),
        "failure_severity": "high"
    },
    {
        "id": "adv_003",
        "name": "inconsistent_self_employed",
        "description": "Self-employed claims perfect monthly regularity "
                       "inconsistent with income type",
        "profile": generate_inconsistent_profile(
            employment_type="self_employed_business",
            income_pattern="suspiciously_regular"
        ),
        "expected_behavior": "flags_income_inconsistency",
        "test_fn": lambda output: "RISK FLAGS" in output and
                                  len(extract_risk_flags(output)) >= 2,
        "failure_severity": "medium"
    },
    {
        "id": "adv_004",
        "name": "ood_loan_amount",
        "description": "Loan 50x monthly income — must DECLINE",
        "profile": generate_profile_override(loan_to_income_ratio=50),
        "expected_decision": "DECLINE",
        "test_fn": lambda output: "DECLINE" in extract_decision(output),
        "failure_severity": "critical"
    },
    {
        "id": "adv_005",
        "name": "perfect_profile_short_tenure",
        "description": "CIBIL 820, FOIR 30%, but only 3 months employed — "
                       "should be CONDITIONAL not APPROVE",
        "profile": generate_profile_override(
            cibil_score=820, foir_post_loan=0.30, tenure_years=0.25
        ),
        "expected_decision": "CONDITIONAL_APPROVE",
        "test_fn": lambda output: "CONDITIONAL" in extract_decision(output),
        "failure_severity": "high"
    },
    {
        "id": "adv_006",
        "name": "active_delinquency",
        "description": "5 missed payments in 24 months — must DECLINE",
        "profile": generate_profile_override(missed_payments_24m=5),
        "expected_decision": "DECLINE",
        "test_fn": lambda output: "DECLINE" in extract_decision(output),
        "failure_severity": "critical"
    },
    {
        "id": "adv_007",
        "name": "debt_consolidation_red_flag",
        "description": "Debt consolidation loan with already high FOIR — "
                       "must flag purpose risk",
        "profile": generate_profile_override(
            loan_purpose="debt_consolidation",
            foir_pre_loan=0.45
        ),
        "expected_behavior": "flags_consolidation_risk",
        "test_fn": lambda output: any(
            phrase in output.lower() for phrase in
            ["consolidation", "existing debt", "debt burden"]
        ),
        "failure_severity": "medium"
    },
    {
        "id": "adv_008",
        "name": "boundary_foir",
        "description": "FOIR exactly at policy boundary 50% — "
                       "must not be clean APPROVE",
        "profile": generate_profile_override(foir_post_loan=0.50),
        "expected_behavior": "not_clean_approve",
        "test_fn": lambda output: extract_decision(output) != "APPROVE",
        "failure_severity": "high"
    }
]


def run_adversarial_suite(generate_fn, adversarial_cases):
    results = []
    for case in adversarial_cases:
        output = generate_fn(case["profile"])
        passed = case["test_fn"](output)
        results.append({
            "case_id": case["id"],
            "name": case["name"],
            "description": case["description"],
            "passed": passed,
            "failure_severity": case["failure_severity"] if not passed else None,
            "output_decision": extract_decision(output),
            "output_snippet": output[:400]
        })

    critical_failures = [r for r in results
                         if not r["passed"]
                         and r["failure_severity"] == "critical"]

    return {
        "results": results,
        "pass_rate": sum(r["passed"] for r in results) / len(results),
        "critical_failure_count": len(critical_failures),
        "critical_failures": critical_failures
    }
```

---

## Tier 4 — Three-Way Baseline Comparison

Run all 80 test cases through 3 models. Log all to MLflow.

```python
BASELINES = [
    {
        "name": "finetuned_rinlekha",
        "description": "Fine-tuned Gemma 4 E4B (RinLekha v1.0)",
        "generate_fn": generate_via_vllm,
        "prompt_overhead_tokens": 50   # minimal prompt
    },
    {
        "name": "prompted_gemma4_base",
        "description": "Gemma 4 E4B base model, full system prompt",
        "generate_fn": generate_via_vllm_base,
        "prompt_overhead_tokens": 850  # full format instructions
    },
    {
        "name": "prompted_claude_sonnet",
        "description": "Claude Sonnet 4.6 via API, full system prompt",
        "generate_fn": generate_via_claude_api,
        "prompt_overhead_tokens": 850,
        "cost_per_call_usd": 0.0045,   # approximate
        "note": "Run on 30 examples only — cost management"
    }
]
```

**Expected comparison table (fill with real numbers after eval):**

| Metric | RinLekha (FT) | Base (prompted) | Claude Sonnet |
|---|---|---|---|
| Structural Compliance | ~94% | ~61% | ~89% |
| Rec Format Compliance | ~97% | ~58% | ~91% |
| Forbidden Language Rate | ~1% | ~8% | ~2% |
| Risk Flags Count Valid | ~95% | ~71% | ~93% |
| GEval Quality (0-1) | — | — | — |
| Faithfulness | — | — | — |
| Adversarial Pass Rate | — | — | — |
| Avg Inference Latency | <3s (local) | <3s (local) | ~4-8s (API) |
| Cost per 1000 calls | $0 | $0 | ~$4.50 |

---

## Kubernetes Evaluation Setup

### Docker Images

```dockerfile
# Dockerfile.eval
FROM python:3.11-slim
WORKDIR /app
COPY requirements-eval.txt .
RUN pip install --no-cache-dir -r requirements-eval.txt
COPY evaluation/ /app/evaluation/
ENTRYPOINT ["python", "/app/evaluation/run_eval_shard.py"]
```

### Kubernetes Manifests

```yaml
# k8s/eval-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: rinlekha-evaluation
  namespace: rinlekha
spec:
  completions: 3
  parallelism: 3
  completionMode: Indexed
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: eval-worker
        image: rinlekha/eval:latest
        env:
        - name: POD_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: TOTAL_SHARDS
          value: "3"
        - name: VLLM_URL
          value: "http://vllm-service.rinlekha.svc.cluster.local:8000"
        - name: MLFLOW_URL
          value: "http://mlflow-service.rinlekha.svc.cluster.local:5000"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rinlekha-secrets
              key: openai-api-key    # for judge model
        volumeMounts:
        - name: results-volume
          mountPath: /results
        - name: test-data
          mountPath: /data
      volumes:
      - name: results-volume
        persistentVolumeClaim:
          claimName: eval-results-pvc
      - name: test-data
        configMap:
          name: test-cases-config
```

### Eval Shard Script (runs inside each pod)

```python
# evaluation/run_eval_shard.py

import os
import json
import mlflow
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

POD_INDEX = int(os.environ["POD_INDEX"])
TOTAL_SHARDS = int(os.environ["TOTAL_SHARDS"])
VLLM_URL = os.environ["VLLM_URL"]
MLFLOW_URL = os.environ["MLFLOW_URL"]


def main():
    print(f"[Pod {POD_INDEX}] Starting evaluation shard")

    # Load test cases
    with open("/data/test_cases.json") as f:
        all_cases = json.load(f)

    # Compute this pod's shard
    shard_size = len(all_cases) // TOTAL_SHARDS
    start = POD_INDEX * shard_size
    end = (start + shard_size
           if POD_INDEX < TOTAL_SHARDS - 1
           else len(all_cases))
    my_cases = all_cases[start:end]

    print(f"[Pod {POD_INDEX}] Evaluating cases {start}–{end} "
          f"({len(my_cases)} cases)")

    # Initialize metrics
    metrics = [
        StructuralComplianceMetric(),
        RecommendationFormatMetric(),
        ForbiddenLanguageMetric(),
        RiskFlagsCountMetric(),
        credit_quality_metric,
        faithfulness_metric
    ]

    # Build DeepEval test cases
    test_cases = []
    for case in my_cases:
        actual_output = generate_via_vllm(
            case["input"], vllm_url=VLLM_URL
        )
        test_cases.append(LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case["expected_output"]
        ))

    # Run evaluation
    results = evaluate(test_cases, metrics)

    # Aggregate shard scores
    shard_scores = aggregate_scores(results, metrics)

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_URL)
    with mlflow.start_run(
        run_name=f"eval_shard_{POD_INDEX}",
        experiment_id=get_experiment_id("rinlekha-evaluation")
    ):
        mlflow.log_metrics(shard_scores)
        mlflow.log_param("pod_index", POD_INDEX)
        mlflow.log_param("cases_evaluated", len(my_cases))

    # Write to shared volume
    output_path = f"/results/shard_{POD_INDEX}.json"
    with open(output_path, "w") as f:
        json.dump({
            "pod_index": POD_INDEX,
            "cases_start": start,
            "cases_end": end,
            "scores": shard_scores,
            "raw_results": serialize_results(results)
        }, f, indent=2)

    print(f"[Pod {POD_INDEX}] Complete. Results at {output_path}")


if __name__ == "__main__":
    main()
```

### Results Aggregation (after job completes)

```python
# evaluation/aggregate_results.py

import json
import glob
import mlflow

def aggregate_all_shards(results_dir: str = "/results"):
    shard_files = sorted(glob.glob(f"{results_dir}/shard_*.json"))
    all_shards = [json.load(open(f)) for f in shard_files]

    # Average scores across shards (weighted by shard size)
    total_cases = sum(s["cases_end"] - s["cases_start"]
                      for s in all_shards)
    aggregated = {}
    for metric in all_shards[0]["scores"]:
        weighted_sum = sum(
            s["scores"][metric] * (s["cases_end"] - s["cases_start"])
            for s in all_shards
        )
        aggregated[metric] = weighted_sum / total_cases

    # Log final aggregated results to MLflow
    mlflow.set_tracking_uri(MLFLOW_URL)
    with mlflow.start_run(run_name="eval_aggregated_final"):
        mlflow.log_metrics(aggregated)
        mlflow.log_param("total_cases_evaluated", total_cases)
        mlflow.log_param("shards_aggregated", len(all_shards))

    print("Final evaluation results:")
    for metric, score in aggregated.items():
        print(f"  {metric}: {score:.4f}")

    return aggregated
```

---

## Failure Mode Documentation

Document every failure type observed during evaluation.
This goes directly into the model card.

```python
FAILURE_MODES_TEMPLATE = {
    "structural_failures": {
        "description": "Sections missing or out of order",
        "frequency": "X% of test cases",
        "most_common_pattern": "",
        "recommended_handling": "Parse failure → fallback to raw text + flag"
    },
    "decision_errors": {
        "description": "Wrong decision for profile risk level",
        "frequency": "X% of test cases",
        "boundary_cases": "FOIR 48-52% most error-prone",
        "recommended_handling": "Human review mandatory for boundary cases"
    },
    "hallucinated_figures": {
        "description": "Numbers in memo don't match input profile",
        "frequency": "X% of test cases",
        "example": "",
        "recommended_handling": "Post-processing numeric validation"
    },
    "adversarial_critical_failures": {
        "description": "Failed hard decline cases",
        "cases_failed": [],
        "recommended_handling": "Override with rule-based hard filters"
    }
}
```

---

## Deliverables Checklist

```
□ evaluation/structural_eval.py         — Tier 1 metrics (all custom)
□ evaluation/semantic_eval.py           — Tier 2 metrics (DeepEval)
□ evaluation/adversarial_suite.py       — 8 adversarial cases
□ evaluation/baseline_comparison.py    — 3-model comparison
□ evaluation/run_eval_shard.py         — Kubernetes pod entry point
□ evaluation/aggregate_results.py      — Post-job aggregation
□ k8s/eval-job.yaml                    — Kubernetes Job manifest
□ k8s/vllm-deployment.yaml            — vLLM Deployment + Service
□ k8s/mlflow-deployment.yaml          — MLflow Deployment + Service
□ k8s/pvcs.yaml                       — PersistentVolumeClaims
□ helm/rinlekha/                     — Helm chart
□ MLflow: eval runs logged             — all 3 models compared
□ evaluation_report.md                 — human-readable summary
□ failure_modes.json                   — documented failure taxonomy
```