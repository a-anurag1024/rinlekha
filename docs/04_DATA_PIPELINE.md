# RinLekha — Data Pipeline Gameplan

## Objective

Generate 800 high-quality (borrower profile → credit memo) training pairs
using a systematic, reproducible pipeline. No real borrower data used.
Final dataset versioned on HuggingFace Datasets Hub.

---

## Step 1 — Borrower Profile Schema Design

Define all dimensions that vary across profiles. Every dimension must
be independent enough to sample from, but constrained enough to produce
realistic combinations.

```python
PROFILE_DIMENSIONS = {
    # Demographics
    "age": (22, 65),
    "city_tier": ["tier1", "tier2", "tier3"],

    # Employment
    "employment_type": [
        "salaried_private", "salaried_govt",
        "self_employed_professional", "self_employed_business"
    ],
    "monthly_income": (15_000, 5_00_000),     # INR
    "employment_tenure_years": (0.25, 25),
    "sector": ["IT", "manufacturing", "healthcare",
               "finance", "retail", "education", "govt"],

    # Credit Profile
    "cibil_score": (550, 900),
    "existing_emi_monthly": (0, 80_000),       # INR
    "loan_accounts_active": (0, 5),
    "missed_payments_24m": (0, 6),
    "settled_accounts_ever": (0, 3),
    "credit_vintage_years": (0, 15),

    # Loan Request
    "loan_amount": (50_000, 50_00_000),        # INR
    "loan_tenure_months": [12, 24, 36, 48, 60, 84],
    "loan_purpose": [
        "home_renovation", "medical_emergency",
        "education", "debt_consolidation",
        "business_expansion", "wedding",
        "travel", "vehicle_purchase"
    ],
    "annual_interest_rate": (10.5, 26.0)
}
```

**Derived fields (computed, not sampled):**

```python
def compute_derived_fields(profile: dict) -> dict:
    r = profile["annual_interest_rate"] / 12 / 100
    n = profile["loan_tenure_months"]
    p = profile["loan_amount"]
    emi = p * r * (1 + r)**n / ((1 + r)**n - 1)

    profile["proposed_emi"] = round(emi, 2)
    profile["foir_pre_loan"] = round(
        profile["existing_emi_monthly"] / profile["monthly_income"], 3
    )
    profile["foir_post_loan"] = round(
        (profile["existing_emi_monthly"] + emi) / profile["monthly_income"], 3
    )
    profile["loan_to_income_ratio"] = round(
        profile["loan_amount"] / profile["monthly_income"], 1
    )
    return profile
```

---

## Step 2 — Underwriting Decision Logic

Decision logic must reflect real NBFC underwriting policy.
This ensures the model learns meaningful credit risk patterns,
not random label assignments.

```python
def determine_outcome(profile: dict) -> tuple[str, list[str]]:
    """
    Returns: (decision, list_of_conditions)
    Decision: APPROVE | CONDITIONAL_APPROVE | DECLINE
    """

    # ── HARD DECLINES ──────────────────────────────────────────────
    if profile["cibil_score"] < 620:
        return "DECLINE", ["credit_score_below_minimum_threshold"]

    if profile["foir_post_loan"] > 0.55:
        return "DECLINE", ["foir_exceeds_policy_ceiling_of_55pct"]

    if profile["missed_payments_24m"] >= 4:
        return "DECLINE", ["excessive_recent_delinquency"]

    if profile["settled_accounts_ever"] >= 2:
        return "DECLINE", ["multiple_settled_loan_accounts"]

    if profile["loan_to_income_ratio"] > 60:
        return "DECLINE", ["loan_amount_grossly_disproportionate_to_income"]

    # ── CLEAN APPROVES ─────────────────────────────────────────────
    if (
        profile["cibil_score"] >= 760
        and profile["foir_post_loan"] <= 0.38
        and profile["missed_payments_24m"] == 0
        and profile["employment_tenure_years"] >= 2
        and profile["settled_accounts_ever"] == 0
    ):
        return "APPROVE", []

    # ── CONDITIONAL APPROVES ───────────────────────────────────────
    conditions = []

    if profile["foir_post_loan"] > 0.46:
        conditions.append("income_proof_last_6_months_salary_slips")

    if profile["missed_payments_24m"] in [1, 2]:
        conditions.append("written_explanation_for_missed_payments")

    if profile["employment_tenure_years"] < 1.0:
        conditions.append("employment_confirmation_from_hr")

    if profile["loan_to_income_ratio"] > 30:
        conditions.append("additional_income_proof_or_co_applicant")

    if profile["cibil_score"] < 680:
        conditions.append("guarantor_or_collateral_security")

    if profile["settled_accounts_ever"] == 1:
        conditions.append("noc_from_previous_lender_for_settled_account")

    if not conditions:
        conditions.append("standard_documentation_verification")

    return "CONDITIONAL_APPROVE", conditions
```

**Distribution targets (enforce during generation):**

| Decision | Target % | Rationale |
|---|---|---|
| APPROVE | 20% | Reflects minority of clean applicants |
| CONDITIONAL_APPROVE | 55% | Majority of real applications |
| DECLINE | 25% | Common enough to train on |

---

## Step 3 — Ray Parallel Profile Generation

```python
import ray
import random

@ray.remote
def generate_profile_batch(batch_size: int, seed: int) -> list[dict]:
    random.seed(seed)
    profiles = []
    for i in range(batch_size):
        profile = sample_profile_from_dimensions(seed + i)
        profile = compute_derived_fields(profile)
        decision, conditions = determine_outcome(profile)
        profile["outcome"] = decision
        profile["conditions"] = conditions
        profile["profile_id"] = f"profile_{seed}_{i:04d}"
        profiles.append(profile)
    return profiles


def generate_all_profiles(total: int = 1000) -> list[dict]:
    ray.init(ignore_reinit_error=True)
    n_workers = 8
    batch_size = total // n_workers

    futures = [
        generate_profile_batch.remote(batch_size, seed=i * 10_000)
        for i in range(n_workers)
    ]
    batches = ray.get(futures)
    profiles = [p for batch in batches for p in batch]

    # Enforce decision distribution
    profiles = enforce_distribution(profiles, targets={
        "APPROVE": 0.20,
        "CONDITIONAL_APPROVE": 0.55,
        "DECLINE": 0.25
    })
    return profiles[:total]
```

**Why Ray here:**
Generating 1000 profiles is trivial sequentially. Ray is used because
the same pattern — `@ray.remote` batched workers — scales directly to
production data pipelines processing millions of records. This is the
learnable pattern, not the speed gain at this scale.

---

## Step 4 — Credit Memo Synthesis Prompt

The synthesis prompt is the single most important engineering artifact
in the data pipeline. Spend disproportionate time on this.

```
SYSTEM:
You are a senior credit analyst at an Indian NBFC (Non-Banking Financial
Company) writing internal credit assessment memos. Your memos are read
by credit managers making final lending decisions.

Write credit memos that are:
- Factually accurate: every figure stated must match the input profile
- Appropriately hedged: never use "definitely", "certainly", "guaranteed",
  "will", "100%", "no doubt", or any language implying certainty
- Analytically substantive: interpret the data, do not merely restate it
- Structurally strict: follow the exact 6-section format below

USER:
Write a credit memo for this borrower. Follow the format EXACTLY.

## APPLICANT SUMMARY
[2-3 sentences. Employment stability, income source, demographic context.
Factual statements only. No risk assessment in this section.]

## DEBT SERVICEABILITY
[FOIR analysis, EMI burden, income adequacy, step-up risk if applicable.
Quantify: state the actual FOIR percentage and policy threshold.
Hedged language mandatory.]

## CREDIT BEHAVIOR
[CIBIL score interpretation with band context (e.g., "724 — acceptable tier").
Payment history — distinguish old vs recent delinquency explicitly.
Credit vintage and account diversity where relevant.]

## RISK FLAGS
[Bulleted list. Minimum 2 items, maximum 4 items.
Each flag must reference a specific data point from the profile.
No generic flags like "loan default risk" without specific grounding.]

## RECOMMENDATION
DECISION: [APPROVE / CONDITIONAL APPROVE / DECLINE]
CONDITIONS: [numbered list if conditional; "None" if clean approve]
RISK GRADE: [A / B+ / B / B- / C]
DECISION AUTHORITY: [Branch Credit Manager / Regional Credit Head /
                     HO Credit Committee]
REVIEW TRIGGER: [one sentence — what new information would change
                 this recommendation]

## ANALYST NOTES
[1-2 sentences. What additional information would materially change
the credit assessment. Be specific.]

---
BORROWER PROFILE:
{profile_json}

EXPECTED DECISION: {outcome}
CONDITIONS IF APPLICABLE: {conditions_list}

Write the memo now. Do not add any text before ## APPLICANT SUMMARY.
```

---

## Step 5 — Parallel Synthesis with Ray

```python
@ray.remote
def synthesize_memo_batch(
    profiles: list[dict],
    openai_api_key: str
) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
    results = []

    for profile in profiles:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=900,
                messages=[{
                    "role": "user",
                    "content": build_synthesis_prompt(profile)
                }]
            )
            results.append({
                "profile_id": profile["profile_id"],
                "input_profile": profile,
                "output_memo": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "synthesis_status": "success"
            })
        except Exception as e:
            results.append({
                "profile_id": profile["profile_id"],
                "input_profile": profile,
                "output_memo": None,
                "synthesis_status": f"error: {str(e)}"
            })
    return results
```

**Cost estimate (gpt-4o-mini):**
- Avg input tokens per call: ~600
- Avg output tokens per call: ~650
- 800 calls total
- gpt-4o-mini pricing: ~$0.15/M input + $0.60/M output
- Total estimate: (800 × 600 / 1M × $0.15) + (800 × 650 / 1M × $0.60)
- ≈ $0.07 + $0.31 = **~$0.38 total (~₹32)**

---

## Step 6 — Automated Quality Review

Run on all 800 examples before any manual review.

```python
import re

REQUIRED_SECTIONS = [
    "## APPLICANT SUMMARY",
    "## DEBT SERVICEABILITY",
    "## CREDIT BEHAVIOR",
    "## RISK FLAGS",
    "## RECOMMENDATION",
    "## ANALYST NOTES"
]

FORBIDDEN_PHRASES = [
    "definitely", "certainly", "guaranteed", "will definitely",
    "100%", "no doubt", "absolutely", "without question"
]

REQUIRED_REC_FIELDS = [
    r"DECISION:\s*(APPROVE|CONDITIONAL APPROVE|DECLINE)",
    r"RISK GRADE:\s*[ABC][+-]?",
    r"DECISION AUTHORITY:",
    r"REVIEW TRIGGER:",
    r"CONDITIONS:"
]


def quality_check(example: dict) -> dict:
    memo = example.get("output_memo", "")
    checks = {}

    # Structural checks
    for section in REQUIRED_SECTIONS:
        checks[f"section_{section[3:].lower().replace(' ', '_')}"] = (
            section in memo
        )

    # Recommendation format checks
    for pattern in REQUIRED_REC_FIELDS:
        field_name = pattern.split(r"\s*")[0].replace("\\", "").lower()
        checks[f"rec_{field_name}"] = bool(re.search(pattern, memo))

    # Language checks
    memo_lower = memo.lower()
    forbidden_found = [f for f in FORBIDDEN_PHRASES if f in memo_lower]
    checks["no_forbidden_language"] = len(forbidden_found) == 0
    checks["forbidden_phrases_found"] = forbidden_found

    # Risk flags count
    flags_section = extract_section(memo, "## RISK FLAGS")
    flag_bullets = re.findall(r"^[-•]\s+.+", flags_section, re.MULTILINE)
    checks["risk_flags_count_valid"] = 2 <= len(flag_bullets) <= 4
    checks["risk_flags_count"] = len(flag_bullets)

    # Decision consistency
    stated_decision = extract_decision(memo)
    expected_decision = example["input_profile"]["outcome"]
    checks["decision_matches_expected"] = (
        stated_decision == expected_decision
    )

    # Overall pass/fail
    critical_checks = [v for k, v in checks.items()
                       if isinstance(v, bool)
                       and k not in ["no_forbidden_language",
                                     "decision_matches_expected"]]
    checks["structural_pass"] = all(critical_checks)
    checks["needs_review"] = not checks["structural_pass"]

    return checks
```

**Expected quality check results on first pass:**
- Structural pass: ~82-85% without review
- After prompt iteration + regeneration: target ≥95%

**Manual review protocol:**
- Review only flagged examples (not all 800)
- For each flagged: identify root cause (prompt ambiguity vs. edge case)
- If prompt issue: fix prompt, regenerate entire batch for that failure mode
- If edge case: fix individual example or discard

---

## Step 7 — Instruction Tuning Format Conversion

```python
def format_as_instruction_pair(example: dict) -> dict:
    """Convert raw example to Alpaca-style instruction format"""
    profile_text = format_profile_as_readable_text(
        example["input_profile"]
    )
    return {
        "instruction": (
            "You are a senior credit analyst at an Indian NBFC. "
            "Write a structured credit memo for the borrower profile "
            "below following institutional format exactly."
        ),
        "input": profile_text,
        "output": example["output_memo"],
        # Metadata (not used in training, useful for analysis)
        "outcome": example["input_profile"]["outcome"],
        "cibil_band": get_cibil_band(example["input_profile"]["cibil_score"]),
        "foir_band": get_foir_band(example["input_profile"]["foir_post_loan"]),
        "employment_type": example["input_profile"]["employment_type"],
        "loan_purpose": example["input_profile"]["loan_purpose"],
        "profile_id": example["input_profile"]["profile_id"]
    }
```

---

## Step 8 — HuggingFace Dataset Push

```python
from datasets import Dataset, DatasetDict

def create_and_push_dataset(
    examples: list[dict],
    hf_repo: str = "your-username/rinlekha-training-data"
) -> DatasetDict:

    formatted = [format_as_instruction_pair(e) for e in examples]
    full_dataset = Dataset.from_list(formatted)

    # Stratified split by outcome
    splits = stratified_split(full_dataset, label_col="outcome",
                               train=0.80, val=0.10, test=0.10)

    dataset_dict = DatasetDict({
        "train": splits["train"],       # 640 examples
        "validation": splits["val"],    # 80 examples
        "test": splits["test"]          # 80 examples
    })

    dataset_dict.push_to_hub(
        hf_repo,
        commit_message="v1.0 — 800 examples, stratified split"
    )

    print(f"Dataset pushed: https://huggingface.co/datasets/{hf_repo}")
    print(f"Train: {len(splits['train'])} | "
          f"Val: {len(splits['val'])} | "
          f"Test: {len(splits['test'])}")

    return dataset_dict
```

---

## Deliverables Checklist

```
□ profile_generator.py        — Ray parallel profile generation
□ memo_synthesizer.py         — Ray parallel Claude API synthesis  
□ quality_reviewer.py         — Automated structural QA
□ dataset_builder.py          — Format conversion + HF push
□ data/raw/profiles_v1.jsonl  — Raw profiles (800)
□ data/raw/memos_v1.jsonl     — Raw memos (800)
□ data/processed/train.jsonl  — 640 training examples
□ data/processed/val.jsonl    — 80 validation examples
□ data/processed/test.jsonl   — 80 test examples (hold-out)
□ data/qc_report_v1.json      — Quality check results
□ HuggingFace Dataset         — rinlekha-training-data@v1.0
```