# RinLekha — Serving Pipeline Gameplan

## Objective

Build a production-grade local serving stack: vLLM for inference,
LangChain for pipeline orchestration and output parsing, Langfuse
for observability, and Gradio for the demo interface.

Zero borrower data transmitted externally at any point.

---

## Serving Stack Overview

```
Gradio UI
    ↓ (borrower form input)
LangChain Pipeline
    ├── Input validation
    ├── Profile → text formatter
    ├── Prompt assembly
    ├── vLLM call (local REST API)
    ├── CreditMemo output parser
    └── Error handling + fallback
    ↓ (structured CreditMemo object)
Langfuse Observability
    └── Trace: latency, tokens, quality scores, metadata
    ↓
Gradio UI
    └── Render: memo sections, risk dashboard, compliance check
```

---

## Component 1 — vLLM Inference Server

vLLM provides an OpenAI-compatible REST API for the fine-tuned model.
Runs as a standalone process in WSL2 (not inside Kubernetes — GPU
passthrough to kind pods on Windows/WSL2 is unreliable).

### Start Command

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/rinlekha-run-best \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --disable-log-requests
```

### Verify Server

```bash
curl http://localhost:8000/health
# {"status": "ok"}

curl http://localhost:8000/v1/models
# {"data": [{"id": "rinlekha-run-best", ...}]}
```

### Configuration Notes

```python
VLLM_CONFIG = {
    "temperature": 0.1,      # low temp for format consistency
    "max_tokens": 900,       # sufficient for full memo
    "top_p": 0.9,
    "frequency_penalty": 0.1  # mild penalty to reduce repetition
}
```

Temperature 0.1 is a deliberate choice — not 0.0 (greedy) because
credit language benefits from slight variation, not 0.7+ because
format consistency is the primary objective.

---

## Component 2 — LangChain Pipeline

### Output Schema (Pydantic)

```python
# serving/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class CreditDecision(str, Enum):
    APPROVE = "APPROVE"
    CONDITIONAL_APPROVE = "CONDITIONAL APPROVE"
    DECLINE = "DECLINE"
    UNKNOWN = "UNKNOWN"

class RiskGrade(str, Enum):
    A = "A"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C = "C"
    UNKNOWN = "UNKNOWN"

class CreditMemo(BaseModel):
    applicant_summary: str = Field(description="2-3 sentence applicant overview")
    debt_serviceability: str = Field(description="FOIR and EMI analysis")
    credit_behavior: str = Field(description="CIBIL and payment history analysis")
    risk_flags: list[str] = Field(description="2-4 specific risk bullets")
    decision: CreditDecision
    conditions: list[str] = Field(default_factory=list)
    risk_grade: RiskGrade
    decision_authority: str
    review_trigger: str
    analyst_notes: str
    raw_output: str = Field(description="Unmodified model output")
    parse_success: bool
    parse_errors: list[str] = Field(default_factory=list)

    @validator("risk_flags")
    def validate_flag_count(cls, v):
        if not (2 <= len(v) <= 4):
            raise ValueError(f"Expected 2-4 risk flags, got {len(v)}")
        return v
```

### Output Parser

```python
# serving/parser.py
import re
from serving.schemas import CreditMemo, CreditDecision, RiskGrade

class CreditMemoParser:
    """
    Deterministic parser for credit memo structured output.
    Returns a CreditMemo object with parse_success=False
    and populated parse_errors if parsing fails — never raises.
    """

    SECTION_HEADERS = [
        "## APPLICANT SUMMARY",
        "## DEBT SERVICEABILITY",
        "## CREDIT BEHAVIOR",
        "## RISK FLAGS",
        "## RECOMMENDATION",
        "## ANALYST NOTES"
    ]

    def parse(self, raw_output: str) -> CreditMemo:
        errors = []
        sections = self._extract_all_sections(raw_output)

        # Parse recommendation fields
        rec_section = sections.get("## RECOMMENDATION", "")
        decision = self._extract_decision(rec_section, errors)
        conditions = self._extract_conditions(rec_section, decision)
        risk_grade = self._extract_risk_grade(rec_section, errors)
        authority = self._extract_authority(rec_section, errors)
        review_trigger = self._extract_review_trigger(rec_section, errors)

        # Parse risk flags
        flags_section = sections.get("## RISK FLAGS", "")
        risk_flags = self._extract_bullets(flags_section)
        if not (2 <= len(risk_flags) <= 4):
            errors.append(
                f"Risk flags count {len(risk_flags)} outside 2-4 range"
            )

        return CreditMemo(
            applicant_summary=sections.get("## APPLICANT SUMMARY", ""),
            debt_serviceability=sections.get("## DEBT SERVICEABILITY", ""),
            credit_behavior=sections.get("## CREDIT BEHAVIOR", ""),
            risk_flags=risk_flags if risk_flags else ["[PARSE ERROR]"],
            decision=decision,
            conditions=conditions,
            risk_grade=risk_grade,
            decision_authority=authority,
            review_trigger=review_trigger,
            analyst_notes=sections.get("## ANALYST NOTES", ""),
            raw_output=raw_output,
            parse_success=len(errors) == 0,
            parse_errors=errors
        )

    def _extract_all_sections(self, text: str) -> dict:
        sections = {}
        for i, header in enumerate(self.SECTION_HEADERS):
            if header not in text:
                continue
            start = text.index(header) + len(header)
            remaining = self.SECTION_HEADERS[i+1:]
            next_positions = [text.find(h, start) for h in remaining
                              if text.find(h, start) > 0]
            end = min(next_positions) if next_positions else len(text)
            sections[header] = text[start:end].strip()
        return sections

    def _extract_decision(self, rec_text: str, errors: list) -> CreditDecision:
        match = re.search(
            r"DECISION:\s*(APPROVE|CONDITIONAL APPROVE|DECLINE)",
            rec_text, re.IGNORECASE
        )
        if not match:
            errors.append("Could not extract DECISION field")
            return CreditDecision.UNKNOWN
        value = match.group(1).upper().strip()
        mapping = {
            "APPROVE": CreditDecision.APPROVE,
            "CONDITIONAL APPROVE": CreditDecision.CONDITIONAL_APPROVE,
            "DECLINE": CreditDecision.DECLINE
        }
        return mapping.get(value, CreditDecision.UNKNOWN)

    def _extract_conditions(self, rec_text: str,
                            decision: CreditDecision) -> list[str]:
        if decision == CreditDecision.APPROVE:
            return []
        if decision == CreditDecision.DECLINE:
            return []
        # Extract numbered conditions
        conditions_match = re.search(
            r"CONDITIONS:(.*?)(?:RISK GRADE:|$)", rec_text,
            re.DOTALL | re.IGNORECASE
        )
        if not conditions_match:
            return []
        conditions_text = conditions_match.group(1).strip()
        if conditions_text.lower() in ["none", "n/a", "-"]:
            return []
        items = re.findall(r"\d+\.\s+(.+)", conditions_text)
        return [item.strip() for item in items]

    def _extract_risk_grade(self, rec_text: str,
                            errors: list) -> RiskGrade:
        match = re.search(r"RISK GRADE:\s*([ABC][+-]?)", rec_text)
        if not match:
            errors.append("Could not extract RISK GRADE field")
            return RiskGrade.UNKNOWN
        grade_map = {
            "A": RiskGrade.A, "B+": RiskGrade.B_PLUS,
            "B": RiskGrade.B, "B-": RiskGrade.B_MINUS,
            "C": RiskGrade.C
        }
        return grade_map.get(match.group(1), RiskGrade.UNKNOWN)

    def _extract_authority(self, rec_text: str, errors: list) -> str:
        match = re.search(r"DECISION AUTHORITY:\s*(.+)", rec_text)
        if not match:
            errors.append("Could not extract DECISION AUTHORITY field")
            return "Unknown"
        return match.group(1).strip()

    def _extract_review_trigger(self, rec_text: str,
                                errors: list) -> str:
        match = re.search(r"REVIEW TRIGGER:\s*(.+)", rec_text,
                          re.DOTALL)
        if not match:
            errors.append("Could not extract REVIEW TRIGGER field")
            return "Unknown"
        return match.group(1).strip().split("\n")[0]

    def _extract_bullets(self, section_text: str) -> list[str]:
        bullets = re.findall(r"^[-•*]\s+(.+)", section_text,
                             re.MULTILINE)
        return [b.strip() for b in bullets]
```

### LangChain Pipeline

```python
# serving/pipeline.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from openai import OpenAI
from serving.parser import CreditMemoParser
from serving.schemas import CreditMemo

SYSTEM_PROMPT = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write structured credit memos following institutional format exactly."
)

USER_PROMPT = (
    "Write a credit memo for this borrower profile:\n\n{profile_text}"
)

parser = CreditMemoParser()

# vLLM is OpenAI-API-compatible — use ChatOpenAI with custom base_url
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="rinlekha-run-best",
    temperature=0.1,
    max_tokens=900
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT)
])

def build_pipeline():
    """Returns the full LangChain pipeline as a runnable chain"""
    return (
        RunnablePassthrough()
        | prompt_template
        | llm
        | (lambda response: parser.parse(response.content))
    )

# Initialize once, reuse across requests
pipeline = build_pipeline()


def generate_credit_memo(profile: dict) -> CreditMemo:
    """
    Public entry point. Called by Gradio app and Langfuse wrapper.
    """
    profile_text = format_profile_as_text(profile)
    result = pipeline.invoke({"profile_text": profile_text})
    return result
```

---

## Component 3 — Langfuse Observability

### Setup

```python
# serving/observability.py
import os
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from serving.schemas import CreditMemo
from serving.pipeline import generate_credit_memo as _generate
from evaluation.structural_eval import (
    StructuralComplianceMetric,
    RecommendationFormatMetric
)

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host="https://cloud.langfuse.com"
)


@observe(name="generate_credit_memo")
def generate_credit_memo_observed(profile: dict) -> CreditMemo:
    """
    Wraps the pipeline call with full Langfuse tracing.
    Every call is traced: latency, tokens, quality scores, metadata.
    """

    # Add input metadata to trace
    langfuse_context.update_current_observation(
        input=profile,
        metadata={
            "cibil_score": profile.get("cibil_score"),
            "cibil_band": get_cibil_band(profile.get("cibil_score")),
            "foir_post_loan": profile.get("foir_post_loan"),
            "foir_band": get_foir_band(profile.get("foir_post_loan")),
            "employment_type": profile.get("employment_type"),
            "loan_purpose": profile.get("loan_purpose"),
            "outcome_expected": profile.get("outcome"),
            "model": "rinlekha-gemma4-e4b-finetuned"
        }
    )

    # Generate
    memo: CreditMemo = _generate(profile)

    # Score: parse success (binary)
    langfuse_context.score_current_observation(
        name="parse_success",
        value=1.0 if memo.parse_success else 0.0,
        comment="; ".join(memo.parse_errors) if memo.parse_errors else None
    )

    # Score: structural compliance (fast, no LLM)
    structural_score = compute_fast_structural_score(memo.raw_output)
    langfuse_context.score_current_observation(
        name="structural_compliance",
        value=structural_score
    )

    # Score: decision present
    langfuse_context.score_current_observation(
        name="decision_extracted",
        value=1.0 if memo.decision.value != "UNKNOWN" else 0.0
    )

    # Add output to trace
    langfuse_context.update_current_observation(
        output=memo.model_dump(exclude={"raw_output"})
    )

    return memo
```

### What Langfuse Captures Per Call

```
Trace fields (automatic via @observe):
├── trace_id          — unique identifier
├── name              — "generate_credit_memo"
├── start_time        — timestamp
├── end_time          — timestamp
├── latency_ms        — end-to-end latency
├── input             — full borrower profile dict
├── output            — structured CreditMemo dict
└── metadata
    ├── cibil_score
    ├── cibil_band
    ├── foir_post_loan
    ├── foir_band
    ├── employment_type
    ├── loan_purpose
    └── model

Scores (custom, logged in function):
├── parse_success         (0.0 or 1.0)
├── structural_compliance (0.0 – 1.0)
└── decision_extracted    (0.0 or 1.0)
```

### Langfuse Dashboard — Key Views to Show in Portfolio

```
1. Traces table        — every inference call, latency, scores
2. Score distribution  — histogram of structural_compliance over time
3. Latency P50/P95     — inference performance
4. Metadata filter     — filter traces by cibil_band, foir_band
5. Parse failure filter — traces where parse_success=0.0
```

---

## Component 4 — Gradio Demo App

```python
# app/gradio_app.py
import gradio as gr
from serving.observability import generate_credit_memo_observed
from serving.pipeline import generate_credit_memo as generate_base
from serving.schemas import CreditDecision, RiskGrade

DECISION_COLORS = {
    CreditDecision.APPROVE: "🟢",
    CreditDecision.CONDITIONAL_APPROVE: "🟡",
    CreditDecision.DECLINE: "🔴",
    CreditDecision.UNKNOWN: "⚪"
}

GRADE_COLORS = {
    RiskGrade.A: "🟢", RiskGrade.B_PLUS: "🟢",
    RiskGrade.B: "🟡", RiskGrade.B_MINUS: "🟠",
    RiskGrade.C: "🔴", RiskGrade.UNKNOWN: "⚪"
}


def run_analysis(
    age, income, employment_type, tenure_years, sector,
    cibil_score, existing_emis, missed_payments,
    loan_amount, tenure_months, loan_purpose, interest_rate
):
    profile = {
        "age": age,
        "monthly_income": income,
        "employment_type": employment_type.lower().replace(" ", "_"),
        "employment_tenure_years": tenure_years,
        "sector": sector.lower(),
        "cibil_score": cibil_score,
        "existing_emi_monthly": existing_emis,
        "missed_payments_24m": missed_payments,
        "loan_amount": loan_amount,
        "loan_tenure_months": tenure_months,
        "loan_purpose": loan_purpose.lower().replace(" ", "_"),
        "annual_interest_rate": interest_rate
    }

    # Fine-tuned model (with Langfuse tracing)
    ft_memo = generate_credit_memo_observed(profile)

    # Base model (for comparison tab, no tracing)
    base_memo = generate_base(profile)  # uses base model prompt

    # Format outputs
    ft_display = format_memo_markdown(ft_memo)
    base_display = format_base_output(base_memo)
    risk_dashboard = format_risk_dashboard(ft_memo)
    compliance = format_compliance_check(ft_memo)

    return ft_display, base_display, risk_dashboard, compliance


def format_memo_markdown(memo) -> str:
    decision_icon = DECISION_COLORS[memo.decision]
    grade_icon = GRADE_COLORS[memo.risk_grade]

    return f"""
## Applicant Summary
{memo.applicant_summary}

## Debt Serviceability
{memo.debt_serviceability}

## Credit Behavior
{memo.credit_behavior}

## Risk Flags
{chr(10).join(f"• {flag}" for flag in memo.risk_flags)}

## Recommendation
**Decision:** {decision_icon} {memo.decision.value}
**Risk Grade:** {grade_icon} {memo.risk_grade.value}
**Decision Authority:** {memo.decision_authority}
**Review Trigger:** {memo.review_trigger}
{"**Conditions:**" + chr(10) + chr(10).join(f"{i+1}. {c}" for i, c in enumerate(memo.conditions)) if memo.conditions else ""}

## Analyst Notes
{memo.analyst_notes}
"""


def format_risk_dashboard(memo) -> dict:
    return {
        "decision": memo.decision.value,
        "risk_grade": memo.risk_grade.value,
        "decision_authority": memo.decision_authority,
        "conditions_count": len(memo.conditions),
        "risk_flags": memo.risk_flags,
        "parse_success": memo.parse_success,
        "parse_errors": memo.parse_errors
    }


def format_compliance_check(memo) -> dict:
    return {
        "all_sections_parsed": memo.parse_success,
        "decision_extracted": memo.decision != "UNKNOWN",
        "risk_grade_extracted": memo.risk_grade != "UNKNOWN",
        "conditions_parsed": True,
        "risk_flags_count": len(memo.risk_flags),
        "risk_flags_valid": 2 <= len(memo.risk_flags) <= 4,
        "parse_errors": memo.parse_errors
    }


# ── UI Layout ──────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="RinLekha — Local Credit Memo Generator"
) as app:

    gr.Markdown("# 🏦 RinLekha")
    gr.Markdown(
        "**Fine-tuned credit memo generation. "
        "Runs fully locally — no borrower data transmitted externally.**"
    )

    with gr.Row():
        # ── Input Panel ───────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Borrower Profile")

            with gr.Group():
                gr.Markdown("**Demographics & Employment**")
                age = gr.Slider(22, 65, value=34, step=1, label="Age")
                income = gr.Number(value=85000, label="Monthly Income (₹)")
                employment_type = gr.Dropdown(
                    ["Salaried Private", "Salaried Govt",
                     "Self Employed Professional", "Self Employed Business"],
                    value="Salaried Private",
                    label="Employment Type"
                )
                tenure_years = gr.Slider(
                    0.25, 25, value=4.5, step=0.25,
                    label="Employment Tenure (years)"
                )
                sector = gr.Dropdown(
                    ["IT", "Manufacturing", "Healthcare",
                     "Finance", "Retail", "Education", "Govt"],
                    value="IT", label="Sector"
                )

            with gr.Group():
                gr.Markdown("**Credit Profile**")
                cibil_score = gr.Slider(
                    550, 900, value=724, step=1, label="CIBIL Score"
                )
                existing_emis = gr.Number(
                    value=18000, label="Existing EMIs/month (₹)"
                )
                missed_payments = gr.Slider(
                    0, 6, value=0, step=1,
                    label="Missed Payments (last 24 months)"
                )

            with gr.Group():
                gr.Markdown("**Loan Request**")
                loan_amount = gr.Number(
                    value=800000, label="Loan Amount (₹)"
                )
                tenure_months = gr.Dropdown(
                    [12, 24, 36, 48, 60, 84],
                    value=48, label="Tenure (months)"
                )
                loan_purpose = gr.Dropdown(
                    ["Home Renovation", "Medical Emergency",
                     "Education", "Debt Consolidation",
                     "Business Expansion", "Wedding",
                     "Travel", "Vehicle Purchase"],
                    value="Home Renovation", label="Loan Purpose"
                )
                interest_rate = gr.Slider(
                    10.5, 26.0, value=14.5, step=0.5,
                    label="Interest Rate (% per annum)"
                )

            generate_btn = gr.Button(
                "Generate Credit Memo", variant="primary", size="lg"
            )

        # ── Output Panel ──────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Output")

            with gr.Tabs():
                with gr.Tab("📄 RinLekha (Fine-tuned)"):
                    ft_output = gr.Markdown()

                with gr.Tab("🔵 Base Model (Prompted)"):
                    base_output = gr.Markdown()
                    gr.Markdown(
                        "*Same input, base Gemma 4 E4B with full format prompt.*"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Risk Dashboard**")
                    risk_dashboard_output = gr.JSON()

                with gr.Column():
                    gr.Markdown("**Format Compliance**")
                    compliance_output = gr.JSON()

    generate_btn.click(
        fn=run_analysis,
        inputs=[age, income, employment_type, tenure_years, sector,
                cibil_score, existing_emis, missed_payments,
                loan_amount, tenure_months, loan_purpose, interest_rate],
        outputs=[ft_output, base_output,
                 risk_dashboard_output, compliance_output]
    )

    gr.Markdown(
        "---\n"
        "**Model:** `rinlekha-gemma4-e4b-finetuned` | "
        "**Inference:** vLLM (local) | "
        "**Observability:** Langfuse | "
        "[GitHub](#) | [HuggingFace](#)"
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

---

## Error Handling Strategy

```python
# serving/error_handling.py

class ServingErrorHandler:
    """
    Graceful degradation strategy.
    Never crash the user-facing app.
    Always return something useful.
    """

    @staticmethod
    def handle_vllm_timeout(profile: dict) -> CreditMemo:
        """vLLM didn't respond within timeout"""
        return CreditMemo(
            applicant_summary="[SERVICE TIMEOUT — please retry]",
            parse_success=False,
            parse_errors=["vLLM server timeout"],
            decision=CreditDecision.UNKNOWN,
            risk_grade=RiskGrade.UNKNOWN,
            # ... other fields empty
        )

    @staticmethod
    def handle_parse_failure(raw_output: str) -> CreditMemo:
        """Model output unparseable — return raw with flag"""
        return CreditMemo(
            raw_output=raw_output,
            parse_success=False,
            parse_errors=["Structural parse failure — raw output preserved"],
            decision=CreditDecision.UNKNOWN,
            applicant_summary=raw_output[:500],  # show partial output
            risk_grade=RiskGrade.UNKNOWN
        )

    @staticmethod
    def handle_ood_input(profile: dict) -> CreditMemo:
        """Input outside training distribution — warn but proceed"""
        warnings = validate_input_range(profile)
        memo = generate_credit_memo_observed(profile)
        if warnings:
            memo.parse_errors.extend([
                f"Input warning: {w}" for w in warnings
            ])
        return memo
```

---

## Human Review Trigger Rules

Post-processing rules that flag memos requiring mandatory human review.
Document these in the model card.

```python
HUMAN_REVIEW_TRIGGERS = [
    {
        "condition": lambda p, m: 0.48 <= p["foir_post_loan"] <= 0.52,
        "reason": "FOIR at policy boundary — model most error-prone here"
    },
    {
        "condition": lambda p, m: p.get("settled_accounts_ever", 0) >= 1,
        "reason": "Settled account — requires NOC verification"
    },
    {
        "condition": lambda p, m: m.decision == CreditDecision.UNKNOWN,
        "reason": "Decision could not be parsed — full human review"
    },
    {
        "condition": lambda p, m: p["loan_purpose"] == "debt_consolidation",
        "reason": "Consolidation loan — risk pattern review recommended"
    },
    {
        "condition": lambda p, m: p["cibil_score"] < 670,
        "reason": "Below 670 CIBIL — boundary credit behavior"
    },
    {
        "condition": lambda p, m: not m.parse_success,
        "reason": "Parse failure — output format non-standard"
    }
]
```

---

## Deliverables Checklist

```
□ serving/schemas.py              — Pydantic output schema
□ serving/parser.py               — Deterministic memo parser
□ serving/pipeline.py             — LangChain pipeline
□ serving/observability.py        — Langfuse @observe wrapper
□ serving/error_handling.py       — Graceful degradation
□ app/gradio_app.py               — Full Gradio UI
□ vLLM server verified running    — health check passing
□ LangChain parse success ≥92%    — measured on 80 test cases
□ Langfuse dashboard live         — 50+ traces visible, scores logged
□ Gradio app running locally      — both tabs functional
□ Screenshot: Langfuse dashboard  — portfolio artifact
□ Screenshot: Gradio comparison   — FT vs base side-by-side
```