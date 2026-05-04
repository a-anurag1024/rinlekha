"""
RinLekha — Gradio demo app.

Left panel : borrower profile input form
Right panel: two tabs (RinLekha fine-tuned | GPT-4o-mini baseline) + risk dashboard

All inference runs locally via llama-cpp-python.
No borrower data is transmitted externally except to the GPT-4o-mini baseline tab
(optional — click only if you want the comparison).

Usage:
  python app/gradio_app.py

Requires:
  bash serving/start_server.sh outputs/rinlekha-q8.gguf
"""
import json, os, sys, time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import openai

from serving.observability import generate_credit_memo_observed
from serving.pipeline import format_profile
from serving.schemas import CreditDecision, CreditMemo, RiskGrade

_DECISION_ICON = {
    CreditDecision.APPROVE:             "🟢 APPROVE",
    CreditDecision.CONDITIONAL_APPROVE: "🟡 CONDITIONAL APPROVE",
    CreditDecision.DECLINE:             "🔴 DECLINE",
    CreditDecision.UNKNOWN:             "⚪ UNKNOWN",
}
_GRADE_ICON = {
    RiskGrade.A:       "🟢 A",
    RiskGrade.B_PLUS:  "🟢 B+",
    RiskGrade.B:       "🟡 B",
    RiskGrade.B_MINUS: "🟠 B-",
    RiskGrade.C:       "🔴 C",
    RiskGrade.UNKNOWN: "⚪ ?",
}

_BASELINE_SYSTEM = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly.\n\n"
    "Use this exact section order:\n"
    "## APPLICANT SUMMARY\n## DEBT SERVICEABILITY\n## CREDIT BEHAVIOR\n"
    "## RISK FLAGS\n## RECOMMENDATION\n## ANALYST NOTES\n\n"
    "The RECOMMENDATION section must end with:\n"
    "DECISION: <APPROVE|CONDITIONAL APPROVE|DECLINE>\n"
    "CONDITIONS: <list or N/A>\nRISK GRADE: <A/B/C+/->\n"
    "DECISION AUTHORITY: <role>\nREVIEW TRIGGER: <condition>"
)


def _build_profile(age, income, emp_type, tenure, sector, city_tier,
                   cibil, existing_emi, missed, settled, active_loans, vintage,
                   loan_amount, loan_tenure, purpose, rate) -> dict:
    return {
        "age":                     int(age),
        "city_tier":               city_tier,
        "employment_type":         emp_type,
        "sector":                  sector,
        "monthly_income":          float(income),
        "employment_tenure_years": float(tenure),
        "cibil_score":             int(cibil),
        "existing_emi_monthly":    float(existing_emi),
        "missed_payments_24m":     int(missed),
        "settled_accounts_ever":   int(settled),
        "active_loans":            int(active_loans),
        "credit_vintage_years":    float(vintage),
        "loan_amount":             float(loan_amount),
        "loan_tenure_months":      int(loan_tenure),
        "loan_purpose":            purpose,
        "annual_interest_rate":    float(rate),
    }


def _format_memo_md(memo: CreditMemo) -> str:
    flags_md = "\n".join(f"- {f}" for f in memo.risk_flags)
    conds_md = (
        "\n".join(f"{i+1}. {c}" for i, c in enumerate(memo.conditions))
        if memo.conditions else "N/A"
    )
    return f"""## APPLICANT SUMMARY
{memo.applicant_summary}

## DEBT SERVICEABILITY
{memo.debt_serviceability}

## CREDIT BEHAVIOR
{memo.credit_behavior}

## RISK FLAGS
{flags_md}

## RECOMMENDATION
**Decision:** {_DECISION_ICON[memo.decision]}
**Risk Grade:** {_GRADE_ICON[memo.risk_grade]}
**Decision Authority:** {memo.decision_authority}
**Review Trigger:** {memo.review_trigger}
**Conditions:**
{conds_md}

## ANALYST NOTES
{memo.analyst_notes}
"""


def _risk_dashboard(memo: CreditMemo) -> dict:
    return {
        "decision":           memo.decision.value,
        "risk_grade":         memo.risk_grade.value,
        "decision_authority": memo.decision_authority,
        "conditions_count":   len(memo.conditions),
        "risk_flags":         memo.risk_flags,
        "parse_success":      memo.parse_success,
        "parse_errors":       memo.parse_errors,
    }


def _compliance_check(memo: CreditMemo) -> dict:
    return {
        "all_sections_parsed":  memo.parse_success,
        "decision_extracted":   memo.decision != CreditDecision.UNKNOWN,
        "risk_grade_extracted": memo.risk_grade != RiskGrade.UNKNOWN,
        "risk_flags_count":     len(memo.risk_flags),
        "risk_flags_valid":     2 <= len(memo.risk_flags) <= 4,
        "parse_errors":         memo.parse_errors,
    }


def generate_rinlekha(age, income, emp_type, tenure, sector, city_tier,
                      cibil, existing_emi, missed, settled, active_loans, vintage,
                      loan_amount, loan_tenure, purpose, rate):
    profile = _build_profile(age, income, emp_type, tenure, sector, city_tier,
                             cibil, existing_emi, missed, settled, active_loans, vintage,
                             loan_amount, loan_tenure, purpose, rate)
    try:
        memo = generate_credit_memo_observed(profile)
        return (
            _format_memo_md(memo),
            _risk_dashboard(memo),
            _compliance_check(memo),
        )
    except Exception as exc:
        err = f"**Error:** {exc}\n\nIs the inference server running?\n`bash serving/start_server.sh outputs/rinlekha-q8.gguf`"
        return err, {}, {}


def generate_baseline(age, income, emp_type, tenure, sector, city_tier,
                      cibil, existing_emi, missed, settled, active_loans, vintage,
                      loan_amount, loan_tenure, purpose, rate):
    profile = _build_profile(age, income, emp_type, tenure, sector, city_tier,
                             cibil, existing_emi, missed, settled, active_loans, vintage,
                             loan_amount, loan_tenure, purpose, rate)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return "**Error:** OPENAI_API_KEY not set in environment.", {}, {}
    try:
        client = openai.OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=700,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _BASELINE_SYSTEM},
                {"role": "user",   "content": format_profile(profile)},
            ],
        )
        return resp.choices[0].message.content.strip(), {}, {}
    except Exception as exc:
        return f"**Error:** {exc}", {}, {}


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="RinLekha — Credit Memo Generator") as app:

    gr.Markdown("# RinLekha — NBFC Credit Memo Generator")
    gr.Markdown(
        "Fine-tuned Gemma 3 4B · Runs fully locally · "
        "**No borrower data transmitted externally** (except optional GPT-4o-mini comparison tab)"
    )

    with gr.Row():

        # ── Input panel ───────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Borrower Profile")

            with gr.Group():
                gr.Markdown("**Demographics & Employment**")
                age          = gr.Slider(22, 65, value=34, step=1, label="Age")
                city_tier    = gr.Dropdown(["Tier 1", "Tier 2", "Tier 3"], value="Tier 2", label="City Tier")
                income       = gr.Number(value=85000, label="Monthly Income (₹)")
                emp_type     = gr.Dropdown(
                    ["Salaried Private", "Salaried Govt",
                     "Self Employed Professional", "Self Employed Business"],
                    value="Salaried Private", label="Employment Type",
                )
                tenure       = gr.Slider(0.25, 25, value=4.5, step=0.25, label="Employment Tenure (years)")
                sector       = gr.Dropdown(
                    ["Technology", "Manufacturing", "Healthcare",
                     "Finance", "Retail", "Education", "Govt"],
                    value="Technology", label="Sector",
                )

            with gr.Group():
                gr.Markdown("**Credit Profile**")
                cibil        = gr.Slider(550, 900, value=724, step=1, label="CIBIL Score")
                existing_emi = gr.Number(value=12000, label="Existing EMIs/month (₹)")
                missed       = gr.Slider(0, 6, value=0, step=1, label="Missed Payments (last 24m)")
                settled      = gr.Slider(0, 3, value=0, step=1, label="Settled Accounts (ever)")
                active_loans = gr.Slider(0, 8, value=2, step=1, label="Active Loans")
                vintage      = gr.Slider(0, 20, value=5.0, step=0.5, label="Credit Vintage (years)")

            with gr.Group():
                gr.Markdown("**Loan Request**")
                loan_amount  = gr.Number(value=500000, label="Loan Amount (₹)")
                loan_tenure  = gr.Dropdown([12, 24, 36, 48, 60, 84], value=48, label="Tenure (months)")
                purpose      = gr.Dropdown(
                    ["Home Renovation", "Medical", "Education",
                     "Debt Consolidation", "Business Expansion",
                     "Wedding", "Vehicle Purchase"],
                    value="Home Renovation", label="Loan Purpose",
                )
                rate         = gr.Slider(10.5, 26.0, value=14.5, step=0.5, label="Interest Rate (% p.a.)")

            inputs = [age, income, emp_type, tenure, sector, city_tier,
                      cibil, existing_emi, missed, settled, active_loans, vintage,
                      loan_amount, loan_tenure, purpose, rate]

            generate_btn  = gr.Button("Generate Credit Memo", variant="primary", size="lg")
            baseline_btn  = gr.Button("Compare with GPT-4o-mini", variant="secondary", size="sm")

        # ── Output panel ──────────────────────────────────────────────────────
        with gr.Column(scale=2):

            with gr.Tabs():
                with gr.Tab("RinLekha (Fine-tuned Gemma 3 4B)"):
                    rl_output = gr.Markdown(label="Credit Memo")

                with gr.Tab("GPT-4o-mini (Prompted Baseline)"):
                    bl_output = gr.Markdown(label="Baseline Memo")
                    gr.Markdown("*Same borrower profile, prompted GPT-4o-mini — no fine-tuning.*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Risk Dashboard**")
                    risk_output = gr.JSON()
                with gr.Column():
                    gr.Markdown("**Format Compliance**")
                    compliance_output = gr.JSON()

    rl_outputs = [rl_output, risk_output, compliance_output]
    bl_outputs = [bl_output, risk_output, compliance_output]

    generate_btn.click(fn=generate_rinlekha,  inputs=inputs, outputs=rl_outputs)
    baseline_btn.click(fn=generate_baseline,  inputs=inputs, outputs=bl_outputs)

    gr.Markdown(
        "---\n"
        "Model: `a-anurag1024/rinlekha-gemma3-4b-finetuned` · "
        "Serving: llama-cpp-python (GGUF Q8_0) · "
        "Observability: Langfuse · "
        "[GitHub](https://github.com/a-anurag1024/rinlekha) · "
        "[HuggingFace](https://huggingface.co/a-anurag1024/rinlekha-gemma3-4b-finetuned)"
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False,
               theme=gr.themes.Soft())
