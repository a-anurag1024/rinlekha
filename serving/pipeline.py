"""
LangChain serving pipeline — profile dict → CreditMemo.

Uses the llama-cpp-python OpenAI-compatible server via the completions
endpoint (not chat) to match the Alpaca format the model was trained on.

Requires:
  bash serving/start_server.sh outputs/rinlekha-q8.gguf
"""
import math
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

from serving import parser
from serving.schemas import CreditMemo

_INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)

_ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{profile_text}\n\n"
    "### Response:\n"
)

_prompt = PromptTemplate(
    input_variables=["instruction", "profile_text"],
    template=_ALPACA_TEMPLATE,
)

# llama-cpp-python exposes an OpenAI-compatible completions endpoint.
# api_key is required by the client but unused by the server.
_llm = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="rinlekha",
    temperature=0.1,
    top_p=0.9,
    max_tokens=700,
    frequency_penalty=0.1,
    stop=["### Instruction:"],
)

_chain = _prompt | _llm


def _emi(principal: float, annual_rate: float, months: int) -> float:
    if annual_rate <= 0:
        return round(principal / months, 2)
    r = annual_rate / 12 / 100
    return round(principal * r * (1 + r) ** months / ((1 + r) ** months - 1), 2)


def format_profile(profile: dict) -> str:
    """Convert a structured profile dict into the borrower profile text block."""
    income = profile.get("monthly_income", 0)
    existing_emi = profile.get("existing_emi_monthly", 0)
    loan_amount = profile.get("loan_amount", 0)
    tenure_months = profile.get("loan_tenure_months", 36)
    annual_rate = profile.get("annual_interest_rate", 14.0)
    cibil = profile.get("cibil_score", 0)

    proposed_emi = _emi(loan_amount, annual_rate, tenure_months)
    foir_pre = round(existing_emi / income * 100, 1) if income else 0
    foir_post = round((existing_emi + proposed_emi) / income * 100, 1) if income else 0
    lti = round(loan_amount / income, 1) if income else 0

    city_tier = profile.get("city_tier", "Tier 1")
    emp_type = profile.get("employment_type", "Salaried Private").replace("_", " ").title()
    sector = profile.get("sector", "").replace("_", " ").title()

    return f"""\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : {profile.get("age", "N/A")} years
  City Tier       : {city_tier}

EMPLOYMENT
  Type            : {emp_type}
  Sector          : {sector}
  Monthly Income  : ₹{income:,.0f}
  Tenure          : {profile.get("employment_tenure_years", 0)} years

CREDIT PROFILE
  CIBIL Score     : {cibil}
  Missed Payments (last 24m): {profile.get("missed_payments_24m", 0)}
  Settled Accounts: {profile.get("settled_accounts_ever", 0)}
  Active Loans    : {profile.get("active_loans", 0)}
  Credit Vintage  : {profile.get("credit_vintage_years", 0)} years
  Existing EMI    : ₹{existing_emi:,.0f}/month

LOAN REQUEST
  Amount          : ₹{loan_amount:,.0f}
  Tenure          : {tenure_months} months
  Purpose         : {profile.get("loan_purpose", "").replace("_", " ").title()}
  Annual Rate     : {annual_rate}%

DERIVED METRICS
  Proposed EMI    : ₹{proposed_emi:,.0f}/month
  FOIR (pre-loan) : {foir_pre}%
  FOIR (post-loan): {foir_post}%  [policy ceiling: 55%]
  Loan-to-Income  : {lti}×"""


def generate_credit_memo(profile: dict) -> CreditMemo:
    """Profile dict → CreditMemo. Entry point for the observability wrapper."""
    profile_text = format_profile(profile)
    raw = _chain.invoke({"instruction": _INSTRUCTION, "profile_text": profile_text})
    return parser.parse(raw.strip())
