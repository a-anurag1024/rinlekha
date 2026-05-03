"""
Quick sanity check: generate one credit memo directly from the LoRA adapter via
Unsloth (slow, but accurate). Prints first 600 chars so you can see whether the
output starts with ## APPLICANT SUMMARY (fine-tuned) or numbered/bolded headers
(base model). If the adapter produces the right format but the GGUF does not, the
merge failed and needs to be rerun.

Usage:
  python scripts/test_adapter_output.py
  python scripts/test_adapter_output.py --adapter outputs/run3-r16-lr1e4-ep5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)

SAMPLE_INPUT = """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 34 years
  City Tier       : Tier 2

EMPLOYMENT
  Type            : Salaried Private
  Sector          : Private
  Monthly Income  : ₹45,000
  Tenure          : 3.5 years

CREDIT PROFILE
  CIBIL Score     : 720
  Missed Payments (last 24m): 1
  Settled Accounts: 0
  Active Loans    : 2
  Credit Vintage  : 5.2 years
  Existing EMI    : ₹8,000/month

LOAN REQUEST
  Amount          : ₹3,00,000
  Tenure          : 36 months
  Purpose         : Home Renovation
  Annual Rate     : 14.5%

DERIVED METRICS
  Proposed EMI    : ₹10,290/month
  FOIR (pre-loan) : 17.8%
  FOIR (post-loan): 40.6%  [policy ceiling: 55%]
  Loan-to-Income  : 6.7×"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/run3-r16-lr1e4-ep5")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    args = parser.parse_args()

    prompt = (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{SAMPLE_INPUT}\n\n"
        f"### Response:\n"
    )

    print(f"Loading adapter: {args.adapter}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    print("\n" + "=" * 60)
    print("ADAPTER OUTPUT (first 600 chars):")
    print("=" * 60)
    print(generated[:600])
    print("=" * 60)
    print(f"\nStarts with '## APPLICANT SUMMARY': {generated.startswith('## APPLICANT SUMMARY')}")
    print(f"Contains '## DEBT SERVICEABILITY': {'## DEBT SERVICEABILITY' in generated}")


if __name__ == "__main__":
    main()
