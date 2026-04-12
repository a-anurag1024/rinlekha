#!/usr/bin/env python3
"""
pipeline/quality_reviewer.py — Phase 1, Step 3

Automated structural QA for synthesised credit memos.

Each memo is checked against:
  - Section presence  (all 6 required sections must exist)
  - Recommendation fields  (DECISION / RISK GRADE / DECISION AUTHORITY /
                            REVIEW TRIGGER / CONDITIONS regex patterns)
  - Language hygiene  (no overconfident phrases)
  - Risk-flag count   (2–4 bullets in ## RISK FLAGS)
  - Decision consistency  (DECISION: field matches input profile outcome)

Critical checks (structural + rec fields + flag count) determine
`structural_pass`. Language hygiene and decision consistency are
advisory — tracked in the report but never block `structural_pass`.

Usage:
    python pipeline/quality_reviewer.py                          # default paths
    python pipeline/quality_reviewer.py --memos path/to/memos.jsonl
    python pipeline/quality_reviewer.py --memos m.jsonl --report out.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.memo_synthesizer import load_memos

# ─── Constants ────────────────────────────────────────────────────────────────

REQUIRED_SECTIONS: list[str] = [
    "## APPLICANT SUMMARY",
    "## DEBT SERVICEABILITY",
    "## CREDIT BEHAVIOR",
    "## RISK FLAGS",
    "## RECOMMENDATION",
    "## ANALYST NOTES",
]

# Each tuple: (check_key_suffix, compiled_regex)
REQUIRED_REC_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("decision",          re.compile(r"DECISION:\s*(APPROVE|CONDITIONAL APPROVE|DECLINE)")),
    ("risk_grade",        re.compile(r"RISK GRADE:\s*[ABC][+-]?")),
    ("decision_authority",re.compile(r"DECISION AUTHORITY:")),
    ("review_trigger",    re.compile(r"REVIEW TRIGGER:")),
    ("conditions",        re.compile(r"CONDITIONS:")),
]

FORBIDDEN_PHRASES: list[str] = [
    "definitely",
    "certainly",
    "guaranteed",
    "will definitely",
    "100%",
    "no doubt",
    "absolutely",
    "without question",
]

# Maps profile outcome values to the text expected in DECISION:
_OUTCOME_TO_DECISION: dict[str, str] = {
    "APPROVE":             "APPROVE",
    "CONDITIONAL_APPROVE": "CONDITIONAL APPROVE",
    "DECLINE":             "DECLINE",
}

# ─── Section extraction helpers ───────────────────────────────────────────────

def extract_section(memo: str, header: str) -> str:
    """Return the text between *header* and the next ## heading (or EOF)."""
    pattern = re.compile(
        rf"^{re.escape(header)}\s*\n(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(memo)
    return match.group(1) if match else ""


def extract_decision(memo: str) -> str | None:
    """
    Return the normalised decision string from the RECOMMENDATION section,
    or None if not found.

    Normalises "CONDITIONAL APPROVE" → "CONDITIONAL_APPROVE" to match
    profile outcome keys.
    """
    match = re.search(
        r"DECISION:\s*(APPROVE|CONDITIONAL APPROVE|DECLINE)", memo
    )
    if not match:
        return None
    raw = match.group(1).strip()
    return raw.replace(" ", "_")  # "CONDITIONAL APPROVE" → "CONDITIONAL_APPROVE"


# ─── Core check ───────────────────────────────────────────────────────────────

def quality_check(example: dict) -> dict:
    """
    Run all QC checks on a single synthesised example.

    Returns a flat dict of check results.  Boolean values whose keys are
    in the critical set determine ``structural_pass``.

    Args:
        example: dict with keys ``output_memo`` (str) and
                 ``input_profile`` (dict containing ``outcome``).

    Returns:
        checks dict with at minimum:
          - section_<name>: bool (one per REQUIRED_SECTIONS entry)
          - rec_<name>: bool (one per REQUIRED_REC_PATTERNS entry)
          - no_forbidden_language: bool
          - forbidden_phrases_found: list[str]
          - risk_flags_count: int
          - risk_flags_count_valid: bool
          - decision_matches_expected: bool
          - structural_pass: bool
          - needs_review: bool
    """
    memo: str = example.get("output_memo") or ""
    checks: dict = {}

    # ── 1. Section presence ──────────────────────────────────────────────────
    for section in REQUIRED_SECTIONS:
        key = "section_" + section[3:].lower().replace(" ", "_")
        checks[key] = section in memo

    # ── 2. Recommendation field patterns ─────────────────────────────────────
    for suffix, pattern in REQUIRED_REC_PATTERNS:
        checks[f"rec_{suffix}"] = bool(pattern.search(memo))

    # ── 3. Forbidden language (advisory) ─────────────────────────────────────
    memo_lower = memo.lower()
    found_phrases = [phrase for phrase in FORBIDDEN_PHRASES if phrase in memo_lower]
    checks["no_forbidden_language"] = len(found_phrases) == 0
    checks["forbidden_phrases_found"] = found_phrases

    # ── 4. Risk-flag bullet count ─────────────────────────────────────────────
    flags_text = extract_section(memo, "## RISK FLAGS")
    flag_bullets = re.findall(r"^[-•*]\s+.+", flags_text, re.MULTILINE)
    checks["risk_flags_count"] = len(flag_bullets)
    checks["risk_flags_count_valid"] = 2 <= len(flag_bullets) <= 4

    # ── 5. Decision consistency (advisory) ───────────────────────────────────
    stated = extract_decision(memo)
    expected = example.get("input_profile", {}).get("outcome", "")
    checks["decision_matches_expected"] = stated == expected

    # ── 6. Aggregate: critical checks → structural_pass ──────────────────────
    # Advisory checks excluded from the critical set.
    _ADVISORY = {"no_forbidden_language", "decision_matches_expected"}
    critical_bools = [
        v for k, v in checks.items()
        if isinstance(v, bool) and k not in _ADVISORY
    ]
    checks["structural_pass"] = all(critical_bools)
    checks["needs_review"] = not checks["structural_pass"]

    return checks


# ─── Batch review ─────────────────────────────────────────────────────────────

def review_all(examples: list[dict]) -> list[dict]:
    """Run quality_check on every example and attach results in-place.

    Returns a new list of dicts, each original example enriched with
    a ``qc`` key containing the checks dict.
    """
    results = []
    for ex in examples:
        enriched = {**ex, "qc": quality_check(ex)}
        results.append(enriched)
    return results


# ─── Report generation ────────────────────────────────────────────────────────

def generate_report(reviewed: list[dict]) -> dict:
    """
    Summarise QC results across the full batch.

    Returns a dict suitable for JSON serialisation containing:
      - total, passed, failed, pass_rate
      - per-check failure counts
      - advisory failure counts (language, decision consistency)
      - lists of flagged profile_ids (structural failures)
    """
    total = len(reviewed)
    if total == 0:
        return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

    passed = sum(1 for r in reviewed if r["qc"]["structural_pass"])
    failed = total - passed

    # Count failures per check key (bool checks only)
    check_failures: dict[str, int] = {}
    for r in reviewed:
        for k, v in r["qc"].items():
            if isinstance(v, bool) and not v:
                check_failures[k] = check_failures.get(k, 0) + 1

    # Advisory sub-counts
    language_failures = sum(
        1 for r in reviewed if not r["qc"]["no_forbidden_language"]
    )
    decision_mismatches = sum(
        1 for r in reviewed if not r["qc"]["decision_matches_expected"]
    )

    # Profile IDs that need manual review (structural failures only)
    flagged_ids = [
        r.get("profile_id", r.get("input_profile", {}).get("profile_id", "unknown"))
        for r in reviewed if r["qc"]["needs_review"]
    ]

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4),
        "check_failures": dict(sorted(check_failures.items())),
        "advisory": {
            "language_failures": language_failures,
            "decision_mismatches": decision_mismatches,
        },
        "flagged_profile_ids": flagged_ids,
    }


def save_report(report: dict, path: str | Path) -> None:
    """Write the QC report to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    passed = report.get("passed", "?")
    total = report.get("total", "?")
    pass_rate = report.get("pass_rate")
    rate_str = f"{pass_rate:.1%}" if pass_rate is not None else "?"
    print(f"QC report saved → {p}  ({passed}/{total} passed, pass_rate={rate_str})")


def print_report(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n── QC Summary ──────────────────────────────────────")
    print(f"  Total:     {report['total']}")
    print(f"  Passed:    {report['passed']}")
    print(f"  Failed:    {report['failed']}")
    print(f"  Pass rate: {report['pass_rate']:.1%}")

    if report.get("check_failures"):
        print("\n  Per-check failure counts (structural):")
        for k, v in report["check_failures"].items():
            if k not in ("structural_pass", "needs_review"):
                print(f"    {k}: {v}")

    adv = report.get("advisory", {})
    print(f"\n  Advisory — forbidden language: {adv.get('language_failures', 0)}")
    print(f"  Advisory — decision mismatches: {adv.get('decision_mismatches', 0)}")

    if report.get("flagged_profile_ids"):
        n = len(report["flagged_profile_ids"])
        print(f"\n  Flagged for review: {n} profile(s)")
        for pid in report["flagged_profile_ids"][:10]:
            print(f"    • {pid}")
        if n > 10:
            print(f"    … and {n - 10} more (see full report JSON)")
    print("────────────────────────────────────────────────────\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-review synthesised memos.")
    parser.add_argument(
        "--memos",
        type=str,
        default="data/raw/memos.jsonl",
        help="Path to synthesised memos JSONL (default: data/raw/memos.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/qc_report.json",
        help="Output path for QC report JSON (default: data/qc_report.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading memos from {args.memos} …")
    examples = load_memos(args.memos)
    print(f"Loaded {len(examples)} examples.")

    reviewed = review_all(examples)
    report = generate_report(reviewed)
    print_report(report)
    save_report(report, args.report)


if __name__ == "__main__":
    main()
