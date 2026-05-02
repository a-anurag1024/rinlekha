"""Custom DeepEval metrics for RinLekha credit memo evaluation."""
import re
from deepeval.metrics import BaseMetric, GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class StructuralComplianceMetric(BaseMetric):
    threshold = 0.90
    name = "Structural Compliance"

    SECTIONS_IN_ORDER = [
        "## APPLICANT SUMMARY",
        "## DEBT SERVICEABILITY",
        "## CREDIT BEHAVIOR",
        "## RISK FLAGS",
        "## RECOMMENDATION",
        "## ANALYST NOTES",
    ]

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output
        checks = []

        for section in self.SECTIONS_IN_ORDER:
            checks.append(section in output)

        positions = [output.find(s) for s in self.SECTIONS_IN_ORDER if s in output]
        checks.append(positions == sorted(positions))

        for i, section in enumerate(self.SECTIONS_IN_ORDER):
            if section in output:
                start = output.index(section) + len(section)
                next_positions = [
                    output.find(s, start)
                    for s in self.SECTIONS_IN_ORDER
                    if output.find(s, start) > 0
                ]
                end = min(next_positions) if next_positions else len(output)
                content = output[start:end].strip()
                checks.append(len(content) > 30)

        self.score = sum(checks) / len(checks)
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


class RecommendationFormatMetric(BaseMetric):
    threshold = 1.0
    name = "Recommendation Format"

    REQUIRED_PATTERNS = {
        "decision":           r"DECISION:\s*(APPROVE|CONDITIONAL APPROVE|DECLINE)",
        "conditions":         r"CONDITIONS:",
        "risk_grade":         r"RISK GRADE:\s*[ABC][+-]?",
        "decision_authority": r"DECISION AUTHORITY:",
        "review_trigger":     r"REVIEW TRIGGER:",
    }

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output
        results = {
            field: bool(re.search(pattern, output, re.IGNORECASE))
            for field, pattern in self.REQUIRED_PATTERNS.items()
        }
        self.score = sum(results.values()) / len(results)
        self.field_results = results
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


class ForbiddenLanguageMetric(BaseMetric):
    threshold = 1.0
    name = "Forbidden Language"

    FORBIDDEN = [
        "definitely", "certainly", "guaranteed",
        "will definitely", "100%", "no doubt",
        "absolutely certain", "without question",
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

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


class RiskFlagsCountMetric(BaseMetric):
    threshold = 1.0
    name = "Risk Flags Count"

    def measure(self, test_case: LLMTestCase) -> float:
        flags_section = self._extract_section(test_case.actual_output, "## RISK FLAGS")
        bullets = re.findall(r"^[-•*]\s+.+", flags_section, re.MULTILINE)
        valid = 2 <= len(bullets) <= 4
        self.score = 1.0 if valid else 0.0
        self.flag_count = len(bullets)
        self.success = valid
        return self.score

    def _extract_section(self, text: str, header: str) -> str:
        if header not in text:
            return ""
        start = text.index(header) + len(header)
        next_headers = [
            "## APPLICANT SUMMARY", "## DEBT SERVICEABILITY",
            "## CREDIT BEHAVIOR", "## RISK FLAGS",
            "## RECOMMENDATION", "## ANALYST NOTES",
        ]
        positions = [text.find(h, start) for h in next_headers if text.find(h, start) > 0]
        end = min(positions) if positions else len(text)
        return text[start:end]

    def is_successful(self) -> bool:
        return self.success

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


def build_geval_metric(judge_model: str = "gpt-4o-mini") -> GEval:
    return GEval(
        name="Credit Memo Quality",
        model=judge_model,
        criteria="""
Evaluate this credit memo on 5 dimensions (1–5 scale each):

1. Factual Accuracy: Do all figures (FOIR, EMI, CIBIL) match the input?
   Check at least 3 specific numbers.

2. Risk Appropriateness: Does the risk grade match the profile risk?
   CIBIL 800+ profile getting C grade = wrong. FOIR 52%+ getting A grade = wrong.

3. Condition Specificity: Are conditions verifiable and specific?
   "Provide income proof" = generic. "Last 6 months salary slips + Form 16" = specific.

4. Analytical Depth: Does the analyst interpret or merely restate?
   "FOIR is 44%" = restatement. "FOIR of 44% is within policy but represents
   significant EMI step-up — income stability is therefore critical" = analysis.

5. Hedging Compliance: Is language appropriately uncertain?
   Good: indicates, suggests, warrants, appears.
   Bad: will, definitely, guaranteed.

Score each 1–5. Final score = average / 5 (normalized 0–1).
""",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )


def build_faithfulness_metric(judge_model: str = "gpt-4o-mini") -> FaithfulnessMetric:
    return FaithfulnessMetric(
        threshold=0.85,
        model=judge_model,
        include_reason=True,
    )
