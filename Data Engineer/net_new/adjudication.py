"""Validate and aggregate human adjudications of development cases.

Judgments are explicit input data; this module never creates them, never matches
findings automatically, and computes no composite score. References are
non-exhaustive, so generated findings without a reference mapping are reviewed
separately rather than counted as false positives.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, get_args

from pydantic import BaseModel, ConfigDict, model_validator

from .evaluation import rate

Coverage = Literal["covered", "partial", "missed"]
ComparisonQuality = Literal["supported", "partial", "missing", "not_applicable"]
EvidenceSupport = Literal["supported", "partial", "unsupported", "not_applicable"]
ExtraAssessment = Literal["supported_useful", "supported_low_value", "unsupported_or_overstated"]


class ReferenceAdjudication(BaseModel):
    """A human judgment of how one reference finding is reflected in the output."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    reference_finding_id: str
    mapped_model_finding_indices: list[int]
    coverage: Coverage
    comparison_quality: ComparisonQuality
    evidence_support: EvidenceSupport
    notes: str

    @model_validator(mode="after")
    def consistent_mapping(self) -> ReferenceAdjudication:
        indices = self.mapped_model_finding_indices
        if any(i < 0 for i in indices):
            raise ValueError("Model finding indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate model finding index in mapping")
        if self.coverage == "missed":
            if indices:
                raise ValueError("A missed reference finding cannot map to model findings")
            if self.comparison_quality != "not_applicable":
                raise ValueError("A missed reference finding has no comparison to assess")
            if self.evidence_support != "not_applicable":
                raise ValueError("A missed reference finding has no evidence to assess")
        else:
            if not indices:
                raise ValueError("Covered or partial findings must map to a model finding")
            if self.evidence_support == "not_applicable":
                raise ValueError("Covered or partial findings must have evidence assessed")
        return self


class ExtraFindingReview(BaseModel):
    """A human judgment of a generated finding not mapped to any reference finding."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    model_finding_index: int
    assessment: ExtraAssessment
    notes: str


@dataclass(frozen=True)
class ReferenceResult:
    adjudication: ReferenceAdjudication
    acceptable_classifications: tuple[str, ...]
    # Derived from the output, not entered by the reviewer. Every mapped
    # classification is kept; None means nothing was mapped, so not evaluable.
    mapped_classifications: tuple[str, ...]
    classification_all_acceptable: bool | None


@dataclass(frozen=True)
class CaseAdjudication:
    case_id: str
    output_state: str
    reference_finding_count: int
    results: tuple[ReferenceResult, ...]
    extras: tuple[ExtraFindingReview, ...]
    # Completed-output indices not mapped to any reference finding.
    unmapped_model_findings: tuple[int, ...]


def adjudicate_case(
    row: dict,
    adjudications: list[ReferenceAdjudication],
    extras: list[ExtraFindingReview],
) -> CaseAdjudication:
    """Validate one aligned case's judgments and derive classification agreement."""
    case = row["case_id"]
    if any(a.case_id != case for a in [*adjudications, *extras]):
        raise ValueError(f"Adjudication belongs to another case than {case}")
    references = {f["finding_id"]: f for f in row["reference"]["findings"]}
    completed = row["output_state"] == "completed"
    if (adjudications or extras) and not completed:
        # Missing/error outputs stay execution failures, never semantic misses.
        raise ValueError(f"Only completed outputs can be adjudicated: {case}")
    findings = row["output"]["prediction"]["findings"] if completed else []

    seen: set[str] = set()
    mapped: set[int] = set()
    results = []
    for a in adjudications:
        reference = references.get(a.reference_finding_id)
        if reference is None:
            raise ValueError(f"Unknown reference finding {a.reference_finding_id} in {case}")
        if a.reference_finding_id in seen:
            raise ValueError(f"Duplicate adjudication for {a.reference_finding_id}")
        seen.add(a.reference_finding_id)
        if any(i >= len(findings) for i in a.mapped_model_finding_indices):
            raise ValueError(f"Mapped model finding index out of range in {case}")
        mapped.update(a.mapped_model_finding_indices)
        acceptable = tuple(reference["acceptable_classifications"])
        classes = tuple(findings[i]["classification"] for i in a.mapped_model_finding_indices)
        results.append(
            ReferenceResult(
                adjudication=a,
                acceptable_classifications=acceptable,
                mapped_classifications=classes,
                classification_all_acceptable=(
                    all(c in acceptable for c in classes) if classes else None
                ),
            )
        )

    reviewed: set[int] = set()
    for e in extras:
        if not 0 <= e.model_finding_index < len(findings):
            raise ValueError(f"Extra finding index out of range in {case}")
        if e.model_finding_index in mapped:
            raise ValueError(f"Model finding {e.model_finding_index} is mapped to a reference")
        if e.model_finding_index in reviewed:
            raise ValueError(f"Duplicate extra-finding review in {case}")
        reviewed.add(e.model_finding_index)

    return CaseAdjudication(
        case_id=case,
        output_state=row["output_state"],
        reference_finding_count=len(references),
        results=tuple(results),
        extras=tuple(extras),
        unmapped_model_findings=tuple(i for i in range(len(findings)) if i not in mapped),
    )


def adjudicate_run(
    rows: list[dict],
    adjudications: list[ReferenceAdjudication],
    extras: list[ExtraFindingReview],
) -> list[CaseAdjudication]:
    """Group judgments by aligned case; judgments for unknown cases are rejected."""
    known = {r["case_id"] for r in rows}
    unknown = {x.case_id for x in [*adjudications, *extras]} - known
    if unknown:
        raise ValueError(f"Adjudications for unknown cases: {sorted(unknown)}")
    return [
        adjudicate_case(
            row,
            [a for a in adjudications if a.case_id == row["case_id"]],
            [e for e in extras if e.case_id == row["case_id"]],
        )
        for row in rows
    ]


def _counts(values: list[str], categories: type) -> dict[str, int]:
    counter = Counter(values)
    return {c: counter[c] for c in get_args(categories)}


def _rates(counts: dict[str, int]) -> dict[str, float | None]:
    total = sum(counts.values())
    return {k: rate(v, total) for k, v in counts.items()}


@dataclass(frozen=True)
class AdjudicationSummary:
    cases: int
    reference_findings: int
    # Reference findings in missing/error cases: execution failures, not misses.
    reference_findings_not_adjudicable: int
    # Reference findings in completed cases with no judgment yet.
    reference_findings_pending: int
    reference_findings_adjudicated: int
    coverage: dict[str, int]
    coverage_rates: dict[str, float | None]
    # Comparison and evidence are judged only where something was mapped
    # (covered or partial); a miss has no comparison or evidence to assess.
    comparison_quality: dict[str, int]
    comparison_quality_rates: dict[str, float | None]
    evidence_support: dict[str, int]
    evidence_support_rates: dict[str, float | None]
    classification_evaluable: int
    classification_all_acceptable: int
    classification_agreement_rate: float | None
    unmapped_model_findings: int
    extras_reviewed: int
    extras_pending: int
    extra_assessments: dict[str, int]
    extra_assessment_rates: dict[str, float | None]


def summarize(cases: list[CaseAdjudication]) -> AdjudicationSummary:
    """Raw counts and pooled rates for one case (pass a single item) or a run."""
    results = [r for c in cases for r in c.results]
    matched = [r.adjudication for r in results if r.adjudication.coverage != "missed"]
    evaluable = [r for r in results if r.classification_all_acceptable is not None]
    extras = [e for c in cases for e in c.extras]
    total = sum(c.reference_finding_count for c in cases)
    not_adjudicable = sum(c.reference_finding_count for c in cases if c.output_state != "completed")
    coverage = _counts([r.adjudication.coverage for r in results], Coverage)
    comparison = _counts([a.comparison_quality for a in matched], ComparisonQuality)
    evidence = _counts([a.evidence_support for a in matched], EvidenceSupport)
    assessments = _counts([e.assessment for e in extras], ExtraAssessment)
    agreed = sum(bool(r.classification_all_acceptable) for r in evaluable)
    unmapped = sum(len(c.unmapped_model_findings) for c in cases)
    return AdjudicationSummary(
        cases=len(cases),
        reference_findings=total,
        reference_findings_not_adjudicable=not_adjudicable,
        reference_findings_pending=total - not_adjudicable - len(results),
        reference_findings_adjudicated=len(results),
        coverage=coverage,
        coverage_rates=_rates(coverage),
        comparison_quality=comparison,
        comparison_quality_rates=_rates(comparison),
        evidence_support=evidence,
        evidence_support_rates=_rates(evidence),
        classification_evaluable=len(evaluable),
        classification_all_acceptable=agreed,
        classification_agreement_rate=rate(agreed, len(evaluable)),
        unmapped_model_findings=unmapped,
        extras_reviewed=len(extras),
        extras_pending=unmapped - len(extras),
        extra_assessments=assessments,
        extra_assessment_rates=_rates(assessments),
    )


Record = TypeVar("Record", ReferenceAdjudication, ExtraFindingReview)


def load_jsonl(path: Path, model: type[Record]) -> list[Record]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line]
