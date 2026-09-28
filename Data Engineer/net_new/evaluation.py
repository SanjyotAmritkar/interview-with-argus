"""Deterministic, label-free diagnostics for run records.

These checks describe execution, citation traceability and input exposure. Quote
anchoring only means a cited quote appears contiguously in the supplied chunk it
names after conservative normalization. It is not factual correctness, entailment,
or semantic evidence support, and nothing here is an accuracy score.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Literal

OutputState = Literal["missing", "error", "completed"]

_QUOTE_CHARS = str.maketrans(
    {
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark / apostrophe
        "‚": "'",
        "‛": "'",
        "′": "'",  # prime
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',  # double prime
    }
)


def rate(numerator: int, denominator: int) -> float | None:
    """A zero denominator is undefined, not 0%."""
    return numerator / denominator if denominator else None


def resolve_citation(evidence: dict, context: dict) -> dict | None:
    """Return the supplied chunk a citation names, or None.

    Only chunks actually given to the model (`current` and `prior`) are searched.
    Accepts full `<document_id>:<hash>` IDs and legacy suffix-only `<hash>` IDs.
    The chunk must belong to the cited document.
    """
    chunks = {c["id"]: c for c in [*context.get("current", []), *context.get("prior", [])]}
    document_id, chunk_id = evidence["document_id"], evidence["chunk_id"]
    chunk = chunks.get(chunk_id) or chunks.get(f"{document_id}:{chunk_id}")
    if chunk is None or chunk["document_id"] != document_id:
        return None
    return chunk


def normalize_text(text: str) -> str:
    """Conservative normalization for textual anchoring; never drops words."""
    text = html.unescape(text)
    text = text.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
    text = text.translate(_QUOTE_CHARS)
    return re.sub(r"\s+", " ", text).strip()


def quote_anchored(quote: str, chunk_text: str) -> bool:
    """True when the normalized quote occurs contiguously in the normalized chunk.

    Ellipsis-compressed or paraphrased quotes are not anchored unless the exact
    normalized string is present. This is textual anchoring, not semantic support.
    """
    needle = normalize_text(quote)
    return bool(needle) and needle in normalize_text(chunk_text)


@dataclass(frozen=True)
class CitationCheck:
    side: Literal["current", "prior"]
    finding_index: int
    document_id: str
    chunk_id: str
    resolved_chunk_id: str | None
    anchored: bool

    @property
    def resolvable(self) -> bool:
        return self.resolved_chunk_id is not None


@dataclass(frozen=True)
class CaseDiagnostics:
    case_id: str
    output_state: OutputState
    # Execution. Finding/evidence fields are None unless output completed.
    finding_count: int | None
    completed_empty: bool | None
    # Evidence
    current_citations: int | None
    prior_citations: int | None
    total_citations: int | None
    resolvable_citations: int | None
    resolvable_citation_rate: float | None
    anchored_citations: int | None
    anchored_rate_of_resolvable: float | None
    # Input exposure. Available for error records too, since input is built first.
    current_chunks_supplied: int | None
    current_chunks_total: int | None
    current_context_coverage: float | None
    current_context_truncated: bool | None
    prior_chunks_supplied: int | None
    eligible_history_documents: int | None
    citations: tuple[CitationCheck, ...] = ()


def check_citations(prediction: dict, context: dict) -> list[CitationCheck]:
    checks = []
    for index, finding in enumerate(prediction.get("findings", [])):
        for side in ("current", "prior"):
            for evidence in finding.get(f"{side}_evidence", []):
                chunk = resolve_citation(evidence, context)
                checks.append(
                    CitationCheck(
                        side=side,
                        finding_index=index,
                        document_id=evidence["document_id"],
                        chunk_id=evidence["chunk_id"],
                        resolved_chunk_id=chunk["id"] if chunk else None,
                        anchored=bool(chunk) and quote_anchored(evidence["quote"], chunk["text"]),
                    )
                )
    return checks


def diagnose_case(case_id: str, record: dict | None) -> CaseDiagnostics:
    """Diagnostics for one case. `record` is a records.jsonl row, or None if missing.

    For aligned rows use `diagnose_case(row["case_id"], row["output"])`; the state
    rule matches `references.align_cases`.
    """
    state: OutputState = (
        "missing" if record is None else "completed" if record["status"] == "completed" else "error"
    )
    context = (record or {}).get("input")
    exposure: dict = dict.fromkeys(
        [
            "current_chunks_supplied",
            "current_chunks_total",
            "current_context_coverage",
            "current_context_truncated",
            "prior_chunks_supplied",
            "eligible_history_documents",
        ]
    )
    if context is not None:
        supplied, total = len(context["current"]), context["current_chunks_total"]
        history = context.get("eligible_history_document_ids")
        exposure.update(
            current_chunks_supplied=supplied,
            current_chunks_total=total,
            current_context_coverage=rate(supplied, total),
            current_context_truncated=supplied < total,
            prior_chunks_supplied=len(context["prior"]),
            eligible_history_documents=None if history is None else len(history),
        )
    if state != "completed":
        return CaseDiagnostics(
            case_id=case_id,
            output_state=state,
            finding_count=None,
            completed_empty=None,
            current_citations=None,
            prior_citations=None,
            total_citations=None,
            resolvable_citations=None,
            resolvable_citation_rate=None,
            anchored_citations=None,
            anchored_rate_of_resolvable=None,
            **exposure,
        )
    prediction = record["prediction"]
    checks = check_citations(prediction, context or {})
    resolvable = sum(c.resolvable for c in checks)
    anchored = sum(c.anchored for c in checks)
    findings = len(prediction["findings"])
    return CaseDiagnostics(
        case_id=case_id,
        output_state=state,
        finding_count=findings,
        completed_empty=findings == 0,
        current_citations=sum(c.side == "current" for c in checks),
        prior_citations=sum(c.side == "prior" for c in checks),
        total_citations=len(checks),
        resolvable_citations=resolvable,
        resolvable_citation_rate=rate(resolvable, len(checks)),
        anchored_citations=anchored,
        anchored_rate_of_resolvable=rate(anchored, resolvable),
        citations=tuple(checks),
        **exposure,
    )


@dataclass(frozen=True)
class RunDiagnostics:
    cases: int
    completed: int
    errors: int
    missing: int
    completed_empty: int
    findings: int
    citations: int
    resolvable_citations: int
    resolvable_citation_rate: float | None
    anchored_citations: int
    anchored_rate_of_resolvable: float | None
    completed_truncated_current: int
    completed_truncation_rate: float | None


def aggregate(cases: list[CaseDiagnostics]) -> RunDiagnostics:
    """Pool underlying counts across cases; per-case rates are never averaged."""
    done = [c for c in cases if c.output_state == "completed"]
    citations = sum(c.total_citations for c in done)
    resolvable = sum(c.resolvable_citations for c in done)
    anchored = sum(c.anchored_citations for c in done)
    truncated = sum(bool(c.current_context_truncated) for c in done)
    return RunDiagnostics(
        cases=len(cases),
        completed=len(done),
        errors=sum(c.output_state == "error" for c in cases),
        missing=sum(c.output_state == "missing" for c in cases),
        completed_empty=sum(bool(c.completed_empty) for c in done),
        findings=sum(c.finding_count for c in done),
        citations=citations,
        resolvable_citations=resolvable,
        resolvable_citation_rate=rate(resolvable, citations),
        anchored_citations=anchored,
        anchored_rate_of_resolvable=rate(anchored, resolvable),
        completed_truncated_current=truncated,
        completed_truncation_rate=rate(truncated, len(done)),
    )
