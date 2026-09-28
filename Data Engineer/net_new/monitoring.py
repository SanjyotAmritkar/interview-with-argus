"""Label-free monitoring of replay runs.

Reads only a run directory, a frozen baseline run and source-corpus metadata; never
references, adjudications or aligned outputs. Checks are tiered: invariants are
actionable on any failure, warnings are heuristics against a small frozen
development baseline, and info is reported without thresholds. Textual anchoring
is not evidence correctness, and nothing here is an accuracy measure.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from .dataset import Dataset
from .evaluation import CaseDiagnostics, aggregate, diagnose_case
from .pipeline import load_records
from .references import case_id

# Mirrors the generation cap currently hard-coded in pipeline.fresh_prediction;
# run metadata does not record it.
MAX_OUTPUT_TOKENS = 6000
NEAR_LIMIT_TOKENS = 5400
ANCHORING_FLOOR = 0.5
ANCHORING_MIN_CITATIONS = 20
LATENCY_RATIO = 2.0
INPUT_TOKEN_RATIO = 1.25
INPUT_TOKEN_MIN_CASES = 3
# Selected-configuration fields compared with the frozen baseline. The
# implementation hash covers analysis-only modules too, so it is report-only.
FROZEN_CONFIG = [
    "prompt_hash",
    "model",
    "pipeline_version",
    "parser_version",
    "history_chunks",
    "current_char_budget",
]

Tier = Literal["invariant", "warning", "info"]
Status = Literal["ok", "alert", "insufficient_data", "not_observed", "info"]


@dataclass(frozen=True)
class Check:
    name: str
    tier: Tier
    status: Status
    observed: object = None
    threshold: object = None
    detail: object = None


@dataclass(frozen=True)
class Operational:
    """Per-record operational metadata. `source` says whether it was measured now."""

    source: Literal["live", "historical_cache", "unavailable"]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    latency_ms: float | None
    requested_model: str | None
    reported_model: str | None
    resolved_model: str | None
    catalog_version: str | None
    response_status: str | None
    incomplete_details: object = None


def operational(record: dict) -> Operational:
    meta = record.get("model_metadata") or {}
    usage = meta.get("usage") or {}
    raw = meta.get("raw_response") or {}
    cached = bool(record.get("cache_hit"))
    source = "historical_cache" if cached else "live" if meta else "unavailable"
    cost = meta.get("cost_usd")
    return Operational(
        source=source,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost_usd=None if cost is None else float(cost),
        # Cached latency is the original generation's, never replay time.
        latency_ms=None if cached else record.get("generation_latency_ms"),
        requested_model=meta.get("requested_model"),
        reported_model=meta.get("reported_model"),
        resolved_model=meta.get("resolved_model"),
        catalog_version=meta.get("catalog_version"),
        response_status=raw.get("status") if raw else None,
        incomplete_details=raw.get("incomplete_details") if raw else None,
    )


@dataclass(frozen=True)
class BaselineProfile:
    """Frozen reference values from a full development run (no labels)."""

    run_id: str
    original_run_ids: list[str]
    execution_mode: str
    dataset_fingerprint: str
    config: dict
    catalog_versions: list[str]
    mean_input_tokens: float | None
    median_ms_per_output_token: float | None
    latency_source: str

    @classmethod
    def from_run(cls, run: Path) -> BaselineProfile:
        meta = json.loads((run / "run.json").read_text())
        records = load_records(run)
        completed = [r for r in records if r["status"] == "completed"]
        ops = [operational(r) for r in completed]
        inputs = [o.input_tokens for o in ops if o.input_tokens is not None]
        # A cache replay's latency is the original generation's latency: a real
        # historical measurement, usable as a reference but not as replay time.
        per_token = [
            r["generation_latency_ms"] / o.output_tokens
            for r, o in zip(completed, ops)
            if o.output_tokens and r.get("generation_latency_ms") is not None
        ]
        return cls(
            run_id=meta["run_id"],
            original_run_ids=sorted({r["original_run_id"] for r in records if r.get("original_run_id")}),
            execution_mode=meta["execution_mode"],
            dataset_fingerprint=meta["dataset_fingerprint"],
            config=meta["config"],
            catalog_versions=sorted({o.catalog_version for o in ops if o.catalog_version}),
            mean_input_tokens=statistics.mean(inputs) if inputs else None,
            median_ms_per_output_token=statistics.median(per_token) if per_token else None,
            latency_source="historical_generation" if meta["execution_mode"] == "cache" else "live",
        )


@dataclass
class Case:
    key: str
    filing_id: str
    available_at: datetime
    record: dict | None
    diagnostics: CaseDiagnostics = field(init=False)
    ops: Operational | None = field(init=False)

    def __post_init__(self) -> None:
        self.diagnostics = diagnose_case(self.key, self.record)
        self.ops = operational(self.record) if self.record else None


def month(value: datetime) -> str:
    return value.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m")


def month_batches(cases: list[Case]) -> list[tuple[str, list[Case]]]:
    """Chronological monthly batches, keeping empty months inside the observed span."""
    if not cases:
        return []
    ordered = sorted(cases, key=lambda c: (c.available_at, c.key))
    first, last = month(ordered[0].available_at), month(ordered[-1].available_at)
    y, m = map(int, first.split("-"))
    batches = []
    while f"{y:04d}-{m:02d}" <= last:
        label = f"{y:04d}-{m:02d}"
        batches.append((label, [c for c in ordered if month(c.available_at) == label]))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return batches


def _alert(condition: bool) -> Status:
    return "alert" if condition else "ok"


def _offenders(
    name: str,
    tier: Tier,
    offenders: list,
    observed: object = None,
    threshold: object = None,
    observable: bool = True,
) -> Check:
    """Alert when any offending case exists; not_observed when nothing could be checked."""
    status: Status = _alert(bool(offenders)) if observable else "not_observed"
    return Check(name, tier, status, observed, threshold, offenders)


def case_checks(cases: list[Case], profile: BaselineProfile) -> list[Check]:
    """Checks that apply to any set of expected cases (a whole run or one batch)."""
    diags = [c.diagnostics for c in cases]
    run = aggregate(diags)
    done = [c for c in cases if c.diagnostics.output_state == "completed"]
    zero_current = [
        c.key
        for c in done
        if c.diagnostics.current_chunks_supplied == 0
        and (c.diagnostics.current_chunks_total or 0) > 0
    ]
    checks = [
        _offenders(
            "missing_records",
            "invariant",
            [c.key for c in cases if c.diagnostics.output_state == "missing"],
            run.missing,
            0,
        ),
        _offenders(
            "error_records",
            "invariant",
            [
                {"case": c.key, "error_type": c.record.get("error_type")}
                for c in cases
                if c.diagnostics.output_state == "error"
            ],
            run.errors,
            0,
        ),
        _offenders(
            "unresolved_citations",
            "invariant",
            [
                {"case": c.key, **asdict(x)}
                for c in done
                for x in c.diagnostics.citations
                if not x.resolvable
            ],
            {"resolvable": run.resolvable_citations, "citations": run.citations},
            "all citations resolve to a supplied chunk of the cited document",
        ),
        _offenders("zero_current_context", "invariant", zero_current),
    ]

    responses = [c for c in done if c.ops and c.ops.response_status is not None]
    bad = [
        {"case": c.key, "status": c.ops.response_status, "incomplete": c.ops.incomplete_details}
        for c in responses
        if c.ops.response_status != "completed" or c.ops.incomplete_details is not None
    ]
    checks.append(
        _offenders(
            "response_incomplete",
            "invariant",
            bad,
            {"checked": len(responses), "incomplete": len(bad)},
            'status == "completed" and incomplete_details is None',
            observable=bool(responses),
        )
    )

    # Warnings.
    with_usage = [c for c in done if c.ops and c.ops.output_tokens is not None]
    near = [
        {"case": c.key, "output_tokens": c.ops.output_tokens, "source": c.ops.source}
        for c in with_usage
        if c.ops.output_tokens >= NEAR_LIMIT_TOKENS
    ]
    checks.append(
        _offenders(
            "near_output_token_limit",
            "warning",
            near,
            max((c.ops.output_tokens for c in with_usage), default=None),
            {"warn_at": NEAR_LIMIT_TOKENS, "limit": MAX_OUTPUT_TOKENS},
            observable=bool(with_usage),
        )
    )

    with_meta = [c for c in done if c.ops and c.ops.source != "unavailable"]
    drift = []
    for c in with_meta:
        o = c.ops
        reasons = []
        if o.requested_model != profile.config["model"]:
            reasons.append("requested_model differs from frozen model")
        if o.reported_model != o.requested_model:
            reasons.append("reported_model differs from requested_model")
        if o.resolved_model is not None:
            reasons.append("resolved_model set")
        if profile.catalog_versions and o.catalog_version not in profile.catalog_versions:
            reasons.append("catalog_version differs from baseline")
        if reasons:
            drift.append({"case": c.key, "source": o.source, "reasons": reasons})
    checks.append(
        _offenders(
            "model_metadata_drift",
            "warning",
            drift,
            sorted({f"{c.ops.reported_model}|{c.ops.catalog_version}" for c in with_meta}),
            {"model": profile.config["model"], "catalog_versions": profile.catalog_versions},
            observable=bool(with_meta),
        )
    )

    live = [c for c in done if c.ops and c.ops.source == "live"]
    lacking = [c.key for c in live if c.ops.input_tokens is None or c.ops.cost_usd is None]
    checks.append(
        _offenders(
            "live_usage_metadata_missing",
            "warning",
            lacking,
            {"live_completed": len(live), "missing_usage_or_cost": len(lacking)},
            0,
            observable=bool(live),
        )
    )

    anchored_rate = run.anchored_rate_of_resolvable
    checks.append(
        Check(
            "textual_anchoring_rate",
            "warning",
            "insufficient_data"
            if run.resolvable_citations < ANCHORING_MIN_CITATIONS
            else _alert(anchored_rate < ANCHORING_FLOOR),
            {
                "anchored": run.anchored_citations,
                "resolvable": run.resolvable_citations,
                "rate": anchored_rate,
            },
            {"floor": ANCHORING_FLOOR, "min_resolvable": ANCHORING_MIN_CITATIONS},
            "Quote textual anchoring only; not evidence correctness.",
        )
    )

    timed = [
        c.ops.latency_ms / c.ops.output_tokens
        for c in live
        if c.ops.latency_ms is not None and c.ops.output_tokens
    ]
    base_latency = profile.median_ms_per_output_token
    observed_latency = statistics.median(timed) if timed else None
    checks.append(
        Check(
            "latency_per_output_token",
            "warning",
            "not_observed"
            if not timed
            else "insufficient_data"
            if base_latency is None
            else _alert(observed_latency > LATENCY_RATIO * base_latency),
            {"median_ms_per_output_token": observed_latency, "live_cases": len(timed)},
            {
                "max_ratio": LATENCY_RATIO,
                "baseline_median": base_latency,
                "baseline_source": profile.latency_source,
            },
        )
    )

    with_input = [c for c in done if c.ops and c.ops.input_tokens is not None]
    inputs = [c.ops.input_tokens for c in with_input]
    base_input = profile.mean_input_tokens
    mean_input = statistics.mean(inputs) if inputs else None
    checks.append(
        Check(
            "mean_input_tokens_per_case",
            "warning",
            "not_observed"
            if not inputs
            else "insufficient_data"
            if len(inputs) < INPUT_TOKEN_MIN_CASES or base_input is None
            else _alert(mean_input > INPUT_TOKEN_RATIO * base_input),
            {
                "mean": mean_input,
                "cases": len(inputs),
                "ratio": mean_input / base_input
                if mean_input is not None and base_input
                else None,
                "sources": sorted({c.ops.source for c in with_input}),
            },
            {
                "max_ratio": INPUT_TOKEN_RATIO,
                "baseline_mean": base_input,
                "min_cases": INPUT_TOKEN_MIN_CASES,
            },
        )
    )

    # Info only.
    checks += [
        Check(
            "completed_empty",
            "info",
            "info",
            run.completed_empty,
            detail=[c.key for c in done if c.diagnostics.completed_empty],
        ),
        Check(
            "findings_per_case",
            "info",
            "info",
            {
                "findings": run.findings,
                "completed": run.completed,
                "per_case": {c.key: c.diagnostics.finding_count for c in done},
            },
        ),
        Check(
            "current_context_truncation",
            "info",
            "info",
            {
                "truncated": run.completed_truncated_current,
                "completed": run.completed,
                "min_coverage": min(
                    (
                        c.diagnostics.current_context_coverage
                        for c in done
                        if c.diagnostics.current_context_coverage is not None
                    ),
                    default=None,
                ),
            },
        ),
        Check(
            "prior_chunks_supplied",
            "info",
            "info",
            sorted({c.diagnostics.prior_chunks_supplied for c in cases if c.record}),
        ),
    ]
    return checks


def run_checks(meta: dict, records: list[dict], expected: set[str], profile: BaselineProfile):
    """Run-level checks that are not meaningful per batch."""
    ids = [r["filing_id"] for r in records]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    unexpected = sorted(set(ids) - expected)
    config = meta["config"]
    drift = {
        k: {"run": config.get(k), "frozen": profile.config.get(k)}
        for k in FROZEN_CONFIG
        if config.get(k) != profile.config.get(k)
    }
    if meta["dataset_fingerprint"] != profile.dataset_fingerprint:
        drift["dataset_fingerprint"] = {
            "run": meta["dataset_fingerprint"],
            "frozen": profile.dataset_fingerprint,
        }
    return [
        Check(
            "run_status",
            "invariant",
            _alert(meta.get("status") != "completed" or bool(meta.get("errors"))),
            {"status": meta.get("status"), "errors": meta.get("errors")},
            {"status": "completed", "errors": 0},
        ),
        Check(
            "duplicate_or_unexpected_records",
            "invariant",
            _alert(bool(duplicates or unexpected)),
            {"duplicates": duplicates, "unexpected": unexpected},
            "none",
        ),
        Check(
            "config_drift",
            "invariant",
            _alert(bool(drift)),
            sorted(drift),
            FROZEN_CONFIG + ["dataset_fingerprint"],
            drift,
        ),
        Check(
            "implementation_hash",
            "info",
            "info",
            {
                "run": config.get("implementation_hash"),
                "baseline": profile.config.get("implementation_hash"),
                "same": config.get("implementation_hash")
                == profile.config.get("implementation_hash"),
            },
            detail="Report-only: the hash also covers analysis-only modules.",
        ),
    ]


def overall_status(checks: list[Check]) -> str:
    if any(c.tier == "invariant" and c.status == "alert" for c in checks):
        return "invariant_alert"
    if any(c.tier == "warning" and c.status == "alert" for c in checks):
        return "warning"
    return "ok"


def case_view(c: Case) -> dict:
    d = asdict(c.diagnostics)
    citations = d.pop("citations")
    return {
        **d,
        "filing_id": c.filing_id,
        "available_at": c.available_at.isoformat(),
        "batch": month(c.available_at),
        "unresolved_citations": [x for x in citations if x["resolved_chunk_id"] is None],
        "unanchored_resolvable_citations": [
            {k: x[k] for k in ("side", "finding_index", "document_id", "chunk_id")}
            for x in citations
            if x["resolved_chunk_id"] is not None and not x["anchored"]
        ],
        "operational": asdict(c.ops) if c.ops else None,
    }


def operational_totals(cases: list[Case]) -> dict:
    ops = [c.ops for c in cases if c.ops]
    live = [o for o in ops if o.source == "live"]
    latencies = [o.latency_ms for o in live if o.latency_ms is not None]

    def total(values):
        values = [v for v in values if v is not None]
        return sum(values) if values else None

    return {
        "sources": {s: sum(o.source == s for o in ops) for s in sorted({o.source for o in ops})},
        "input_tokens": total(o.input_tokens for o in ops),
        "output_tokens": total(o.output_tokens for o in ops),
        "total_tokens": total(o.total_tokens for o in ops),
        "live_cost_usd": total(o.cost_usd for o in live),
        "historical_cost_usd": total(o.cost_usd for o in ops if o.source == "historical_cache"),
        "live_latency_ms_median": statistics.median(latencies) if latencies else None,
        "live_latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "note": "Historical cache metadata describes the original generation, not this replay.",
    }


def monitor(run: Path, baseline: Path, dataset: Dataset) -> dict:
    meta = json.loads((run / "run.json").read_text())
    if meta["dataset_fingerprint"] != dataset.fingerprint:
        raise ValueError("Run dataset fingerprint does not match --dataset")
    records = load_records(run)
    profile = BaselineProfile.from_run(baseline)
    primaries = {d.filing_id: d for d in dataset.documents if d.role == "primary"}
    expected = meta["expected_filing_ids"]
    unknown = [f for f in expected if f not in primaries]
    if unknown:
        raise ValueError(f"Expected filings absent from dataset: {unknown}")
    by_filing: dict[str, dict] = {}
    for r in records:
        by_filing.setdefault(r["filing_id"], r)
    cases = [
        Case(
            key=case_id(primaries[f].ticker, f),
            filing_id=f,
            available_at=primaries[f].available_at,
            record=by_filing.get(f),
        )
        for f in expected
    ]
    overall = run_checks(meta, records, set(expected), profile) + case_checks(cases, profile)
    batches = []
    for label, members in month_batches(cases):
        checks = case_checks(members, profile) if members else []
        batches.append(
            {
                "batch": label,
                "cases": [c.key for c in members],
                "status": overall_status(checks) if members else "no_expected_filings",
                "checks": [asdict(c) for c in checks],
                "operational": operational_totals(members),
            }
        )
    return {
        "monitoring_schema": 1,
        "run": {
            "run_id": meta["run_id"],
            "execution_mode": meta["execution_mode"],
            "status": meta.get("status"),
            "dataset_fingerprint": meta["dataset_fingerprint"],
            "config": meta["config"],
        },
        "baseline": asdict(profile),
        "thresholds": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "near_limit_tokens": NEAR_LIMIT_TOKENS,
            "anchoring_floor": ANCHORING_FLOOR,
            "anchoring_min_resolvable": ANCHORING_MIN_CITATIONS,
            "latency_ratio": LATENCY_RATIO,
            "input_token_ratio": INPUT_TOKEN_RATIO,
            "input_token_min_cases": INPUT_TOKEN_MIN_CASES,
        },
        "status": overall_status(overall),
        "checks": [asdict(c) for c in overall],
        "batches": batches,
        "operational": operational_totals(cases),
        "cases": [case_view(c) for c in sorted(cases, key=lambda c: (c.available_at, c.key))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-free monitoring for a replay run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output file exists; choose a new path")
        report = monitor(args.run, args.baseline, Dataset(args.dataset))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        alerts = [
            f"{c['tier']}:{c['name']}" for c in report["checks"] if c["status"] == "alert"
        ]
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "alerts": alerts,
                    "batches": {b["batch"]: b["status"] for b in report["batches"]},
                    "output": str(args.output),
                }
            )
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
