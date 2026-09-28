from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from net_new.dataset import Dataset
from net_new.monitoring import Case, monitor, month_batches
from net_new.pipeline import load_records, run_replay

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def live_metadata(output_tokens: int = 500, latency_ms: float = 10_000.0) -> dict:
    return {
        "requested_model": "extractive-preview",
        "reported_model": "extractive-preview",
        "resolved_model": None,
        "catalog_version": "catalog-1",
        "cost_usd": "0.01",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": output_tokens,
            "total_tokens": 1000 + output_tokens,
        },
        "raw_response": {"status": "completed", "incomplete_details": None},
    }


def rewrite(run: Path, edit_records=None, edit_meta=None) -> None:
    records = load_records(run)
    if edit_records:
        records = edit_records(records)
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if edit_meta:
        meta = json.loads((run / "run.json").read_text())
        edit_meta(meta)
        (run / "run.json").write_text(json.dumps(meta))


@pytest.fixture
def dataset():
    return Dataset(DEMO)


@pytest.fixture
def runs(dataset, tmp_path):
    """A frozen baseline and an identical candidate run with live-style metadata."""
    base = tmp_path / "baseline"
    run_replay(dataset, "extractive-preview", base)

    def add_live(records):
        for r in records:
            r["model_metadata"] = live_metadata()
            r["generation_latency_ms"] = 10_000.0
        return records

    rewrite(base, add_live)
    run = tmp_path / "run"
    shutil.copytree(base, run)
    return base, run


def check(report: dict, name: str, scope: dict | None = None) -> dict:
    return next(c for c in (scope or report)["checks"] if c["name"] == name)


def alerts(report: dict) -> set[str]:
    return {c["name"] for c in report["checks"] if c["status"] == "alert"}


def test_clean_run_has_no_alerts(runs, dataset):
    report = monitor(runs[1], runs[0], dataset)
    assert report["status"] == "ok" and alerts(report) == set()
    assert check(report, "response_incomplete")["status"] == "ok"
    assert check(report, "latency_per_output_token")["status"] == "ok"


def test_missing_error_and_empty_outputs_stay_distinct(runs, dataset):
    def edit(records):
        records[1].update(status="error", prediction=None, error_type="RuntimeError")
        records[1]["model_metadata"] = {}
        records[2]["prediction"]["findings"] = []
        return records[1:]  # drop the first record entirely

    rewrite(runs[1], edit)
    report = monitor(runs[1], runs[0], dataset)
    assert check(report, "missing_records")["detail"] == ["ASTER-DEMO:guidance"]
    assert check(report, "error_records")["detail"][0]["case"] == "ASTER-DEMO:repeat"
    empty = check(report, "completed_empty")
    assert empty["tier"] == "info" and empty["detail"] == ["ASTER-DEMO:ceo"]
    assert {"missing_records", "error_records"} <= alerts(report)
    assert "completed_empty" not in alerts(report)
    assert report["status"] == "invariant_alert"
    # The missing case still appears in its batch from dataset metadata.
    batch = report["batches"][0]
    assert "ASTER-DEMO:guidance" in batch["cases"]
    assert check(report, "missing_records", batch)["status"] == "alert"


def test_unresolved_citation_is_an_invariant_with_evidence(runs, dataset):
    def edit(records):
        evidence = records[0]["prediction"]["findings"][0]["current_evidence"][0]
        evidence["document_id"] = "some-other-document"
        return records

    rewrite(runs[1], edit)
    report = monitor(runs[1], runs[0], dataset)
    result = check(report, "unresolved_citations")
    assert result["tier"] == "invariant" and result["status"] == "alert"
    assert result["detail"][0]["document_id"] == "some-other-document"


def test_changed_quote_affects_only_the_anchoring_warning(runs, dataset):
    def edit(records):
        finding = records[0]["prediction"]["findings"][0]
        # Inflate resolvable citations past the minimum, all with unanchored quotes.
        for evidence in finding["current_evidence"]:
            evidence["quote"] = "A paraphrase that does not appear in the source."
        finding["prior_evidence"] = []
        records[0]["prediction"]["findings"] = [finding] * 25
        return records

    rewrite(runs[1], edit)
    report = monitor(runs[1], runs[0], dataset)
    anchoring = check(report, "textual_anchoring_rate")
    assert anchoring["tier"] == "warning" and anchoring["status"] == "alert"
    assert "not evidence correctness" in anchoring["detail"]
    assert not any(
        c["tier"] == "invariant" and c["status"] == "alert" for c in report["checks"]
    )


def test_anchoring_needs_enough_resolvable_citations(runs, dataset):
    report = monitor(runs[1], runs[0], dataset)
    anchoring = check(report, "textual_anchoring_rate")
    assert anchoring["observed"]["resolvable"] < 20
    assert anchoring["status"] == "insufficient_data"


def test_cached_operational_metadata_is_not_treated_as_replay_time(runs, dataset):
    rewrite(runs[1], lambda rs: [{**r, "cache_hit": True, "original_run_id": "x"} for r in rs])
    report = monitor(runs[1], runs[0], dataset)
    assert check(report, "latency_per_output_token")["status"] == "not_observed"
    assert check(report, "live_usage_metadata_missing")["status"] == "not_observed"
    ops = report["operational"]
    assert ops["live_cost_usd"] is None and ops["live_latency_ms_median"] is None
    assert ops["historical_cost_usd"] == pytest.approx(0.04)
    assert all(c["operational"]["latency_ms"] is None for c in report["cases"])


def test_live_record_without_usage_or_cost_warns(runs, dataset):
    def edit(records):
        records[0]["model_metadata"].pop("cost_usd")
        return records

    rewrite(runs[1], edit)
    result = check(monitor(runs[1], runs[0], dataset), "live_usage_metadata_missing")
    assert result["status"] == "alert" and result["detail"] == ["ASTER-DEMO:guidance"]


def test_near_output_limit_and_incomplete_response(runs, dataset):
    def edit(records):
        records[0]["model_metadata"]["usage"]["output_tokens"] = 5400
        records[1]["model_metadata"]["raw_response"]["incomplete_details"] = {
            "reason": "max_output_tokens"
        }
        return records

    rewrite(runs[1], edit)
    report = monitor(runs[1], runs[0], dataset)
    near = check(report, "near_output_token_limit")
    assert near["tier"] == "warning" and near["status"] == "alert"
    assert near["detail"][0]["case"] == "ASTER-DEMO:guidance"
    incomplete = check(report, "response_incomplete")
    assert incomplete["tier"] == "invariant" and incomplete["status"] == "alert"


def test_model_metadata_drift_warns(runs, dataset):
    def edit(records):
        records[0]["model_metadata"]["reported_model"] = "something-else"
        records[1]["model_metadata"]["catalog_version"] = "catalog-2"
        return records

    rewrite(runs[1], edit)
    result = check(monitor(runs[1], runs[0], dataset), "model_metadata_drift")
    assert result["tier"] == "warning" and result["status"] == "alert"
    assert {d["case"] for d in result["detail"]} == {"ASTER-DEMO:guidance", "ASTER-DEMO:repeat"}


def test_latency_and_input_token_growth_warn(runs, dataset):
    def edit(records):
        for r in records:
            r["generation_latency_ms"] = 25_000.0
            r["model_metadata"]["usage"]["input_tokens"] = 1300
        return records

    rewrite(runs[1], edit)
    report = monitor(runs[1], runs[0], dataset)
    assert check(report, "latency_per_output_token")["status"] == "alert"
    assert check(report, "mean_input_tokens_per_case")["status"] == "alert"


def test_input_token_warning_needs_enough_cases(runs, dataset):
    rewrite(runs[1], lambda rs: rs[:2])
    result = check(monitor(runs[1], runs[0], dataset), "mean_input_tokens_per_case")
    assert result["status"] == "insufficient_data"


def test_zero_current_context_is_an_invariant(runs, dataset):
    def edit(records):
        records[0]["input"]["current"] = []
        records[0]["prediction"]["findings"] = []
        return records

    rewrite(runs[1], edit)
    result = check(monitor(runs[1], runs[0], dataset), "zero_current_context")
    assert result["status"] == "alert" and result["detail"] == ["ASTER-DEMO:guidance"]


def test_duplicate_and_unexpected_records(runs, dataset):
    def edit(records):
        return records + [records[0], {**records[1], "filing_id": "not-expected"}]

    rewrite(runs[1], edit)
    result = check(monitor(runs[1], runs[0], dataset), "duplicate_or_unexpected_records")
    assert result["status"] == "alert"
    assert result["observed"] == {"duplicates": ["guidance"], "unexpected": ["not-expected"]}


def test_config_drift_fails_but_implementation_hash_is_report_only(runs, dataset):
    rewrite(runs[1], edit_meta=lambda m: m["config"].update(implementation_hash="changed"))
    report = monitor(runs[1], runs[0], dataset)
    assert check(report, "config_drift")["status"] == "ok"
    assert check(report, "implementation_hash")["observed"]["same"] is False
    assert report["status"] == "ok"

    rewrite(runs[1], edit_meta=lambda m: m["config"].update(history_chunks=24))
    drift = check(monitor(runs[1], runs[0], dataset), "config_drift")
    assert drift["status"] == "alert" and drift["observed"] == ["history_chunks"]


def test_monthly_batches_are_chronological_and_keep_empty_months():
    def case(key, when):
        return Case(key=key, filing_id=key, available_at=when, record=None)

    cases = [
        case("late", datetime(2024, 11, 26, 21, tzinfo=UTC)),
        case("early", datetime(2024, 7, 2, 20, tzinfo=UTC)),
        # 03:00 UTC on Sep 1 is still August in New York.
        case("edge", datetime(2024, 9, 1, 3, tzinfo=UTC)),
    ]
    batches = month_batches(cases)
    assert [b for b, _ in batches] == ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]
    assert [[c.key for c in members] for _, members in batches] == [
        ["early"],
        ["edge"],
        [],
        [],
        ["late"],
    ]


def test_report_never_names_accuracy_or_score(runs, dataset):
    report = monitor(runs[1], runs[0], dataset)

    def names(value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield k
                yield from names(v)
                if k == "name":
                    yield v
        elif isinstance(value, list):
            for v in value:
                yield from names(v)

    found = {str(n).lower() for n in names(report)}
    assert not any("accuracy" in n or "score" in n for n in found)


def test_cli_writes_deterministic_report_without_prediction_text(runs, tmp_path):
    base, run = runs
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for out in outputs:
        subprocess.run(
            [
                sys.executable, "-m", "net_new.monitoring",
                "--run", str(run), "--baseline", str(base),
                "--dataset", str(DEMO), "--output", str(out),
            ],
            check=True, cwd=ROOT, capture_output=True,
        )
    assert outputs[0].read_text() == outputs[1].read_text()
    finding = load_records(run)[0]["prediction"]["findings"][0]
    assert finding["announced"] not in outputs[0].read_text()
    again = subprocess.run(
        [
            sys.executable, "-m", "net_new.monitoring",
            "--run", str(run), "--baseline", str(base),
            "--dataset", str(DEMO), "--output", str(outputs[0]),
        ],
        cwd=ROOT, capture_output=True, check=False,
    )
    assert again.returncode == 2
