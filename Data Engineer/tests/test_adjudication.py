from __future__ import annotations

from dataclasses import asdict

import pytest

from net_new.adjudication import (
    ExtraFindingReview,
    ReferenceAdjudication,
    adjudicate_case,
    adjudicate_run,
    load_jsonl,
    summarize,
)


def row(case_id="ACME:1", state="completed", classifications=("changed", "new", "repeated")):
    """An aligned case shaped like `references.align_cases` output."""
    return {
        "case_id": case_id,
        "reference": {
            "findings": [
                {"finding_id": "guidance", "acceptable_classifications": ["new", "changed"]},
                {"finding_id": "ceo", "acceptable_classifications": ["new"]},
                {"finding_id": "buyback", "acceptable_classifications": ["repeated"]},
            ]
        },
        "output_state": state,
        "output": None
        if state == "missing"
        else {
            "status": state,
            "prediction": None
            if state == "error"
            else {"findings": [{"classification": c} for c in classifications]},
        },
    }


def adj(finding_id, indices, coverage="covered", case_id="ACME:1", **overrides):
    return ReferenceAdjudication(
        case_id=case_id,
        reference_finding_id=finding_id,
        mapped_model_finding_indices=indices,
        coverage=coverage,
        comparison_quality=overrides.get(
            "comparison_quality", "not_applicable" if coverage == "missed" else "supported"
        ),
        evidence_support=overrides.get(
            "evidence_support", "not_applicable" if coverage == "missed" else "supported"
        ),
        notes="",
    )


def extra(index, assessment="supported_useful", case_id="ACME:1"):
    return ExtraFindingReview(
        case_id=case_id, model_finding_index=index, assessment=assessment, notes=""
    )


@pytest.mark.parametrize(
    ("indices", "coverage", "message"),
    [
        ([0], "missed", "missed"),
        ([], "covered", "must map"),
        ([], "partial", "must map"),
        ([1, 1], "covered", "Duplicate"),
        ([-1], "covered", "non-negative"),
    ],
)
def test_record_level_mapping_rules(indices, coverage, message):
    with pytest.raises(ValueError, match=message):
        adj("guidance", indices, coverage)


@pytest.mark.parametrize(
    ("coverage", "indices", "overrides", "message"),
    [
        ("missed", [], {"comparison_quality": "missing"}, "no comparison"),
        ("missed", [], {"comparison_quality": "supported"}, "no comparison"),
        ("missed", [], {"evidence_support": "unsupported"}, "no evidence"),
        ("covered", [0], {"evidence_support": "not_applicable"}, "evidence assessed"),
        ("partial", [0], {"evidence_support": "not_applicable"}, "evidence assessed"),
    ],
)
def test_assessments_must_fit_coverage(coverage, indices, overrides, message):
    with pytest.raises(ValueError, match=message):
        adj("guidance", indices, coverage, **overrides)


def test_valid_assessments_for_each_coverage():
    missed = adj("guidance", [], "missed")
    assert (missed.comparison_quality, missed.evidence_support) == (
        "not_applicable",
        "not_applicable",
    )
    # A matched finding may lack a meaningful prior comparison, but not evidence.
    for coverage in ("covered", "partial"):
        assert adj("guidance", [0], coverage, comparison_quality="not_applicable")
        assert adj("guidance", [0], coverage, evidence_support="unsupported")


def test_mapping_must_fit_the_aligned_case():
    with pytest.raises(ValueError, match="Unknown reference finding"):
        adjudicate_case(row(), [adj("invented", [0])], [])
    with pytest.raises(ValueError, match="out of range"):
        adjudicate_case(row(), [adj("guidance", [3])], [])
    with pytest.raises(ValueError, match="Duplicate adjudication"):
        adjudicate_case(row(), [adj("guidance", [0]), adj("guidance", [1])], [])
    with pytest.raises(ValueError, match="another case"):
        adjudicate_case(row(), [adj("guidance", [0], case_id="ACME:2")], [])


@pytest.mark.parametrize("state", ["missing", "error"])
def test_failed_outputs_are_not_adjudicated_as_semantic_misses(state):
    with pytest.raises(ValueError, match="Only completed"):
        adjudicate_case(row(state=state), [adj("guidance", [], "missed")], [])
    summary = summarize([adjudicate_case(row(state=state), [], [])])
    assert summary.reference_findings_not_adjudicable == 3
    assert summary.coverage["missed"] == 0 and summary.reference_findings_adjudicated == 0


def test_classification_agreement_is_derived_from_output():
    case = adjudicate_case(
        row(), [adj("guidance", [0]), adj("ceo", [2], "partial"), adj("buyback", [], "missed")], []
    )
    guidance, ceo, buyback = case.results
    assert guidance.mapped_classifications == ("changed",)
    assert guidance.classification_all_acceptable is True
    assert ceo.mapped_classifications == ("repeated",)
    assert ceo.classification_all_acceptable is False
    # Nothing mapped: agreement is not evaluable rather than a disagreement.
    assert buyback.classification_all_acceptable is None


def test_many_to_one_mapping_preserves_every_classification():
    agreed = adjudicate_case(row(), [adj("guidance", [0, 1])], []).results[0]
    assert agreed.mapped_classifications == ("changed", "new")
    assert agreed.classification_all_acceptable is True
    mixed = adjudicate_case(row(), [adj("guidance", [0, 2])], []).results[0]
    assert mixed.mapped_classifications == ("changed", "repeated")
    assert mixed.classification_all_acceptable is False


def test_one_model_finding_may_support_several_reference_findings():
    case = adjudicate_case(row(), [adj("guidance", [0]), adj("ceo", [0], "partial")], [])
    assert case.unmapped_model_findings == (1, 2)


def test_unmatched_extras_are_reviewed_not_assumed_false_positive():
    case = adjudicate_case(row(), [adj("guidance", [0])], [extra(1)])
    summary = summarize([case])
    assert case.unmapped_model_findings == (1, 2)
    assert summary.extras_reviewed == 1 and summary.extras_pending == 1
    assert summary.extra_assessments == {
        "supported_useful": 1,
        "supported_low_value": 0,
        "unsupported_or_overstated": 0,
    }
    assert "false_positive" not in str(asdict(summary))
    with pytest.raises(ValueError, match="mapped to a reference"):
        adjudicate_case(row(), [adj("guidance", [0])], [extra(0)])
    with pytest.raises(ValueError, match="out of range"):
        adjudicate_case(row(), [], [extra(3)])
    with pytest.raises(ValueError, match="Duplicate extra"):
        adjudicate_case(row(), [], [extra(1), extra(1, "supported_low_value")])


def test_run_aggregation_pools_counts_and_keeps_states_distinct():
    rows = [row("A:1"), row("B:1", classifications=("repeated", "new")), row("C:1", "error")]
    adjudications = [
        adj("guidance", [0], case_id="A:1", evidence_support="partial"),
        adj("ceo", [1], "partial", case_id="A:1", comparison_quality="not_applicable"),
        adj("buyback", [], "missed", case_id="A:1"),
        adj("guidance", [0, 1], case_id="B:1"),
        adj("ceo", [], "missed", case_id="B:1"),
    ]
    extras = [
        extra(2, "unsupported_or_overstated", case_id="A:1"),
    ]
    summary = summarize(adjudicate_run(rows, adjudications, extras))
    assert summary.cases == 3 and summary.reference_findings == 9
    assert summary.reference_findings_not_adjudicable == 3
    assert summary.reference_findings_pending == 1  # B:1 buyback
    assert summary.coverage == {"covered": 2, "partial": 1, "missed": 2}
    assert summary.coverage_rates["missed"] == 2 / 5
    # Comparison/evidence judged only on the 3 covered/partial findings.
    assert summary.comparison_quality == {
        "supported": 2,
        "partial": 0,
        "missing": 0,
        "not_applicable": 1,
    }
    # Missed findings' not_applicable values are excluded from quality counts.
    assert summary.evidence_support == {
        "supported": 2,
        "partial": 1,
        "unsupported": 0,
        "not_applicable": 0,
    }
    assert summary.evidence_support_rates["supported"] == 2 / 3
    # A:1 guidance ok, A:1 ceo ("new") ok, B:1 guidance ("repeated","new") not all acceptable.
    assert (summary.classification_evaluable, summary.classification_all_acceptable) == (3, 2)
    assert summary.classification_agreement_rate == 2 / 3
    assert summary.unmapped_model_findings == 1 and summary.extras_reviewed == 1
    assert not {"score", "accuracy", "precision", "f1"} & set(asdict(summary))
    with pytest.raises(ValueError, match="unknown cases"):
        adjudicate_run(rows, [adj("guidance", [0], case_id="Z:1")], [])


def test_empty_summary_uses_undefined_rates():
    summary = summarize([adjudicate_case(row(), [], [])])
    assert summary.reference_findings_pending == 3
    assert summary.classification_agreement_rate is None
    assert set(summary.coverage_rates.values()) == {None}


def test_judgments_load_from_jsonl(tmp_path):
    path = tmp_path / "adjudications.jsonl"
    path.write_text(adj("guidance", [0]).model_dump_json() + "\n")
    assert load_jsonl(path, ReferenceAdjudication) == [adj("guidance", [0])]
    path.write_text('{"case_id": "A:1", "model_finding_index": 0, "assessment": "wrong"}\n')
    with pytest.raises(ValueError):
        load_jsonl(path, ExtraFindingReview)
