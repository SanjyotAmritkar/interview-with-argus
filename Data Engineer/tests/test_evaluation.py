from __future__ import annotations

from dataclasses import asdict

from net_new.evaluation import (
    aggregate,
    diagnose_case,
    normalize_text,
    quote_anchored,
    resolve_citation,
)

RELEASE = "ACME:0001:acme-ex991.htm"
ANNUAL = "ACME:0000:acme-10k.htm"


def chunk(document_id: str, suffix: str, text: str) -> dict:
    return {"id": f"{document_id}:{suffix}", "document_id": document_id, "text": text}


CURRENT = chunk(RELEASE, "aaaaaaaaaaaa", "Revenue guidance is now “about $120 million” for 2024.")
PRIOR = chunk(ANNUAL, "bbbbbbbbbbbb", "We expect revenue of about $100 million. The CEO’s term ends.")


def context(current_total: int = 1, current=(CURRENT,), prior=(PRIOR,)) -> dict:
    return {
        "current_document_ids": [RELEASE],
        "eligible_history_document_ids": [ANNUAL, "ACME:0000:other.htm"],
        "current_chunks_total": current_total,
        "history_chunks_total": 5,
        "current": list(current),
        "prior": list(prior),
    }


def evidence(document_id: str, chunk_id: str, quote: str) -> dict:
    return {"document_id": document_id, "chunk_id": chunk_id, "quote": quote}


def finding(current: list[dict], prior: list[dict]) -> dict:
    return {
        "title": "Guidance raised",
        "announced": "Guidance raised",
        "change": "Raised from $100 million",
        "classification": "changed",
        "current_evidence": current,
        "prior_evidence": prior,
    }


def record(findings: list[dict] | None, status: str = "completed", ctx: dict | None = None):
    return {
        "filing_id": "0001",
        "ticker": "ACME",
        "status": status,
        "input": ctx or context(),
        "prediction": None if findings is None else {"findings": findings, "limitations": ""},
    }


def test_full_and_suffix_only_chunk_ids_resolve_to_supplied_chunk():
    ctx = context()
    full = evidence(RELEASE, CURRENT["id"], "x")
    suffix = evidence(ANNUAL, "bbbbbbbbbbbb", "x")
    assert resolve_citation(full, ctx) is CURRENT
    assert resolve_citation(suffix, ctx) is PRIOR


def test_wrong_document_or_unsupplied_chunk_cannot_resolve():
    ctx = context()
    # A real chunk ID attributed to a different document is not traceable.
    assert resolve_citation(evidence(ANNUAL, CURRENT["id"], "x"), ctx) is None
    assert resolve_citation(evidence(ANNUAL, "aaaaaaaaaaaa", "x"), ctx) is None
    # Chunks that exist in the corpus but were not supplied to the model do not count.
    assert resolve_citation(evidence(ANNUAL, "cccccccccccc", "x"), ctx) is None
    assert resolve_citation(evidence(ANNUAL, PRIOR["id"], "x"), context(prior=())) is None


def test_normalization_is_conservative():
    assert normalize_text("  a&amp;b \n\t c d ") == "a&b c d"
    assert normalize_text('say \\"hi\\" it\\\'s') == "say \"hi\" it's"
    assert normalize_text("“quoted” ‘x’") == "\"quoted\" 'x'"
    # Words, punctuation and case are preserved.
    assert normalize_text("Revenue, up 5%.") == "Revenue, up 5%."


def test_quote_anchoring_accepts_typography_and_escaping_differences():
    assert quote_anchored('guidance is now "about $120  million"', CURRENT["text"])
    assert quote_anchored("The CEO\\'s term ends.", PRIOR["text"])
    assert quote_anchored("The CEO&#8217;s term ends.", PRIOR["text"])


def test_quote_anchoring_rejects_ellipsis_paraphrase_and_empty_quotes():
    text = PRIOR["text"]
    assert not quote_anchored("We expect revenue ... The CEO’s term ends.", text)
    assert not quote_anchored("We expect revenue… $100 million", text)
    assert not quote_anchored("We expect revenue of roughly $100 million.", text)
    assert not quote_anchored("we expect revenue of about $100 million.", text)
    assert not quote_anchored("   ", text)


def test_missing_output_has_no_invented_counts():
    diag = diagnose_case("ACME:0001", None)
    assert diag.output_state == "missing"
    assert diag.finding_count is None and diag.completed_empty is None
    assert diag.total_citations is None and diag.current_chunks_supplied is None


def test_error_output_keeps_input_exposure_but_not_findings():
    diag = diagnose_case("ACME:0001", record(None, status="error", ctx=context(current_total=4)))
    assert diag.output_state == "error"
    assert diag.finding_count is None and diag.completed_empty is None
    assert diag.resolvable_citation_rate is None
    assert diag.current_chunks_supplied == 1 and diag.current_context_truncated is True


def test_completed_empty_is_distinct_and_rates_are_undefined_not_zero():
    diag = diagnose_case("ACME:0001", record([]))
    assert diag.output_state == "completed"
    assert diag.finding_count == 0 and diag.completed_empty is True
    assert diag.total_citations == 0 and diag.resolvable_citations == 0
    assert diag.resolvable_citation_rate is None
    assert diag.anchored_rate_of_resolvable is None


def test_completed_with_findings_counts_citations_by_side():
    findings = [
        finding(
            [
                evidence(RELEASE, CURRENT["id"], "about $120 million"),  # resolvable, anchored
                evidence(RELEASE, "aaaaaaaaaaaa", "about $120 ... 2024"),  # resolvable only
            ],
            [
                evidence(ANNUAL, "bbbbbbbbbbbb", "revenue of about $100 million"),  # both
                evidence(RELEASE, "bbbbbbbbbbbb", "revenue of about $100 million"),  # neither
            ],
        ),
        finding([evidence(RELEASE, "zzzzzzzzzzzz", "about $120 million")], []),  # neither
    ]
    diag = diagnose_case("ACME:0001", record(findings))
    assert diag.finding_count == 2 and diag.completed_empty is False
    assert (diag.current_citations, diag.prior_citations, diag.total_citations) == (3, 2, 5)
    assert diag.resolvable_citations == 3 and diag.resolvable_citation_rate == 3 / 5
    assert diag.anchored_citations == 2 and diag.anchored_rate_of_resolvable == 2 / 3
    # Anchoring is only ever credited to a resolvable citation.
    assert all(c.resolvable for c in diag.citations if c.anchored)
    assert [c.finding_index for c in diag.citations] == [0, 0, 0, 0, 1]


def test_current_context_exposure_full_and_truncated():
    full = diagnose_case("ACME:0001", record([], ctx=context(current_total=1)))
    assert full.current_context_coverage == 1.0 and full.current_context_truncated is False
    assert full.prior_chunks_supplied == 1 and full.eligible_history_documents == 2
    cut = diagnose_case("ACME:0001", record([], ctx=context(current_total=4)))
    assert cut.current_context_coverage == 0.25 and cut.current_context_truncated is True
    empty = diagnose_case("ACME:0001", record([], ctx=context(current_total=0, current=())))
    assert empty.current_context_coverage is None and empty.current_context_truncated is False


def test_run_aggregation_pools_counts_instead_of_averaging_case_rates():
    one_of_one = [finding([evidence(RELEASE, CURRENT["id"], "about $120 million")], [])]
    one_of_four = [
        finding(
            [
                evidence(RELEASE, CURRENT["id"], "about $120 million"),
                evidence(RELEASE, "nope", "x"),
                evidence(RELEASE, "nope", "x"),
                evidence(RELEASE, "nope", "x"),
            ],
            [],
        )
    ]
    cases = [
        diagnose_case("A", record(one_of_one)),
        diagnose_case("B", record(one_of_four, ctx=context(current_total=3))),
        diagnose_case("C", record([])),
        diagnose_case("D", record(None, status="error", ctx=context(current_total=9))),
        diagnose_case("E", None),
    ]
    run = aggregate(cases)
    assert (run.cases, run.completed, run.errors, run.missing) == (5, 3, 1, 1)
    assert run.completed_empty == 1 and run.findings == 2
    assert run.citations == 5 and run.resolvable_citations == 2
    # Pooled 2/5, not the mean of per-case rates (1.0 + 0.25) / 2.
    assert run.resolvable_citation_rate == 2 / 5
    assert run.anchored_citations == 2 and run.anchored_rate_of_resolvable == 1.0
    # Truncation is measured over completed cases; the truncated error case is excluded.
    assert run.completed_truncated_current == 1 and run.completed_truncation_rate == 1 / 3
    assert "score" not in asdict(run) and "accuracy" not in asdict(run)


def test_aggregation_of_no_completed_cases_is_undefined():
    run = aggregate([diagnose_case("A", None)])
    assert run.completed == 0 and run.missing == 1
    assert run.resolvable_citation_rate is None
    assert run.anchored_rate_of_resolvable is None
    assert run.completed_truncation_rate is None
