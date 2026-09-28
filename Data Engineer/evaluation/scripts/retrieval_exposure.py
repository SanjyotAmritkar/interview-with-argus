"""Exact reference-quote exposure under the starter's lexical prior retrieval.

For each development reference finding whose human comparison_quality rating is
`partial` or `missing`, locate every reference prior-evidence quote in the parsed
chunks of its (point-in-time eligible) document, rank those chunks with the
existing retrieval ordering, and report whether they were supplied at k=12 and the
minimum k that would supply them.

Read-only; no inference. Development data only (references + human adjudications).

Limitations: this measures exposure of the EXACT reference quote after conservative
normalization. It is not general retrieval recall, and lexical absence does not
prove that semantically equivalent evidence was unavailable. The denominator is only
the weak-comparison findings that have reference prior evidence.

Usage (from `Data Engineer/`):
  uv run python evaluation/scripts/retrieval_exposure.py \
    --dataset data/pilot --references references/development.jsonl \
    --adjudications evaluation/baseline/reference_adjudications.jsonl \
    --run data/pilot/baseline
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net_new.dataset import Dataset
from net_new.evaluation import normalize_text
from net_new.parsing import parse, tokens
from net_new.pipeline import build_input

WEAK = {"partial", "missing"}


def read_records(run: Path) -> list[dict]:
    path = run / "records.jsonl"
    if path.exists():
        text = path.read_text()
    else:
        with gzip.open(run / "records.jsonl.gz", "rt") as stream:
            text = stream.read()
    return [json.loads(line) for line in text.splitlines() if line]


def ranked_history(context: dict, history) -> list:
    """The starter's retrieve() ordering over all eligible chunks, without the k cut."""
    wanted = tokens(" ".join(c["text"] for c in context["current"]))
    ranked = sorted(history, key=lambda c: (-len(wanted & tokens(c.text)), c.id))
    return [c for c in ranked if wanted & tokens(c.text)]


def quote_chunks(chunks, quote: str) -> list[str] | None:
    """IDs of the chunk(s) a normalized quote overlaps, or None if not found contiguously."""
    norm = [normalize_text(c.text) for c in chunks]
    starts, position = [], 0
    for n in norm:
        starts.append(position)
        position += len(n) + 1
    needle = normalize_text(quote)
    at = " ".join(norm).find(needle)
    if at < 0:
        return None
    end = at + len(needle)
    return [c.id for c, s, n in zip(chunks, starts, norm) if s < end and s + len(n) > at]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="Run whose config/input to use")
    parser.add_argument("--ks", type=int, nargs="+", default=[12, 16, 20, 24, 32])
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    config = json.loads((args.run / "run.json").read_text())["config"]
    supplied = {r["filing_id"]: r["input"] for r in read_records(args.run)}
    refs = [json.loads(line) for line in args.references.read_text().splitlines() if line]
    if any(r["split"] != "development" for r in refs):
        raise SystemExit("Development references only")
    ratings = {
        a["reference_finding_id"]: a["comparison_quality"]
        for a in map(json.loads, args.adjudications.read_text().splitlines())
    }
    filings = {f.filing_id: f for f in dataset.replay_filings()}
    documents = {d.id: d for d in dataset.documents}

    rows = []
    for ref in refs:
        filing = filings[ref["filing_id"]]
        context = build_input(dataset, filing, config)
        # Rebuilt input must match what the model actually received.
        if [c["id"] for c in context["prior"]] != [c["id"] for c in supplied[filing.filing_id]["prior"]]:
            raise SystemExit(f"Rebuilt prior context differs from run input: {ref['case_id']}")
        history_docs = dataset.history(filing)
        eligible = {d.id for d in history_docs}
        rank = {
            c.id: i
            for i, c in enumerate(
                ranked_history(context, [c for d in history_docs for c in parse(dataset, d)]), 1
            )
        }
        prior_docs = {c["document_id"] for c in context["prior"]}
        for finding in ref["findings"]:
            rating = ratings[finding["finding_id"]]
            if rating not in WEAK:
                continue
            quotes = []
            for q in finding["prior_evidence"]:
                chunks = parse(dataset, documents[q["document_id"]])
                ids = quote_chunks(chunks, q["quote"])
                quotes.append(
                    {
                        "document": q["document_id"].split(":", 1)[1],
                        "eligible": q["document_id"] in eligible,
                        "document_in_prior_k12": q["document_id"] in prior_docs,
                        "best_document_chunk_rank": min(
                            (rank[c.id] for c in chunks if c.id in rank), default=None
                        ),
                        "quote_chunk_ranks": [rank.get(i) for i in ids] if ids else None,
                    }
                )
            needed = [r for q in quotes for r in (q["quote_chunk_ranks"] or [None])]
            min_k = None if not quotes or None in needed else max(needed)
            rows.append(
                {
                    "finding_id": finding["finding_id"],
                    "comparison_quality": rating,
                    "quotes": quotes,
                    "min_k_exact_quote": min_k,
                }
            )

    print("finding_id | rating | prior document | eligible | doc in PRIOR@k12 | "
          "best doc-chunk rank | exact-quote chunk rank(s) | min k")
    for row in rows:
        if not row["quotes"]:
            print(f"{row['finding_id']} | {row['comparison_quality']} | "
                  "(no reference prior evidence) | - | - | - | - | n/a")
        for q in row["quotes"]:
            print(f"{row['finding_id']} | {row['comparison_quality']} | {q['document']} | "
                  f"{q['eligible']} | {q['document_in_prior_k12']} | "
                  f"{q['best_document_chunk_rank']} | {q['quote_chunk_ranks']} | "
                  f"{row['min_k_exact_quote']}")

    with_evidence = [r for r in rows if r["quotes"]]
    print(f"\nWeak comparisons: {len(rows)}; with reference prior evidence: {len(with_evidence)}")
    print("Exact reference-quote exposure (all quote chunks within top k) under existing ranking:")
    for k in args.ks:
        exposed = [
            r["finding_id"]
            for r in with_evidence
            if r["min_k_exact_quote"] is not None and r["min_k_exact_quote"] <= k
        ]
        print(f"  k={k}: {len(exposed)}/{len(with_evidence)} {exposed}")


if __name__ == "__main__":
    main()
