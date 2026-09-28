"""Preregistered deterministic effect check for the history_chunks 12 -> 24 experiment.

Compares a baseline run and an experiment run on the same development filings:
execution states, recorded configuration and model metadata, preservation of the
baseline's top-12 prior context, exposure of the preregistered target chunks,
operational metadata, and label-free evaluation diagnostics. It reads no generated
finding text and makes no semantic judgment.

Read-only; no inference. Accepts run directories containing records.jsonl or the
tracked records.jsonl.gz. Target chunks are identified by rank under the existing
lexical ranking, recomputed from the supplied corpus and the baseline input.

Limitations: one sample per arm. A baseline replayed from cache carries its original
generation latency and cost; runs generated on different days may differ in gateway
load. Alias-level model metadata cannot prove the provider model is identical.

Usage (from `Data Engineer/`):
  uv run python evaluation/scripts/k24_effect_check.py --dataset data/pilot \
    --baseline data/pilot/baseline --experiment evaluation/run-outputs/history-k24-01
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net_new.dataset import Dataset
from net_new.evaluation import aggregate, diagnose_case
from net_new.parsing import parse, tokens
from net_new.references import case_id

NEAR_LIMIT = 5400
# Preregistered before adjudication: filing -> {rank under existing ranking: values}.
TARGETS = {
    "0001535527-24-000024": {
        15: ["3,976", "4,010", "961.2"],  # June FY and Q2 guidance ranges
        23: ["979.2", "984.7", "3,890.0", "3,902.2"],  # August Q3 and FY guidance
    },
    "0001535527-24-000017": {9: ["961.2"]},  # already exposed at k=12
}


def load(run: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((run / "run.json").read_text())
    path = run / "records.jsonl"
    if path.exists():
        text = path.read_text()
    else:
        with gzip.open(run / "records.jsonl.gz", "rt") as stream:
            text = stream.read()
    return meta, [json.loads(line) for line in text.splitlines() if line]


def usage(records: list[dict]) -> dict:
    rows = [(r["model_metadata"].get("usage") or {}, r) for r in records]
    latencies = [r["generation_latency_ms"] for _, r in rows]
    return {
        "input_tokens": sum(u["input_tokens"] for u, _ in rows),
        "output_tokens": sum(u["output_tokens"] for u, _ in rows),
        "total_tokens": sum(u["total_tokens"] for u, _ in rows),
        "cost_usd": round(sum(float(r["model_metadata"]["cost_usd"]) for _, r in rows), 6),
        "latency_ms_median": round(statistics.median(latencies)),
        "latency_ms_mean": round(statistics.mean(latencies)),
        "max_output_tokens": max(u["output_tokens"] for u, _ in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    dataset = Dataset(args.dataset)
    (bm, br), (em, er) = load(args.baseline), load(args.experiment)
    failures = []

    print("## Execution")
    for name, m, r in [("baseline", bm, br), ("experiment", em, er)]:
        states = [x["status"] for x in r]
        print(f"  {name}: run_status={m['status']} errors={m.get('errors')} "
              f"expected={len(m['expected_filing_ids'])} records={len(r)} "
              f"completed={states.count('completed')} mode={m['execution_mode']}")
    if bm["expected_filing_ids"] != em["expected_filing_ids"]:
        failures.append("expected filing IDs differ")

    print("\n## Configuration differences (run.json config)")
    for key in sorted(set(bm["config"]) | set(em["config"])):
        if bm["config"].get(key) != em["config"].get(key):
            print(f"  {key}: {bm['config'].get(key)!r} -> {em['config'].get(key)!r}")
    print(f"  prompt text identical: {bm['prompt'] == em['prompt']}")
    print(f"  dataset fingerprint identical: {bm['dataset_fingerprint'] == em['dataset_fingerprint']}")
    for field in ("requested_model", "reported_model", "resolved_model", "catalog_version"):
        values = [
            sorted({str(x["model_metadata"].get(field)) for x in r}) for r in (br, er)
        ]
        print(f"  {field}: baseline={values[0]} experiment={values[1]}")

    print("\n## Prefix preservation (experiment PRIOR[:len(baseline)] == baseline PRIOR)")
    base_input = {x["filing_id"]: x["input"] for x in br}
    for x in er:
        b = [c["id"] for c in base_input[x["filing_id"]]["prior"]]
        e = [c["id"] for c in x["input"]["prior"]]
        ok = e[: len(b)] == b
        failures += [] if ok else [f"prefix violated: {x['filing_id']}"]
        print(f"  {x['ticker']}:{x['filing_id']} baseline={len(b)} experiment={len(e)} prefix_ok={ok}")

    print("\n## Target exposure (preregistered chunks, by rank under existing ranking)")
    filings = {f.filing_id: f for f in dataset.replay_filings()}
    exp_input = {x["filing_id"]: x["input"] for x in er}
    for filing_id, targets in TARGETS.items():
        f = filings[filing_id]
        wanted = tokens(" ".join(c["text"] for c in base_input[filing_id]["current"]))
        history = [c for d in dataset.history(f) for c in parse(dataset, d)]
        ranked = [
            c
            for c in sorted(history, key=lambda c: (-len(wanted & tokens(c.text)), c.id))
            if wanted & tokens(c.text)
        ]
        supplied = {c["id"]: (i, c["text"]) for i, c in enumerate(exp_input[filing_id]["prior"], 1)}
        for rank, values in targets.items():
            chunk = ranked[rank - 1]
            position, text = supplied.get(chunk.id, (None, ""))
            present = {v: v in text for v in values}
            if position is None or not all(present.values()):
                failures.append(f"target not exposed: {filing_id} rank {rank}")
            print(f"  {f.ticker}:{filing_id} rank={rank} chunk={chunk.id} "
                  f"experiment_position={position} values_present={present}")

    print("\n## Operational (baseline cached values are original-generation measurements)")
    bu, eu = usage(br), usage(er)
    for key in bu:
        change = f" ({(eu[key] - bu[key]) / bu[key]:+.1%})" if bu[key] else ""
        print(f"  {key}: {bu[key]} -> {eu[key]}{change}")
    base_in = {x["filing_id"]: x["model_metadata"]["usage"]["input_tokens"] for x in br}
    for x in er:
        u = x["model_metadata"]["usage"]
        near = "  <-- >= near-limit threshold" if u["output_tokens"] >= NEAR_LIMIT else ""
        print(f"  {x['ticker']}:{x['filing_id']} input +{u['input_tokens'] - base_in[x['filing_id']]} "
              f"output={u['output_tokens']}{near}")

    print("\n## Label-free diagnostics (not accuracy; anchoring is textual only)")
    for name, records in [("baseline", br), ("experiment", er)]:
        run = aggregate([diagnose_case(case_id(x["ticker"], x["filing_id"]), x) for x in records])
        print(f"  {name}: completed={run.completed} errors={run.errors} missing={run.missing} "
              f"empty={run.completed_empty} findings={run.findings} citations={run.citations} "
              f"resolvable={run.resolvable_citations} anchored={run.anchored_citations} "
              f"truncated_current={run.completed_truncated_current}")

    print("\n## Verdict")
    print("  valid for adjudication" if not failures else f"  FAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
