# Findings

## What I focused on

I focused on whether the workflow reliably tells an investor what actually changed, rather than just whether it returns valid structured output.

I evaluated the supplied development cases along four dimensions: whether the reference development was surfaced, whether the historical comparison was useful, whether the cited evidence supported the finding, and whether the model's classification was among the reference's acceptable classifications. I kept deterministic checks such as successful execution, citation resolution, textual anchoring, and context coverage separate from these judgments. A citation resolving correctly is useful operationally, but it does not mean the conclusion drawn from it is correct.

The development set contains 9 AMD/CrowdStrike cases and 23 reference findings (reference version 1.0.1). The references are non-exhaustive, so I did not treat model findings without a reference match as false positives.

My priority was the evaluation and monitoring loop: establish the baseline, understand its main failure mode, test one targeted change, and freeze the selected configuration before looking at the holdout. I intentionally did not add a judge model or a composite quality score because the small, non-exhaustive reference set does not support that level of precision.

## Baseline: where the workflow breaks down

All 9 development cases completed successfully. The baseline surfaced all 23 reference developments at least partially, but the more important failures appeared in the historical comparisons.

| Measure | Baseline |
| --- | ---: |
| Reference coverage | 19 covered / 4 partial / 0 missed |
| Historical comparison | 10 supported / 8 partial / 5 missing |
| Evidence support | 17 supported / 6 partial / 0 unsupported |
| Classification agreement | 22 / 23 |
| Model findings | 74 |
| Citation resolution | 140 / 140 |
| Textually anchored citations | 97 / 140 (69.3%) |
| Current filings truncated by input budget | 5 / 9 |

I do not interpret `0 missed` as perfect recall. The supplied references are intentionally non-exhaustive, so this only means every supplied reference finding had at least some corresponding model output.

The recurring weakness was comparison with what an investor already knew. For example, in CRWD 024 the output surfaced the current guidance and results but omitted the historical deltas against earlier guidance and the prior quarter. CRWD 017 showed a similar problem: relevant prior evidence was available, but the output did not turn it into the expected comparison.

This distinction mattered more to me than whether the model found the current number. For this product, saying "$X was announced" and saying "$X changed from $Y" answer different investor questions.

There were also many findings outside the supplied references. I reviewed 8 of the 40 unmatched baseline findings and found those 8 supported but relatively low-value. I left the other 32 unreviewed rather than labeling them false positives. Because of that, I do not report an overall unsupported-claim rate.

## Retrieval experiment: more context wasn't enough

The comparison failures could have come from either retrieval or reasoning, so I first checked whether the historical evidence was reaching the model.

I investigated 13 weak comparisons. Eleven had explicit prior evidence in the reference set. I then measured whether the exact reference evidence appeared in the retrieved prior chunks:

| Prior chunks (`k`) | Reference evidence exposed |
| ---: | ---: |
| 12 | 1 / 11 |
| 16 | 1 / 11 |
| 20 | 1 / 11 |
| 24 | 3 / 11 |
| 32 | 4 / 11 |

This is exact-reference-quote exposure, not a general retrieval-recall metric.

Based on that diagnostic, I tested one change: increasing `history_chunks` from 12 to 24. Before running it, I specified the expected effect around the CRWD 024 failures: exposing the missing prior evidence should improve at least one of those historical comparisons without degrading reference coverage or evidence support.

Everything else in the generation configuration was held fixed: model alias, prompt, parser, current-document budget, schema, and output limit. The same 9 development cases were used, and the original top-12 historical context remained a prefix of the k=24 context in all 9 cases.

The result was mixed:

| Measure | k=12 baseline | k=24 |
| --- | ---: | ---: |
| Reference coverage | 19 covered / 4 partial | 18 covered / 5 partial |
| Historical comparison | 10 supported / 8 partial / 5 missing | 10 supported / 9 partial / 4 missing |
| Evidence support | 17 supported / 6 partial | 19 supported / 4 partial |
| Classification agreement | 22 / 23 | 22 / 23 |
| Findings | 74 | 79 |
| Citations | 140 | 174 |
| Input tokens | baseline | +46.0% |
| Reported cost | baseline | +28.8% |

The extra context exposed the two CRWD 024 target chunks that were outside k=12. One missing comparison moved to partial, but the number of fully supported comparisons stayed at 10 and one reference finding moved from covered to partial.

More importantly, CRWD 024 showed that retrieval was not the whole problem. The model now had the relevant prior ranges in context and still described the direction of the change incorrectly. Making the evidence available did not guarantee that the model would reason over it correctly.

Given the quality result and the 46% increase in input tokens and 28.8% increase in reported cost, I did not keep k=24. I restored the original k=12 configuration.

I only ran one sample per condition, and the baseline and experiment were run on different dates using an alias whose underlying provider model cannot be independently verified. I therefore treat the differences above as evidence for the engineering decision, not as a statistically significant improvement or regression.

## Monitoring without labels

For production monitoring, I separated failures into things that can be observed without labels and things that cannot.

The monitor replays historical runs as monthly batches and checks hard invariants including missing or duplicate cases, inference errors, incomplete responses, unresolved citations, empty current context, and drift from the frozen generation configuration. It also reports warning-level signals for output-limit pressure, model/catalog changes, missing live usage metadata, low textual anchoring, and large shifts in mean input tokens per case or latency per output token.

Hard invariant failures alert directly; they do not use thresholds. The warning thresholds are:

- output near-limit: >= 5,400 output tokens, based on the pipeline's configured 6,000-token output cap
- textual anchoring: < 50% of resolvable citations, evaluated only with at least 20 resolvable citations
- latency per output token: > 2x the baseline median
- mean input tokens per case: > 1.25x the baseline, evaluated only with at least 3 cases

The near-limit threshold follows from the output cap. The other thresholds are deliberately simple heuristics based on the 9 development cases rather than statistically calibrated control limits.

On the baseline replay, the hard invariants passed. The only warning was the near-limit check in the 2024-10 batch (an AMD case with 5,799 output tokens); no other development batch crossed a warning threshold. The baseline-relative warnings compare the baseline with itself here, so their silence is expected rather than evidence of health. Running the same monitor against the k=24 experiment caught the expected configuration drift and increased input usage (1.46x the baseline mean), and also raised a near-limit warning for an AMD case with 5,750 output tokens.

What this monitor cannot tell me is at least as important as what it can tell me. Citation resolution does not establish semantic support. Normal latency and token usage do not tell me whether a development was missed. The monitor also cannot reliably detect an incorrect historical comparison, an overstated novelty judgment, or a causal explanation that is not supported by the filing.

Those failures need labeled evaluation, stronger semantic checks, or periodic human review.

## Frozen holdout

I restored k=12 and froze the implementation at commit `c24393c` before running the supplied holdout. I ran the holdout once and made no generation, retrieval, prompt, or monitoring-threshold changes after inspecting it.

The holdout contained 3 NVDA cases:

- 3/3 completed with no errors
- 22 findings
- 47/47 citations resolved
- 43/47 citations were conservatively textually anchored
- 2/3 current filings reached the current-context budget
- 48,780 total tokens
- $0.2578 reported cost
- median latency: 63.2 seconds

The label-free monitor returned `ok` with no alerts.

I then reviewed the generated output against the supplied filing text. This was a source review, not evaluation against hidden holdout labels, so I do not report a holdout accuracy number.

The current-period facts I checked were generally grounded in the supplied filing text. The weaker behavior was again around history and comparison.

In the Q2 case, the output reported $30.0B revenue but did not compare it with the earlier $28.0B ±2% guidance. It also described Blackwell-related information as new even though earlier filings in the supplied corpus already discussed its production and ramp. Neither that Blackwell material nor the earlier guidance was in the model's retrieved prior context.

In the Q3 case, the output reported $35.1B revenue without comparing it with the earlier $32.5B ±2% guidance, which was also in the supplied corpus but not in the retrieved prior context. One finding also attributed the lower Q4 gross-margin guidance to Blackwell even though the source supplied the figures but not that causal explanation.

This was useful because the operational monitor passed while these semantic issues remained. That is the main boundary I would keep in mind for production: the automated monitor can detect whether the system is behaving normally, but normal system behavior is not the same as a correct investor-facing interpretation.

## Limitations and what I would do next

This is a small evaluation: 9 development cases with 23 supplied reference findings, followed by a 3-case holdout. The development references are non-exhaustive, and the semantic adjudication used a single reviewer.

I also did not finish reviewing every unmatched model finding. Of the 40 unmatched baseline findings, 8 were reviewed and 32 remain pending; all 48 unmatched k=24 findings remain pending. I therefore cannot estimate the unsupported-claim rate for extra findings.

There was one model run per experimental condition, so I cannot separate all observed changes from run-to-run model variance. The model is also exposed through the `anthropic-fast-v1` alias, so I can verify the requested/reported alias but not the underlying provider model identity.

The next experiment I would run is on retrieval ranking rather than simply increasing k. I would test whether prioritizing recent, same-type disclosures and down-weighting boilerplate exposes the relevant historical guidance with less context. I would then run the prompt-only comparison experiment I deferred here, with repeated runs to get a better sense of variance.

After that, I would finish the unmatched-finding review and add a second reviewer for a subset of cases. That would make unsupported-claim estimates and reviewer agreement more defensible.

I did not identify any reference corrections to propose during this evaluation. I also did not independently re-audit the full reference set, so I am not treating that as evidence that the references contain no errors.

## Reproducing the results

### Environment and selected configuration

- Starter commit: `b556c69`
- k=24 experiment commit: `638ee23`
- Selected/frozen implementation: `c24393c`
- Development references: version `1.0.1`
- Selected configuration: original k=12 configuration
- Provider: `litellm`
- Model alias: `anthropic-fast-v1`
- Model catalog: `candidate-models-v1`
- Pipeline: `net-new-v2`
- Parser: `html-text-v1`
- History chunks: `12`
- Current-document budget: `24000`
- Maximum output tokens: `6000`
- Python: `3.11.8`
- uv: `0.12.10`
- No dependencies were added; `uv.lock` is unchanged.

The supplied baseline is `data/pilot/baseline`. The tracked k=24 and holdout outputs are under `evaluation/run-outputs/`, the baseline and k=24 adjudications under `evaluation/baseline/` and `evaluation/history-k24/`, and the monitoring reports under `evaluation/monitoring/`.

### Commands

Run from `Data Engineer/`. Reproduction outputs go under the ignored `runs/repro/`.

```bash
uv sync --frozen
uv run pytest -q
```

#### Evaluation

The tracked records are compressed, and `compare` and `monitoring` read `records.jsonl`. The last command reproduces both adjudication tables (there is no CLI for this).

```bash
for r in history-k24-01 holdout-final; do
  mkdir -p runs/repro/$r
  cp evaluation/run-outputs/$r/run.json runs/repro/$r/
  gzip -dc evaluation/run-outputs/$r/records.jsonl.gz > runs/repro/$r/records.jsonl
done

uv run python -m net_new.compare --dataset data/pilot --references references/development.jsonl \
  --run data/pilot/baseline --output runs/repro/baseline-aligned.jsonl
uv run python -m net_new.compare --dataset data/pilot --references references/development.jsonl \
  --run runs/repro/history-k24-01 --output runs/repro/history-k24-aligned.jsonl

uv run python -c '
import json
from pathlib import Path
from net_new.adjudication import (ExtraFindingReview, ReferenceAdjudication,
                                  adjudicate_run, load_jsonl, summarize)
for name in ("baseline", "history-k24"):
    rows = [json.loads(line) for line in open(f"runs/repro/{name}-aligned.jsonl")]
    d = Path("evaluation", name)
    s = summarize(adjudicate_run(
        rows,
        load_jsonl(d / "reference_adjudications.jsonl", ReferenceAdjudication),
        load_jsonl(d / "extra_reviews.jsonl", ExtraFindingReview)))
    print(name, s.coverage, s.comparison_quality, s.evidence_support,
          f"classification {s.classification_all_acceptable}/{s.classification_evaluable}",
          f"extras reviewed {s.extras_reviewed} pending {s.extras_pending}")
'
```

#### Retrieval and experiment

```bash
uv run python evaluation/scripts/retrieval_exposure.py --dataset data/pilot \
  --references references/development.jsonl \
  --adjudications evaluation/baseline/reference_adjudications.jsonl --run data/pilot/baseline
uv run python evaluation/scripts/k24_effect_check.py --dataset data/pilot \
  --baseline data/pilot/baseline --experiment evaluation/run-outputs/history-k24-01
```

#### Monitoring

```bash
uv run python -m net_new.monitoring --run data/pilot/baseline --baseline data/pilot/baseline \
  --dataset data/pilot --output runs/repro/monitoring-baseline.json
uv run python -m net_new.monitoring --run runs/repro/history-k24-01 --baseline data/pilot/baseline \
  --dataset data/pilot --output runs/repro/monitoring-history-k24-01.json
uv run python -m net_new.monitoring --run runs/repro/holdout-final --baseline data/pilot/baseline \
  --dataset data/holdout --output runs/repro/monitoring-holdout-final.json
```

`data/pilot/baseline` is the supplied evaluation baseline; replaying its cache is not required. The tracked monitoring reports used that cache replay as their baseline profile. Regenerating them against `data/pilot/baseline` gives the same statuses and alerts; only the baseline report differs, in its live-metadata observations.

## Time and AI assistance

**Time:** approximately 10–12 hours over the weekend.

I used AI as an engineering assistant during the take-home. Claude Code helped with repository navigation, implementation and tests, building repeatable analysis tooling, running diagnostic analyses (such as the retrieval-exposure and k=24 effect checks and a draft of the holdout source review), and first-draft wording for some adjudication notes. I used ChatGPT mainly to pressure-test the evaluation approach, keep track of the assessment requirements, and review the experiment design and evidence.

I reviewed the assignment and outputs myself, made the case-level adjudication decisions, chose and reviewed the experiment, rejected the k=24 change after looking at its quality and operational trade-offs, froze the final configuration, and reviewed the holdout against the supplied source material. I treated AI suggestions and drafts as engineering input rather than decisions: I checked what was being changed, why it was being changed, and the resulting evidence before deciding whether to use it. The decisions and conclusions in this document are mine.