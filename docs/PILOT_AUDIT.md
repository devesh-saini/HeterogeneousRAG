# Audit of the existing results database

This audit describes the local `evaluation/results.db` as inspected on 2026-10-31.
It is not a confirmatory result and should not be quoted as the paper's final
experiment.

## What is currently available

- All 275 stored rows have no experiment ID and therefore no immutable protocol
  manifest.
- Under scoring version `3.0-official-plus`, the homogeneous configuration has
  220 rows representing 100 unique questions; the heterogeneous configuration has
  19 rows representing 19 unique questions.
- Only 19 question IDs can be paired. The comparison therefore selects the latest
  homogeneous duplicate for each of those questions and is vulnerable to run-order,
  duplicate-selection, and protocol drift.
- The paired rows are labelled `legacy-unpaced`; they have no corpus manifest.
- These rows were generated before source-aware retrieval. QASPER queries could
  retrieve chunks from unrelated papers, which violates the intended
  closed-document evaluation and independently requires rerunning the experiment.

## Exploratory signal

Using the current paired analysis with seed 1729, 10,000 bootstrap repetitions,
and the exact paired randomization test where feasible:

- Official answer F1 averages 0.0093 for homogeneous and 0.0985 for heterogeneous.
  The paired absolute difference is +0.0892 (95% bootstrap interval 0.0216 to
  0.1747; paired dz 0.50; wins/ties/losses 6/13/0; raw two-sided p=0.0312).
- The relative change is very large only because the baseline is close to zero;
  the absolute F1 of both systems remains low.
- None of the secondary endpoints passes the Holm-adjusted secondary-family
  criterion at alpha 0.05.
- Only three paired observations have categorical-reference accuracy, so that
  estimate is not stable.
- Self-verifier rates cannot establish independent groundedness because the
  verifier model is part of the configuration being compared.

The apparent F1 signal is useful for planning a clean experiment, not for making a
publication claim. In particular, calling F1 "primary" after these outcomes were
already observed does not retroactively make it pre-specified.

## Required replacement experiment

Follow `docs/RESEARCH_PROTOCOL.md`: rebuild the corpus, use the corrected
source-paper-restricted retrieval, predeclare the question
count and primary endpoint, run both configurations with one new shared experiment
ID, complete every planned pair, export the manifest and machine-readable analysis,
reverse configuration order in a robustness run, and use independent blinded review
for any groundedness or hallucination claim.
