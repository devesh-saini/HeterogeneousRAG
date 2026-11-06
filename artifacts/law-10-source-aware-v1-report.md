# Exploratory law evaluation

Experiment: `law-10-source-aware-v1`  
Domain: CUAD law QA  
Sample: 10 paired questions  
Retrieval: restricted to each question's source contract  
Analysis seed: 1729

## Main result

The heterogeneous configuration did not improve the pre-specified primary metric
in this small sample. Mean official answer F1 was 0.6945 for homogeneous and 0.5013
for heterogeneous. The paired heterogeneous-minus-homogeneous difference was
-0.1933 (95% paired bootstrap interval -0.4933 to 0.0223; paired randomization
p=0.375; paired dz=-0.45; wins/ties/losses=1/6/3).

This is not evidence of a difference at alpha 0.05. The point estimate favors the
homogeneous configuration, but the interval is wide and includes zero.

## Diagnostics

- Extractive-question F1: 0.2364 homogeneous vs 0.2532 heterogeneous.
- Unanswerable-question F1: 1.0000 homogeneous vs 0.6667 heterogeneous.
- Retrieval recall: 0.4000 for both configurations.
- Execution latency excluding quota waits: 104.2 s homogeneous vs 167.0 s
  heterogeneous; the paired interval includes zero.
- Observed latency including quota waits: 104.2 s homogeneous vs 295.5 s
  heterogeneous. Heterogeneous was slower on all 10 questions; its Holm-adjusted
  p-value was 0.0254, although the comparison is deployment- and quota-specific.
- Heterogeneous used fewer retries on average (0.4 vs 1.0), but this secondary
  difference did not survive multiplicity correction.
- Self-verifier pass rates are operational telemetry, not independent quality
  judgments, because the verifier model differs between configurations.
- Recorded cost was zero because provider pricing/local compute accounting is not
  configured; no cost claim is supported.

## Interpretation

This 10-question run is an exploratory engineering check, not an adequately powered
law-domain publication result. Six questions were unanswerable and only four were
extractive, so the overall F1 difference is sensitive to a few cases. A larger,
predeclared sample and blinded independent review are required before making a law
domain claim.

Machine-readable estimates are in `law-10-source-aware-v1-analysis.json`; exact
code, corpus, sample, model, environment, and pacing fingerprints are in
`law-10-source-aware-v1-manifest.json`. Per-question records are stored in
`evaluation/results.db`.
