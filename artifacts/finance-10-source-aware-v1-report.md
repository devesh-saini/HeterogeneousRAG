# Exploratory finance evaluation

Experiment: `finance-10-source-aware-v1`  
Domain: FinanceBench  
Sample: 10 paired questions  
Retrieval: restricted to each question's source evidence  
Analysis seed: 1729

## Main result

The heterogeneous configuration improved the pre-specified primary metric in this
small sample. Mean official answer F1 was 0.0377 for homogeneous and 0.2911 for
heterogeneous. The paired heterogeneous-minus-homogeneous difference was +0.2534
(95% paired bootstrap interval 0.0945 to 0.4535; paired randomization p=0.0156;
paired dz=0.81; wins/ties/losses=7/3/0).

The interval excludes zero and the raw primary-endpoint p-value is below 0.05.
However, absolute heterogeneous F1 remains modest and the ten-question sample is
too small and narrow for a general finance-domain publication claim.

## Diagnostics

- Retrieval recall was 1.0000 for both configurations, suggesting the observed
  quality difference arose after retrieval rather than from evidence availability.
- Semantic similarity was 0.1304 homogeneous vs 0.6539 heterogeneous; its
  Holm-adjusted p-value was 0.0234.
- Correctness pass rate was 0.0000 homogeneous vs 0.4000 heterogeneous, but this
  secondary comparison did not survive multiplicity correction.
- Average retries were 1.8 homogeneous vs 0.0 heterogeneous; Holm-adjusted
  p=0.0391.
- Execution latency excluding quota waits was 53.3 s homogeneous vs 7.8 s
  heterogeneous; Holm-adjusted p=0.0234.
- Observed latency including quota waits was 53.3 s homogeneous vs 21.7 s
  heterogeneous; Holm-adjusted p=0.0469. This is deployment-specific and compares
  local CPU inference with a hosted service under the recorded quota policy.
- Self-verifier rates are endogenous operational telemetry, not independent quality
  judgments, because the verifier model differs between configurations.
- Recorded cost was zero because provider pricing and local compute accounting are
  not configured; no cost claim is supported.

## Interpretation

The run provides a promising exploratory finance signal for heterogeneous model
assignment, with higher F1 and lower measured latency on these ten questions. It
does not establish broad FinanceBench superiority. A larger predeclared sample,
reversed-order replication, and blinded independent review are required before a
publication-level claim.

Machine-readable estimates are in `finance-10-source-aware-v1-analysis.json`; exact
code, corpus, sample, model, environment, and pacing fingerprints are in
`finance-10-source-aware-v1-manifest.json`. Per-question records are stored in
`evaluation/results.db`.
