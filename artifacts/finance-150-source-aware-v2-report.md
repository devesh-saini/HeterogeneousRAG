# FinanceBench evaluation: 150 paired questions

Experiment: `finance-150-source-aware-v2`  
Sample: all 150 questions in the locally available FinanceBench split  
Analysis: paired bootstrap (10,000 repetitions) and paired randomization (20,000 repetitions), seed 1729

## Results

| Metric | Homogeneous | Heterogeneous |
|---|---:|---:|
| Mean maximum-reference token F1 | 0.0677 | 0.3244 |
| Answer F2 | 0.0602 | 0.3312 |
| ROUGE-L F1 | 0.0658 | 0.3069 |
| Semantic similarity | 0.1761 | 0.6110 |
| Correctness pass rate | 5.33% | 35.33% |
| Mean retries | 1.7400 | 0.2533 |
| Mean execution latency, excluding quota wait | 49.46 s | 8.82 s |
| Mean observed latency, including quota wait | 49.46 s | 29.81 s |

The primary F1 difference (heterogeneous minus homogeneous) was **+0.2567**,
with a 95% paired bootstrap interval of **[0.2046, 0.3111]**, paired effect size
dz of **0.76**, and wins/ties/losses of **84/59/7**. The two-sided Monte Carlo
paired-randomization p-value was approximately **0.00005** (one extreme count
after the simulation's +1 correction, 20,000 permutations). This is an
exploratory comparison, because the smaller pilot results were observed before
this full-sample run.

The analysis file reports raw and Holm-adjusted p-values for all available
secondary metrics. F2, ROUGE-L, semantic similarity, correctness pass rate,
retry count, and both latency measures have Holm-adjusted p-values of about
0.00065 in this analysis. Self-verifier results describe the systems' own
decisions and do not independently establish groundedness: the verifier model
differs between configurations.

## Interpretation and limitations

- The corpus is built from FinanceBench's supplied evidence passages and
  retrieval is filtered to the question's own evidence source. Report this as
  evidence-provided answer evaluation, not end-to-end retrieval across complete
  financial filings. The 100% retrieval-recall result follows from that setup.
- Token F1 is a lexical overlap metric; it does not verify financial arithmetic or
  numerical correctness. Independent blinded review and task-specific numerical
  evaluation are needed before making a broad quality claim.
- The full sample contains 150 questions from the cached dataset split; it is not
  evidence about all finance questions or unseen datasets.
- Homogeneous inference ran locally, while heterogeneous inference used local and
  hosted models. Latency comparisons depend on hardware, network, provider service,
  quota pacing, and the dates of execution. The heterogeneous run was resumed
  after a daily quota interruption, so its conditions span multiple sessions.
- Recorded dollar cost is zero because pricing and local compute accounting are
  not configured; no cost conclusion is supported.
- All 150 paired rows are present, and both configurations had zero verifier
  parse failures. The metrics and uncertainty are reported for the complete
  planned sample.

The detailed output is in `finance-150-source-aware-v2-analysis.json`, the run
configuration is in `finance-150-source-aware-v2-manifest.json`, and individual
rows are in `evaluation/results.db` under the experiment ID above. The earlier
10-question pilot remains in its separately named files.
