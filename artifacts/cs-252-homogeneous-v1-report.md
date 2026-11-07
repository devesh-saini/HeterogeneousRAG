# Computer Science evaluation: 252 homogeneous questions

Experiment: `cs-252-homogeneous-v1`  
Sample: first 252 questions in the locally available QASPER validation split (not a random sample).  
Analysis: descriptive, homogeneous configuration only.

## Results

| Metric | Mean | Non-missing questions |
|---|---:|---:|
| Exact match | 0.1349 | 252 |
| Token F1 | 0.2301 | 252 |
| Token F2 | 0.2338 | 252 |
| ROUGE-L F1 | 0.2253 | 252 |
| Semantic similarity | 0.3379 | 252 |
| Retrieval recall | 0.9501 | 252 |
| Verifier correctness pass | 0.2897 | 252 |
| Verifier groundedness pass | 0.6905 | 252 |
| Verifier relevance pass | 0.9722 | 252 |
| Verifier completeness pass | 0.7619 | 252 |
| Verifier parse success | 1.0000 | 252 |
| Retries | 0.8373 | 252 |
| Observed latency (ms) | 74578.1746 | 252 |
| Execution latency (ms) | 74578.1746 | 252 |
| Rate-limit wait (ms) | 0.0000 | 252 |
| Recorded cost (USD) | 0.0000 | 252 |

## Limitations

- No heterogeneous run is available for this experiment, so no paired comparison or significance claim is made.
- Verifier pass rates are model judgments, not independent human assessments.
- The sample is the first 252 validation questions, not a randomly selected or necessarily representative sample.
- Recorded cost excludes unconfigured pricing and local compute; it does not represent total operating cost.

Individual results are stored in `evaluation/results.db` under `cs-252-homogeneous-v1`. The accompanying analysis JSON contains metric counts and failure categories; the manifest records the experiment configuration.
