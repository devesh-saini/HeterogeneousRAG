# Research protocol and reporting plan

This document defines the confirmatory analysis before a publication run. Do not
change it after inspecting outcomes from that run. If the protocol changes, use a
new experiment ID and describe the old analysis as exploratory.

## Research question

Does heterogeneous assignment of language models to fixed RAG pipeline roles
improve answer quality relative to the homogeneous baseline when both systems are
evaluated on exactly the same questions, retrieval corpus, topology, prompts,
generation limits, and rate-limit policy?

The independent variable is the role-to-model assignment. The unit of analysis is
one question. The design is paired: every included question must have one result
from each configuration. Runs without a shared, named experiment ID are pilot data
and must not be used for the confirmatory claim.

## Endpoints

- Primary: maximum-reference, SQuAD-normalized token F1 (`f1_score`). This is the
  official-compatible answer metric and is the only confirmatory endpoint.
- Secondary: F2, ROUGE-L, semantic similarity, categorical accuracy, correctness,
  retrieval recall, verifier dimensions, retries, execution latency, observed
  latency, and cost. These explain trade-offs; they do not replace the primary.
- Diagnostic metrics must be labelled as such. In particular, semantic similarity
  and LLM-verifier verdicts are not official dataset metrics. The verifier model is
  part of the tested configuration, so its verdicts are endogenous self-verification
  telemetry and cannot support an independent hallucination or groundedness claim.
  Such a claim requires blinded human annotation or a fixed external judge applied
  identically after generation, with reliability and judge limitations reported.

For human evaluation, export randomized answer pairs while keeping the assignment
key away from reviewers:

```bash
python -m evaluation.export_blinded_review \
  --experiment-id STUDY_ID --domain computerScience \
  --review-sheet artifacts/STUDY_ID-review.csv \
  --assignment-key artifacts/STUDY_ID-review-key.csv
```

Use at least two independent reviewers. Before reviewing, define the 1--5 anchors:
1 is wholly incorrect/unsupported/incomplete, 3 is mixed or partially adequate,
and 5 is fully correct/supported/complete. Reviewers may use reference answers and
gold evidence but must not see the assignment key. Report agreement for ordinal
scores (for example, weighted kappa or an appropriate intraclass correlation),
adjudication rules, reviewer qualifications, and both raw and adjudicated outcomes.
The human analysis plan and sample must be fixed before unblinding.

The primary hypothesis is two-sided at alpha 0.05. Report the paired mean
difference, 95% paired bootstrap interval, paired effect size (dz), wins/ties/losses,
and paired-randomization p-value. Secondary p-values are Holm-adjusted as one
family. Report all endpoints, including unfavorable and null results. Do not use a
post-hoc composite score.

## Sample and exclusions

Choose the sample size before running either configuration and justify it from a
prospective power analysis or a resource-constrained precision target. Do not use
the tool's post-hoc `n@80%` estimate as a sample-size justification. The loader's
first `N` eligible questions form the planned sample and are fingerprinted in the
experiment manifest, including question text, references, answer types, and
evidence.

No completed question may be removed based on its score. Permissible exclusions
are limited to pre-specified mechanical failures (corrupt source record, unavailable
model service, or irrecoverable parser/runtime failure), and every exclusion must be
counted with a reason. A question missing from either configuration is absent from
the paired analysis and must be disclosed. Named comparison fails when any planned
pair is missing; use `--resume` to complete it. `--allow-incomplete` is strictly for
exploratory progress checks and must not produce a confirmatory claim.

## Execution controls

1. Rebuild the corpus immediately before the experiment. Ingestion replaces the
   old Chroma collection and writes `ingestion_manifest.json` with a corpus hash,
   chunking settings, embedding model, and dataset ID. For QASPER, retrieval is
   restricted to the source paper associated with the question, matching the
   closed-document task rather than searching unrelated papers.
2. Commit or archive the exact code. The experiment manifest fingerprints all
   Python and Markdown files, prompts, and requirements, including uncommitted files.
3. Use the same experiment ID, question count, corpus, semantic-scoring setting,
   and runtime policy for both configurations. The runner rejects mismatches.
4. Keep temperature at zero and keep role-specific output caps fixed. Record model
   provider/version details that are not exposed by the API in the paper's artifact
   notes.
5. To reduce temporal service-load bias, repeat the complete experiment with the
   configuration order reversed under a second pre-declared experiment ID. Treat
   the first experiment as primary and the reversed-order run as a robustness check,
   or specify a multi-run model before seeing results.
6. Do not run competing processes against the same remote quota. Report quota wait
   separately from execution latency.

## Reproducible commands

Replace the placeholders once, record them in a preregistration, and do not change
them mid-run.

```bash
python -m data.ingestion.ingest_cs

python main.py --domain computerScience --config homogeneous \
  --n_questions 100 --experiment_id STUDY_ID --resume
python main.py --domain computerScience --config heterogeneous \
  --n_questions 100 --experiment_id STUDY_ID --resume

python -m evaluation.compare_configs --domain computerScience \
  --experiment-id STUDY_ID --primary-metric f1_score \
  --bootstrap 10000 --permutations 20000 --seed 1729 \
  --output-json artifacts/STUDY_ID-analysis.json

python -m evaluation.export_experiment STUDY_ID \
  --output artifacts/STUDY_ID-manifest.json
```

Use the full planned sample rather than the example size if the prospective design
requires more questions.

## Required paper disclosures

- Dataset names, source versions/URLs, splits, planned and paired sample sizes.
- Corpus construction, chunk size/overlap, embedding model, and retrieval depth.
- Exact model identifiers, role assignment, prompts, temperature, token caps,
  retry topology, and verifier stopping rule.
- Hardware, Python/dependency environment, remote inference service, execution
  dates, and rate-limit policy.
- Primary and secondary endpoint designation, multiplicity correction, analysis
  seed, bootstrap/randomization repetitions, and missing-data handling.
- Absolute scores and paired differences, not only relative percentages. Relative
  changes are unstable when the baseline is near zero.
- All negative/null results, parse failures, API failures, exclusions, and any
  deviations from this protocol.
- Limitations: model-service nondeterminism, LLM-verifier subjectivity, dataset
  representativeness, possible benchmark contamination in pretrained models, and
  generalization beyond the tested domains and model versions.
- Latency comparisons between local and hosted models are deployment-specific and
  confound architecture with hardware/network/service conditions. Cost is reportable
  only after real provider prices and local compute accounting are supplied.

## Artifact release checklist

- [ ] Named experiment completed for both configurations with no missing pairs.
- [ ] Corpus and experiment manifests exported.
- [ ] Machine-readable comparison JSON archived.
- [ ] Prompts, configuration, code, and dependency lock file archived.
- [ ] Results database shared after checking licensing and removing secrets.
- [ ] Dataset licenses and model terms checked for redistribution restrictions.
- [ ] README contains the exact reproduction command and expected artifact hashes.
- [ ] Independent rerun or reversed-order robustness run completed.
- [ ] Any human evaluation used independent blinded raters and reports agreement.
- [ ] Claims match the primary endpoint and uncertainty; no metric was selected
      because it looked favorable.
