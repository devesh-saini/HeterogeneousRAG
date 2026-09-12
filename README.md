# Heterogeneous RAG Benchmark

Benchmarking scaffold for comparing homogeneous and heterogeneous model assignment in an agentic RAG pipeline.

The experiment asks whether assigning stronger models only to high-reasoning roles improves answer quality, hallucination control, latency, and cost relative to a single-model baseline.

## Pipeline

The pipeline has five roles with a fixed topology:

```text
Query -> Rewriter -> Retriever -> Reranker -> Synthesizer -> Verifier
                                                              |
                                  if verifier fails and retries < 2
                                                              |
                                                        Rewriter
```

The verifier is the only conditional node. If it returns `verdict: false`, the graph loops back to the rewriter and increments `retry_count`. After two failed retries, the run exits with the final failed answer state.

## Model Configurations

Configurations live in `config/models.py`.

Homogeneous baseline:

```python
HOMOGENEOUS_CONFIG = {
    "rewriter": "ollama/qwen2.5:latest",
    "retriever": "ollama/qwen2.5:latest",
    "reranker": "ollama/qwen2.5:latest",
    "synthesizer": "ollama/qwen2.5:latest",
    "verifier": "ollama/qwen2.5:latest",
}
```

Heterogeneous config:

```python
HETEROGENEOUS_CONFIG = {
    "rewriter": "ollama/qwen2.5:latest",
    "retriever": "ollama/qwen2.5:latest",
    "reranker": "groq/openai/gpt-oss-20b",
    "synthesizer": "groq/openai/gpt-oss-120b",
    "verifier": "groq/openai/gpt-oss-120b",
}
```

Local models are served through Ollama. Remote models are served through Groq and require `GROQ_API_KEY`.

## Domains

Domain configuration lives in `config/domains.py`.

| Domain | Dataset | Hugging Face ID |
|---|---|---|
| CS | QASPER | `allenai/qasper` |
| Medical | BioASQ | `rag-datasets/rag-mini-bioasq` |
| Law | CUAD | `theatticusproject/cuad-qa` |
| Finance | FinanceBench | `PatronusAI/financebench` |

QASPER and BioASQ are loaded as corpus plus QA pairs. CUAD and FinanceBench are treated as context/question/answer triples, with contexts deduplicated during ingestion.

## Project Layout

```text
config/                 Model and domain configuration
data/ingestion/         Dataset ingestion into ChromaDB
data/evaluation/        QA-pair loaders for evaluation
evaluation/             Metrics, runner, and SQLite result logging
pipeline/               LangGraph state, graph, prompts, and nodes
vectordb/               Persisted ChromaDB collections by domain
main.py                 CLI entry point
requirements.txt        Python dependencies
```

## Setup

Create or activate a virtual environment, then install dependencies:

```bash
python -m venv heterorag
source heterorag/bin/activate
pip install -r requirements.txt
```

Install and run Ollama, then pull the local model:

```bash
ollama pull qwen2.5:7b
```

For heterogeneous runs, create a `.env` file with:

```bash
GROQ_API_KEY=your_key_here
```

## Ingest Corpora

Run ingestion once per domain before evaluating that domain:

```bash
python -m data.ingestion.ingest_cs
python -m data.ingestion.ingest_medical
python -m data.ingestion.ingest_law
python -m data.ingestion.ingest_finance
```

Each script downloads the Hugging Face dataset, chunks documents, embeds chunks locally with `sentence-transformers/all-MiniLM-L6-v2`, and persists a ChromaDB collection under `vectordb/<domain>/`.

Re-run ingestion after changes to corpus metadata or train/test split coverage. Retrieval recall for medical uses stored passage IDs, so older medical vector DBs created before metadata support should be regenerated.

## Run Experiments

Run a homogeneous baseline:

```bash
python main.py --domain computerScience --config homogeneous --n_questions 100 \
  --experiment_id cs-100-groq-free-v1 --resume
```

Run a heterogeneous experiment:

```bash
python main.py --domain computerScience --config heterogeneous --n_questions 100 \
  --experiment_id cs-100-groq-free-v1 --resume
```

### Groq free-tier pacing

Remote requests are protected by a process-wide rolling token limiter. The
default profile uses Groq's 8,000 TPM free-tier ceiling with a 90% working budget
(7,200 estimated tokens in any rolling 60-second window). The limiter reserves a
conservative token estimate before each Groq call, replaces it with exact usage
reported by Groq afterward, and honors `retry-after` when HTTP 429 is returned.
Reranker, synthesizer, and verifier calls share this budget, including calls made
during graph retries. GPT-OSS calls use `reasoning_effort="low"`; role-specific
completion caps are 512 tokens for reranking and 768 tokens each for synthesis
and verification. These settings prevent hidden reasoning output from consuming
an unbounded portion of the free-tier quota. The reranker sees an 18,000-character
budgeted view of the five candidate chunks (the beginning and end are retained
when truncation is necessary); synthesis still receives the selected chunks in
full.

The defaults can be stated explicitly for a reproducible run:

```bash
python main.py --domain computerScience --config heterogeneous --n_questions 100 \
  --experiment_id cs-100-groq-free-v1 --resume \
  --groq_tpm_limit 8000 --groq_rate_limit_utilization 0.90 \
  --groq_max_429_retries 4
```

This is adaptive pacing, so it waits only when the next estimated request would
exceed the rolling budget. A fixed two-minute delay is unnecessary. Set the TPM
limit to `0` only when client-side pacing is intentionally disabled. TPM pacing
does not bypass Groq's daily token/request limits; use `--resume` on another day
if the daily allowance is exhausted.

The limiter coordinates calls within one Python process. Do not run multiple
heterogeneous benchmark processes against the same Groq organization at once.

Suggested methods wording:

> Experiments were executed in a resource-constrained local environment. Remote
> inference used the Groq free service tier, so Groq requests were serialized by
> a client-side rolling-window limiter configured for an 8,000-token-per-minute
> ceiling and a 90% working budget. Quota-induced waiting time was logged
> separately from pipeline execution latency, and runs were resumable. GPT-OSS
> inference used low reasoning effort and fixed role-specific completion limits;
> reranking used a bounded view of candidate passages while selected passages
> were supplied to answer synthesis without this truncation.

Use the same experiment ID and question count for both configurations. `--resume`
skips completed question/config pairs, which makes long experiments safe to stop
and continue on a small machine. The database prevents duplicate rows within a
named experiment. Use a new experiment ID for this rate-limited protocol; the
runner rejects resuming a named experiment under a different pacing profile.

For untracked rows, `--resume` without an experiment ID skips only questions with
the same configuration, evaluation version, and pacing profile. Legacy unpaced
rows are not silently mixed into a new rate-limited run.

Valid domains:

```text
computerScience, medical, law, finance

`cs` is also accepted as a backward-compatible alias for `computerScience`.
```

Valid configs:

```text
homogeneous, heterogeneous
```

## Logged Metrics

Results are written to `evaluation/results.db`.

Summarize results without the `sqlite3` CLI:

```bash
python -m evaluation.summarize_results --recent 10
```

Inspect one result row:

```bash
python -m evaluation.inspect_result 34
```

The runner logs:

- `question_id`
- `domain`
- `config_name`
- `model_assignment`
- `em_score`
- `f1_score`
- `f2_score`
- `rouge_l_f1`
- `semantic_similarity`
- `type_accuracy`
- `retrieval_recall`
- `retry_count`
- `hallucination`
- `total_cost_usd`
- `total_latency_ms`
- `execution_latency_ms`
- `rate_limit_wait_ms`
- `groq_input_tokens`, `groq_output_tokens`, and `groq_total_tokens`
- `groq_requests` and `groq_429_retries`
- `rate_limit_profile` and `rate_limit_policy`
- `node_metadata`
- `synthesized_answer`
- `gold_answer`

`node_metadata` contains each node's model, provider-reported token usage when
available, cost estimate, model-service latency, quota wait, retry count, and
configured output cap. `total_latency_ms` is observed wall-clock latency including
quota delays; `execution_latency_ms` removes deliberate quota waiting for a fairer
architecture comparison.

### Reference-aware evaluation

New runs use evaluation version `2.0-reference-aware`. Legacy rows remain in the
database and are reported separately.

- `f1_score` follows the official QASPER convention: SQuAD-normalized token F1
  against every human reference, retaining the maximum. Keep it as the primary
  paper metric.
- `f2_score` weights answer coverage more strongly, so it exposes correct but
  verbose answers without hiding the precision penalty.
- `rouge_l_f1` measures ordered phrase overlap. `semantic_similarity` uses the
  same small MiniLM encoder already used for retrieval and is diagnostic, not a
  replacement for official F1.
- Boolean and unanswerable references use QASPER's official Yes/No/Unanswerable
  representation; `type_accuracy` also accepts standard equivalent forms.
- `answer_precision` measures concision; `answer_recall` measures how much of the
  reference answer was covered.
- `reference_contained` identifies answers that contain a complete reference but
  add extra material, and `verbosity_ratio` reports generated length relative to
  the best-matching reference.
- The verifier reports independent groundedness, relevance, and completeness
  verdicts. Graph retries require all three to pass.
- `hallucination` now means the groundedness check found unsupported content. It
  is no longer inferred from a low lexical answer score.
- Retry metadata is stored as an ordered `events` list with an attempt number;
  `latest` provides convenient access to the final invocation of each node.

Summaries keep legacy and reference-aware runs separate:

```bash
python -m evaluation.summarize_results --full --failures
```

Aggregates automatically use only the latest row for each unique question. To
compare configurations on exactly the same questions with confidence intervals,
effect sizes, win/tie/loss counts, and paired randomization tests:

```bash
python -m evaluation.compare_configs --domain computerScience \
  --experiment-id cs-100-groq-free-v1
```

Existing stored answers can be upgraded to the current scorecard without any LLM
calls:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m evaluation.rescore_results
```

Inspect all reference answers, verifier dimensions, and post-run scores for one
result:

```bash
python -m evaluation.inspect_result RESULT_ID
```

## Notes

- Model pricing defaults to `0.0` in `config/models.py`. Update `MODEL_PRICING_USD_PER_1M` for cost-accurate Groq accounting.
- Token counts are approximate word-based estimates, not provider billing token counts.
- The retriever model assignment is logged for comparison consistency, but retrieval itself uses ChromaDB plus local sentence-transformer embeddings.
- Dataset schemas can change. The loaders use conservative field fallbacks, but verify sampled rows before large benchmark runs.
