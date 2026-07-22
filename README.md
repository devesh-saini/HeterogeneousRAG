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
    "rewriter": "ollama/qwen2.5:7b",
    "retriever": "ollama/qwen2.5:7b",
    "reranker": "ollama/qwen2.5:7b",
    "synthesizer": "ollama/qwen2.5:7b",
    "verifier": "ollama/qwen2.5:7b",
}
```

Heterogeneous config:

```python
HETEROGENEOUS_CONFIG = {
    "rewriter": "ollama/qwen2.5:7b",
    "retriever": "ollama/qwen2.5:7b",
    "reranker": "ollama/qwen2.5:7b",
    "synthesizer": "groq/llama-3.1-70b-versatile",
    "verifier": "groq/llama-3.1-70b-versatile",
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
python data/ingestion/ingest_cs.py
python data/ingestion/ingest_medical.py
python data/ingestion/ingest_law.py
python data/ingestion/ingest_finance.py
```

Each script downloads the Hugging Face dataset, chunks documents, embeds chunks locally with `sentence-transformers/all-MiniLM-L6-v2`, and persists a ChromaDB collection under `vectordb/<domain>/`.

## Run Experiments

Run a homogeneous baseline:

```bash
python main.py --domain cs --config homogeneous --n_questions 200
```

Run a heterogeneous experiment:

```bash
python main.py --domain cs --config heterogeneous --n_questions 200
```

Valid domains:

```text
cs, medical, law, finance
```

Valid configs:

```text
homogeneous, heterogeneous
```

## Logged Metrics

Results are written to `evaluation/results.db`.

The runner logs:

- `question_id`
- `domain`
- `config_name`
- `model_assignment`
- `em_score`
- `f1_score`
- `retrieval_recall`
- `retry_count`
- `hallucination`
- `total_cost_usd`
- `total_latency_ms`
- `node_metadata`
- `synthesized_answer`
- `gold_answer`

`node_metadata` contains per-node model, token estimate, cost estimate, and latency.

## Notes

- Model pricing defaults to `0.0` in `config/models.py`. Update `MODEL_PRICING_USD_PER_1M` for cost-accurate Groq accounting.
- Token counts are approximate word-based estimates, not provider billing token counts.
- The retriever model assignment is logged for comparison consistency, but retrieval itself uses ChromaDB plus local sentence-transformer embeddings.
- Dataset schemas can change. The loaders use conservative field fallbacks, but verify sampled rows before large benchmark runs.
