# Search-Relevance Benchmark

A reproducible harness for measuring how much different retrieval strategies improve search relevance over a **BM25 baseline**, on a fixed dataset and qrel set. It indexes documents into ElasticSearch, runs each variant (semantic, hybrid + RRF, and reranked combinations) through **one shared pipeline**, scores them with graded relevance metrics (`avg_relevance`, `ndcg@10`, `recall@10`, `precision@10`), and emits per-run CSV artifacts plus a paired statistical comparison against the baseline. The first concrete instantiation uses **WANDS** (Wayfair ANnotation Dataset for Search) on ElasticSearch **>= 8.15**, used as a **plain vector + BM25 index**: the harness embeds the corpus and queries via provider connectors (Cohere / Voyage / OpenAI) and retrieves with BM25 `match` + `knn` over `dense_vector` fields.

This README is the operational guide for **running the evals**. For the full experimental design — abstractions, metric definitions, statistics, and the DRY single-execution-path argument — read [`docs/experiment.md`](docs/experiment.md). Where this guide and the design doc differ on a name, the design doc wins.

---

## TL;DR (Quickstart)

```bash
# 0. Get the harness, then enter it
git clone <this-repo> && cd <repo>

# Prereqs: Docker + Compose, Python 3.11+, hatch, ~8 GB RAM free for ES

# 1. Install hatch (the eval env is auto-created on first `hatch run eval:*`)
pipx install hatch          # or: pip install --user hatch

# 2. Bring up ElasticSearch (single-node, eval-friendly)
docker compose up -d
hatch run eval:wait-for-es   # blocks until cluster health is yellow/green

# 3. Configure inference provider keys (only those you actually use)
export ES_URL=http://localhost:9200
export COHERE_KEY=...           # Cohere embedder / reranker
export VOYAGE_KEY=...           # Voyage embedder / reranker
export OPENAI_KEY=sk-...        # OpenAI embedder (no reranker)

# 4. Provision the dataset into dataset/wands/  (see "Dataset" below)
hatch run eval:fetch-data       # downloads query.csv / product.csv / label.csv

# 5. Embed the corpus + build the index
hatch run eval:index

# 6. Run the full experiment matrix (baseline + 5 variants)
hatch run eval:run

# 7. Inspect results
ls results/                     # result_* / metrics_* / comparison_bm25_* / run_config_*

# 8. Tear down
docker compose down -v
```

`hatch run eval:run` reads `config.yaml` — a set of **explicit, named pipelines** (a `baseline` plus named `variants`; no matrix expansion, no sweep) — executes every pipeline through the single `ExperimentRunner` path (baseline first), and writes the three CSV artifact types described below. Commands shown above are the intended CLI surface that the implementation satisfies; they mirror the design in [`docs/experiment.md`](docs/experiment.md).

---

## What this repo does

The harness supports **six conceptual retrieval shapes** — the BM25 baseline plus five strategies scored against it. You realize the ones you want as **explicit named pipelines** in `config.yaml`:

| # | Strategy | Description |
|---|----------|-------------|
| 0 | `bm25` (baseline) | Lexical BM25 over the canonical `search_text` field. |
| 1 | `semantic` | Dense/sparse vector retrieval over a chosen embedder. |
| 2 | `hybrid` | RRF fusion of BM25 + semantic at a chosen `rank_constant`. |
| 3 | `bm25_rerank` | BM25 candidates → reranker. |
| 4 | `semantic_rerank` | Semantic candidates → reranker. |
| 5 | `hybrid_rerank` | RRF(BM25, semantic) → reranker. |

Every pipeline is an explicit named entry in the config — there is no per-variant code and **no matrix expansion or sweep**. Each named variant produces one `result_*` + `metrics_*` + `comparison_bm25_*` triple. The baseline is not compared to itself, so only the named variants yield `comparison_bm25_*` CSVs.

---

## Prerequisites

- **Docker + Docker Compose** (Compose v2, the `docker compose` subcommand).
- **Python 3.11+**.
- **[Hatch](https://hatch.pypa.io)** for environment + script management (`pipx install hatch`).
- **~8 GB free RAM** recommended for a comfortable single-node ES. The compose file pins the JVM heap (default `-Xms2g -Xmx2g`); raise it for large corpora. ES stores + searches vectors only — embeddings are computed by the provider, so no ES ML-node memory is required.
- **ElasticSearch >= 8.15** — ES is a plain vector + BM25 index (`dense_vector` + `knn`, available in ES well before 8.15). The `>= 8.15` pin matches the shipped `elasticsearch` client; it is a convenience version pin, not a hard feature floor. The compose file ships a compatible image.
- API keys for the **inference providers** you enable — Cohere, Voyage, and/or OpenAI (see the env-var table below). Every embedder and reranker is an external provider connector; there are no local models.

---

## Repo layout

```
.
├── benchmark/                 # the harness package
│   ├── models.py              # Query, Document, Qrel, ScoredDoc, RankedResult, FieldSchema, IndexMapping
│   ├── protocols.py           # Searcher/Fuser/Reranker + Dataset ABCs; Embedder, RerankClient, Indexer, SearchBackend, SearcherFactory
│   ├── providers.py           # provider connectors: OpenAI/Cohere/Voyage embedders + Cohere/Voyage rerankers (stdlib-HTTP)
│   ├── pipeline.py            # RRFFuser, HybridSearch, SearchPipeline (the composers)
│   ├── fusion.py              # fuse_rrf_local (client-side RRF, windowed)
│   ├── rerank.py              # rerank_local (client-side score+reorder helper, windowed)
│   ├── metrics.py             # Evaluator, Metrics, QrelIndex
│   ├── stats.py               # Comparator (bootstrap CI, Wilcoxon/permutation, FDR/BH-BY)
│   ├── runner.py              # ExperimentRunner — the single execution path
│   ├── io_csv.py              # write_result_csv / write_metrics_csv / write_comparison_csv / write_run_config
│   ├── config.py              # config value types (PipelineCfg/Services/ResolvedConfig, NO expansion) + build_pipeline; YAML/JSON load + resolve; lazy adapter factories
│   ├── datasets/
│   │   └── wands.py           # WandsDataset (label→gain, search_text concat)
│   └── backends/
│       └── elasticsearch.py   # ElasticsearchBackend, ESIndexer, LexicalSearcher/VectorSearch/ESReranker
├── dataset/
│   └── wands/                 # query.csv, product.csv, label.csv  (NOT in the repo; see below)
├── results/                   # run artifacts land here (gitignored)
├── config.yaml                # explicit named pipelines (baseline + variants)
├── docker-compose.yml         # single-node ElasticSearch
├── pyproject.toml             # hatch envs + scripts
├── docs/
│   └── experiment.md          # authoritative design doc
└── LICENSE                    # MIT
```

Module names and responsibilities follow §11 of the design doc. `pipeline`, `metrics`, `stats`, `runner`, and `io_csv` depend only on `models`/`protocols`; `config.py` (the config layer — value types + `build_pipeline` + loader) imports `pipeline` for the composers, a one-way wiring edge; adapters are selected by `config.py`'s lazy factories.
---

## Install

```bash
pipx install hatch        # recommended; or: pip install --user hatch
```

Hatch reads the environments and scripts from `pyproject.toml`. The eval scripts referenced throughout this guide live in the `eval` environment, which Hatch **auto-provisions on first `hatch run eval:*`** — there is no separate install step. If you prefer to materialize it ahead of time:

```bash
hatch env create eval     # optional; otherwise created on first `hatch run eval:*`
hatch env show            # list environments and their scripts
```

---

## Bring up ElasticSearch

The bundled `docker-compose.yml` runs a single-node ES (>= 8.15) configured for local evaluation (security relaxed, exposed on `localhost:9200`).

```bash
docker compose up -d
docker compose ps
```

### Wait for cluster health

```bash
hatch run eval:wait-for-es
# equivalent to polling:  curl -s "$ES_URL/_cluster/health?wait_for_status=yellow&timeout=60s"
```

A single-node cluster reports **yellow** (replicas unassigned), which is expected and fine for evals.

### Common local-ES gotchas

- **`vm.max_map_count` too low** — ES refuses to start with a `max virtual memory areas vm.max_map_count [65530] is too low` bootstrap error. On Linux:
  ```bash
  sudo sysctl -w vm.max_map_count=262144         # persist in /etc/sysctl.conf
  ```
  On Docker Desktop (macOS/Windows) this is handled inside the VM; if you hit it, raise it via your Docker VM settings.
- **Heap / OOM** — if ES is killed at startup, give Docker more memory or lower the heap. The heap is pinned in compose (`ES_JAVA_OPTS=-Xms2g -Xmx2g`); set both `-Xms` and `-Xmx` to the same value.
- **Port already in use** — something else is on `9200`; stop it or change the published port in compose and `ES_URL`.

### Configure provider connectors

Embedding and reranker models are **provider connectors** (`benchmark/providers.py`): the harness calls Cohere / Voyage / OpenAI directly over HTTP — ES runs no inference. You declare each as a `services` entry in `config.yaml`; secrets are injected from environment variables at load time. A connector is just a `provider` + a `settings` block (`api_key`, `model_id`, optional `rate_limit.requests_per_minute`, `batch_size`, `dims`). **OpenAI has no reranker** — a `reranker` with `provider: openai` is rejected at load.

Set only the variables for providers you actually use:

```bash
export ES_URL=http://localhost:9200

# Inference providers (skip any you don't use):
export COHERE_KEY=...             # Cohere embedder (embed-english-v3.0) + reranker (rerank-v3.5)
export VOYAGE_KEY=...             # Voyage embedder (voyage-3.5) + reranker (rerank-2.5)
export OPENAI_KEY=sk-...          # OpenAI embedder (text-embedding-3-small); no reranker
```

Concrete connector shapes, as they appear in `config.yaml` under `services`:

```yaml
services:
  # Embedder — auth via env var; the harness embeds the corpus into a dense_vector field:
  - embedder: { name: cohere, provider: cohere,
      settings: { api_key: ${COHERE_KEY}, model_id: embed-english-v3.0 } }
  # Reranker — top_n (the rank-window cap) is a plain settings key:
  - reranker: { name: co-rr, provider: cohere,
      settings: { api_key: ${COHERE_KEY}, model_id: rerank-v3.5, top_n: 100 } }
  # Searchers reference the services above by name:
  - searcher: { name: bm25,        provider: elasticsearch, kind: lexical }
  - searcher: { name: semantic_co, provider: elasticsearch, kind: vector, embedder: cohere }
```

The harness embeds the whole corpus with each configured embedder **before** searching — one `dense_vector` field per embedder (see §5.2 of the design doc). Rerankers need no setup: `ESReranker` calls the provider connector per query, and the harness asserts `rerank_window_size <= settings.top_n` before running each rerank pipeline.

---

## Dataset

The eval expects the **WANDS** files in `dataset/wands/` (tab-separated):

```
dataset/wands/
├── query.csv      # query_id, query, query_class
├── product.csv    # product_id, product_name, product_description, product_features, ...
└── label.csv      # id, query_id, product_id, label   (label is a STRING: Exact / Partial / Irrelevant)
```

These files are **not** committed to the repo. Obtain them from the upstream project — [github.com/wayfair/WANDS](https://github.com/wayfair/WANDS) — and drop them into `dataset/wands/`:

```bash
# Convenience script (clones/downloads upstream into dataset/wands/):
hatch run eval:fetch-data

# Or manually:
git clone https://github.com/wayfair/WANDS.git /tmp/WANDS
mkdir -p dataset/wands
cp /tmp/WANDS/dataset/query.csv /tmp/WANDS/dataset/product.csv /tmp/WANDS/dataset/label.csv dataset/wands/
```

Note that the raw `label.csv` stores the **string** label (`Exact` / `Partial` / `Irrelevant`) and carries a leading `id` column; the numeric gains are **not** in the file. The `WandsDataset` adapter applies the `Exact/Partial/Irrelevant → 1.0/0.5/0.0` label-to-gain mapping at qrel emission (design §7) and concatenates name + description (+ features) into the canonical `search_text` field, so every variant ranks the same input text.

---

## Build the index

This creates the index mapping (a `text` `search_text` field + one `dense_vector` field per embedder), embeds the corpus **client-side** with each configured embedder connector, and bulk-indexes the documents with their vectors attached (ES stores the vectors; it computes none).

```bash
hatch run eval:index
```

Indexing is idempotent (`_id = product_id`), so re-running is safe. Adding a new embedder later requires a reindex (its `dense_vector` field must be embedded for the whole corpus).

---

## Run the evals

```bash
hatch run eval:run
```

This drives the entire `ExperimentRunner` path from `config.yaml`:

1. Loads the dataset, builds (or reuses) the index, freezes the shared query set.
2. Reads the explicit pipelines from config, **baseline first**.
3. For each pipeline: (for a rerank pipeline) asserts `rerank_window_size <= settings.top_n`, builds the `SearchPipeline` graph (`build_pipeline`), runs it over all queries, writes the result CSV, scores it, writes the metrics CSV.
4. Compares every named variant against the baseline on the identical query set and writes the comparison CSV.
5. Serializes the fully-resolved config + seed to `run_config_{timestamp}.json`.

The pipelines run are **exactly** those you wrote in `config.yaml` (a `baseline` plus a map of named `variants`). There is no matrix expansion and no sweep — if you want two RRF `k` values or two embedders, add two named pipelines.

Useful invocations:

```bash
hatch run eval:run                       # all pipelines from config.yaml
hatch run eval:run -- --config myrun.yaml  # alternate config
hatch run eval:run -- --dry-run            # print the pipeline list, run nothing
```

---

## Outputs

Artifacts are written to `results/` with a single per-run UTC timestamp `{timestamp} = YYYYMMDDTHHMMSSZ`. `{variant}` is the pipeline's name from config (e.g. `hybrid_e5_k60`); `{baseline}` is the baseline pipeline's id (`bm25`). All CSVs are UTF-8, comma-separated, with a header. **Field names and order are fixed.**

### `result_{variant}_{timestamp}.csv`

```
query_id,product_id,score,position
```

One row per returned doc; `position` is the 1-based rank; at most `top_k` rows per query.

### `metrics_{variant}_{timestamp}.csv`

```
query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_scored,n_missing
```

One row per query. Metrics use **condensed-list** evaluation: a returned doc with **no qrel entry (a MISSING judgement)** is **skipped** (not scored as 0); only a **judged-irrelevant** doc (`gain 0.0`) counts as a zero. `n_scored` = judged docs the metrics were computed over (`<= 10`); `n_missing` = missing docs skipped to collect them; both are non-negative integers, always present. Any of the four metric cells is written as an **empty field** (two adjacent commas) when its value is `NaN` — `avg_relevance`/`ndcg@10`/`precision@10` when `n_scored=0`, `recall@10` when `R=0` — meaning "excluded from that metric's aggregation", not zero.

### `comparison_{baseline}_{variant}_{timestamp}.csv`

```
variant,metric,delta,delta_ci_lo,delta_ci_high,p_value,significant_raw,p_value_adjusted,significant
```

One row per metric ∈ {`avg_relevance`, `ndcg@10`, `recall@10`, `precision@10`}.

- `delta` — mean paired difference (variant − baseline) over the shared query set.
- `delta_ci_lo` / `delta_ci_high` — the **per-comparison, unadjusted 2.5/97.5 percentile bootstrap interval**. This is effect-size context only; it is **not** a significance gate.
- `p_value` — the **raw** (uncorrected) Wilcoxon signed-rank (or permutation) p-value.
- `significant_raw` ∈ {`true`,`false`} — the **uncorrected** per-test decision (`p_value <= α`), independent of the family.
- `p_value_adjusted` — the **FDR-adjusted p-value (q-value)**: Benjamini-Hochberg by default (Benjamini-Yekutieli if `correction: by`), computed over the family of all `(variant × metric)` tests in the run.
- `significant` ∈ {`true`,`false`} — the **FDR decision** (`p_value_adjusted <= α`) over that family, at level `q = α = 0.05`.

> The CI lives in a different role from the significance flags and **may disagree** with them — this is expected under a step-up FDR procedure (see §8.3 of the design doc). The CI is descriptive; `significant` is the FDR gate, `significant_raw` is the uncorrected view. The design uses FDR (not FWER/Holm) because this is an **exploratory** search for the best pipeline among many **correlated** configurations, where the cost of a false positive is low and asymmetric and BH keeps far more power than Bonferroni-style FWER.

### `run_config_{timestamp}.json`

The fully-resolved config + seed: the resolved services registry and named pipelines (baseline + variants, each with its retrievers/fuser/reranker/window), bootstrap `B`, CI level, family `α` and size `m`, correction method, test + its zero/tie params, dataset version, ES + endpoint versions, cutoff, and any degenerate-paired-set notes. Given the recorded seed, the statistics are reproducible.

```bash
ls -1 results/
# result_bm25_20260630T120000Z.csv
# metrics_bm25_20260630T120000Z.csv
# comparison_bm25_semantic_e5_20260630T120000Z.csv
# ...
# run_config_20260630T120000Z.json
```

---

## Configuration reference

The config lives in `config.yaml` (YAML or JSON). It declares **explicit, named building blocks** — `services` (embedders, rerankers, searchers) and `pipelines` (a `baseline` plus a map of named `variants`). No axes, no expander, no sweep: the pipelines run are exactly the ones written.

```yaml
dataset:
  name: wands
  path: ./dataset/wands
services:                       # named, typed, reusable building blocks (embedders/rerankers are provider connectors)
  - embedder: { name: cohere, provider: cohere, settings: { api_key: ${COHERE_KEY}, model_id: embed-english-v3.0 } }
  - embedder: { name: voyage, provider: voyage, settings: { api_key: ${VOYAGE_KEY}, model_id: voyage-3.5 } }
  - reranker: { name: co-rr,  provider: cohere, settings: { api_key: ${COHERE_KEY}, model_id: rerank-v3.5, top_n: 100 } }
  - searcher: { name: bm25,        provider: elasticsearch, kind: lexical }
  - searcher: { name: semantic_co, provider: elasticsearch, kind: vector, embedder: cohere }
  - searcher: { name: semantic_vo, provider: elasticsearch, kind: vector, embedder: voyage }
indexer:
  provider: elasticsearch
  index: wands_bench
  settings: { url: ${ES_URL} }
pipelines:
  baseline:                      # the reference every variant is compared against
    retriever: bm25
  variants:                      # each is one explicit run; the map key is its id
    semantic_co:   { retriever: semantic_co }
    semantic_vo:   { retriever: semantic_vo }
    hybrid_co_k60:
      retrievers: [bm25, semantic_co]
      fuser: { type: rrf, rank_constant: 60, window: 100 }
    bm25_rerank:
      retriever: bm25
      reranker: co-rr
      rerank_window_size: 100
    hybrid_co_rerank:
      retrievers: [bm25, semantic_co]
      fuser: { type: rrf, rank_constant: 60, window: 100 }
      reranker: co-rr
      rerank_window_size: 100
stats:
  test: wilcoxon
  correction: bh                 # Benjamini-Hochberg FDR (default); by = Benjamini-Yekutieli
  alpha: 0.05                    # BOTH the raw per-test threshold AND the FDR target level q
  bootstrap_B: 10000
  ci_level: 0.95                 # UNADJUSTED per-comparison effect-size CI; NOT a gate
  seed: 1234
cutoff: 10                       # metrics @10
top_k: 100                       # results retrieved per query
```

**Pipeline field rules** (validated at load; a violation raises a clear error):

- exactly one of `retriever` (a single searcher name) XOR `retrievers` (a list of 2+ names);
- `retrievers` (2+) requires a `fuser`; a `fuser` is only allowed with `retrievers` (`type: rrf` only);
- `reranker` requires `rerank_window_size`, and vice-versa;
- every referenced service must exist and be the right type; a vector searcher must reference an existing embedder;
- a variant id must not duplicate the baseline id.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `ES_URL` | ElasticSearch endpoint (e.g. `http://localhost:9200`). |
| `COHERE_KEY` | API key for the Cohere embedder / reranker. |
| `VOYAGE_KEY` | API key for the Voyage embedder / reranker. |
| `OPENAI_KEY` | API key for the OpenAI embedder (OpenAI has no reranker). |

Keys are referenced via `${VAR}` in `config.yaml` and resolved at load time — secrets never live in the config file. A reranker's `top_n` must be `>= rerank_window_size`.

---

## Troubleshooting

- **ES won't start / `vm.max_map_count` error** — raise it: `sudo sysctl -w vm.max_map_count=262144` (Linux), or via the Docker VM on Desktop. See the compose section above.
- **ES killed at startup / OOM** — give Docker more RAM, or lower `ES_JAVA_OPTS` heap in `docker-compose.yml` (set `-Xms` and `-Xmx` equal).
- **Mapping rejects `dense_vector` / `knn` unsupported** — your ES is very old; upgrade the image (`dense_vector` + `knn` predate the `>= 8.15` pin). `VectorSearch` embeds the query and issues a `knn` query, so ES must support `dense_vector`.
- **Provider auth failures (401/403)** — the relevant provider env var is unset or wrong. Confirm `COHERE_KEY` / `VOYAGE_KEY` / `OPENAI_KEY` are exported in the shell that runs `hatch run eval:*`, and that the `model_id` in `config.yaml` matches the provider. A failed call raises a `ProviderError` carrying the provider, HTTP status, and raw body.
- **`W <= top_n` assertion fails before a rerank variant** — a reranker's `settings.top_n` is smaller than the pipeline's `rerank_window_size`. Raise `settings.top_n` (the number of candidates the provider scores per request).
- **Empty results / all-zero metrics** — usually the index is empty or the wrong index name. Re-run `hatch run eval:index`, confirm `dataset/wands/` is populated, and check `indexer.index` in `config.yaml`. Verify doc count: `curl -s "$ES_URL/wands_bench/_count"`.
- **First run is slow / rate-limited** — `eval:index` embeds the whole corpus through the provider, so a large corpus is many calls (bounded by `settings.rate_limit.requests_per_minute`); a `429` is retried with backoff automatically.

---

## Teardown

```bash
docker compose down -v        # stop ES and delete its volume (index data)
```

Run artifacts in `results/` and the dataset in `dataset/wands/` are untouched by teardown; remove them manually if you want a clean slate.

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 TensorOpt.
