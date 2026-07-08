# Search-Relevance Benchmark

A reproducible harness for measuring how much different retrieval strategies improve search relevance over a **BM25 baseline**, on a fixed dataset and qrel set. It indexes documents into ElasticSearch, runs each variant (semantic, hybrid + RRF, and reranked combinations) through **one shared pipeline**, scores them with graded relevance metrics (`avg_relevance`, `ndcg@10`, `recall@10`, `recall@50`, `recall@100`, `precision@10`), and emits per-run CSV artifacts plus a paired statistical comparison against the baseline. The first concrete instantiation uses **WANDS** (Wayfair ANnotation Dataset for Search) on ElasticSearch **>= 8.15**, used as a **plain vector + BM25 index**: the harness embeds the corpus and queries via provider connectors (Cohere / Voyage / OpenAI) and retrieves with BM25 `match` + `knn` over `dense_vector` fields.

This README is the operational guide for **running the evals**. The design lives in two authoritative docs: the **science** (metric definitions, statistics, variants-as-hypotheses) → [`docs/methodology.md`](docs/methodology.md); the **internals** (abstractions, ES mapping, caching, the DRY single-execution-path, artifact schemas, config, module layout) → [`docs/architecture.md`](docs/architecture.md). Where this guide and a design doc differ on a name, the design doc wins.

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
ls results/                     # result_* / metrics_* / comparison_* / run_config_*

# 8. Tear down
docker compose down -v
```

`hatch run eval:run` reads `config.yaml` — a set of **explicit, named pipelines** (a `baseline` plus named `variants`; no matrix expansion, no sweep) — executes every pipeline through the single `ExperimentRunner` path (baseline first), and writes the three CSV artifact types described below. Commands shown above are the intended CLI surface that the implementation satisfies; they mirror the design in [`docs/architecture.md`](docs/architecture.md).

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

Every pipeline is an explicit named entry in the config — there is no per-variant code and **no matrix expansion or sweep**. A run produces one `result_*` + `metrics_*` + `comparison_*` file, each holding all pipelines (the `result`/`metrics` files include the baseline via a leading `variant` column). The baseline is not compared to itself, so it has no row in `comparison_*`.

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
│   ├── common/                # (g) shared bottom layer — depends on nothing
│   │   ├── models.py          # Query, Document, Qrel, ScoredDoc, RankedResult, FieldSchema, IndexMapping, enums
│   │   ├── protocols.py       # Searcher/Fuser/Reranker + Dataset ABCs; Embedder, RerankClient, IndexWriter Protocols
│   │   ├── ranking.py         # fuse_rrf_local + rerank_local (pure windowed ranking primitives)
│   │   └── logging_setup.py   # console + file logging (logs/run_{timestamp}.log)
│   ├── providers/             # (f) concrete adapters — depend ONLY on common
│   │   ├── inference.py       # OpenAI/Cohere/Voyage Embedders + Cohere/Voyage RerankClients (stdlib-HTTP)
│   │   └── elasticsearch.py   # LexicalSearcher, VectorSearch, ESReranker, ESIndexWriter, build_searchers/build_rerankers
│   ├── embedding.py           # (c) make_embedder + EMBEDDER_PROVIDERS (dispatch provider -> providers.inference)
│   ├── reranking.py           # (d) make_reranker + RERANKER_PROVIDERS (dispatch provider -> providers.inference)
│   ├── indexing.py            # (a) Indexer (backend-agnostic build orchestration) + embed-at-ingest streaming
│   ├── search.py              # (b) RRFFuser, HybridSearch, SearchPipeline (the composers)
│   ├── evaluation/            # (e) scoring + statistics
│   │   ├── metrics.py         # Evaluator, Metrics, QrelIndex
│   │   └── stats.py           # Comparator (bootstrap CI, Wilcoxon/permutation, FDR/BH-BY)
│   ├── datasets/
│   │   └── wands.py           # WandsDataset (label→gain, search_text concat)
│   ├── config.py              # config value types (PipelineCfg/Services/ResolvedConfig, NO expansion) + build_pipeline; YAML/JSON load + resolve; lazy dotted adapter factories
│   ├── runner.py              # ExperimentRunner — the single execution path
│   └── io_csv.py              # write_results_csv / write_metrics_csv / write_comparison_csv / write_run_config
├── dataset/
│   └── wands/                 # query.csv, product.csv, label.csv  (NOT in the repo; see below)
├── results/                   # run artifacts land here (gitignored)
├── config.yaml                # explicit named pipelines (baseline + variants)
├── docker-compose.yml         # single-node ElasticSearch
├── pyproject.toml             # hatch envs + scripts
├── docs/
│   ├── methodology.md         # authoritative: evaluation science (metrics, statistics)
│   └── architecture.md        # authoritative: blueprint (abstractions, ES, caching, schemas, layout)
└── LICENSE                    # MIT
```

Module names and responsibilities follow architecture.md §11. The `a–g` layers form a strict acyclic engine (`common` ← `providers` ← `embedding`/`reranking` ← `indexing`/`search`/`evaluation`); the domain layers import only `common` abstractions at import time and consume concrete `providers` pieces injected at runtime. `config.py`, `runner.py`, `io_csv.py` are the composition layer above the engine: `config.py` (config value types + `build_pipeline` + loader) imports `search` for the composers, a one-way wiring edge, and selects adapters via its lazy dotted-target factories, so no engine module names an adapter.
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

Embedding and reranker models are **provider connectors** (`benchmark/providers/inference.py`): the harness calls Cohere / Voyage / OpenAI directly over HTTP — ES runs no inference. You declare each as a `services` entry in `config.yaml`; secrets are injected from environment variables at load time. A connector is just a `provider` + a `settings` block (`api_key`, `model_id`, optional `rate_limit.requests_per_minute`, `batch_size`, `dims`). **OpenAI has no reranker** — a `reranker` with `provider: openai` is rejected at load.

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

The harness embeds the whole corpus with each configured embedder **before** searching — one `dense_vector` field per embedder (see architecture.md §5.2). Rerankers need no setup: `ESReranker` calls the provider connector per query, and the harness asserts `rerank_window_size <= settings.top_n` before running each rerank pipeline.

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

Note that the raw `label.csv` stores the **string** label (`Exact` / `Partial` / `Irrelevant`) and carries a leading `id` column; the numeric gains are **not** in the file. The `WandsDataset` adapter applies the `Exact/Partial/Irrelevant → 1.0/0.5/0.0` label-to-gain mapping at qrel emission (methodology.md §7) and concatenates name + description (+ features) into the canonical `search_text` field, so every variant ranks the same input text.

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

`eval:run` does **not** index — build the index first with `eval:index`. It drives the entire `ExperimentRunner` path from `config.yaml`:

1. Loads the dataset and **verifies the index is fully built**: it must exist and its doc count must equal the dataset's document count, otherwise `eval:run` exits non-zero with a clear message (build it with `eval:index`). This never re-embeds — re-running the eval reuses the index and makes no document-embedding calls. (Changed the dataset or an embedder? Re-run `eval:index` first — the count check can't detect same-count-but-different content.)
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

### Caching (optional)

The harness can persist provider inference (query/document embeddings, rerank scores) and searcher result lists to a local sqlite cache, so repeated work — within a run (variants share embedders/rerankers/searchers) and across runs (re-indexing an unchanged corpus, re-running evals) — is served from disk instead of re-paid. The cache is a **pure-function cache**: cache-on and cache-off produce byte-identical metrics (the key captures every value-determining input), so it changes speed, never numbers.

```yaml
cache:
  enabled: true      # default when the block is ABSENT: false (opt-in cold run)
  dir: .cache        # single sqlite file at .cache/inference.sqlite (gitignored)
```

- **Enable / disable:** set `cache.enabled` (or omit the block for a cold run — disabled).
- **Clear:** `rm -rf .cache/` (or delete `.cache/inference.sqlite`). It is gitignored; there is no dedicated command.
- **Note:** the key uses the model *name*, not its weights — if a provider silently retrains a model behind a stable alias, clear the cache. A corrupt cache never crashes a run: it is logged and the run proceeds without it.

---

## Outputs

Artifacts are written to `results/` with a single per-run UTC timestamp `{timestamp} = YYYYMMDDTHHMMSSZ`. Each run produces exactly three CSVs (plus `run_config`), each holding **all** pipelines in one file: the `variant` column is the pipeline's name from config (e.g. `hybrid_e5_k60`), and `baseline` is the baseline pipeline's id (`bm25`). All CSVs are UTF-8, comma-separated, with a header. **Field names and order are fixed.**

### `result_{timestamp}.csv`

```
variant,query_id,product_id,score,position
```

One file for all pipelines (baseline included). One row per returned doc; `variant` is the pipeline id; `position` is the 1-based rank; at most `top_k` rows per (variant, query). Rows are grouped by variant, baseline first.

### `metrics_{timestamp}.csv`

```
variant,query_id,avg_relevance,ndcg@10,recall@10,recall@50,recall@100,precision@10,n_results,n_scored,n_missing
```

One file for all pipelines (baseline included). One row per (variant, query); `variant` is the pipeline id, baseline first. The **condensed** metrics (`avg_relevance`/`ndcg@10`/`precision@10`) use **condensed-list** evaluation: a returned doc with **no qrel entry (a MISSING judgement)** is **skipped** (not scored as 0); only a **judged-irrelevant** doc (`gain 0.0`) counts as a zero. **Recall is standard/coverage** (`|judged-relevant ∩ actual top-k| / R`, `R` from qrels) at cutoffs `{10, 50, 100}` — it looks at the true retrieved positions, never scores a MISSING doc as irrelevant, and scores an empty/failed result `0` (not NaN) when `R > 0`. `n_results` = docs the pipeline returned for the query (`<= top_k`); `n_scored` = judged docs the condensed metrics were computed over (`<= 10`); `n_missing` = missing docs skipped to collect them; all three are non-negative integers, always present. Any of the six metric cells is written as an **empty field** (two adjacent commas) when its value is `NaN` — `avg_relevance`/`ndcg@10`/`precision@10` when `n_scored=0`, every `recall@k` when `R=0` — meaning "excluded from that metric's aggregation", not zero.

### `comparison_{timestamp}.csv`

```
system_a,system_b,metric,value_a,value_b,delta,delta_ci_lo,delta_ci_high,p_value,significant_raw,in_family,p_value_adjusted,significant,n_common
```

One file for all contrasts. Each row is a contrast between two systems (the baseline is just a system, so variant-vs-variant contrasts look the same; the default all-vs-baseline run never compares the baseline to itself). One row per (contrast, metric ∈ {`avg_relevance`, `ndcg@10`, `recall@10`, `recall@50`, `recall@100`, `precision@10`}).

- `system_a` / `system_b` — the contrast's two system ids (`delta = value_a − value_b`, positive = `a` wins).
- `value_a` / `value_b` — the per-metric means of `a` and `b` over the shared **family-wide common subset** (finite in every contrast-referenced system) for that metric.
- `delta` — mean paired difference (`value_a − value_b`) over that subset.
- `delta_ci_lo` / `delta_ci_high` — the **per-comparison, unadjusted 2.5/97.5 percentile bootstrap interval**. Effect-size context only; **not** a significance gate.
- `p_value` — the **raw** (uncorrected) p-value from the **mean-δ sign-flip permutation** test (default; Wilcoxon signed-rank if `test: wilcoxon`).
- `significant_raw` ∈ {`true`,`false`} — the **uncorrected** per-test decision (`p_value <= α`), independent of the family.
- `in_family` ∈ {`true`,`false`} — FDR-family membership: `contrast.family AND metric ∈ fdr_metrics AND non-degenerate` (default `fdr_metrics = {ndcg@10, recall@100}`).
- `p_value_adjusted` — the **FDR-adjusted p-value (q-value)** (Benjamini-Hochberg by default, Benjamini-Yekutieli if `correction: by`), computed over the family rows only. **Empty when `in_family=false`.**
- `significant` ∈ {`true`,`false`} — the **FDR decision** (`p_value_adjusted <= α`) at level `q = α = 0.05`. **Empty when `in_family=false`** (rule: `in_family=false ⟺ both `p_value_adjusted` and `significant` empty`).
- `n_common` — the common-subset size for that metric (always present).

> The CI is descriptive effect-size context in a different role from the significance flags, and **may disagree** with them — expected under a step-up FDR procedure. `significant` is the FDR gate; `significant_raw` is the uncorrected view. The FDR family is deliberately small — only the headline metrics (`ndcg@10` = ranking quality, `recall@100` = coverage) enter it. Why FDR (not FWER/Holm), why a small family, the CI/flag disagreement, and the exact mean-δ estimand → methodology.md §8.

### `run_config_{timestamp}.json`

The fully-resolved config + seed: the resolved services registry and named pipelines (baseline + variants, each with its retrievers/fuser/reranker/window), the stats block (incl. `contrasts` and `fdr_metrics`), bootstrap `B`, CI level, family `α`, correction method, test + its zero/tie params, dataset version, ES + endpoint versions, cutoff, uniform retrieval depth (`top_k`), and a `diagnostics` block (per-metric common-subset `n_common`/`n_excluded` + per-system retrieval-failure counts). Given the recorded seed, the statistics are reproducible.

```bash
ls -1 results/
# result_20260630T120000Z.csv
# metrics_20260630T120000Z.csv
# comparison_20260630T120000Z.csv
# run_config_20260630T120000Z.json
```

---

## Crafting `config.yaml`

The config lives in `config.yaml` (YAML or JSON) and declares **explicit, named building blocks**.
No axes, no expander, no sweep: the pipelines run are exactly the ones written, so you read the file
top to bottom and see the whole experiment. `benchmark/config.py` parses + validates it into a
`ResolvedConfig`; the runner iterates the baseline + explicit variants (baseline first). Full schema
+ validation rules → architecture.md §10.

```yaml
dataset:
  name: wands
  path: ./dataset/wands
services:                       # named, typed, reusable building blocks
  - embedder: { name: cohere, provider: cohere, settings: { api_key: ${COHERE_KEY}, model_id: embed-english-v3.0 } }
  - reranker: { name: co-rr,  provider: cohere, settings: { api_key: ${COHERE_KEY}, model_id: rerank-v3.5, top_n: 100 } }
  - searcher: { name: bm25,        provider: elasticsearch, kind: lexical }
  - searcher: { name: semantic_co, provider: elasticsearch, kind: vector, embedder: cohere }
indexer:
  provider: elasticsearch
  index: wands_bench
  settings: { url: ${ES_URL} }
pipelines:
  baseline:                      # the reference every variant is compared against
    retriever: bm25
  variants:                      # each is one explicit run; the map key is its id
    semantic_co:   { retriever: semantic_co }
    hybrid_co_k60:
      retrievers: [bm25, semantic_co]
      fuser: { type: rrf, rank_constant: 60, window: 100 }
    bm25_rerank:        { retriever: bm25,        reranker: co-rr, rerank_window_size: 100 }
    semantic_co_rerank: { retriever: semantic_co, reranker: co-rr, rerank_window_size: 100 }
    hybrid_co_rerank:
      retrievers: [bm25, semantic_co]
      fuser: { type: rrf, rank_constant: 60, window: 100 }
      reranker: co-rr
      rerank_window_size: 100
stats:
  test: permutation              # mean-δ sign-flip permutation (default); wilcoxon selectable
  correction: bh                 # Benjamini-Hochberg FDR (default); by = Benjamini-Yekutieli
  alpha: 0.05                    # BOTH the raw per-test threshold AND the FDR target level q
  bootstrap_B: 10000             # CI resamples AND permutation count; p-resolution floor 1/(B+1)
  ci_level: 0.95                 # UNADJUSTED per-comparison effect-size CI; NOT a gate
  seed: 1234
  contrasts:                     # explicit system_a vs system_b; absent => every variant vs baseline
    - { a: semantic_co,        b: bm25,          family: true }
    - { a: hybrid_co_k60,      b: semantic_co,   family: true }
    - { a: hybrid_co_rerank,   b: semantic_co,   family: true }
    - { a: bm25_rerank,        b: bm25,          family: true }
    - { a: semantic_co_rerank, b: semantic_co,   family: true }
    - { a: hybrid_co_rerank,   b: hybrid_co_k60, family: true }
  fdr_metrics: [ndcg@10, recall@100]  # the ONLY metrics that enter the BH family
cutoff: 10                       # point/quality metrics @10 (avg_relevance, ndcg@10, precision@10)
top_k: 100                       # uniform retrieval depth: fuser.window == rerank_window_size == top_k
cache:                           # OPTIONAL: memoize embeddings/rerank/searcher results to .cache/
  enabled: true
  dir: .cache
```

**`services` — the reusable building blocks.** Three kinds, each `name`d and referenced by name
elsewhere:
- **`embedder`** / **`reranker`** are **provider connectors** — the harness calls Cohere / Voyage /
  OpenAI directly (ES runs no inference). `provider` picks the connector; `settings` carries
  `model_id`, `api_key`, and optional `rate_limit.requests_per_minute` / `batch_size` / `dims`. A
  reranker's **`top_n`** (the rank-window cap = candidates scored per request) is a plain `settings`
  key; the harness asserts `rerank_window_size <= top_n` before each rerank pipeline. **OpenAI has no
  reranker** (rejected at load).
- **`searcher`** is a leaf retriever: `kind: lexical` (BM25) or `kind: vector` (references an
  `embedder` by name). Add an embedding model by adding an `embedder` + a `vector` searcher, then
  reindex (architecture.md §5.2/§12).
- Secrets are `${VAR}` placeholders resolved from the environment at load — they never live in the
  file.

**`pipelines` — the baseline + named variants.** `baseline` is the reference every default comparison
subtracts from; `variants` is a map of `id → pipeline spec` (the map key is the pipeline's artifact
id). The six entries above are the full **{bm25, semantic, hybrid} × {rerank off, rerank on}**
factorial — all six cells present, including the dense-only `semantic_co_rerank`. Each spec is one of:
a single `retriever`; or `retrievers: [...]` (2+) with a `fuser` (RRF); optionally a `reranker` +
`rerank_window_size`. There is no per-variant code — every spec is composed by the same
`build_pipeline` (architecture.md §4). Want two RRF `k`s or two embedders? Write two named variants.

**`stats` — the comparison regime.** `contrasts` turns the factorial into explicit hypotheses
(each `a` vs `b`; `family: true` puts the row in the FDR family); absent, it defaults to
every-variant-vs-baseline. `fdr_metrics` is the small set of headline metrics that enter the
Benjamini-Hochberg family (default `ndcg@10` = ranking quality, `recall@100` = coverage) — everything
else is reported as descriptive context. `test` is the default mean-δ sign-flip `permutation`
(`wilcoxon` selectable), `correction` is `bh` (or `by`), `alpha` is both the raw and FDR threshold,
`bootstrap_B` drives both the CI resamples and the permutation count, and `seed` makes it all
reproducible. The science behind these choices → methodology.md §8.

**`cutoff` / `top_k` — uniform depth (a load-bearing invariant).** `cutoff` is the point/quality
cutoff (10). `top_k` is the **single retrieval depth every system shares**: the config keeps
`fuser.window == rerank_window_size == top_k` so depth never co-varies with the rerank/fusion
treatment (and enables `recall@100`). See architecture.md §5.3.

**`cache` — an optional speed knob.** `cache: { enabled, dir }` memoizes embeddings / rerank scores /
searcher result lists to a local sqlite file. It is a **pure-function** cache — enabling it never
changes the numbers, only the speed — so it is not part of the experiment definition. Absent → a cold
(disabled) run. See architecture.md §5.5 (and the [Caching](#caching-optional) section above).

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
