# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

A reproducible **search-relevance benchmark harness**. It measures how much each retrieval
strategy improves over a **BM25 baseline** on a fixed dataset + qrel set. First instantiation:
**WANDS** dataset on **ElasticSearch (>= 8.15)** via native `_inference` endpoints and retrievers.

- **`docs/experiment.md`** — authoritative experimental design (abstractions, metrics, statistics).
  **This is the source of truth.** When code and this doc disagree on a name or schema, the doc wins.
- **`README.md`** — operational guide: how to run the evals end to end.

Status: **Phase 0 done** (scaffolding: `pyproject.toml` + hatch envs, `docker-compose.yml`,
`benchmark/` package skeleton, `config.yaml`). Build proceeds phase-by-phase per
[`docs/plan.md`](docs/plan.md); each phase ends in a user sign-off + commit.

## Stack

- Python, managed with **hatch** (envs + scripts; eval commands run as `hatch run eval:<script>`).
- **docker-compose** spins up single-node ElasticSearch (>= 8.15, hard floor).
- Dataset lives in **`dataset/wands/`** (`query.csv`, `product.csv`, `label.csv`, tab-separated; gitignored, not in repo).

## Load-bearing invariants — do not drift

- **DRY / one path.** All 6 variants (`bm25`, `semantic`, `hybrid`, `bm25_rerank`,
  `semantic_rerank`, `hybrid_rerank`) are config rows through a single `SearchPipeline` +
  `ExperimentRunner`. No per-variant code.
- **Generality.** Swapping dataset, backend/vector index, embedding model, or reranker must be a
  new adapter + config only — never edits to pipeline/evaluator/stats. ES + WANDS are adapters
  behind Protocols.
- **Relevance gains** (float; WANDS): `Exact=1.0`, `Partial=0.5`, `Irrelevant=0.0`. Binary-relevance
  threshold for precision/recall is `gain >= 0.5` (Partial or Exact). A **MISSING** judgement (no
  qrel entry for a returned doc) is **SKIPPED** via condensed-list evaluation (§7) — **NOT** treated
  as irrelevant; only a **judged** `0.0` is irrelevant. Per-query `n_scored`/`n_missing` are recorded.
- **Exact CSV artifact schemas (do not rename/reorder fields):**
  - `result_{variant}_{timestamp}.csv` — `query_id, product_id, score, position`
  - `metrics_{variant}_{timestamp}.csv` — `query_id, avg_relevance, ndcg@10, recall@10, precision@10, n_scored, n_missing`
  - `comparison_{baseline}_{variant}_{timestamp}.csv` — `variant, metric, delta, delta_ci_lo, delta_ci_high, p_value, significant_raw, p_value_adjusted, significant`
- **RRF k-sweep** is over `rank_constant` ∈ {10,20,…,100}.

## Conventions

- Match the style of surrounding code; keep abstractions minimal (favor stdlib/native over deps).
- **Use logging, not `print()`.** Get a logger via `benchmark.logging_setup.get_logger(__name__)`
  and call `setup_logging()` once at each entry point — it logs to the console and to
  `logs/run_{timestamp}.log`. Pass the run's timestamp so the log lines up with that run's artifacts.
- Before changing a name/schema, check it against `docs/experiment.md` and keep both files consistent.
- Don't commit `dataset/`, `results/`, or `logs/` artifacts.
