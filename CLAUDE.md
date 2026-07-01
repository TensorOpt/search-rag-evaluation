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
- **Move with certainty.** Prefer resolving unknowns at build time over runtime. If a dependency's
  capability/version/behavior is uncertain, pin the version that guarantees it and call it directly —
  do NOT ship runtime feature-detection (`getattr(mod, "feature", None)` probes) or best-effort
  fallbacks. Verify and validate up front, not at runtime.
- **Never test float equality.** Do not compare floats with `==`/`!=` (incl. `== 0.0`). Use
  `math.isclose(a, b, abs_tol=1e-6)` — or `np.isclose(x, y, rtol=0.0, atol=1e-6)` for numpy arrays.
  Name the tolerance (e.g. `ZERO_ABS_TOL = 1e-6`).
- **Exhaustive branching on enumerated values.** When behavior branches on an enumerated/config
  value, handle every valid value explicitly and `raise` a clear error on no match. Never fall
  through to a silent default when the configuration is invalid.
- **Descriptive names, no cryptic abbreviations — but don't overdo it.** Prefer `mean_delta`/
  `baseline_value` over `obs`/`b_val`; a reader should not have to guess what a variable holds.
  Equally, don't pad obvious names: a short/idiomatic name is fine when context makes it plain
  (`idx`/`rank` for a loop counter, `k` for the RRF rank constant inside `fuse_rrf_local`). Avoid
  both extremes — cryptic (`obs`) and needlessly verbose (`zero_based_index`).
- **Use logging, not `print()`.** Get a logger via `benchmark.logging_setup.get_logger(__name__)`
  and call `setup_logging()` once at each entry point — it logs to the console and to
  `logs/run_{timestamp}.log`. Pass the run's timestamp so the log lines up with that run's artifacts.
- Before changing a name/schema, check it against `docs/experiment.md` and keep both files consistent.
- Don't commit `dataset/`, `results/`, or `logs/` artifacts.
