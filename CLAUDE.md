# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

A reproducible **search-relevance benchmark harness**. It measures how much each retrieval
strategy improves over a **BM25 baseline** on a fixed dataset + qrel set. First instantiation:
**WANDS** dataset on **ElasticSearch** as a plain BM25 + `dense_vector` index. The harness owns
inference: it computes embeddings and rerank scores via **provider connectors** (Cohere, Voyage,
OpenAI) in `benchmark/providers/inference.py` — ES runs no `_inference`.

- **`docs/methodology.md`** — authoritative for the evaluation **science**: objective, variants-as-
  hypotheses, the IR model/glossary, metrics, statistics (the WHAT/WHY).
- **`docs/architecture.md`** — authoritative for the **blueprint**: abstractions/class hierarchy, ES
  mapping, caching, data flow + single execution path, artifact schemas, config, module layout,
  extension guide (the HOW).
- **These two are the source of truth.** When code and a doc disagree on a name or schema, the doc
  wins. (The former `experiment.md` and the per-feature design docs have been folded into these two
  and removed.)
- **`README.md`** — operational guide: how to run the evals end to end.

## Stack

- Python, managed with **hatch** (envs + scripts; eval commands run as `hatch run eval:<script>`).
- **docker-compose** spins up single-node ElasticSearch (>= 8.15, hard floor).
- Dataset lives in **`dataset/wands/`** (`query.csv`, `product.csv`, `label.csv`, tab-separated; gitignored, not in repo).

## Load-bearing invariants — do not drift

- **DRY / one path.** All 6 variants (`bm25`, `semantic`, `hybrid`, `bm25_rerank`,
  `semantic_rerank`, `hybrid_rerank`) are config rows through a single `SearchPipeline` +
  `ExperimentRunner`. No per-variant code. The full {bm25, semantic, hybrid} × {rerank off/on}
  factorial is realized on WANDS as `bm25`, `semantic_co`, `hybrid_co_k60`, `bm25_rerank`,
  **`semantic_co_rerank`**, `hybrid_co_rerank` — all six cells present (none omitted).
- **Generality.** Swapping dataset, backend/vector index, embedding model, or reranker must be a
  new adapter + config only — never edits to the domain engine (`search`/`indexing`/`evaluation`).
  ES + WANDS are adapters behind Protocols; a new backend is an `IndexWriter` +
  `build_searchers`/`build_rerankers` in `providers/` + config target-table rows, no per-backend
  `Indexer`/factory (the domain `Indexer` is backend-agnostic and shared).
- **Relevance gains** (float; WANDS): `Exact=1.0`, `Partial=0.5`, `Irrelevant=0.0`. **ONE
  `metrics.unjudged` policy governs ALL SIX metrics identically** (methodology.md §7; config keys
  `metrics.unjudged` + `metrics.relevance_threshold`) — there is **no per-metric carve-out**. Under
  the shipped `condensed` default, a **MISSING** judgement (no qrel entry for a returned doc) is
  **DROPPED** from the eval list — **NOT** treated as irrelevant; only a **judged** `0.0` is
  irrelevant. Under `irrelevant` (trec_eval) a MISSING doc is scored `0.0` in place. Every metric —
  recall included — slices the one policy-built eval list (recall is condensed under `condensed`,
  standard/positional under `irrelevant`). The binary-relevance threshold for precision/recall AND
  the recall denominator `R` (from qrels) is `gain >= metrics.relevance_threshold` (default `0.5`:
  Partial or Exact). Recall (cutoffs `{10, 50, 100}`) still penalizes retrieval failures (empty
  result → `0`, not NaN, when `R > 0`) and is NaN iff `R == 0`. Per-query `n_results` (docs
  returned), `n_scored`/`n_missing` (always-condensed top-10 diagnostics), and `n_relevant` (= `|R|`
  under the resolved threshold) are recorded.
- **Uniform retrieval depth (kill the confound).** Every system retrieves/returns to ONE depth:
  `fuser.window == rerank_window_size == top_k` (WANDS: 100), reranker `top_n >= W`. No knob may make
  depth co-vary with the rerank/fusion treatment (architecture.md §5.3). This is a **config
  convention for `eval:run`** — recorded in the manifest, deliberately NOT hard-validated at load
  (the `eval:sweep --axis=rerank_window` diagnostic must vary `rerank_window_size` per cell); only
  `W <= top_n` is asserted (R0).
- **Default significance test = mean-δ sign-flip permutation** (methodology.md §8.2): the p-value, point estimate,
  and CI share one estimand (the mean paired difference). `wilcoxon` stays selectable.
- **Exact CSV artifact schemas (do not rename/reorder fields) — one file per run, all pipelines:**
  - `result_{timestamp}.csv` — `variant, query_id, product_id, score, position`
  - `metrics_{timestamp}.csv` — `variant, query_id, avg_relevance, ndcg@10, recall@10, recall@50, recall@100, precision@10, n_results, n_scored, n_missing, n_relevant`
    (`n_relevant` = the relevant-set size `|R|` under the resolved threshold; non-negative int,
    always present.)
  - `comparison_{timestamp}.csv` — `system_a, system_b, metric, value_a, value_b, delta, delta_ci_lo, delta_ci_high, p_value, significant_raw, in_family, p_value_adjusted, significant, n_common`
    (FDR family = `contrast.family × fdr_metrics`; **`in_family=false ⟺ p_value_adjusted and significant BOTH empty`**, methodology.md §8.3.)

## Conventions

- Match the style of surrounding code; keep abstractions minimal (favor stdlib/native over deps).
- **Declare the interface a concrete class satisfies (preferred).** Every concrete implementation —
  or its shared implementation base — explicitly subclasses the ABC or `Protocol` it fulfills (e.g.
  `ESIndexWriter(IndexWriter)`, `_BaseEmbedder(_Connector, Embedder)`, `CohereReranker` via
  `_BaseReranker(_Connector, RerankClient)`, `LexicalSearcher(Searcher)`), **even for structural
  `Protocol`s** where Python does not require it. This makes the interface→implementations mapping
  greppable/navigable (find an interface's implementors by its subclasses) and lets mypy verify
  conformance at the class definition — the lazy dotted-target factories (architecture.md §11) return `Any`, so an
  undeclared structural implementation's drift would otherwise go statically unchecked.
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
- **Handle exceptions meaningfully — never `except …: pass`.** Catch narrowly (the specific
  exception, not bare `except`). If the caught condition is *expected/benign*, make that intent
  explicit — log at `debug`/`info` and continue (prefer `try/except/else`, e.g. `NotFoundError` →
  "not found; creating it"). If it is *unexpected*, log a `warning`/`error` with context **or** let
  it bubble up. Don't widen the `try` to swallow errors you did not mean to catch. A silent `pass`
  hides both the intent and real failures.
- **Use logging, not `print()`.** Get a logger via `benchmark.common.logging_setup.get_logger(__name__)`
  and call `setup_logging()` once at each entry point — it logs to the console and to
  `logs/run_{timestamp}.log`. Pass the run's timestamp so the log lines up with that run's artifacts.
- Before changing a name/schema, check it against `docs/methodology.md` (metrics/stats) or
  `docs/architecture.md` (abstractions/schemas/config/layout) and keep code and docs consistent.
- Cross-references in code and docs may point **only** to `docs/methodology.md` and
  `docs/architecture.md` (by section, e.g. `§7` / `architecture.md §5.5`); never to ephemeral
  fix-tracking IDs (`P0-x`, `MF-x`, etc.) or any other transient document.
- Don't commit `dataset/`, `results/`, or `logs/` artifacts.
