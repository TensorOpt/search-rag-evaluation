# Search-Relevance Benchmark — Phased Development Plan

> Status: build plan v1 · Owner: TensorOpt · License: MIT
> Authoritative design: [`docs/experiment.md`](experiment.md). Operational guide: [`README.md`](../README.md). Repo invariants: [`CLAUDE.md`](../CLAUDE.md).
> Where this plan and `docs/experiment.md` disagree on a name, schema, or sequencing, **the design doc wins** — this plan only schedules the build; it does not redefine it.

---

## 1. Purpose & how to read this plan

This document turns the (already complete) design in `docs/experiment.md` into a sequence of **small, independently reviewable, bottom-up phases**. The design is authoritative and exhaustive about *what* to build (module names §11, import DAG §11, CSV schemas §9, metrics §7, statistics §8, single execution path §8.0, config matrix §10). This plan is only about *the order in which we build it and how we verify each step*.

Read it top to bottom: §2 explains the phasing principles, §3 gives the dependency graph and ordered phase list, §4 is the per-phase detail (each phase uses the same template), §5 is cross-cutting concerns reused by every phase, and §6 is the traceability map proving every §11 module and every §1.4 success criterion is covered.

### Per-phase workflow (every phase, no exceptions)

```
developer implements  →  reviewer reviews  →  USER signs off  →  USER commits
```

1. **Developer agent** implements exactly the deliverables of one phase against the cited design sections, plus its tests.
2. **Reviewer agent** reviews for design conformance (names/schemas/section citations), correctness, DRY/generality invariants, and that the acceptance criteria actually pass.
3. **User** personally inspects the phase via its **User sign-off gate** and decides.
4. **User** commits — and only the user commits.

### Standing rule — no commit without consent

> **NOTHING is committed to git without the user's explicit consent.** No phase, no sub-task, no "quick fix" is committed by an agent. Every phase ends in a **User sign-off gate** that terminates in *"commit on user consent only"*. Agents may stage/show diffs and propose a commit message; the user runs (or explicitly authorizes) the commit. This rule overrides any other instinct to "wrap up" by committing.

---

## 2. Guiding principles for phasing

1. **Build bottom-up along the §11 import DAG.** §11 fixes the dependency direction: `pipeline`, `metrics`, `stats`, `matrix`, `runner`, `io_csv` import only `models`/`protocols`; adapters (`datasets/*`, `backends/*`) are selected by `config.py` factories and depended on by nobody upstream. We build leaves first (`models`, `protocols`), then each pure consumer, then adapters, then the wiring. **No phase may depend on a later phase.**
2. **Each phase is independently testable WITHOUT later phases.** Where a phase needs a collaborator that is not yet built (notably the ES backend), it is tested against a **fake/stub** that implements the relevant Protocol (`FakeBackend`, `FakeReranker`, tiny in-memory dataset fixture). This is exactly the seam the design promises (§1.4(3), §3.7): the pure core never imports an adapter.
3. **Keep each phase reviewable in one sitting.** One coherent module (or a tightly-coupled pair) plus its tests per phase. The single risky, large module — `backends/elasticsearch.py` — is split into two phases (BM25/execute first, then semantic/RRF/rerank).
4. **Isolate ES integration risk late and behind the backend adapter.** Everything except the two ES phases and the end-to-end phases is **pure unit-testable offline** (no Docker). Docker-dependent work (compose ES ≥ 8.15, `register_inference`, `ensure_index`, `bulk_index`, retriever execution) lands in Phases 10–12 only. By then the entire pure core is proven against `FakeBackend`, so ES work reduces to making the real backend satisfy the same contract.

---

## 3. Phase dependency graph & ordered list

```mermaid
flowchart TB
  P0["Phase 0<br/>Scaffolding<br/>pyproject/compose/tooling"] --> P1["Phase 1<br/>models.py + protocols.py"]
  P1 --> P2["Phase 2<br/>metrics.py"]
  P1 --> P3["Phase 3<br/>stats.py"]
  P1 --> P4["Phase 4<br/>fusion.py + rerank.py"]
  P1 --> P5["Phase 5<br/>pipeline.py + spec_for"]
  P4 --> P5
  P1 --> P6["Phase 6<br/>matrix.py + config.py"]
  P1 --> P7["Phase 7<br/>io_csv.py"]
  P2 --> P7
  P3 --> P7
  P1 --> P8["Phase 8<br/>datasets/wands.py"]
  P5 --> P9["Phase 9<br/>backends/elasticsearch.py — BM25 + execute (Docker ES)"]
  P6 --> P9
  P9 --> P10["Phase 10<br/>backends/elasticsearch.py — semantic + RRF + rerank + indexer (Docker ES)"]
  P2 --> P11["Phase 11<br/>runner.py + hatch CLI scripts (Docker ES)"]
  P3 --> P11
  P5 --> P11
  P6 --> P11
  P7 --> P11
  P8 --> P11
  P10 --> P11
  P11 --> P12["Phase 12<br/>full-WANDS end-to-end validation vs §1.4 (Docker ES)"]
```

| Phase | Title | Depends on | Docker ES? |
|------:|-------|-----------|:----------:|
| 0 | Scaffolding & tooling | — | builds compose, not run |
| 1 | `models.py` + `protocols.py` | 0 | no (pure) |
| 2 | `metrics.py` | 1 | no (pure) |
| 3 | `stats.py` | 1 | no (pure) |
| 4 | `fusion.py` + `rerank.py` (harness-side fallbacks) | 1 | no (pure) |
| 5 | `pipeline.py` (`SearchPipeline`, `PipelineSpec`, `spec_for`) | 1, 4 | no (FakeBackend) |
| 6 | `matrix.py` + `config.py` | 1 | no (pure) |
| 7 | `io_csv.py` | 1, 2, 3 | no (golden files) |
| 8 | `datasets/wands.py` | 1 | no (fixture) |
| 9 | `backends/elasticsearch.py` — BM25 + lifecycle + `execute` + `capabilities` | 5, 6 | **yes** |
| 10 | `backends/elasticsearch.py` — semantic + RRF + rerank + `ElasticsearchIndexer` | 9 | **yes** |
| 11 | `runner.py` + hatch CLI scripts (end-to-end on a small subset) | 2,3,5,6,7,8,10 | **yes** |
| 12 | Full-WANDS end-to-end validation vs success criteria §1.4 | 11 | **yes** |

Phases 1–8 are **pure offline unit work** (no Docker, no network). Phases 9–12 require dockerized ES ≥ 8.15.

---

## 4. Phases

Each phase uses the same template: **Objective · Deliverables · Depends on · Implementation notes · Test/acceptance criteria · Developer/reviewer responsibilities · User sign-off gate**.

---

### Phase 0 — Scaffolding & tooling

**Objective.** Stand up the buildable, lintable, testable skeleton with no business logic.

**Deliverables (only what this phase adds).**
- `pyproject.toml` — hatch project; an `eval` environment exposing scripts `wait-for-es`, `fetch-data`, `index`, `run` (wired to placeholders that import cleanly and exit non-destructively for now); a `dev`/`test` environment with `pytest`, `ruff`, `mypy`. Python 3.11+.
- `benchmark/` package skeleton: empty-but-importable `models.py`, `protocols.py`, `pipeline.py`, `fusion.py`, `rerank.py`, `metrics.py`, `stats.py`, `matrix.py`, `runner.py`, `io_csv.py`, `config.py`, `logging_setup.py`, `datasets/__init__.py`, `datasets/wands.py`, `backends/__init__.py`, `backends/elasticsearch.py` (modules may be stubs; names must match §11 exactly). `logging_setup.py` is the one module with real content this phase: console + `logs/run_{timestamp}.log` logging used everywhere instead of `print()`.
- `docker-compose.yml` — single-node ElasticSearch **≥ 8.15** (hard floor, §1.1), security relaxed for local eval, `9200` published, `ES_JAVA_OPTS=-Xms2g -Xmx2g` pinned.
- `.gitignore` — ignores `results/` and `dataset/` (CLAUDE.md "don't commit dataset/ or results/").
- `eval:fetch-data` script — downloads/copies WANDS `query.csv`/`product.csv`/`label.csv` into `dataset/wands/` (README "Dataset").
- `eval:wait-for-es` script — polls `${ES_URL}/_cluster/health?wait_for_status=yellow` (README).
- `config.yaml` — the §10 / README config matrix, verbatim axes (embedding_models, rerankers, `rrf_k_sweep` {10..100}, `variants`, `hybrid_rerank_k: 60`) and the §10 `stats` block verbatim: `bootstrap_B: 10000`, `ci_level: 0.95`, `alpha: 0.05`, `correction: holm`, `test: wilcoxon`, `wilcoxon_zero_method: wilcox`, `wilcoxon_correction: true`, `seed: 1234`.
- `tests/` layout + shared fixtures scaffold (see §5).

**Depends on.** —

**Implementation notes.** Module/file names are load-bearing — copy them from §11 and README "Repo layout" exactly. Compose image tag must be ≥ 8.15; do not pin below the floor (§1.1, README prereqs). `config.yaml` mirrors §10 exactly, including the comment that `top_n` is a `task_settings` key and `ci_level` is not a gate. No metric/stat/pipeline code in this phase.

**Test / acceptance criteria.** *(pure / offline)*
- `hatch env create eval` and `hatch env show` succeed; `hatch run dev:ruff check` and `hatch run dev:mypy benchmark` run clean over the (empty) package.
- `python -c "import benchmark.models, benchmark.protocols, benchmark.pipeline, benchmark.fusion, benchmark.rerank, benchmark.metrics, benchmark.stats, benchmark.matrix, benchmark.runner, benchmark.io_csv, benchmark.config, benchmark.datasets.wands, benchmark.backends.elasticsearch"` imports with no error (skeleton importability).
- `docker compose config` validates; image tag ≥ 8.15. (Compose is validated, not necessarily started, in this phase.)
- **`config.yaml` stats block matches §10 verbatim:** assert the keys/values `bootstrap_B: 10000`, `alpha: 0.05`, `correction: holm`, `test: wilcoxon`, `wilcoxon_zero_method: wilcox`, `wilcoxon_correction: true`, and a `seed` are present (these are load-bearing for Phases 3/6/7/11 reproducibility); assert the inline comments are present that `ci_level` is the **unadjusted per-comparison effect-size CI, not a gate** and that `top_n` is a `task_settings` key. Catches drift at sign-off rather than in Phase 11.
- `.gitignore` excludes `results/` and `dataset/`; `git status` shows neither after creating them.

**Developer / reviewer responsibilities.** Developer creates the tree, tooling, compose, and `config.yaml`. Reviewer checks every filename against §11/README, the ES image floor, the gitignore entries, and that lint/type/import all pass.

**User sign-off gate.** Inspect `pyproject.toml` (envs + the 4 eval scripts), `docker-compose.yml` (ES tag ≥ 8.15, heap), `.gitignore`, `config.yaml` against §10, and the empty package tree against §11. Approve → **commit on user consent only.**

---

### Phase 1 — `models.py` + `protocols.py` (the seams)

**Objective.** Define all pure data models and the Protocol seams the entire harness depends on.

**Deliverables.**
- `benchmark/models.py` — frozen dataclasses / enums: `Query`, `Document`, `Qrel`, `ScoredDoc`, `RankedResult`, `FieldRole`, `FieldSpec`, `FieldSchema`, `IndexMapping`, `InferenceTaskType`, `InferenceEndpoint`, `BackendCapabilities`. **The pipeline-config types (`StageCfg`/`FuseCfg`/`RerankCfg`/`PipelineSpec`) are NOT here — they live in `pipeline.py` (Phase 5) per §11/§3.6.**
- `benchmark/protocols.py` — `Dataset`, `SearchBackend`, `RetrieverSpec`, `EmbeddingModel`, `Reranker`, `Indexer` Protocols.

> Note: §11 lists `PipelineSpec`/`FuseCfg`/`RerankCfg`/`StageCfg`/`spec_for` under `pipeline.py`. Keep them in `pipeline.py` per §11; only the §3.1 plain data models (`Query`/`Document`/`Qrel`/`ScoredDoc`/`RankedResult`/`FieldSchema`/`InferenceEndpoint`) live in `models.py`. Shared enums/dataclasses referenced by both (`FieldRole`, `InferenceTaskType`, `IndexMapping`, `BackendCapabilities`) go in `models.py`. Reviewer enforces this split against §11's per-module comments.

**Depends on.** Phase 0.

**Implementation notes.** Exact field names/types per §3.1–§3.6: `Query(query_id, text, query_class=None)`, `Qrel(query_id, doc_id, gain:int)`, `ScoredDoc(doc_id, score)`, `RankedResult(query_id, docs)`. `position` is **not** a field on `ScoredDoc` (§3.1 — derived at CSV write time so it cannot drift). `FieldSchema.search_text_field` and `rerank_field` both default to `"search_text"` (§3.2, §5.1). `IndexMapping.sem_field(model_id)` getter (§3.5). `InferenceEndpoint` carries **separate** `service_settings` and `task_settings` maps (§3.4 — `top_n` lives in `task_settings`). All dataclasses `frozen=True`. Protocols are structural (`typing.Protocol`), no implementation.

**Test / acceptance criteria.** *(pure / offline)*
- Construction + immutability tests for every dataclass (frozen → `FrozenInstanceError` on mutate).
- `FieldSchema()` defaults `search_text_field == rerank_field == "search_text"`.
- A trivial in-test class structurally satisfies each Protocol (mypy `--strict` passes; a runtime `isinstance` check against `runtime_checkable` Protocols where used).
- `mypy benchmark/models.py benchmark/protocols.py` clean.

**Developer / reviewer responsibilities.** Developer writes the models/Protocols verbatim from §3. Reviewer diffs every field name/type/default against §3.1–§3.6 and confirms the `models.py`-vs-`pipeline.py` split matches §11.

**User sign-off gate.** Inspect `models.py`/`protocols.py` field-by-field against §3; confirm `ScoredDoc` has no `position`, `task_settings` is separate, defaults are `"search_text"`. Approve → **commit on user consent only.**

---

### Phase 2 — `metrics.py`

**Objective.** Implement the four per-query metrics and the qrel index, exactly per §7.

**Deliverables.**
- `benchmark/metrics.py` — `QrelIndex` (`dict[query_id, dict[doc_id, gain]]`), `MetricVector`, `Evaluator` with `score_run(results) -> per-query MetricVectors` (joining each `RankedResult` to qrels by `query_id`).

**Depends on.** Phase 1.

**Implementation notes (§7).**
- **avg_relevance** = `(1/10)·Σ_{i=1..10} gain(d_i)`; lists shorter than 10 zero-padded at gain level, **denominator stays 10**.
- **ndcg@10 (graded):** `DCG@10 = Σ (2^{gain}−1)/log2(i+1)`; **IDCG truncated to top-10 of the ideal ordering** (`Σ_{i=1..min(10,#judged)}`), not over all judged gains; `nDCG=0` when `IDCG=0`.
- **relevant iff gain ≥ 1** (Partial or Exact).
- **precision@10** = `|relevant ∩ top10| / 10` (denominator fixed at 10).
- **recall@10** = `|relevant ∩ top10| / R`, `R = #relevant judged over all of label.csv`. **`R=0` → `recall = NaN`** in the in-memory `MetricVector` (excluded from aggregation/deltas, §8.1). The empty-cell serialization is Phase 7's concern, not here.
- Unjudged docs treated as `gain=0` (§7).

**Test / acceptance criteria.** *(pure / offline)*
- **Hand-computed** values on tiny fixtures for all four metrics, including: a perfect ranking (nDCG=1.0); a short list (< 10 docs) confirming zero-padding and fixed-10 denominators; **R=0 → recall is `NaN`**; a query with **> 10 relevant docs** confirming IDCG truncation to 10 (not deflated); `IDCG=0 → nDCG=0`.
- A graded case with mixed `{0,1,2}` gains where DCG/IDCG are computed by hand in the test and asserted to a tight tolerance.
- Unjudged doc in the ranked list scored as gain 0.

**Developer / reviewer responsibilities.** Developer implements per §7. Reviewer recomputes at least the nDCG and IDCG-truncation cases independently and confirms the NaN policy.

**User sign-off gate.** Inspect the hand-computed test table (especially IDCG truncation and recall-NaN). Approve → **commit on user consent only.**

---

### Phase 3 — `stats.py`

**Objective.** Implement the `Comparator`: bootstrap CI, Wilcoxon/permutation p-value, Holm decision, and degenerate-set handling — one coherent regime (§8).

**Deliverables.**
- `benchmark/stats.py` — `Comparator(stats_cfg).compare(baseline_vectors, variant_vectors)` returning per-`(variant, metric)` rows with `delta, delta_ci_lo, delta_ci_high, significant, p_value` (+ degenerate-set `note`). Holm applied across the family of `(variant × metric)` tests within a run.

**Depends on.** Phase 1.

**Implementation notes (§8).**
- **Pairing (§8.1):** paired by `query_id`; recall@10 pairs further restricted to non-`NaN` queries; detection is by in-memory `NaN`, **never** by re-reading CSV.
- **Degenerate sets (§8.1 table), short-circuited before any scipy/bootstrap call:** *empty paired set* → `delta`/CI empty, `p_value=1.0`, `significant=false`, `note=empty_paired_set`; *all-zero deltas* → `delta=0.0`, CI `0.0/0.0`, `p_value=1.0`, `significant=false`, `note=all_zero_delta`.
- **CI (§8.2):** percentile bootstrap, **B=10000**, seeded `numpy.random.default_rng(seed)`, resample **paired query indices** (preserve pairing), recompute mean δ, take **2.5/97.5** percentiles. This CI is **effect-size context only, not a gate**.
- **p_value (§8.2):** two-sided **Wilcoxon signed-rank**, `zero_method="wilcox"`, `correction=True` (both recorded); seeded **paired-permutation** test selectable as primary via `stats.test`. Raw p written to CSV.
- **Holm (§8.3):** family = all `(variant × metric)` tests in the run, `α=0.05`, step-down on **raw** p; `significant` is exactly the Holm reject/retain outcome. **The CI is not a second gate and may legitimately disagree** — do not reconcile.

**Test / acceptance criteria.** *(pure / offline)*
- **Seeded determinism:** same `seed` → byte-identical `delta_ci_lo/high` across repeated runs; different seed → (generally) different CI; recorded B=10000 honored.
- **Degenerate sets:** empty paired set and all-zero deltas each produce the exact §8.1-table outputs **without** calling scipy (assert via monkeypatch/spy that the bootstrap/test is never invoked).
- **Holm:** a hand-constructed family of raw p-values verifies step-down reject/retain (including the "first failure stops the sequence" behavior) and that `significant` matches.
- **CI vs significant may disagree:** a constructed case where an unadjusted CI excludes 0 but Holm retains (and vice versa) — assert no exception, both reported as designed.
- Recall pairing excludes `NaN` queries; Wilcoxon zero/tie params are passed through and recorded.

**Developer / reviewer responsibilities.** Developer implements per §8.1–§8.3. Reviewer verifies the short-circuits fire *before* scipy, the seeded RNG is `default_rng(seed)`, Holm is on raw p, and that no "adjusted alpha" or CI-as-gate logic sneaks in.

**User sign-off gate.** Inspect the Holm step-down test, the seeded-CI determinism test, and the degenerate-set table assertions. Approve → **commit on user consent only.**

---

### Phase 4 — `fusion.py` + `rerank.py` (harness-side fallbacks)

**Objective.** Implement the pure-Python windowed RRF and rerank fallbacks used by non-server-side backends (§3.7).

**Deliverables.**
- `benchmark/fusion.py` — `fuse_rrf_local(lists, *, rank_constant, rank_window_size)`.
- `benchmark/rerank.py` — `rerank_local(reranker, query, candidates, *, rank_window_size, doc_text)`.

**Depends on.** Phase 1.

**Implementation notes (§3.7).**
- `fuse_rrf_local`: **truncate each input list to its top `rank_window_size` BEFORE fusing**, then `score(d) = Σ 1/(rank_constant + rank_d)`, rank **1-based** within the truncated list; return merged list sorted by fused score **desc, tie-break doc_id** (§9.1). Must mirror ES `rrf` window semantics — dropping the window (fusing full lists) is the explicit v2 bug to avoid.
- `rerank_local`: take only the top `rank_window_size` candidates, call reranker over `(query.text, doc_text(doc_id))`, re-sort by model score; candidates **beyond the window keep input order, appended after the reranked head** (as ES does).

**Test / acceptance criteria.** *(pure / offline)*
- **Hand-computed RRF:** two short lists with a known overlap, `rank_constant=10`, small window → assert exact fused scores and order, including the **doc_id tie-break** on equal fused score.
- **Window truncation:** a doc present only beyond `rank_window_size` in every list is excluded from fusion (proves the truncate-before-fuse rule).
- **rerank_local:** with a fake reranker returning fixed scores, the top-W head is re-sorted by model score and the tail (> W) retains input order appended after the head.

**Developer / reviewer responsibilities.** Developer implements both helpers. Reviewer confirms windowing matches ES semantics and the tie-break is on `doc_id`.

**User sign-off gate.** Inspect the hand-computed RRF test and the window-truncation test. Approve → **commit on user consent only.**

---

### Phase 5 — `pipeline.py` (`SearchPipeline`, `PipelineSpec`, `spec_for`)

**Objective.** Implement the single DRY pipeline and the variant→spec composition, tested entirely against a `FakeBackend`.

**Deliverables.**
- `benchmark/pipeline.py` — `StageCfg`/`FuseCfg`/`RerankCfg`/`PipelineSpec` (the pipeline-config dataclasses per §3.6), `SearchPipeline` (`plan()` + `run()`), and `spec_for(variant_cfg, mapping)` (§4).

**Depends on.** Phases 1, 4.

**Implementation notes (§3.6, §3.7, §4).**
- `plan()` is **pure composition**: retrieve → [fuse] → [rerank], **no per-variant branching beyond presence/absence** of `fuse`/`rerank`. Uses server-side combinators when `capabilities()` allows, else wraps the §3.7 harness-side helpers (Phase 4) with the **same** `rank_constant`/`rank_window_size`.
- `run()` builds the plan once, then per query calls `backend.execute(plan, q, top_k=...)` (binding happens in the backend, §3.3).
- `spec_for` matches §4 exactly: `use_bm25` → bm25 stage on `mapping.search_text_field`; `embedding_model_id` → semantic stage on `mapping.sem_field(id)`; `fuse` → `FuseCfg(rrf_k, window)`; `reranker_id` → `RerankCfg(reranker_id, mapping.rerank_field, window)`. **`spec_for` never performs k-selection** — `v.rrf_k` is already a concrete int (§4, §8.0a).

**Test / acceptance criteria.** *(pure / offline, via `FakeBackend`)*
- **All 6 variant specs compose:** `spec_for` produces the §4-table `PipelineSpec` for `bm25`, `semantic`, `hybrid`, `bm25_rerank`, `semantic_rerank`, `hybrid_rerank`; assert retrievers/fuse/rerank presence per the table.
- **DRY proof:** a `FakeBackend` (server-side caps true) and a second fake (caps false → exercises §3.7 fallbacks) both run all 6 specs through the **same** `run()` with no variant branching; outputs are `RankedResult`s.
- **Capability gating:** with `server_side_rrf=false`/`server_side_rerank=false`, `plan()` routes through `fuse_rrf_local`/`rerank_local` with identical window/constant; with caps true it composes server-side combinators.
- `run()` yields one `RankedResult` per query in query order.

**Developer / reviewer responsibilities.** Developer implements pipeline + `spec_for` and the `FakeBackend` test double (promote to shared fixture, §5). Reviewer confirms zero per-variant branching and that `spec_for` does no selection.

**User sign-off gate.** Inspect `spec_for` against the §4 table and the FakeBackend test covering all 6 variants under both capability modes. Approve → **commit on user consent only.**

---

### Phase 6 — `matrix.py` + `config.py`

**Objective.** Implement deterministic matrix expansion (baseline first), the `best_per_model` selection, and config load/resolve.

**Deliverables.**
- `benchmark/matrix.py` — `VariantCfg`, `ResolvedConfig`, `expand_matrix(cfg)`, `resolve_hybrid_rerank_best_per_model(cfg, per_query)`.
- `benchmark/config.py` — YAML/JSON load + resolve, `${VAR}` env substitution, factories `load_dataset`/`make_backend`, and `ConfigInferenceModel` (implements both `EmbeddingModel` and `Reranker` Protocols).

**Depends on.** Phase 1. *(Factories reference adapters by name only; the adapters themselves arrive in Phases 8/9–10. `make_backend`/`load_dataset` may dispatch on `kind`/`name` with the ES/WANDS branches stubbed/imported lazily so this phase stays offline.)*

**Implementation notes (§8.0a, §10, §11).**
- **Expansion order (§10), `bm25` emitted FIRST:** `bm25`(1) → `semantic`(per model) → `hybrid`(models × `rrf_k_sweep`) → `bm25_rerank`(per reranker) → `semantic_rerank`(models × rerankers) → `hybrid_rerank`.
- **`hybrid_rerank` (§10/§8.0a):** if `hybrid_rerank_k` is an **int** (default 60) → emit models × rerankers at that fixed k as static rows; if `best_per_model` → emit **no** `hybrid_rerank` rows (deferred to §8.0a phase).
- `resolve_hybrid_rerank_best_per_model`: per model, **argmax mean nDCG@10** over that model's `hybrid` rows; **tie-break: smallest k, then lexicographically smallest variant id**; **seed-independent, deterministic** (§1.4(2)); emits one row per `(model, reranker)` at the chosen k. Operates **only on in-memory `MetricVector`s** passed in — adds **no adapter dependency** (§11).
- `expand_matrix` is a **pure deterministic function**; the data dependency is confined to the one named selection phase.
- Variant ids match §9 examples (e.g. `hybrid__e5-small__k60`).
- `config.py`: `${VAR}` resolved at load (secrets never in file); `ci_level` parsed but is **not** a gate; records correction/test/zero-tie params for run metadata.

**Test / acceptance criteria.** *(pure / offline)*
- **Expansion counts & order:** from the §10 `config.yaml` (3 embedding models, 2 rerankers, 10-step k-sweep), assert exact variant **count per family** and that **`bm25` is index 0**; order matches §10.
- **best_per_model deferral:** with `hybrid_rerank_k: best_per_model`, `expand_matrix` emits **zero** `hybrid_rerank` rows; with an int, it emits `models×rerankers` rows at that k.
- **Deterministic selection:** `resolve_hybrid_rerank_best_per_model` on hand-built `MetricVector`s returns the argmax-mean-nDCG k with smallest-k then lexicographic tie-break; identical output across runs (seed-independent).
- **Config:** `${VAR}` substitution from env; missing required key errors clearly; `ConfigInferenceModel` satisfies both Protocols (mypy + structural test).
- **Factory dispatch (offline, no adapter import):** assert the `name`→factory registry maps `wands`→the WANDS dataset factory target and `elasticsearch`→the ES backend factory target (as a dotted-path string or lazy importer, **not** by importing the adapter module), and that an unknown `name`/`kind` raises a clear error. The dispatch *logic* is verified here; live resolution (actually importing + constructing the adapter) is deferred to Phase 11.

**Developer / reviewer responsibilities.** Developer implements expander, selector, config loader, factories. Reviewer verifies baseline-first ordering, exact counts, the two-mode `hybrid_rerank` behavior, and the tie-break.

**User sign-off gate.** Inspect the expansion-count/order test, the best_per_model deferral test, and the tie-break determinism test. Approve → **commit on user consent only.**

---

### Phase 7 — `io_csv.py`

**Objective.** Write the three CSV artifact types + run-config JSON with **exact, fixed** schemas (§9), verified by golden files.

**Deliverables.**
- `benchmark/io_csv.py` — `write_result_csv`, `write_metrics_csv`, `write_comparison_csv`, `write_run_config`.

**Depends on.** Phases 1, 2, 3 (consumes `RankedResult`, `MetricVector`, comparator rows).

**Implementation notes (§9, CLAUDE.md invariants).**
- Filenames: `result_{variant}_{timestamp}.csv`, `metrics_{variant}_{timestamp}.csv`, `comparison_{baseline}_{variant}_{timestamp}.csv`, `run_config_{timestamp}.json`; `{timestamp}` = single per-run UTC `YYYYMMDDTHHMMSSZ`.
- **Exact headers / field order (do not rename/reorder):**
  - result → `query_id,product_id,score,position`
  - metrics → `query_id,avg_relevance,ndcg@10,recall@10,precision@10`
  - comparison → `variant,metric,delta,delta_ci_lo,delta_ci_high,significant,p_value`
- **`position` derived** as the 1-based index into `RankedResult.docs` at write time (§3.1, §9); ≤ `top_k` rows/query.
- **recall@10 `NaN` → empty field** (two adjacent commas, no quoting) per §7/§9. `significant` ∈ {`true`,`false`} lowercase.
- **Degenerate comparison rows (§8.1/§9):** empty paired set → `delta`/CI cells empty, `p_value=1.0`, `significant=false`; all-zero → `0.0`/`0.0`/`0.0`, `p_value=1.0`.
- `write_run_config` serializes the fully-resolved config + seed per §9.1 (expanded variants incl. any selected k, selection metric + `hybrid_rerank_selection_bias` flag, B, fixed CI level 2.5/97.5, α, family size m, correction, test + zero/tie params, degenerate notes, dataset/ES/endpoint versions, cutoff, seed). **No per-test adjusted alpha** (Holm defines none).

**Test / acceptance criteria.** *(pure / offline, golden files)*
- **Exact headers** for all three CSVs asserted byte-for-byte against committed golden files.
- **position derivation:** `docs[0]` → `position=1`, ascending; ≤ `top_k` rows.
- **recall empty cell:** a `NaN` recall serializes as an empty field (golden row shows two adjacent commas).
- **Degenerate rows** serialize exactly per the §8.1 table.
- **run_config JSON** round-trips and contains every §9.1 field (and omits any "adjusted alpha").

**Developer / reviewer responsibilities.** Developer implements writers + golden fixtures. Reviewer diffs headers char-for-char against §9 and CLAUDE.md, and checks the NaN→empty and degenerate serializations.

**User sign-off gate.** Inspect the golden CSV headers/rows and the run_config JSON keys against §9/§9.1. Approve → **commit on user consent only.**

---

### Phase 8 — `datasets/wands.py`

**Objective.** Implement the WANDS dataset adapter against a tiny sample fixture (no full corpus, no network).

**Deliverables.**
- `benchmark/datasets/wands.py` — `WandsDataset` implementing `Dataset`: `queries()`, `documents()`, `qrels()`, `field_schema()`, with `name`/`version`.

**Depends on.** Phase 1.

**Implementation notes (§3.2, §5.1, §7, README "Dataset").**
- Parse **tab-separated** `query.csv` (`query_id, query, query_class`), `product.csv` (`product_id, product_name, product_description, product_features, ...`), `label.csv` (leading `id` column + `query_id, product_id, label` where label is the **string** `Exact/Partial/Irrelevant`).
- **label→gain mapping applied at qrel emission:** `Exact=2, Partial=1, Irrelevant=0` (so the rest of the harness only sees integer gains, §3.2/§7).
- **`search_text` concatenation:** name + description (+ features) into the canonical `search_text` field in each `Document`'s field bag (§5.1), so every variant ranks the same input text.
- `field_schema()` returns the §5.1 roles (`product_id`→ID, name/description→bm25+semantic_source, features→bm25, class/category→bm25, ratings→numeric); `search_text_field`/`rerank_field` = `"search_text"`.
- `documents()` is **streamed** (generator) for large corpora.

**Test / acceptance criteria.** *(pure / offline, tiny fixture)*
- Parse a **tiny committed WANDS sample** (a handful of rows of each file): assert `Query`/`Document`/`Qrel` objects round-trip with correct fields.
- **label→gain:** `Exact→2, Partial→1, Irrelevant→0`; assert qrels carry integer gains only.
- **search_text concat:** a document's `search_text` equals the expected name+description(+features) concatenation.
- **field_schema** matches §5.1 roles; `search_text_field == rerank_field == "search_text"`.
- TSV parsing handles the leading `id` column in `label.csv`.

**Developer / reviewer responsibilities.** Developer implements the adapter + the tiny sample fixture. Reviewer verifies TSV (not CSV-comma) parsing, the gain mapping at emission, and the concat formula.

**User sign-off gate.** Inspect the gain-mapping and search_text-concat tests against §5.1/§7 and the sample fixture. Approve → **commit on user consent only.**

---

### Phase 9 — `backends/elasticsearch.py` — BM25 + lifecycle + `execute` + `capabilities` (Docker ES)

**Objective.** Implement the ES backend skeleton end of the contract: lifecycle, the BM25 retriever primitive, `execute` (query binding + tie-break), and `capabilities`. Semantic/RRF/rerank deferred to Phase 10.

**Deliverables.**
- `benchmark/backends/elasticsearch.py` (part 1) — `ElasticsearchBackend` implementing: `register_inference`, `ensure_index`, `bulk_index`, `bm25(...)`, `execute(...)`, `capabilities()`, and the ES `RetrieverSpec` representation. `semantic`/`fuse_rrf`/`rerank` may raise `NotImplementedError` placeholders this phase.

**Depends on.** Phases 5, 6.

**Implementation notes (§3.3, §9.1, §5).**
- **`register_inference` idempotent create-or-get** → `PUT _inference/{task_type}/{inference_id}`, **emitting BOTH `service_settings` and `task_settings`** separately (§3.4); returns `inference_id`.
- **`bm25`** → `{ "standard": { "query": { "match": { "search_text": "$Q" } } } }` plan (query-independent; `$Q` slot only).
- **`execute` (§3.3):** binds `query.text` into **every** query slot, runs, returns docs **score desc with deterministic tie-break on `doc_id`** (§9.1); ≤ `top_k`.
- **`capabilities`** reports `server_side_rrf`/`server_side_rerank`/`semantic_query`; **`semantic_query` is the hard 8.15 gate** (§1.1/§3.3). Optional implicit `match`-on-`semantic_text` form only when cluster ≥ 8.18.
- `ensure_index`/`bulk_index` honor idempotency (`_id = product_id`, §3.5 step 3).

**Test / acceptance criteria.** *(requires dockerized ES ≥ 8.15)*
- `docker compose up -d` + `eval:wait-for-es`; against the small fixture corpus: `ensure_index` + `bulk_index` then a **BM25 `execute`** returns a `RankedResult` ordered score-desc with **doc_id tie-break** verified on a constructed tie.
- **Query binding:** the `$Q` slot is filled from `query.text` (assert a query with a distinctive token retrieves the matching doc).
- **`register_inference`** emits separate `service_settings`/`task_settings` (assert against the registered endpoint body) and is idempotent (second call no-ops/returns same id).
- **`capabilities().semantic_query` is true** on the ≥ 8.15 cluster.
- *(Offline unit slice where feasible: the plan-building of `bm25` and `execute`'s binding/tie-break logic can also be unit-tested without a live cluster via a thin response stub; the live-ES test is the acceptance gate.)*

**Developer / reviewer responsibilities.** Developer implements part 1 against a live ES. Reviewer verifies the tie-break, the `service`/`task` split in registration, and that `capabilities` gates on 8.15.

**User sign-off gate.** Inspect the live BM25 round-trip test output, the tie-break test, and `register_inference` body split. Approve → **commit on user consent only.**

---

### Phase 10 — `backends/elasticsearch.py` — semantic + RRF + rerank + `ElasticsearchIndexer` (Docker ES)

**Objective.** Complete the ES backend: semantic retriever, server-side RRF fuse, `text_similarity_reranker`, and the indexer (semantic_text + copy_to).

**Deliverables.**
- `benchmark/backends/elasticsearch.py` (part 2) — `semantic(field)`, `fuse_rrf(children, rank_constant, rank_window_size)`, `rerank(child, inference_id, field, rank_window_size)`, and `ElasticsearchIndexer` implementing `Indexer.build(...)` → `IndexMapping`.

**Depends on.** Phase 9.

**Implementation notes (§3.5, §5.2, §5.3).**
- **`semantic`** → explicit `{ "semantic": { "field": ..., "query": "$Q" } }` (default, version-robust, ES ≥ 8.15); optional implicit `match` form only when caps ≥ 8.18.
- **`fuse_rrf`** → `{ "rrf": { "retrievers": [...], "rank_constant": $k, "rank_window_size": $W } }`.
- **`rerank`** → `{ "text_similarity_reranker": { "retriever": <child>, "field": "search_text", "inference_id": ..., "inference_text": "$Q", "rank_window_size": $W } }`. **`inference_text` is REQUIRED and injected by `execute()`** — not auto-filled from the child query (§3.3 design note, §5.3).
- **Indexer lifecycle (§3.5 strict order):** (1) `register_inference` for each `EmbeddingModel` **before** `ensure_index` (a `semantic_text` field can't map before its `inference_id` exists); (2) translate `field_schema` → mapping with `search_text` `text` field carrying **`copy_to` → one `semantic_text` field per model** (`copy_to` lives on the **source `text` field**, §5.2 — not `copy_to_source`); each `semantic_text` field sets `inference_id` explicitly; (3) stream `bulk_index` (ES embeds at ingest); (4) return `IndexMapping` with per-model `sem_field` names. **Rerankers are NOT registered here** (lazy at run, §8 R0).

**Test / acceptance criteria.** *(requires dockerized ES ≥ 8.15)*
- **Indexer:** `build` registers an embedding endpoint, creates the mapping with `search_text.copy_to` → `semantic_text` field(s), bulk-indexes the fixture, returns an `IndexMapping` whose `sem_field(model)` resolves; assert the **`copy_to` is on the source field** and `inference_id` is set on each semantic field.
- **semantic execute:** a semantic query returns a `RankedResult` (using a local ES inference model, e.g. ELSER/E5, to avoid external keys).
- **server-side RRF:** `fuse_rrf` over BM25 + semantic executes and returns fused results; `capabilities().server_side_rrf` true.
- **rerank:** `text_similarity_reranker` plan carries an injected `inference_text` (assert the bound request body), and `execute` returns reranked results; **`W <= task_settings["top_n"]`** holds for the registered reranker.
- **Equivalence (cross-check to Phase 4):** for a given `PipelineSpec`, server-side RRF ranking matches `fuse_rrf_local` on the same candidate lists/window (sanity check of §3.7's "identical ranking" claim on a small case).

**Developer / reviewer responsibilities.** Developer completes the backend + indexer against live ES, preferring local inference models for tests. Reviewer verifies the strict lifecycle order, `copy_to` placement, the required `inference_text` injection, and the semantic-query 8.15 form.

**User sign-off gate.** Inspect the indexer mapping (copy_to + per-model semantic_text), the rerank `inference_text` injection test, and the RRF equivalence cross-check. Approve → **commit on user consent only.**

---

### Phase 11 — `runner.py` + hatch CLI scripts (end-to-end on a small subset, Docker ES)

**Objective.** Wire the single execution path (§8.0) and the `eval:*` CLI; produce all three CSV types + run_config on a small-subset live run.

**Deliverables.**
- `benchmark/runner.py` — `ExperimentRunner.run(cfg)` exactly per §8.0 (incl. R0 lazy reranker registration + `W <= top_n` assert, baseline-first, the §8.0a best_per_model phase, in-memory metric vectors, comparator pass, run_config write).
- Hatch `eval:index` and `eval:run` scripts fully wired (entry points), including `--config` and `--dry-run` (README "Useful invocations").
- **Factory-to-adapter binding (owned here):** wire `config.py`'s `load_dataset`/`make_backend` factories — stubbed/lazily-imported in Phase 6 to stay offline — to the **real** adapters now that they exist: `dataset.name == "wands"` → `WandsDataset` (Phase 8), `backend.kind == "elasticsearch"` → `ElasticsearchBackend` (Phase 10). This is the concrete moment the §11 factories resolve to live adapters.

**Depends on.** Phases 2, 3, 5, 6, 7, 8, 10.

**Implementation notes (§8.0, §6, §1.4(4)).**
- **Setup prelude before any `run_one` (first five lines of §8.0):** `runner.run()` must (a) call `ElasticsearchIndexer.build(dataset, backend, cfg.embedding_models)` to obtain the `IndexMapping`, (b) freeze `queries = list(dataset.queries())` as the single shared query set, and (c) build `QrelIndex(dataset.qrels())` once — all before the first `run_one`. Do not under-build this setup phase.
- **`eval:index` entry point** invokes this same indexer path: `ElasticsearchIndexer.build` in §3.5 strict order — **register embedding endpoints → `ensure_index` (semantic_text + `copy_to`) → `bulk_index`**. (`eval:index` builds/populates the index; `eval:run` consumes it.)
- **One `run_one` path for every variant — baseline included** (DRY guarantee, §8.0): R0 register reranker if `reranker_id` and `assert v.window <= ep.task_settings["top_n"]`; `spec_for`; `pipeline.run` over the **frozen shared query set**; `write_result_csv`; `Evaluator.score_run`; `write_metrics_csv`; stash per-query vectors keyed by `v.id`.
- **Baseline materialized first** (§6) so every comparison has its paired reference in memory.
- **§8.0a phase** runs only when `hybrid_rerank_k == "best_per_model"`, after all static variants, feeding selected `VariantCfg`s through the **same** `run_one`.
- Comparator pass: for each non-baseline variant, `Comparator(cfg.stats).compare(baseline, metrics)` → `write_comparison_csv`; then `write_run_config`.
- `--dry-run` prints the expanded variant list and runs nothing (README).

**Test / acceptance criteria.** *(requires dockerized ES ≥ 8.15; small subset)*
- **Factory binding:** a registration/import test asserts `load_dataset` resolves `name == "wands"` → `WandsDataset` and `make_backend` resolves `kind == "elasticsearch"` → `ElasticsearchBackend` (the binding is owned and tested here, not merely implied). The dataset-side resolution can be asserted offline; the backend-side is exercised by the live run below.
- **End-to-end small-subset run** over the fixture corpus + a trimmed config (few queries, 1 embedding model, 1 reranker, short k-sweep) produces **all three CSV types** (`result_*`, `metrics_*`, `comparison_bm25_*`) **and** `run_config_*.json`, with **baseline first** and the comparator **not** comparing baseline to itself.
- **`eval:index` on the fixture corpus** produces a populated index carrying one `semantic_text` field per configured embedding model (verify the mapping + a non-zero doc count), exercising the §3.5 register→ensure_index→bulk_index order.
- **`--dry-run`** prints the expanded variant list (baseline first) and writes nothing.
- **R0 assertion:** a config with `top_n < rank_window_size` fails the `W <= top_n` assert before running that rerank variant.
- **best_per_model path:** with the opt-in mode, the §8.0a phase runs after hybrids, selected k recorded in run_config with `hybrid_rerank_selection_bias: true`.
- **DRY inspection:** code review confirms every variant traverses one `run_one` (no per-variant branches).

**Developer / reviewer responsibilities.** Developer wires the runner + CLI and the small-subset integration test. Reviewer verifies the §8.0 path is literally one code path, baseline-first, the R0 assert, and the §8.0a ordering.

**User sign-off gate.** Inspect a sample `results/` directory from the small-subset run (all three CSV types + run_config), the dry-run output, and the runner's single-path structure. Approve → **commit on user consent only.**

---

### Phase 12 — Full-WANDS end-to-end validation vs success criteria §1.4 (Docker ES)

**Objective.** Run the full matrix on full WANDS and validate against the four §1.4 success criteria.

**Deliverables.**
- No new modules. A validation checklist/run record (kept out of `results/` if it would otherwise be gitignored; documented in the PR/sign-off, not as a report file) demonstrating §1.4 compliance. Any defects found feed back as fixes to the owning phase's module.

**Depends on.** Phase 11.

**Implementation notes (§1.4, §9.1).**
- Full path per README: `docker compose up -d` → `eval:wait-for-es` → `eval:fetch-data` → `eval:index` → `eval:run` on the §10 `config.yaml`.
- Validate the four §1.4 criteria: **Correctness** (all three CSV types with exact §9 schemas for every matrix variant; one §8.3 error-control regime); **Reproducibility** (same config + seed → identical metrics/stats modulo pinned backend nondeterminism; `best_per_model` selection seed-independent, §8.0a); **Generality** (verify by code inspection that pipeline/evaluator/stats never import `datasets/*` or `backends/*` — §11 checklist); **DRY** (variants are config rows through one `run_one`/`SearchPipeline`).

**Test / acceptance criteria.** *(requires dockerized ES ≥ 8.15; full corpus)*
- Full matrix run completes; **schema lint** over every produced CSV asserts exact headers/field order (§9) for every variant.
- **Reproducibility:** two runs with the same seed produce identical `metrics_*`/`comparison_*` stats columns (modulo documented ES nondeterminism mitigated by the doc_id tie-break); `run_config_*.json` captures the seed and all §9.1 fields.
- **Generality check:** an import-graph test asserts `pipeline`/`metrics`/`stats`/`matrix`/`runner`/`io_csv` import only `models`/`protocols` (plus the stdlib-only `logging_setup`) and **no** adapter imports — automating the §11 invariant.
- **DRY check:** automated/inspection confirmation of the single execution path.

**Developer / reviewer responsibilities.** Developer executes the full run and the validation suite; files fixes against the owning phase modules for any defect. Reviewer signs that all four §1.4 criteria are demonstrably met.

**User sign-off gate.** Inspect the full-run artifacts, the reproducibility diff, and the import-graph/DRY checks against §1.4. Approve → **commit on user consent only.**

---

## 5. Cross-cutting concerns (reused by every phase)

### Test layout & fixtures
```
tests/
  conftest.py            # shared fixtures
  fixtures/
    wands_sample/        # tiny TSV sample: a few rows of query.csv/product.csv/label.csv
    golden/              # golden CSV headers/rows + run_config JSON for io_csv tests
  unit/                  # Phases 1–8 pure tests (no Docker)
  integration/           # Phases 9–12 live-ES tests (marked, skipped without ES)
```
- **`FakeBackend`** (built in Phase 5, promoted to `conftest.py`): an in-memory `SearchBackend` returning deterministic `RankedResult`s, configurable `capabilities()` (server-side caps on/off) to exercise both the server-side and §3.7 harness-side paths. **`FakeReranker`/`FakeEmbeddingModel`** structurally satisfy the descriptor Protocols.
- **Tiny WANDS sample fixture** (Phase 8): a handful of rows per file, hand-labeled so metrics are hand-computable; reused by Phases 9–11 as the small live corpus.
- Integration tests are marked (e.g. `@pytest.mark.integration`) and **skipped when `ES_URL` is unreachable**, so the pure suite (Phases 1–8) runs fully offline in CI.

### Lint / type-check (via hatch)
- **`ruff`** (lint + format) and **`mypy`** (prefer `--strict` on the pure core) run in a hatch `dev` env: `hatch run dev:ruff check`, `hatch run dev:mypy benchmark`. Both must pass before any sign-off.

### Docker-compose ES for integration phases
- `docker compose up -d` then `hatch run eval:wait-for-es` (polls cluster health yellow). Image is pinned ≥ 8.15 (hard floor). Integration tests prefer **local ES inference models** (ELSER/E5) so they need no external API keys. Teardown: `docker compose down -v`.

### Definition of Done (every phase)
- [ ] Only this phase's §11 deliverables added; names/schemas match `docs/experiment.md` exactly.
- [ ] No dependency on a later phase; pure phases import no adapter.
- [ ] Tests written and passing; pure phases need no Docker, integration phases run against ES ≥ 8.15.
- [ ] `ruff` clean; `mypy` clean.
- [ ] Cited design sections honored (load-bearing details called out in the phase).
- [ ] Reviewer approved.
- [ ] User signed off via the phase's sign-off gate.
- [ ] **Commit on user consent only** (no agent commits).

---

## 6. Traceability

### §11 modules → phase
| Module (§11) | Phase |
|--------------|:-----:|
| `models.py` | 1 |
| `protocols.py` | 1 |
| `pipeline.py` (incl. `PipelineSpec`/`FuseCfg`/`RerankCfg`/`StageCfg`/`spec_for`) | 5 |
| `fusion.py` | 4 |
| `rerank.py` | 4 |
| `metrics.py` | 2 |
| `stats.py` | 3 |
| `matrix.py` | 6 |
| `runner.py` | 11 |
| `io_csv.py` | 7 |
| `config.py` | 6 |
| `logging_setup.py` | 0 |
| `datasets/wands.py` | 8 |
| `backends/elasticsearch.py` (BM25 + lifecycle + execute + capabilities) | 9 |
| `backends/elasticsearch.py` (semantic + RRF + rerank + `ElasticsearchIndexer`) | 10 |
| `pyproject.toml`, `docker-compose.yml`, `.gitignore`, `config.yaml`, eval scripts | 0 |

### §1.4 success criteria → phase
| §1.4 criterion | Built by | Validated by |
|----------------|----------|--------------|
| (1) **Correctness** — three CSV types, exact §9 schemas, one §8.3 regime | 2, 3, 7, 11 | 7 (golden), 12 (full schema lint) |
| (2) **Reproducibility** — config + seed reproduces metrics/stats; `best_per_model` seed-independent | 3, 6, 7 | 3 (seeded determinism), 6 (selection determinism), 12 (two-run diff) |
| (3) **Generality** — dataset/backend swap = adapter + config only; degenerate metrics defined | 1, 3, 6, 8, 9, 10 | 3 (degenerate sets), 5 (FakeBackend), 12 (import-graph check) |
| (4) **DRY** — 6 variants = one pipeline + one execution path | 5, 11 | 5 (all 6 specs via one `run()`), 11/12 (single `run_one` inspection) |
