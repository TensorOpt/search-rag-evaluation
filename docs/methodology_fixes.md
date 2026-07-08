# Methodology Fixes — Design (review + sign-off before implementation)

> Status: DESIGN ONLY. Nothing in `experiment.md` / `CLAUDE.md` / `config.yaml` / code is edited yet.
> This doc spells out the exact edits so implementation is mechanical. `experiment.md` remains the
> source of truth; the "Proposed edits" section lists every section to rewrite so code and doc stay
> consistent. All findings below are grounded in `results/*_20260707T163016Z.csv`
> (top_k=50, window=25, rerank_window_size=25, top_n=25).

## The 7 problems (one line + the confirmed number)

1. **Missing `semantic_co_rerank` cell.** The factorial is {bm25, semantic_co, hybrid_co} × {rerank off/on} = 6 cells; the config runs only **5** — dense+rerank is absent, so the rerank effect is never isolated on dense-only.
2. **Estimand mismatch.** Point estimate + bootstrap CI are of the **mean** difference, but the default p-value is **Wilcoxon** (pseudo-median, drops zeros) — a different estimand from the magnitude we report.
3. **No variant-vs-variant.** Only `variant vs bm25` is computed; the questions "does RRF-with-bm25 hurt dense?" and "is hybrid+rerank indistinguishable from plain dense?" (`hybrid_co vs semantic_co`, `hybrid_co_rerank vs semantic_co`) are unanswerable.
4. **recall@10 / precision@10 pinned at the ceiling.** recall@10 baseline = **0.0681 for every system** (bounded ≈ 10/|rel|, median |rel| ≈ 178); precision@10 ≈ **0.91-0.92** (ceiling). No coverage signal can move.
5. **Depth confounded with the rerank/fusion factor.** Docs returned/query: bm25 **49.6**, semantic_co **50**, bm25_rerank **24.9**, hybrid_co_rerank **25**, hybrid_co_k60 **41.3 (25-50)** — depth co-varies with the treatment, so any delta mixes "reranked" with "shallower".
6. **Queries silently dropped, set differs per system.** ndcg@10 `baseline_value` takes **3 distinct values** (0.7905 / 0.7914 / 0.7930) across comparison rows because each pairing uses its own finite-in-both mask — one baseline system reported with three numbers. 7-12 `n_scored==0` queries per variant, a different set each.
7. **FDR family full of near-collinear metrics.** All (variant × 4-metric) tests share one BH family; `avg_relevance` and `precision@10` measure nearly the same thing (both ≈ fraction-relevant in a shallow window) and move together, inflating the family with no added information.

---

## Fix 1 — Add the `semantic_co_rerank` variant

**Design.** Config-only. Add the 6th factorial cell so the rerank effect can be isolated on dense-only
retrieval (mirror of `bm25_rerank`/`bm25` and `hybrid_co_rerank`/`hybrid_co_k60`).

**config.yaml addition** (under `pipelines.variants`, using the normalized depth from Fix 5):

```yaml
    semantic_co_rerank:
      retriever: semantic_co
      reranker: co-rr
      rerank_window_size: 100
```

**Modules changed:** none (pure config). `build_pipeline` already composes `VectorSearch + ESReranker`
via the same one-path assembly; no per-variant code. Adds one contrast to Fix 3's family
(`semantic_co_rerank vs semantic_co`).

**Invariants.** DRY/one-path (a config row, not code); generality untouched. Depends on Fix 5 for
`rerank_window_size: 100`.

---

## Fix 2 — One estimand: mean-difference permutation test as default

**Design.** Make the significance test consistent with the reported estimand. The point estimate is
`mean(δ)`, the bootstrap CI is of `mean(δ)` (§8.2), so the p-value must also concern `mean(δ)`. The
**seeded sign-flip paired-permutation test with statistic = `mean(δ)` is already implemented** in
`Comparator._permutation_p_value` (stats.py). Make it the default; demote Wilcoxon to a non-default
opt-in.

**Soundness (confirmed).** The sign-flip test's null is that the paired differences `δ_q` are
**exchangeable in sign** — i.e. symmetric about 0 (each `δ_q` equally likely `+|δ_q|` or `−|δ_q|`),
the standard randomization null for a paired design. With statistic `mean(δ)` and two-sided
`p = (1 + #{|perm_stat| ≥ |obs_stat|}) / (B + 1)`, the test asks exactly whether the **mean** paired
difference departs from 0 — the same quantity as the point estimate and the CI. So all three now
describe **one estimand: the mean paired difference**. Two properties make it strictly better aligned
than Wilcoxon here: (a) it targets the mean, not the pseudo-median; (b) sign-flipping a zero delta
leaves it zero, so zero-deltas are **retained** and contribute to the null (Wilcoxon's `zero_method`
drops them) — correct for the sparse-delta nDCG/recall/precision distributions. Caveat to state in the
doc: the null is symmetry-about-0; a distribution asymmetric-about-0 yet mean-0 is a theoretical corner
that does not arise for paired IR-metric deltas — this is the accepted paired randomization test.

**Config surface.** `stats.test: permutation` (was `wilcoxon`). Keep `wilcoxon` selectable (the
exhaustive branch in `_p_value` stays). The `wilcoxon_zero_method`/`wilcoxon_correction` settings stay
but are inert unless `test: wilcoxon`.

**`bootstrap_B` (S1).** `bootstrap_B` governs **both** the CI resample count (§8.2) **and** the Monte-
Carlo permutation count (the exact-enumeration branch only fires when `2**n ≤ bootstrap_B`, never at
n≈470), so it is the p-value resolution floor: two-sided permutation p is quantized to `1/(B+1)`.
`B = 10000` gives ~1e-4 resolution — adequate below the α=0.05 FDR threshold. **config.yaml ships
`bootstrap_B: 1000`; raise to `10000`** to match the `StatsCfg` default and §10. Add a one-line §8.2
note stating this dual role + the floor.

**Modules changed:**
- `stats.py` — `StatsCfg.test` default `"wilcoxon"` → `"permutation"`. No logic change (both branches
  already exist; `_permutation_p_value` is the one that runs).
- `config.py` — `_resolve_stats` default for `test` → `"permutation"`.
- `config.yaml` — `stats.test: permutation`.

**Invariants.** Reuse over addition (no new code path); reproducibility preserved (permutation is
seeded off the same `default_rng(seed)`; exact enumeration when `2**n ≤ bootstrap_B`, else Monte-Carlo).

---

## Fix 3 — Arbitrary contrasts (system_a vs system_b), not just vs baseline

**Design.** Generalize the comparator from `(baseline, variants)` all-vs-baseline to
`(systems, contrasts)`: all system metric-maps plus an explicit list of contrasts, each a pair of
system ids with `delta = value(a) − value(b)`. The baseline stops being special in the comparator (it
is just another system); "variant vs bm25" becomes one contrast among many. The `baseline` concept
survives only for run ordering (baseline materialized first, §8.0) and for synthesizing the default
contrast set.

**New value type** (in `stats.py`, a plain frozen dataclass — no new dep):

```python
@dataclass(frozen=True)
class Contrast:
    a: str          # system id (delta numerator)
    b: str          # system id (delta subtrahend); delta = value(a) − value(b)
    family: bool    # True => eligible for the FDR family (Fix 7); False => descriptive-only
```

**New `Comparator.compare` signature:**

```python
def compare(
    self,
    systems: Mapping[str, Mapping[str, Mapping[str, float]]],  # system_id -> query_id -> {metric: value}
    contrasts: Sequence[Contrast],
) -> list[ComparisonResult]:
```

Loop over `contrasts × CANONICAL_METRICS`; for each, pull `value_a`/`value_b`/`delta`/CI/`p_value`
over the family-wide common subset (Fix 6), set `in_family` per Fix 7, then FDR-adjust only the family
rows. Rows ordered deterministically: contrasts in config order, metrics in `CANONICAL_METRICS` order.

**Default contrast family** (shipped in config.yaml; the delta convention is "how much better is `a`
than `b`", positive = `a` wins):

| a | b | question |
|---|---|----------|
| `semantic_co` | `bm25` | headline: does dense beat lexical? |
| `hybrid_co_k60` | `semantic_co` | does RRF-with-bm25 help or **hurt** dense? |
| `hybrid_co_rerank` | `semantic_co` | is hybrid+rerank **indistinguishable** from plain dense? |
| `bm25_rerank` | `bm25` | rerank isolation on lexical |
| `semantic_co_rerank` | `semantic_co` | rerank isolation on dense (needs Fix 1) |
| `hybrid_co_rerank` | `hybrid_co_k60` | rerank isolation on hybrid |

All six ship with `family: true`.

**Config surface** (`stats` block; `contrasts` optional):

```yaml
stats:
  contrasts:                       # optional; absent => every variant vs baseline, all family:true
    - { a: semantic_co,        b: bm25,          family: true }
    - { a: hybrid_co_k60,      b: semantic_co,   family: true }
    - { a: hybrid_co_rerank,   b: semantic_co,   family: true }
    - { a: bm25_rerank,        b: bm25,          family: true }
    - { a: semantic_co_rerank, b: semantic_co,   family: true }
    - { a: hybrid_co_rerank,   b: hybrid_co_k60, family: true }
```

**Default when `stats.contrasts` is absent** (backward compatibility): synthesized at config-resolution
time (where baseline/variant ids are known) as `Contrast(a=variant, b=baseline_id, family=True)` for
every variant — reproduces the old all-vs-baseline behavior without hardcoding ids in `StatsCfg`.

**Comparison CSV schema change (old → new).**

Old (12 cols):
```
baseline,variant,metric,baseline_value,variant_value,delta,delta_ci_lo,delta_ci_high,p_value,significant_raw,p_value_adjusted,significant
```
New (14 cols):
```
system_a,system_b,metric,value_a,value_b,delta,delta_ci_lo,delta_ci_high,p_value,significant_raw,in_family,p_value_adjusted,significant,n_common
```
Field-by-field, in order:
1. `system_a` — contrast's `a` id
2. `system_b` — contrast's `b` id
3. `metric` — one of `CANONICAL_METRICS` (Fix 4 grows this set)
4. `value_a` — mean of `a`'s metric over the common subset (Fix 6); empty for empty paired set
5. `value_b` — mean of `b`'s metric over the common subset; empty for empty paired set
6. `delta` — `value_a − value_b`; empty for empty paired set
7. `delta_ci_lo` — 2.5-pct bootstrap of `mean(δ)`; empty for empty paired set
8. `delta_ci_high` — 97.5-pct bootstrap; empty for empty paired set
9. `p_value` — raw permutation (or Wilcoxon) p; `1.0` for degenerate rows
10. `significant_raw` — `true`/`false`, `p_value ≤ α`, independent of the family
11. `in_family` — `true`/`false`, FDR-family membership (Fix 7)
12. `p_value_adjusted` — BH/BY q-value; **empty** when `in_family=false`
13. `significant` — `true`/`false` FDR decision; **empty** when `in_family=false`
14. `n_common` — int, queries in this metric's common subset (Fix 6), always present

**`ComparisonResult` dataclass fields (in order):** `system_a, system_b, metric, value_a, value_b,
delta, delta_ci_lo, delta_ci_high, p_value, significant_raw, in_family, p_value_adjusted (float|None),
significant (bool|None), n_common (int), note`.

**Consistency rule (load-bearing, M3):** `in_family == false ⟺ p_value_adjusted and significant cells
are BOTH empty`. FDR-adjusted values exist **only** for family rows. This one rule governs three row
kinds uniformly:
- **family** (`contrast.family AND metric ∈ fdr_metrics AND non-degenerate`): `in_family=true`,
  `p_value_adjusted` = BH/BY q-value, `significant` = FDR decision.
- **descriptive** (real test, not in family): `in_family=false`, `p_value_adjusted=None`,
  `significant=None` → empty cells; `p_value`/`significant_raw` still populated.
- **degenerate** (`empty_paired_set` / `all_zero_delta`): `in_family=false`, `p_value_adjusted=None`,
  `significant=None` → empty cells (this **supersedes** the old §8.1 `p_value_adjusted=1.0`/
  `significant=false` prescription — see the §8.1 proposed edit). `p_value=1.0`, `significant_raw=false`,
  `note` set, `n_common` populated (0 for `empty_paired_set`).

**Modules changed:**
- `stats.py` — add `Contrast`; rewrite `compare(systems, contrasts)`; rename `ComparisonResult` fields;
  `_paired_values` gets the common mask (Fix 6); FDR step iterates only family rows (Fix 7).
- `config.py` — `StatsCfg` gains `contrasts: tuple[Contrast, ...]` (+ `fdr_metrics`, Fix 7);
  `_resolve_stats` parses `stats.contrasts`; `resolve_config` synthesizes the default from
  (baseline_id, variants) when absent. **Config-time validation (M2, fail fast — CLAUDE.md
  move-with-certainty + the fail-fast invariant):** `resolve_config`/`_resolve_stats` raise
  `ConfigError` if any contrast `a`/`b` ∉ `{baseline_id} ∪ variant ids`, or any `fdr_metrics` entry ∉
  `CANONICAL_METRICS`. A bad contrast id or headline metric is caught at build time, never at run time.
  (Requires `_resolve_stats` to see baseline/variant ids — resolve pipelines first, then stats, passing
  the known id set + `CANONICAL_METRICS`.)
- `runner.py` — build `systems = {vid: {q: m.as_dict() …} for all pipelines}` (baseline no longer
  split out); `Comparator(cfg.stats).compare(systems, cfg.stats.contrasts)`.
- `io_csv.py` — new `_COMPARISON_HEADER` (14 cols); `write_comparison_csv(rows, timestamp, …)` **drops
  the `baseline_id` param**. Only `_bool_cell` needs a `None → ""` path (for `significant` on
  descriptive/degenerate rows, M3); `_float_cell` **already** maps `None`/`NaN → ""`, so `value_a`/
  `value_b`/`p_value_adjusted` empty cells need no change (N5). `_bool_cell` for `in_family` is a plain
  bool (never None); `n_common` is a plain int.

**Invariants.** DRY/one-path (one `compare`, one CSV writer, config-driven contrasts — no per-contrast
code); generality (contrasts are just id pairs, dataset/backend-agnostic); the comparator still imports
only stdlib+numpy+scipy.

---

## Fix 4 — recall at depth (recall@50, @100); standard recall semantics

**Design.** Add `recall@50` and `recall@100` and switch **all** recall to **standard** semantics.
Keep `precision@10`, `ndcg@10`, `avg_relevance` on the condensed-list evaluation unchanged (the
MISSING-skip invariant is load-bearing for those — precision's denominator is `n_scored`, nDCG uses
condensed positions).

**Recall semantics — standard vs condensed (the decision).**
- **Standard recall@k** = `|judged-relevant ∩ actual top-k retrieved (positions 1..k)| / R`, where
  `R` = relevant judged docs over ALL qrels.
- **Condensed recall@k** (current) scans the list skipping MISSING and counts hits among the first `k`
  **judged** docs — which may reach far past retrieved position `k`.

They diverge when MISSING docs occupy top-k slots. At depth 100 with sparse pooling, condensed
recall@100 keeps scanning until 100 judged docs are found (often the whole returned list), so it stops
measuring "did the top-100 contain the relevant docs" — the coverage question. **Standard is correct
for coverage**: it looks at the actual retrieved positions.

Standard recall does **not** violate the MISSING invariant. The invariant forbids scoring a MISSING doc
as irrelevant (gain 0) in a metric. Standard recall@k never does that: the denominator is `R` (fixed,
from qrels), and a MISSING doc in a top-k slot contributes 0 to the numerator exactly as it is neither
counted relevant nor penalized in any denominator — it is simply not a relevant hit. The condensed rule
exists to stop unjudged docs from deflating **precision/nDCG**; recall's denominator is immune, so
recall is the one metric where actual-position (standard) semantics is both correct and invariant-safe.

**recall@10 also switches (DECIDED, decision 3).** recall@10 becomes standard too, so all three recall
columns share one definition (a reader never guesses which recall a column is). Numerically recall@10
barely moves (few MISSING in the top 10) but the value **does** shift — this is why the
`test_condensed_reaches_past_rank_ten` fixture must be recomputed (standard recall@10 = 8/10 = 0.8, not
the condensed 1.0).

**Cutoffs (DECIDED, decision 2): `{10, 50, 100}`** — @10 keeps the headline comparable, @100 is the
coverage story at the normalized depth (Fix 5), @50 shows the curve between. `recall@100` ==
`recall@min(100, n_returned)` (a query returning fewer than 100 caps there — expected).

**NaN condition.** `recall@k = NaN` iff `R == 0` (same as today). Standard recall is otherwise always
defined (numerator 0..R) — including 0 for an empty result set, so recall **penalizes** retrieval
failures (relevant to Fix 6's query 383).

**Metrics CSV schema change (old → new).**

Old:
```
variant,query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_results,n_scored,n_missing
```
New (recall columns contiguous, minimal reorder):
```
variant,query_id,avg_relevance,ndcg@10,recall@10,recall@50,recall@100,precision@10,n_results,n_scored,n_missing
```
`recall@50`/`recall@100` are floats, empty iff `R == 0`. `n_results`/`n_scored`/`n_missing` unchanged
(still condensed-top-10 counts).

**`CANONICAL_METRICS` (stats.py) grows to 6, matching the CSV order:**
```python
("avg_relevance", "ndcg@10", "recall@10", "recall@50", "recall@100", "precision@10")
```
Every contrast then gets delta+CI+raw-p on all six; only the headline metric(s) get FDR (Fix 7).

**Modules changed:**
- `metrics.py` — `Metrics` adds `recall_at_50`, `recall_at_100`; `as_dict()` adds keys `recall@50`,
  `recall@100`. `_score_one` adds a standard-recall pass:
  `for k in RECALL_CUTOFFS: hits_k = sum(1 for d in result.docs[:k] if gain(qid, d.doc_id) >= 0.5); recall_k = hits_k / R if R > 0 else nan`.
  Module constant `RECALL_CUTOFFS = (10, 50, 100)`. The condensed scan (avg/ndcg/precision +
  `n_scored`/`n_missing`) is unchanged; recall no longer reads the condensed hits. Docstring updated to
  state recall is now standard (actual top-k), the other three stay condensed, and why that upholds the
  MISSING invariant.
- `stats.py` — `CANONICAL_METRICS` (6 entries).
- `io_csv.py` — `_METRIC_COLUMNS` = the 6 canonical names in the new order; `_METRICS_HEADER` follows.
- No new config (cutoffs are a fixed constant tied to the frozen CSV schema — no speculative knob).

**Invariants.** Condensed MISSING-skip preserved for precision/nDCG/avg (unchanged); recall's shift is
invariant-safe (argued above); DRY (one evaluator, one loop over cutoffs).

---

## Fix 5 — Normalize retrieval depth (kill the confound)

**Design.** Every system retrieves and returns to the **same target depth = 100**. This removes the
depth confound (all systems compared at equal depth) and enables recall@100 (Fix 4). Note: `experiment.md`
§10 **already prescribes 100** for all four knobs; only the shipped `config.yaml` lags at 25/50. Fix 5 is
therefore mostly aligning config.yaml to the doc + documenting the invariant and its cost.

**Why hybrid varies today, and why 100 fixes it.** `HybridSearch` runs each leaf at `window` (=25),
then `RRFFuser` fuses the two ranked lists and truncates to `top_k` (=50). The union of two 25-doc
lists is **25 (fully overlapping) to 50 (disjoint)** distinct docs → hybrid returns 25-50 (observed
mean 41.3). With `window = 100` and `top_k = 100`, each leaf returns up to 100, the union is 100-200,
truncated to 100 → consistently ~100 (bounded only by `min(100, available matches)`). bm25/semantic/
rerank all likewise return up to 100.

**config.yaml changes** (doc §10 already at these values except confirm all):
- `top_k: 50 → 100`
- every pipeline `fuser.window: 25 → 100`
- every pipeline `rerank_window_size: 25 → 100`
- reranker `co-rr` `settings.top_n: 25 → 100` (so `W ≤ top_n` at R0 still holds)

**Cost (state explicitly).** Cohere rerank now scores **100 docs/query** instead of 25 — ~4× the rerank
requests/tokens per reranked query per run. `_msearch` retrieval cost is negligibly changed. The
optional disk cache (§5.5) absorbs re-runs.

**Reported metrics still cut at 10 for the point/quality metrics:** `cutoff` stays `10`
(`avg_relevance`, `ndcg@10`, `precision@10` are condensed-top-10); the depth-100 retrieval feeds
`recall@10/@50/@100`. A query with fewer than 100 matches returns `min(100, available)` — expected, not
an error (query 383 still returns 0 under bm25; handled by Fix 6).

**Modules changed:** `config.yaml` only (4 knobs). Doc: add a "uniform retrieval depth" invariant note
(§5.3/§8.0) — `fuser.window == rerank_window_size == top_k` so retrieval depth is not confounded with
the rerank/fusion treatment. No code change (the composers already honor these knobs).

**Invariants.** Generality (still config-only knobs); DRY (no new path); reproducibility (depth is
captured in `run_config`).

---

## Fix 6 — Family-wide common subset (one value per system, baseline included)

**Design.** Score every contrast on the **family-wide common subset** for each metric: the set of
queries where that metric is finite (not NaN) for **all** systems in `systems`. Then every system —
baseline included — has exactly **one** mean per metric (the 3-distinct-baseline bug disappears). This
replaces the current pairwise finite-in-both mask, which produces a different query set per contrast.

**`_paired_values` change.** Today it recomputes a pairwise mask per (variant, metric). New flow: in
`compare`, precompute once per metric a single common mask, then select each contrast's
`value_a`/`value_b`/`δ` from that shared mask:

```python
def _common_qids(systems, contrasts, metric) -> list[str]:
    # Scope (S2): only the systems REFERENCED BY THE CONTRASTS — not all of `systems`. A system in no
    # contrast can't shrink everyone's subset; still ONE shared mask, so every referenced system
    # (baseline included) keeps ONE value per metric.
    used = {s for c in contrasts for s in (c.a, c.b)}
    return [qid for qid in sorted(all_qids)
            if all(not isnan(systems[s].get(qid, {}).get(metric, nan)) for s in used)]

def _paired_values(map_a, map_b, metric, common_qids) -> (np.ndarray, np.ndarray):
    # select metric over the SINGLE precomputed common mask (no per-pair recompute)
```

`n_common[metric] = len(common_qids)`; emitted on every row (col 14) and, with `n_excluded = n_queries
− n_common`, recorded per metric in `run_config_{ts}.json`.

**Power cost (S2, state plainly in §8.1).** The family-wide common subset is the **union of per-system
NaN queries** across the referenced systems, so `n_common ≤ any pairwise finite-in-both subset` — the
common subset trades statistical power for coherence (one value per system), **by design**. `n_common`
on every row makes the traded-away queries visible.

**Policy: drop vs score-0 for `n_scored == 0` (DECIDED, decision 4: DROP + retrieval-failure count).**

`n_scored == 0` (no judged doc in the condensed top-10) makes `avg_relevance`/`ndcg@10`/`precision@10`
NaN → excluded. Two distinct causes: (a) **empty result** (system retrieved nothing — bm25/bm25_rerank
on query 383, the only such query in WANDS); (b) **non-empty but all-MISSING top-k** (pooling gap —
7-12 queries/variant). Under the common subset, a query NaN in **any** system is dropped for **all**
systems on that metric.

DECIDED: **keep DROP** (NaN → excluded), because scoring an all-MISSING top-k as 0 would
treat MISSING as irrelevant — a direct violation of the load-bearing condensed-list invariant. Two
things make DROP safe here:
1. **Standard recall already penalizes the failure.** Fix 4's `recall@k` scores an empty/failed
   retrieval as `0` (not NaN, since `R > 0`), so the coverage metrics catch exactly the case DROP is
   accused of hiding — the ranking-quality metrics (nDCG/precision) simply abstain where they cannot
   measure, while recall does the honest coverage accounting.
2. **Visibility.** Report a per-system **retrieval-failure count** (`#queries with n_results == 0`) in
   `run_config_{ts}.json`. For WANDS this is exactly 1 (query 383, bm25/bm25_rerank), so nothing failing
   is invisible.

Asymmetry to STATE in §7/§8: DROP does not penalize a system that returned nothing usable in the
condensed metrics — mitigated by (1) and (2). Rejected alternatives (record only): **score-0**
(penalizes failure but breaks the MISSING invariant for the all-MISSING case and enlarges every
metric's query set), or a **refinement** (empty-result → 0, all-MISSING → drop) that distinguishes the
two causes at the cost of one extra branch.

**Empty common subset (S3).** If a metric's common subset is empty, `_common_qids` returns `[]` →
`_paired_values` returns empty arrays → the existing `deltas.size == 0` branch fires →
`note="empty_paired_set"`, `n_common=0`, `in_family=false`, and (per M3) `p_value_adjusted=None`,
`significant=None` → empty cells. Both degenerate short-circuits (`empty_paired_set`, `all_zero_delta`)
must be updated to populate the NEW `in_family` (always `false`) and `n_common` fields, and to emit
`p_value_adjusted=None`/`significant=None` (not the old `1.0`/`false`).

**Modules changed:**
- `stats.py` — `_paired_values` takes a precomputed `common_qids`; `compare` computes `_common_qids`
  once per metric (scoped to contrast-referenced systems, S2); each `ComparisonResult` carries
  `n_common` + `in_family`; the two degenerate short-circuits emit `in_family=false`, `n_common`, and
  `p_value_adjusted=None`/`significant=None` (M3, S3).
- `runner.py` — collect per-metric `n_excluded` and per-system retrieval-failure counts (from the
  in-memory `Metrics`: `n_results == 0`) and pass to `write_run_config` (or compute in `write_run_config`
  from the config-plus-metrics — simplest: runner assembles a small `diagnostics` dict).
- `io_csv.py` — `write_run_config(cfg, *, diagnostics=None, output_dir=…)` gains a `diagnostics`
  keyword; it merges a top-level `"diagnostics"` key (`{common_subset: {metric: {n_common, n_excluded}},
  retrieval_failures: {system: count}}`) into the `dataclasses.asdict(cfg)` payload before serializing.
  `_RUN_CONFIG_KEYS` (schema-lint) gains `"diagnostics"`. Keyword-only + defaulted so existing callers
  and the round-trip stay valid.

**Invariants.** Condensed MISSING-skip preserved (DROP, not score-0); generality (mask logic operates on
delta arrays only, dataset-agnostic, stays in `stats.py`); reproducibility (sorted qids → deterministic
bootstrap).

---

## Fix 7 — FDR family = contrasts-of-interest × headline metric(s) only

**Design.** Shrink the BH family to the decisions we will act on, carrying independent information. A
comparison row `(contrast, metric)` is in the FDR family iff **`contrast.family` AND `metric ∈
fdr_metrics` AND the row is non-degenerate**. Only those rows get `p_value_adjusted` + `significant`;
every other row is **descriptive** — `delta` + bootstrap CI + raw `p_value`/`significant_raw`, but
`in_family=false` and empty `p_value_adjusted`/`significant`.

**Headline metric(s) (DECIDED, decision 6): `fdr_metrics = ("ndcg@10", "recall@100")`.**
- `ndcg@10` = ranking quality (the "better ordering" claim).
- `recall@100` = coverage (the "semantic finds more" claim) — available after Fix 4, and **nearly
  orthogonal** to nDCG@10 (quality vs coverage answer different questions), so adding it does not inflate
  the family the way `avg_relevance`+`precision@10` (near-collinear, both ≈ fraction-relevant shallow)
  would. These two are the two headline claims and are not near-duplicates.

Alternative: `("ndcg@10",)` alone (smallest family, one headline). `avg_relevance`/`precision@10`/
`recall@10`/`recall@50` stay descriptive regardless — they are context, not decisions.

**Family size.** 6 family contrasts × 2 headline metrics = **12** BH tests (or 6 with the single-metric
alternative) — small, information-dense, no collinear padding.

**CSV marking.** The `in_family` column (Fix 3, col 11) is the explicit flag; descriptive rows write
empty `p_value_adjusted` and empty `significant`. `significant_raw` stays populated on **all** rows (it
is the family-independent per-test decision, §8.3).

**Config surface.** `stats.fdr_metrics` (optional list; default `("ndcg@10", "recall@100")` in
`StatsCfg`). Contrast-level eligibility is the `family:` key on each contrast (Fix 3).

```yaml
stats:
  fdr_metrics: [ndcg@10, recall@100]   # optional; the ONLY metrics that enter the BH family
```

**Modules changed:**
- `config.py` — `StatsCfg.fdr_metrics: tuple[str, ...] = ("ndcg@10", "recall@100")`; `_resolve_stats`
  parses `stats.fdr_metrics`.
- `stats.py` — in `compare`, `in_family = contrast.family and metric in cfg.fdr_metrics and not
  degenerate`; the FDR step (`_fdr_adjust`) runs over family rows only; **every** non-family row
  (descriptive real test AND degenerate) gets `p_value_adjusted=None`, `significant=None` → empty cells
  (M3 rule: `in_family=false ⟺ both empty`). This drops the old degenerate `p_value_adjusted=1.0`/
  `significant=false`.

**Invariants.** DRY (one comparator, one FDR step); the single coherent FDR regime (§8.3) is preserved,
just applied to a well-chosen family; `significant_raw`/CI roles unchanged.

---

## Proposed edits (exact targets)

**`docs/experiment.md`** (source of truth — rewrite these sections):
- **§1.2** — add the 6th variant row; note the factorial is {bm25, semantic_co, hybrid_co} × {rerank
  off/on}. (Fix 1)
- **§7** — recall becomes **standard** at cutoffs `{10,50,100}` (state the standard-vs-condensed
  distinction and why recall is invariant-safe); precision/nDCG/avg stay condensed; add the drop-vs-
  score-0 policy (DROP + retrieval-failure count) and the recall-penalizes-failure note. (Fix 4, Fix 6)
- **§8.1** — pairing uses the **family-wide common subset** (one value per system, scoped to
  contrast-referenced systems, S2), not the pairwise finite-in-both mask; define `n_common`/`n_excluded`
  + the power-cost note (subset = union of per-system NaN queries); retrieval-failure reporting.
  **Rewrite the degenerate-row table (M3):** `empty_paired_set`/`all_zero_delta` now emit
  `p_value_adjusted` and `significant` as **empty** (not `1.0`/`false`), plus `in_family=false` and
  `n_common`. Call out explicitly that this **supersedes** the current §8.1 `1.0`/`false` prescription.
  (Fix 6)
- **§8.2** — permutation (mean-δ) is the **primary/default** test; point estimate + CI + p-value all
  concern the **mean difference**; Wilcoxon demoted to opt-in; add the exchangeability soundness note;
  add the `bootstrap_B` dual-role + p-resolution-floor note (S1). (Fix 2)
- **§8.3** — FDR family = `contrast.family × fdr_metrics` (headline nDCG@10 + recall@100); state the M3
  rule (`in_family=false ⟺ empty `p_value_adjusted`/`significant``); descriptive rows carry
  delta+CI+raw-p only; rationale for dropping near-collinear metrics (avg_relevance≈precision@10) from
  the family. (Fix 3, Fix 7)
- **§5.3 / §8.0** — add the **uniform retrieval depth** invariant (`fuser.window ==
  rerank_window_size == top_k`, target 100) and the rerank-cost note. (Fix 5)
- **§9** — rewrite the two frozen headers (metrics + comparison, below); reproducibility block records
  `contrasts`, `fdr_metrics`, common-subset + retrieval-failure diagnostics, depth. (Fix 3/4/6/7)
- **§10** — config example: 6th variant, depth 100, `stats.contrasts`, `stats.fdr_metrics`,
  `test: permutation`, `top_n: 100`. (Fix 1/2/3/5/7)

**`CLAUDE.md`** (Load-bearing invariants — update these lines):
- Add `semantic_co_rerank` to the 6-variant list.
- **Metrics CSV schema** →
  `variant, query_id, avg_relevance, ndcg@10, recall@10, recall@50, recall@100, precision@10, n_results, n_scored, n_missing`
- **Comparison CSV schema** →
  `system_a, system_b, metric, value_a, value_b, delta, delta_ci_lo, delta_ci_high, p_value, significant_raw, in_family, p_value_adjusted, significant, n_common`
- **MISSING rule** — clarify: condensed-list MISSING-skip is load-bearing for `avg_relevance`/`ndcg@10`/
  `precision@10`; **recall is standard** (actual top-k / R) and invariant-safe (never scores MISSING as
  irrelevant; R from qrels).
- Note the uniform-depth invariant and that the default significance test is the mean-δ permutation.

**`config.yaml`** (changes):
- Add `semantic_co_rerank` variant (Fix 1).
- `top_k: 50 → 100`; every `fuser.window: 25 → 100`; every `rerank_window_size: 25 → 100`; reranker
  `co-rr` `top_n: 25 → 100` (Fix 5).
- `stats.test: wilcoxon → permutation` (Fix 2); `stats.bootstrap_B: 1000 → 10000` (S1).
- Add `stats.contrasts` (6 rows) and `stats.fdr_metrics: [ndcg@10, recall@100]` (Fix 3, Fix 7).

**`README.md`** (operational guide — update):
- `metrics_{ts}.csv` header (add `recall@50,recall@100`) + its prose (recall is standard/coverage, the
  other three stay condensed).
- `comparison_{ts}.csv` header (12 → 14 cols) + its prose (`system_a/system_b`, `in_family`,
  `n_common`, per-metric FDR family).
- The 4-metric enumeration near the top (line 3) → 6 metrics
  (`avg_relevance, ndcg@10, recall@10, recall@50, recall@100, precision@10`).
- The `comparison_*` metric-set enumeration (`{avg_relevance, ndcg@10, recall@10, precision@10}` → the
  6-metric set).

### Tests & fixtures (every target the schema/signature changes break — M1)

- **`tests/unit/test_schema_lint.py`** — update `_METRICS_HEADER` (11 cols), `_COMPARISON_HEADER`
  (14 cols), and `_RUN_CONFIG_KEYS` (exact-set assertion — add the new `diagnostics` key from Fix 6/9).
- **`tests/unit/test_io_csv.py`** — the 12-col comparison-header test → 14-col; `write_comparison_csv`
  loses the `baseline_id` arg; renamed `ComparisonResult` fields (`system_a`/`system_b`/`value_a`/
  `value_b`/`in_family`/`n_common`); `Metrics(...)` now 6 metric fields; degenerate-row cell asserts →
  empty `p_value_adjusted`/`significant` (M3).
- **`tests/unit/evaluation/test_stats.py`** — `compare(baseline, {...})` → `compare(systems,
  contrasts)`; `_paired_values` signature gains `common_qids`; `.variant`/`.baseline_value` reads →
  `.system_a`/`.value_a`; add cases for `in_family`, per-metric common subset, empty-common-subset (S3),
  and the M3 degenerate empties.
- **`tests/unit/evaluation/test_metrics.py`** — `Metrics(...)` 6-field + `as_dict` 6-key; **recompute
  `test_condensed_reaches_past_rank_ten`**: standard recall@10 on that fixture is **8/10 = 0.8**, not
  the condensed `1.0` (Fix 4 — the key behavioral change to assert).
- **`tests/fixtures/golden/metrics_20260101T000000Z.csv`** (11-col header) and
  **`comparison_20260101T000000Z.csv`** (14-col header) — regenerate for the new headers/fields;
  degenerate comparison rows show empty adjusted cells (M3). **`result_*.csv` golden is unchanged.**
- **`tests/unit/test_config.py`** — parse `stats.contrasts`/`fdr_metrics`; assert the M2 `ConfigError`
  on an unknown contrast id and on an `fdr_metrics` entry ∉ `CANONICAL_METRICS`; assert the absent-
  `contrasts` default (every variant vs baseline).

Integration tests (`tests/integration/test_runner.py`) and `tests/unit/test_runner.py` exercise the
runner's `compare(systems, contrasts)` call + the dropped `baseline_id` writer arg — verify their fakes
match the new signatures.

---

## Open decisions — RESOLVED (user signed off; all recommendations accepted as-is)

Every item below is locked to the accepted choice (was the recommendation). Alternatives are retained
only for the record — they are NOT to be implemented.

1. **Uniform depth value.** ACCEPTED **100** — enables recall@100 and equalizes depth. *Rationale:*
   the coverage story needs depth and the confound needs equal depth. *Alt:* 50 (cheaper rerank, caps
   the recall@100 story) / 200 (more coverage, ~2× rerank cost).
2. **Recall cutoffs.** ACCEPTED **{10, 50, 100}** (user explicitly kept @50). *Rationale:* @10 comparable headline, @100 coverage
   at depth, @50 the curve. *Alt:* {10, 100} (leaner) or {10, 20, 50, 100} (finer).
3. **Recall semantics.** ACCEPTED **standard for all recall, incl. recall@10**. *Rationale:* one
   definition across all recall columns; recall is a coverage/position metric and standard is
   invariant-safe. *Alt:* keep recall@10 condensed (backward-compatible headline, but mixed
   definitions across columns).
4. **Drop vs score-0 for `n_scored==0`.** ACCEPTED **DROP (NaN→excluded) + per-system retrieval-failure
   count**. *Rationale:* score-0 would treat all-MISSING top-k as irrelevant (breaks the invariant);
   standard recall already scores empty/failed retrieval as 0, and query 383 is surfaced by the failure
   count. *Alt:* score-0 (penalizes failure, breaks invariant, grows the query set) / hybrid refinement
   (empty→0, all-MISSING→drop — one extra branch).
5. **Contrast family (exact list).** ACCEPTED the **6 contrasts** in Fix 3 (headline dense-vs-bm25,
   RRF-hurts-dense, hybrid+rerank-vs-dense, three rerank isolations). *Rationale:* answers every named
   research question with the minimum set. *Alt:* add pairwise reranker/embedder contrasts later
   (config-only).
6. **FDR headline metric(s).** ACCEPTED **{ndcg@10, recall@100}**. *Rationale:* the two headline claims
   (quality + coverage), nearly orthogonal, no collinear inflation. *Alt:* {ndcg@10} only (smallest
   family).
7. **Comparison-CSV shape.** ACCEPTED the **14-col `system_a/system_b/... in_family, n_common`** header.
   *Rationale:* expresses arbitrary contrasts, self-documents family membership + query count. *Alt:*
   keep `baseline/variant` naming — rejected (cannot express variant-vs-variant).
8. **Significance test default.** ACCEPTED **permutation (mean-δ), Wilcoxon demoted to opt-in**.
   *Rationale:* aligns the test with the reported mean-difference estimand; already implemented and
   seeded. *Alt:* remove Wilcoxon entirely (more deletion, loses a diagnostic + more doc surgery).
9. **`n_common`/`n_excluded` reporting.** ACCEPTED **`n_common` per comparison row + `n_excluded` per
   metric and retrieval-failure counts in `run_config`**. *Rationale:* each row self-documents its query
   set; run-level facts stay in run metadata. *Alt:* run_config only (leaner CSV, less self-documenting).
