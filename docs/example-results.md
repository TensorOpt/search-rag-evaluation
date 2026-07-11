# Example Results — WANDS reference run

A committed reference run so the harness's output can be inspected without a live ElasticSearch or
provider keys. The artifacts live under [`examples/runs/20260708T164458Z/`](../examples/runs/20260708T164458Z/).

This is the **WANDS** dataset on **ElasticSearch 8.x**, embedded with **Cohere `embed-english-v3.0`**
and reranked with **Cohere `rerank-v3.5`**, `seed: 1234`, under a configuration identical to the
shipped [`config.yaml`](../config.yaml) (uniform retrieval depth `top_k = 100`, `cutoff = 10`,
condensed unjudged policy). Metric definitions are in [`docs/methodology.md`](methodology.md); the
artifact schemas are in the README [Outputs](../README.md#outputs) section. WANDS provides 480
queries; 469 form the family-wide common subset for the condensed metrics and 479 for recall
(`R > 0` required), as recorded in the `diagnostics` block of
[`run_config_20260708T164458Z.json`](../examples/runs/20260708T164458Z/run_config_20260708T164458Z.json).

## Headline: mean `ndcg@10` per system

Mean over the non-empty `ndcg@10` cells in
[`metrics_20260708T164458Z.csv`](../examples/runs/20260708T164458Z/metrics_20260708T164458Z.csv)
(`n` = queries with a defined `ndcg@10`; a query is empty when `n_scored = 0`). Baseline first, then
variants in config order.

| System | mean `ndcg@10` | n |
|--------|---------------:|--:|
| `bm25` (baseline) | 0.799 | 470 |
| `semantic_co` | 0.843 | 472 |
| `hybrid_co_k60` | 0.834 | 473 |
| `bm25_rerank` | 0.856 | 470 |
| `semantic_co_rerank` | 0.864 | 472 |
| `hybrid_co_rerank` | 0.865 | 473 |

Reranking lifts every backbone into the same top band (0.856 to 0.865). The paired tests below make
that observation precise: the reranked systems cannot be separated on `ndcg@10`.

## The eight research questions, answered

[methodology.md §1.2](methodology.md) turns the {bm25, semantic, hybrid} × {rerank off, rerank on}
factorial into eight explicit contrasts. The table below is the **complete `ndcg@10` FDR family**
for this run, quoted from
[`comparison_20260708T164458Z.csv`](../examples/runs/20260708T164458Z/comparison_20260708T164458Z.csv)
and rounded for readability (the file carries full precision). `delta = value_a − value_b`
(positive means `a` wins); the 95% CI is the unadjusted percentile bootstrap interval; `raw p` is
the mean-δ sign-flip permutation p-value, whose resolution floor is `1/(B+1) = 0.0001` at
`B = 10000`; `adj p` is the Benjamini-Hochberg q-value over the 8-test family; `sig` is the FDR
decision at `q = 0.05`. `n_common = 469` throughout.

| # | Contrast (`a` vs `b`) | delta | 95% CI | raw p | adj p | sig |
|---|-----------------------|------:|--------|------:|------:|-----|
| 1 | `semantic_co` vs `bm25` | +0.0431 | [0.0285, 0.0579] | 0.0001 | 0.00016 | yes |
| 2 | `hybrid_co_k60` vs `semantic_co` | −0.0062 | [−0.0151, 0.0030] | 0.195 | 0.223 | no |
| 3 | `hybrid_co_rerank` vs `semantic_co` | +0.0250 | [0.0138, 0.0367] | 0.0001 | 0.00016 | yes |
| 4 | `bm25_rerank` vs `bm25` | +0.0576 | [0.0446, 0.0707] | 0.0001 | 0.00016 | yes |
| 5 | `semantic_co_rerank` vs `semantic_co` | +0.0215 | [0.0126, 0.0307] | 0.0001 | 0.00016 | yes |
| 6 | `hybrid_co_rerank` vs `hybrid_co_k60` | +0.0312 | [0.0204, 0.0424] | 0.0001 | 0.00016 | yes |
| 7 | `semantic_co_rerank` vs `bm25_rerank` | +0.0070 | [−0.0028, 0.0169] | 0.165 | 0.220 | no |
| 8 | `hybrid_co_rerank` vs `semantic_co_rerank` | +0.0036 | [−0.0034, 0.0112] | 0.338 | 0.338 | no |

### Q1. Does dense retrieval beat lexical? Yes.

Unreranked, `semantic_co` beats `bm25` by +0.0431 nDCG@10, significant after FDR correction. The
descriptive metrics point the same way: `precision@10` +0.0515 (CI [0.0384, 0.0657], raw p 0.0001)
and `avg_relevance` +0.0370 (CI [0.0260, 0.0484], raw p 0.0001). On this dataset the embedding
backbone is a genuinely better first-stage ranker than BM25.

### Q2. Does RRF-with-BM25 help or hurt dense? Neither, at the cutoff. It trades top-10 precision for depth recall.

On `ndcg@10` the effect is not significant (−0.0062, CI spans 0). The descriptive rows show what
fusion actually does:

| metric | delta | 95% CI | raw p |
|--------|------:|--------|------:|
| `precision@10` | −0.0187 | [−0.0265, −0.0115] | 0.0001 |
| `avg_relevance` | −0.0079 | [−0.0148, −0.0011] | 0.026 |
| `recall@50` | +0.0129 | [0.0059, 0.0209] | 0.0007 |
| `recall@100` | +0.0120 | [0.0039, 0.0206] | 0.0050 |

Fusing BM25 into the dense list pulls additional relevant documents into the top 50 and top 100
while slightly diluting the top 10. On this run, **RRF fusion behaves as a recall mechanism at
depth, not a precision mechanism at the cutoff**. (These rows are outside the FDR family and are
effect-size context, not confirmed claims; the verdict is also stated at the fixed
`rank_constant = 60`, with robustness to that choice delegated to `eval:sweep --axis=rrf_k`,
methodology.md §1.2.)

### Q3. Is hybrid+rerank distinguishable from unreranked dense? Yes, and the factorial attributes the gain to the reranker.

`hybrid_co_rerank` beats `semantic_co` by +0.0250, significant. This is a composite contrast: the
two systems differ by both fusion and reranking. The isolation cells decompose it: reranking alone
on dense is worth +0.0215 (Q5), while fusion under reranking is worth +0.0036 and not significant
(Q8). The composite gain is the reranker's.

### Q4 to Q6. Rerank effect isolated on each shape: significant everywhere, largest on the weakest backbone.

All three rerank isolations are significant at `adj p = 0.00016`: +0.0576 on lexical, +0.0312 on
hybrid, +0.0215 on dense. The ordering mirrors the unreranked headline means (bm25 0.799 < hybrid
0.834 < dense 0.843): the worse the first stage ranks the top of the list, the more the
cross-encoder recovers. After reranking, the three backbones converge into the 0.856 to 0.865 band.

### Q7. Once both are reranked, does dense still beat lexical? Not on top-10 ranking. It still buys coverage at depth.

`semantic_co_rerank` vs `bm25_rerank` on `ndcg@10` is +0.0070, not significant even before
adjustment (raw p 0.165). The descriptive rows split by depth:

| metric | delta | 95% CI | raw p |
|--------|------:|--------|------:|
| `recall@10` | −0.0006 | [−0.0038, 0.0022] | 0.694 |
| `recall@50` | +0.0021 | [−0.0082, 0.0122] | 0.677 |
| `recall@100` | +0.0184 | [0.0061, 0.0305] | 0.0031 |
| `precision@10` | +0.0118 | [0.0025, 0.0217] | 0.0145 |

At depth 100, where the rank order no longer matters and only backbone set membership does, the
dense backbone retrieves more of the relevant set. `precision@10` is raw-significant in dense's
favor but, like the recall rows, sits outside the FDR family; read both as effect-size context. The
practical reading on this dataset: a reranked BM25 pipeline reaches the top nDCG band with no
embedding infrastructure, while the dense backbone's remaining advantage is coverage of the
relevant set at depth.

### Q8. Once both sides are reranked, does RRF fusion still help? No.

`hybrid_co_rerank` vs `semantic_co_rerank` on `ndcg@10` is +0.0036, the weakest contrast in the
family (raw p 0.338). Fusion's depth effect persists descriptively (`recall@50` +0.0147,
CI [0.0076, 0.0225], raw p 0.0001), but with a reranker already ordering the window, fusion adds
nothing measurable at the cutoff.

## What this run shows

Three findings, each carried by multiple contrasts rather than a single row:

1. **Reranking is the precision mechanism.** All three rerank isolations (Q4 to Q6) are large and
   significant; every other significant `ndcg@10` contrast in the family is attributable to the
   presence of a reranker on one side (Q3) or to the unreranked backbone gap (Q1).
2. **Once a reranker is present, the retrieval backbone stops mattering at the cutoff.** Q7 and Q8,
   the two contrasts comparing reranked systems to each other, are both null: neither embeddings
   nor RRF fusion produces a detectable `ndcg@10` difference over reranked BM25 or reranked dense.
3. **Fusion and dense retrieval earn their keep at depth, not at the cutoff.** Fusion adds recall
   at 50 and 100 while diluting the top 10 (Q2); the dense backbone adds recall at 100 over lexical
   even under reranking (Q7). Depth recall is what a reranker consumes, so these effects matter for
   pipelines whose rerank window extends to that depth.

## Descriptive rows and the `recall@100` family exclusion

The FDR family in this run is `ndcg@10` only (`fdr_metrics: [ndcg@10]`, 8 contrasts × 1 metric =
8 BH tests); every other metric row carries a raw p-value and CI but no q-value
(`in_family = false`, `adj p` empty). `recall@100` was removed from the family because it is
**structurally degenerate for the three rerank-only contrasts** (Q4 to Q6): at
`rerank_window_size = top_k = 100` a reranker permutes the top-100 set without changing its
membership, so `recall@100` is identical by construction on both sides. The artifacts show exactly
that: those three rows have `delta = 0.0`, empty CIs, and `p = 1.0`, and the run-config
`diagnostics.stats.excluded` block records each with its identity reason. For contrasts whose
backbones differ (Q1, Q2, Q7, Q8) `recall@100` is well identified and is reported descriptively
(methodology.md §8.3).

## Scope

One dataset (WANDS), one embedder (Cohere `embed-english-v3.0`), one reranker (Cohere
`rerank-v3.5`), one fusion setting (`rank_constant = 60`). These numbers are illustrative of the
harness's output; they are **not** claimed to generalize to other datasets, models, or domains.

## Reproducing the statistics

The [README quickstart](../README.md#tldr-quickstart) plus `seed: 1234` reproduces the comparison
statistics given the same retrieval results (the bootstrap and permutation tests are seeded;
`run_config_20260708T164458Z.json` records the resolved config and seed). Retrieval itself is
subject to ES scoring ties and approximate-kNN nondeterminism, so absolute metric values may differ
slightly run to run; the paired tests are computed over whatever retrieval produced.
