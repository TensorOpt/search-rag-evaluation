# Example Results — WANDS reference run

A committed reference run so the harness's output can be inspected without a live ElasticSearch or
provider keys. The artifacts live under [`examples/runs/20260708T164458Z/`](../examples/runs/20260708T164458Z/).

This is the **WANDS** dataset on **ElasticSearch 8.x**, embedded with **Cohere `embed-english-v3.0`**
and reranked with **Cohere `rerank-v3.5`**, `seed: 1234`, under a configuration identical to the
shipped [`config.yaml`](../config.yaml) (uniform retrieval depth `top_k = 100`, `cutoff = 10`). Metric
definitions are in [`docs/methodology.md`](methodology.md); the artifact schemas are in the README
[Outputs](../README.md#outputs) section.

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

Reranking lifts every backbone into the same top band (~0.856–0.865); the three reranked systems are
close enough that the paired tests below cannot separate them.

## Decisive contrasts

Quoted from
[`comparison_20260708T164458Z.csv`](../examples/runs/20260708T164458Z/comparison_20260708T164458Z.csv),
rounded for readability (the file carries full precision). `delta = value_a − value_b` (positive means
`a` wins); the 95% CI is the unadjusted percentile bootstrap interval; `raw p` is the mean-δ sign-flip
permutation p-value; `adj p` is its Benjamini-Hochberg q-value over the 8-contrast × `ndcg@10` family.

| Contrast (`a` vs `b`) | metric | delta | 95% CI | raw p | adj p | n_common |
|-----------------------|--------|------:|--------|------:|------:|---------:|
| `hybrid_co_rerank` vs `semantic_co_rerank` | `ndcg@10` | +0.0036 | [-0.0034, 0.0112] | 0.338 | 0.338 | 469 |
| `semantic_co_rerank` vs `bm25_rerank` | `ndcg@10` | +0.0070 | [-0.0028, 0.0169] | 0.165 | 0.220 | 469 |
| `semantic_co_rerank` vs `bm25_rerank` | `recall@100` | +0.0184 | [0.0061, 0.0305] | 0.0031 | — | 479 |

**Top-10 ranking quality (the two `ndcg@10` rows).** Both deltas are small and their 95% CIs include
0, and neither is significant even before FDR adjustment. Once a cross-encoder reranker is applied,
the retrieval backbone (lexical vs dense vs hybrid) is not distinguishable on `ndcg@10` on this
dataset: adding RRF fusion on top of a reranked dense system does not move it, and a reranked dense
system does not clearly beat a reranked lexical one.

**Depth-recall counterpoint (the `recall@100` row).** At retrieval depth 100, `semantic_co_rerank`
pulls in more relevant documents than `bm25_rerank` (delta +0.018, CI excludes 0, raw p 0.003) — dense
retrieval improves coverage even where it does not move top-10 nDCG. This is **descriptive only**:
`recall@100` is outside the FDR family (`adj p` empty, `in_family = false`), because for a rerank-only
contrast at `W = 100` the top-100 document set is unchanged by reranking, so the metric is structurally
degenerate for the top-band comparisons (methodology.md §8.3). Read it as effect-size context, not a
confirmed claim.

## Scope

One dataset (WANDS), one embedder (Cohere `embed-english-v3.0`), one reranker (Cohere `rerank-v3.5`).
These numbers are illustrative of the harness's output; they are **not** claimed to generalize to other
datasets, models, or domains.

## Reproducing the statistics

The [README quickstart](../README.md#tldr-quickstart) plus `seed: 1234` reproduces the comparison
statistics given the same retrieval results (the bootstrap and permutation tests are seeded;
`run_config_20260708T164458Z.json` records the resolved config and seed). Retrieval itself is subject
to ES scoring ties and approximate-kNN nondeterminism, so absolute metric values may differ slightly
run to run; the paired tests are computed over whatever retrieval produced.