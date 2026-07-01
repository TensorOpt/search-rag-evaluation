"""Harness-side windowed RRF fallback ``fuse_rrf_local`` (docs/experiment.md §3.7). Phase 4.

Pure Python (dict accumulation); imports only ``benchmark.models`` + stdlib. Used only when a
backend cannot fuse server-side (ES fuses via native ``rrf`` and never touches this path). Mirrors
ES ``rrf`` window semantics: each child list is fused over its own ``rank_window_size`` window.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmark.models import ScoredDoc


def fuse_rrf_local(
    lists: Sequence[Sequence[ScoredDoc]],
    *,
    rank_constant: int,
    rank_window_size: int,
) -> list[ScoredDoc]:
    """Reciprocal-rank-fuse several ranked lists over a fixed window (§3.7).

    Each input list is TRUNCATED to its top ``rank_window_size`` docs BEFORE fusing; a doc's
    contribution from one list is ``1 / (rank_constant + rank)`` with ``rank`` 1-based within that
    truncated list, and a doc appearing in several lists SUMS its contributions. A doc that appears
    only beyond ``rank_window_size`` in every list contributes nothing and is excluded. Returns
    ``ScoredDoc(doc_id, fused_score)`` sorted by fused score DESC, tie-break ``doc_id`` ASC (§9.1).
    """
    fused_scores: dict[str, float] = {}
    for ranked_list in lists:
        for zero_based_index, scored_doc in enumerate(ranked_list[:rank_window_size]):
            rank = zero_based_index + 1
            contribution = 1.0 / (rank_constant + rank)
            fused_scores[scored_doc.doc_id] = fused_scores.get(scored_doc.doc_id, 0.0) + contribution
    return [
        ScoredDoc(doc_id, score)
        for doc_id, score in sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    ]
