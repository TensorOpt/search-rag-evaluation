"""Harness-side windowed rerank fallback ``rerank_local`` (docs/experiment.md §3.7). Phase 4.

Pure Python; imports only ``benchmark.models`` + stdlib. Used only when a backend cannot rerank
server-side (ES reranks via ``text_similarity_reranker`` and never touches this path).

Design note (resolves the §3.7 signature gap): the ``Reranker`` Protocol (§3.4) is a pure
descriptor exposing only ``as_endpoint()`` — it CANNOT score documents locally. So ``rerank_local``
takes a ``score_fn`` instead of a ``Reranker``: a non-ES backend wraps its reranker inference call
into ``score_fn(query, doc_texts) -> one relevance score per text`` (higher = more relevant).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from benchmark.models import Query, ScoredDoc


def rerank_local(
    query: Query,
    candidates: Sequence[ScoredDoc],
    *,
    rank_window_size: int,
    doc_text: Callable[[str], str],
    score_fn: Callable[[Query, Sequence[str]], Sequence[float]],
) -> list[ScoredDoc]:
    """Re-rank the top ``rank_window_size`` candidates with ``score_fn`` (§3.7).

    Only the HEAD (top ``rank_window_size`` candidates) is rescored: ``score_fn`` is called once
    with ``(query, [doc_text(c.doc_id) for c in head])`` and returns one score per doc text (higher
    = more relevant). The head is returned as ``ScoredDoc(doc_id, model_score)`` re-sorted by model
    score DESC, tie-break ``doc_id`` ASC (§9.1). The TAIL (candidates beyond the window) keeps its
    input order and input scores and is appended AFTER the reranked head — matching ES.
    """
    head = list(candidates[:rank_window_size])
    tail = list(candidates[rank_window_size:])

    model_scores = list(score_fn(query, [doc_text(candidate.doc_id) for candidate in head]))
    if len(model_scores) != len(head):
        raise ValueError(
            f"score_fn returned {len(model_scores)} scores for {len(head)} documents; "
            "it must return exactly one score per document."
        )

    reranked_head = [
        ScoredDoc(candidate.doc_id, model_score)
        for candidate, model_score in zip(head, model_scores)
    ]
    reranked_head.sort(key=lambda scored_doc: (-scored_doc.score, scored_doc.doc_id))
    return reranked_head + tail
