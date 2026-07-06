"""Rerank-client factory (d): ``make_reranker`` + the provider->connector dispatch table (docs/experiment.md §5.4).

Layer (d): depends on ``providers`` (the concrete connectors in ``benchmark.providers.inference``) +
``common``. The dispatch table lived in the former ``providers.py``; it moves here so the
provider-selection seam is an explicit domain layer above the connectors. ``RERANKER_PROVIDERS`` is
the source of truth for the valid provider names (``config.py`` mirrors them for offline validation,
§11) — OpenAI is DELIBERATELY absent (it has no reranker, §3.4).
"""

from __future__ import annotations

from typing import Any, Mapping

from benchmark.providers.inference import CohereReranker, VoyageReranker, _BaseReranker

#: Reranker providers — OpenAI is DELIBERATELY absent (it has no reranker, §3.4).
_RERANKER_CLASSES: Mapping[str, type[_BaseReranker]] = {
    "cohere": CohereReranker,
    "voyage": VoyageReranker,
}

RERANKER_PROVIDERS = frozenset(_RERANKER_CLASSES)


def make_reranker(name: str, provider: str, settings: Mapping[str, Any]) -> _BaseReranker:
    """Build the rerank connector for ``provider`` (§5.4). Unknown / openai provider raises."""
    cls = _RERANKER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(
            f"unknown reranker provider {provider!r}; expected one of {sorted(_RERANKER_CLASSES)} "
            "(openai has no reranker)"
        )
    return cls(name, settings)
