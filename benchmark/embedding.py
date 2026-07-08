"""Embedder factory (c): ``make_embedder`` + the provider->connector dispatch table (docs/architecture.md §3.4).

Layer (c): depends on ``providers`` (the concrete connectors in ``benchmark.providers.inference``) +
``common``. The dispatch table lived in the former ``providers.py``; it moves here so the
provider-selection seam is an explicit domain layer above the connectors. ``EMBEDDER_PROVIDERS`` is
the source of truth for the valid provider names (``config.py`` mirrors them for offline validation,
§11).
"""

from __future__ import annotations

from typing import Any, Mapping

from benchmark.providers.inference import (
    CohereEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    _BaseEmbedder,
)

#: Embedder providers. Source of truth for config-time validation (config.py mirrors these names).
_EMBEDDER_CLASSES: Mapping[str, type[_BaseEmbedder]] = {
    "openai": OpenAIEmbedder,
    "cohere": CohereEmbedder,
    "voyage": VoyageEmbedder,
}

EMBEDDER_PROVIDERS = frozenset(_EMBEDDER_CLASSES)


def make_embedder(name: str, provider: str, settings: Mapping[str, Any]) -> _BaseEmbedder:
    """Build the embedding connector for ``provider`` (§3.4). Unknown provider raises (exhaustive)."""
    cls = _EMBEDDER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(
            f"unknown embedder provider {provider!r}; expected one of {sorted(_EMBEDDER_CLASSES)}"
        )
    return cls(name, settings)
