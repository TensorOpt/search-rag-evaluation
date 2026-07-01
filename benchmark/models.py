"""Pure frozen data models and shared enums (docs/experiment.md §3.1-§3.5). Phase 1.

Plain frozen dataclasses + enums ONLY. No Protocols and no business logic live here
(Protocols are in ``protocols.py``; the pipeline-config types ``StageCfg``/``FuseCfg``/
``RerankCfg``/``PipelineSpec`` live in ``pipeline.py`` per §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Query:
    """A search request (§3.1)."""

    query_id: str
    text: str
    query_class: str | None = None


@dataclass(frozen=True)
class Document:
    """A retrievable item: ``doc_id`` + a backend-agnostic field bag (§3.1)."""

    doc_id: str
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class Qrel:
    """A graded relevance judgement ``(query_id, doc_id) -> gain`` (§3.1).

    ``gain`` is a float relevance grade. For WANDS: ``Exact=1.0``, ``Partial=0.5``,
    ``Irrelevant=0.0``. The grade set is a per-dataset modeling choice applied by the
    dataset adapter's label->gain mapping; the rest of the harness only sees floats.
    """

    query_id: str
    doc_id: str
    gain: float


@dataclass(frozen=True)
class ScoredDoc:
    """One scored doc (§3.1).

    Note: there is intentionally NO ``position`` field — position is derived as the
    1-based index into ``RankedResult.docs`` at CSV write time so it cannot drift.
    """

    doc_id: str
    score: float


@dataclass(frozen=True)
class RankedResult:
    """One query's ranked list, ordered by position; ``docs[0]`` is rank 1 (§3.1)."""

    query_id: str
    docs: Sequence[ScoredDoc]


class FieldRole(StrEnum):
    """Declared role of a dataset field — tells the indexer how to map/use it (§3.2).

    Text roles BM25 and SEMANTIC_SOURCE both feed the canonical ``search_text`` field
    (see FieldSchema); because ``search_text`` is *both* the BM25 target and the semantic
    source, a field marked either way becomes searchable both lexically and semantically.
    """

    #: Unique document identifier -> becomes the backend doc id (ES ``_id``); not ranked.
    ID = "id"
    #: Text field concatenated into ``search_text`` for lexical (BM25) matching.
    BM25 = "bm25"
    #: Text field concatenated into ``search_text``, which is embedded as the semantic source.
    SEMANTIC_SOURCE = "semantic_source"
    #: Numeric field stored for filtering/faceting/analysis; not used for text ranking.
    NUMERIC = "numeric"
    #: Field retained in the index for retrieval/display/debug only; never ranked.
    STORED = "stored"


@dataclass(frozen=True)
class FieldSpec:
    """A single field's name + role (§3.2)."""

    name: str
    role: FieldRole


@dataclass(frozen=True)
class FieldSchema:
    """Declares field roles and the canonical text fields (§3.2, §5.1).

    ``search_text_field`` names the canonical text field that the dataset adapter builds
    by CONCATENATING the values of every BM25- and SEMANTIC_SOURCE-role field, in schema
    order, joined by newlines (e.g. WANDS: product_name + product_description +
    product_features). That single field is used as BOTH the BM25 target and the semantic
    source, so every variant ranks the same input text (fair comparison). ``rerank_field``
    is the field text passed to the reranker. Both default to ``"search_text"``.
    """

    fields: Sequence[FieldSpec]
    search_text_field: str = "search_text"
    rerank_field: str = "search_text"


class InferenceTaskType(StrEnum):
    """Inference endpoint task type (§3.4)."""

    TEXT_EMBEDDING = "text_embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    RERANK = "rerank"


@dataclass(frozen=True)
class InferenceEndpoint:
    """Backend-agnostic inference endpoint descriptor (§3.4).

    ``service_settings`` carries auth/model identity (e.g. ``api_key``, ``model_id``);
    ``task_settings`` carries per-task knobs (e.g. rerank ``top_n``, ``return_documents``).
    They are kept SEPARATE maps and emitted separately by ``register_inference`` (§3.4).
    """

    inference_id: str
    task_type: InferenceTaskType
    service: str
    service_settings: Mapping[str, Any] = field(default_factory=dict)
    task_settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do server-side (§3.3).

    ``semantic_query`` (explicit ``{"semantic": {...}}`` query) is the hard 8.15 gate.
    """

    server_side_rrf: bool
    server_side_rerank: bool
    semantic_query: bool


@dataclass(frozen=True)
class IndexMapping:
    """Index identity + per-model semantic field names + backend-native mapping (§3.5)."""

    index_name: str
    search_text_field: str
    sem_fields: Mapping[str, str]
    backend_mapping: Mapping[str, Any]

    def sem_field(self, embedding_model_id: str) -> str:
        return self.sem_fields[embedding_model_id]
