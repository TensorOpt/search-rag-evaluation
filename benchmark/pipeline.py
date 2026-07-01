"""SearchPipeline + pipeline-config dataclasses (docs/experiment.md §3.6, §3.7). Phase 5.

The single DRY retrieval path: every variant is a ``PipelineSpec`` composed and executed
through one ``SearchPipeline``. ``plan()`` composes retrieve -> [fuse] -> [rerank] with no
per-variant branching beyond the presence/absence of ``fuse``/``rerank``; it uses the backend's
server-side combinators when ``capabilities()`` allows, else wires the harness-side windowed RRF
fallback (``fuse_rrf_local``, §3.7). The harness-side RERANK fallback is DEFERRED (§3.7/§13): if a
rerank stage is requested but cannot run server-side, ``plan()`` raises ``NotImplementedError``.

Imports only ``benchmark.models`` / ``benchmark.protocols`` / ``benchmark.fusion`` + stdlib —
never adapters, ``matrix``, ``rerank_local``, or numpy (§11). ``spec_for`` (VariantCfg ->
PipelineSpec) lives in ``matrix.py`` (Phase 6), not here, to avoid a pipeline->matrix forward
dependency (§4).
"""

from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence

from benchmark.fusion import fuse_rrf_local
from benchmark.models import Query, RankedResult
from benchmark.protocols import RetrieverSpec, SearchBackend


@dataclass(frozen=True)
class StageCfg:
    """One retrieval leaf: a BM25 query over ``fields`` or a semantic query over ``field`` (§3.6)."""

    kind: Literal["bm25", "semantic"]
    fields: Sequence[str] = ()
    field: str | None = None

    @classmethod
    def bm25(cls, *, fields: Sequence[str]) -> StageCfg:
        return cls(kind="bm25", fields=tuple(fields))

    @classmethod
    def semantic(cls, *, field: str) -> StageCfg:
        return cls(kind="semantic", field=field)


@dataclass(frozen=True)
class FuseCfg:
    """RRF fusion knobs (§3.6)."""

    rank_constant: int
    rank_window_size: int


@dataclass(frozen=True)
class RerankCfg:
    """Reranker stage knobs (§3.6)."""

    inference_id: str
    field: str
    rank_window_size: int


@dataclass(frozen=True)
class PipelineSpec:
    """A fully-expanded, query-independent variant plan (§3.6)."""

    retrievers: Sequence[StageCfg]
    fuse: FuseCfg | None = None
    rerank: RerankCfg | None = None


@dataclass(frozen=True)
class _LocalFusePlan:
    """Pipeline-private harness-side fuse plan (§3.7): fuse ``leaves`` with ``fuse_rrf_local``.

    Returned by ``plan()`` instead of a backend ``RetrieverSpec`` when the backend cannot fuse
    server-side. ``run()`` executes each leaf per query and fuses the candidate lists locally.
    """

    leaves: tuple[RetrieverSpec, ...]
    fuse: FuseCfg


class SearchPipeline(ABCMeta):
    """The one backend-agnostic pipeline; all six variants run through ``run()`` (§3.6)."""

    def __init__(self, backend: SearchBackend) -> None:
        self.backend = backend

    def plan(self, spec: PipelineSpec) -> RetrieverSpec | _LocalFusePlan:
        """Compose retrieve -> [fuse] -> [rerank]. Pure composition; only branches on
        presence/absence of ``fuse``/``rerank`` and on ``capabilities()`` (§3.6/§3.7)."""
        leaves: list[RetrieverSpec] = []
        for stage in spec.retrievers:
            if stage.kind == "bm25":
                leaves.append(self.backend.bm25(fields=stage.fields))
            elif stage.kind == "semantic":
                if stage.field is None:
                    raise ValueError("semantic StageCfg requires a field")
                leaves.append(self.backend.semantic(field=stage.field))
            else:
                raise ValueError(f"unknown retrieval stage kind: {stage.kind!r}")

        base: RetrieverSpec | _LocalFusePlan
        if spec.fuse is None:
            if len(leaves) != 1:
                raise ValueError(
                    f"a spec without a fuse stage must have exactly one retriever, got {len(leaves)}"
                )
            base = leaves[0]
        elif self.backend.capabilities().server_side_rrf:
            base = self.backend.fuse_rrf(
                leaves,
                rank_constant=spec.fuse.rank_constant,
                rank_window_size=spec.fuse.rank_window_size,
            )
        else:
            base = _LocalFusePlan(leaves=tuple(leaves), fuse=spec.fuse)

        if spec.rerank is None:
            return base

        # Rerank is only supported fully server-side. The harness-side rerank fallback is
        # deferred (§3.7/§13): a local fuse plan or a backend without server_side_rerank cannot
        # be reranked here. ponytail: raise now; wire rerank_local when a non-server-side-rerank
        # backend actually exists.
        if isinstance(base, _LocalFusePlan) or not self.backend.capabilities().server_side_rerank:
            raise NotImplementedError(
                "harness-side rerank fallback is deferred (see docs/experiment.md §3.7/§13); "
                "this backend cannot rerank server-side"
            )
        return self.backend.rerank(
            base,
            inference_id=spec.rerank.inference_id,
            field=spec.rerank.field,
            rank_window_size=spec.rerank.rank_window_size,
        )

    def run(
        self, spec: PipelineSpec, queries: Iterable[Query], *, top_k: int
    ) -> Iterator[RankedResult]:
        """Build the plan once, then yield one ``RankedResult`` per query, in query order (§3.6)."""
        plan = self.plan(spec)
        for query in queries:
            if isinstance(plan, _LocalFusePlan):
                candidate_lists = [
                    self.backend.execute(leaf, query, top_k=plan.fuse.rank_window_size).docs
                    for leaf in plan.leaves
                ]
                fused = fuse_rrf_local(
                    candidate_lists,
                    rank_constant=plan.fuse.rank_constant,
                    rank_window_size=plan.fuse.rank_window_size,
                )
                yield RankedResult(query.query_id, fused[:top_k])
            else:
                yield self.backend.execute(plan, query, top_k=top_k)
