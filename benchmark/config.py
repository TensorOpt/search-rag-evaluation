"""Resolved-config value types + config load/resolve + pipeline assembly + adapter factories (docs/architecture.md §4, §10, §11). Phase 6.

This module is the whole configuration layer. It holds:

- The pure resolved-config value types — :class:`EmbedderCfg`/:class:`RerankerCfg`/:class:`SearcherCfg`
  + the :class:`Services` registry, :class:`FuserCfg`, :class:`PipelineCfg`, :class:`ResolvedConfig`.
  Pipelines are named + explicit (§10): the config declares one ``baseline`` plus a map of named
  ``variants``, each resolved into a :class:`PipelineCfg`. There is NO matrix expansion and NO sweep.
- :func:`build_pipeline`, which composes one ``PipelineCfg`` into a ``SearchPipeline`` object graph
  (§4) from the pre-built leaf ``Searcher`` / ``Reranker`` maps — no factory object.
- The loader: parses the explicit §10 YAML (``dataset`` / ``services`` / ``indexer`` / ``pipelines`` /
  ``stats`` / ``cutoff`` / ``top_k``), substitutes whole-value ``${VAR}`` environment placeholders at
  load (secrets never live in the file), validates it, and resolves it into a :class:`ResolvedConfig`.
  Embedders/rerankers are provider connectors (Cohere/Voyage/OpenAI, §3.4): the config validates the
  ``provider`` offline (no network) and the runner instantiates them lazily via :func:`make_embedders`
  / :func:`make_rerankers`.
- The lazy factories ``load_dataset`` / ``make_index_writer`` / ``make_searchers`` /
  ``make_rerankers_bound`` / ``make_embedders`` / ``make_rerankers``, which dispatch on
  ``dataset.name`` / ``indexer.provider`` / the connector ``provider`` to a dotted-path target. They do
  NOT import the adapter or ``benchmark.providers`` at import time (offline, §11); the live import +
  construct resolves at CALL time. An unknown name/provider raises.

Imports ``benchmark.common.models`` / ``benchmark.common.protocols`` / ``benchmark.evaluation.stats`` /
``benchmark.search`` (the composers, for :func:`build_pipeline` — a one-way wiring edge, §11) + pyyaml
+ stdlib. It NEVER imports an adapter module at import time (§11): the factories resolve their dotted
targets lazily.
"""

from __future__ import annotations

import importlib
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from benchmark.common.cache import CachingEmbedder, CachingRerankClient, DiskCache
from benchmark.common.logging_setup import get_logger
from benchmark.common.models import IndexMapping
from benchmark.common.protocols import Embedder, Reranker, RerankClient, Searcher
from benchmark.evaluation.stats import (
    CANONICAL_METRICS,
    DEFAULT_FDR_METRICS,
    Contrast,
    StatsCfg,
)
from benchmark.search import HybridSearch, RRFFuser, SearchPipeline

logger = get_logger(__name__)

#: Valid embedder/reranker connector providers (§3.4). These MIRROR ``benchmark.providers``
#: (``EMBEDDER_PROVIDERS`` / ``RERANKER_PROVIDERS`` there are the source of truth); duplicated here so
#: config validation stays offline (no ``benchmark.providers`` import at config time, §11). OpenAI is
#: deliberately absent from rerankers — it has no reranker.
_EMBEDDER_PROVIDERS = ("cohere", "voyage", "openai")
_RERANKER_PROVIDERS = ("cohere", "voyage")

#: ``${VAR}`` env placeholder (§10). Whole-value only — secrets are always their own scalar.
_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

#: Config keys whose values are secrets. Any key CONTAINING one of these substrings
#: (case-insensitive) must be supplied as a ``${VAR}`` placeholder at load (never a literal), and is
#: redacted to its ``${VAR}`` name in the run manifest (io_csv ``_redact_secrets``). One definition,
#: shared with ``io_csv`` so the reject-at-load rule and the redact-at-write rule use the same key set.
_SECRET_KEY_RE = re.compile(r"(api_key|token|secret|password|credential)", re.IGNORECASE)

#: Adapter dispatch: dataset.name / indexer.provider -> dotted "module:attr" target. Resolved
#: LAZILY (Phase 11) so this phase imports no adapter and stays offline (§11).
DATASET_TARGETS: Mapping[str, str] = {
    "wands": "benchmark.datasets.wands:WandsDataset",
}
#: The ``IndexWriter`` (§3.5) per provider — the domain ``Indexer`` (built by the runner) delegates
#: mapping/field-naming/ingest to it. Resolved LAZILY so swapping the backend is a config-only edit.
INDEX_WRITER_TARGETS: Mapping[str, str] = {
    "elasticsearch": "benchmark.providers.elasticsearch:ESIndexWriter",
}
#: The leaf ``Searcher`` / ``Reranker`` builders per provider — mint the full configured set over one
#: shared client (§4). Resolved LAZILY like the others so ``config`` names no adapter at import time.
SEARCHER_BUILDER_TARGETS: Mapping[str, str] = {
    "elasticsearch": "benchmark.providers.elasticsearch:build_searchers",
}
RERANKER_BUILDER_TARGETS: Mapping[str, str] = {
    "elasticsearch": "benchmark.providers.elasticsearch:build_rerankers",
}

#: Valid searcher kinds (§10). Exhaustive — anything else is a ConfigError.
_SEARCHER_KINDS = ("lexical", "vector")
#: Valid fuser types (§10). Only RRF today; anything else is a ConfigError.
_FUSER_TYPES = ("rrf",)


# --- resolved-config value types (§4, §10) -----------------------------------------------------


@dataclass(frozen=True)
class EmbedderCfg:
    """A named embedder service (§10 ``services`` block).

    ``name`` is the config-local reference (== the embedder id used for sem-field naming, §3.5);
    ``provider`` selects the connector (``cohere`` / ``voyage`` / ``openai``, §3.4); ``settings``
    carries connector knobs (``model_id``, ``api_key``, ``rate_limit`` …). The runner instantiates
    the connector lazily via :func:`make_embedders`.
    """

    name: str
    provider: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class RerankerCfg:
    """A named reranker service (§10 ``services`` block).

    ``provider`` selects the rerank connector (``cohere`` / ``voyage`` — OpenAI has no reranker,
    §3.4); ``settings`` carries connector knobs plus ``top_n`` (the §5.3 W <= top_n cap the runner
    reads at R0). The runner instantiates the connector lazily via :func:`make_rerankers` and hands
    it to the ES ``searcher_factory`` to build an ``ESReranker`` (``Reranker`` is a behavioral ABC
    realized by the backend).
    """

    name: str
    provider: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class SearcherCfg:
    """A named searcher service (§10 ``services`` block).

    ``kind`` is ``"lexical"`` or ``"vector"``; a vector searcher references an ``embedder`` service
    name (``None`` for lexical). ``build_pipeline`` resolves these into leaf ``Searcher``s.
    """

    name: str
    provider: str
    kind: str
    embedder: str | None


@dataclass(frozen=True)
class Services:
    """The typed registry of named services (§10 ``services`` block).

    Embedders/rerankers/searchers keyed by their config name. The accessors raise a clear
    ``KeyError``-style ``ValueError`` on a missing/mistyped reference so ``build_pipeline`` and
    config validation fail loudly rather than silently.
    """

    embedders: Mapping[str, EmbedderCfg]
    rerankers: Mapping[str, RerankerCfg]
    searchers: Mapping[str, SearcherCfg]

    def embedder(self, name: str) -> EmbedderCfg:
        if name not in self.embedders:
            raise ValueError(f"unknown embedder service {name!r}; known: {sorted(self.embedders)}")
        return self.embedders[name]

    def reranker(self, name: str) -> RerankerCfg:
        if name not in self.rerankers:
            raise ValueError(f"unknown reranker service {name!r}; known: {sorted(self.rerankers)}")
        return self.rerankers[name]

    def searcher(self, name: str) -> SearcherCfg:
        if name not in self.searchers:
            raise ValueError(f"unknown searcher service {name!r}; known: {sorted(self.searchers)}")
        return self.searchers[name]


@dataclass(frozen=True)
class FuserCfg:
    """RRF fusion parameters for a multi-retriever pipeline (§10 ``fuser`` block).

    Only ``rrf`` fusion exists today; ``build_pipeline`` raises on any other ``type`` (exhaustive).
    ``rank_constant`` is the RRF k; ``window`` is the retrieval/fusion candidate depth W.
    """

    type: str
    rank_constant: int
    window: int


@dataclass(frozen=True)
class PipelineCfg:
    """One explicit, named pipeline from the config (§4, §10).

    ``id`` is the pipeline's config name (the map key; the baseline's is ``"baseline"`` by default).
    ``retrievers`` is a tuple of searcher service names (exactly one leaf when ``fuser`` is None; 2+
    when fusing). ``fuser`` is present iff there are multiple retrievers. ``reranker`` is a reranker
    service name (paired with ``rerank_window_size``); both are set together or both None.
    """

    id: str
    retrievers: tuple[str, ...]
    fuser: FuserCfg | None
    reranker: str | None
    rerank_window_size: int | None


@dataclass(frozen=True)
class CacheCfg:
    """Persistent inference/result cache config (docs/architecture.md §5.5).

    ``enabled`` is opt-in — a config with the ``cache`` block ABSENT is disabled (a clean cold run
    by default). ``dir`` is the cache directory (gitignored). A pure config value: it rides in
    ``run_config_{ts}.json`` as provenance but never affects metrics (the cache changes speed, not
    numbers). The live :class:`~benchmark.common.cache.DiskCache` is opened by the runner via
    :func:`open_cache`, never held here (this must serialize cleanly via ``dataclasses.asdict``).
    """

    enabled: bool = False
    dir: str = ".cache"


@dataclass(frozen=True)
class MetricsCfg:
    """The ONE relevance policy governing ALL six metrics (docs/methodology.md §7).

    ``unjudged`` selects how an unjudged (MISSING) returned doc is treated, UNIFORMLY across every
    metric: ``"condensed"`` (Sakai deletion — MISSING dropped from the eval list, the shipped
    default) or ``"irrelevant"`` (trec_eval — MISSING scored as gain ``0.0`` on the raw list,
    ``precision@k`` denom ``k``). ``relevance_threshold`` is the binary-relevance cut applied to
    EVERY binary metric (``precision@k``/``recall@k``) and to ``QrelIndex.relevant_count``/the qrels
    digest. Both keys default (condensed / 0.5) when the ``metrics`` block is absent.
    """

    unjudged: str = "condensed"
    relevance_threshold: float = 0.5


@dataclass(frozen=True)
class ResolvedConfig:
    """The fully-resolved run configuration (§10, §9.1 run metadata).

    ``dataset``/``indexer`` are the raw resolved config sections (this module dispatches the live
    adapter from ``dataset["name"]`` / ``indexer["provider"]``, deferred to Phase 11). ``services``
    is the typed registry. ``baseline`` is the reference pipeline; ``variants`` is the ORDERED list
    of explicit variant pipelines (iterate ``baseline`` first, then ``variants``, via
    :meth:`pipelines`). ``cutoff`` is the metric depth k; ``top_k`` is the retrieval depth.
    ``cache`` is the persistent-cache config (LAST + defaulted so keyword constructors — including
    the test helper — keep working without passing it; disabled by default).
    """

    dataset: Mapping[str, object]
    indexer: Mapping[str, object]
    services: Services
    baseline: PipelineCfg
    variants: Sequence[PipelineCfg]
    stats: StatsCfg
    cutoff: int
    top_k: int
    baseline_id: str
    timestamp: str
    seed: int
    cache: CacheCfg = CacheCfg()
    #: the single relevance policy (unjudged handling + threshold) over ALL six metrics.
    #: Defaulted (condensed / 0.5) so the ``metrics`` config block is optional and existing keyword
    #: constructors keep working; ``resolve_config`` always sets it from the resolved config.
    metrics: MetricsCfg = MetricsCfg()
    #: resolved-secret-value -> ``"${VAR}"`` placeholder name, recorded at load. Popped by
    #: ``io_csv.write_run_config`` BEFORE serialization (never written) and used to reconstruct the
    #: placeholder name for each redacted secret. Defaulted so existing keyword constructors keep
    #: working; empty when the config had no secrets.
    secret_env_refs: Mapping[str, str] = field(default_factory=dict)

    def pipelines(self) -> list[PipelineCfg]:
        """The run's pipelines, baseline first (§6)."""
        return [self.baseline, *self.variants]


# --- pipeline assembly (§4) --------------------------------------------------------------------


def build_pipeline(
    pcfg: PipelineCfg,
    searchers: Mapping[str, Searcher],
    rerankers: Mapping[str, Reranker],
) -> SearchPipeline:
    """Compose ``pcfg``'s ``SearchPipeline`` object graph from the pre-built leaf maps (§4).

    Each retriever name indexes a leaf ``Searcher`` in ``searchers`` (built once by
    :func:`make_searchers`). With a ``fuser`` the leaves are wrapped in a ``HybridSearch`` (RRF at
    ``fuser.rank_constant``, window ``fuser.window``); without one, exactly one leaf is expected
    (else ``ValueError``). A ``reranker`` name indexes a ``Reranker`` in ``rerankers`` and wraps the
    retriever in a ``SearchPipeline`` with a rerank pass at ``rerank_window_size``; else a bare
    pass-through pipeline. This is pure composition over plain provider objects — no factory, no
    adapter import.
    """
    leaves = [searchers[name] for name in pcfg.retrievers]

    if pcfg.fuser is not None:
        if pcfg.fuser.type != "rrf":
            raise ValueError(
                f"pipeline {pcfg.id!r}: unknown fuser type {pcfg.fuser.type!r}; only 'rrf' is supported"
            )
        retriever: Searcher = HybridSearch(
            retrievers=leaves,
            fuser=RRFFuser(rank_constant=pcfg.fuser.rank_constant),
            retrieval_window_size=pcfg.fuser.window,
        )
    else:
        if len(leaves) != 1:
            raise ValueError(
                f"pipeline {pcfg.id!r} has no fuser but built {len(leaves)} leaf retrievers; "
                "expected exactly one"
            )
        (retriever,) = leaves

    if pcfg.reranker is not None:
        return SearchPipeline(
            retriever=retriever,
            reranker=rerankers[pcfg.reranker],
            rerank_window_size=pcfg.rerank_window_size,
        )
    return SearchPipeline(retriever=retriever)


def _searcher_spec(searcher: SearcherCfg, services: Services) -> tuple[str, str, str | None]:
    """Translate one searcher service into a backend-agnostic ``(name, kind, embedder_id)`` spec (§4).

    Resolves a vector searcher's ``embedder_id`` via ``services.embedder`` (as the old ``_build_leaf``
    did); lexical searchers carry ``None``. Exhaustive on ``kind`` — an unknown kind raises.
    """
    if searcher.kind == "lexical":
        return (searcher.name, "lexical", None)
    if searcher.kind == "vector":
        if searcher.embedder is None:
            raise ValueError(f"vector searcher {searcher.name!r} has no embedder reference")
        embedder: EmbedderCfg = services.embedder(searcher.embedder)
        return (searcher.name, "vector", embedder.name)
    raise ValueError(
        f"searcher {searcher.name!r} has unknown kind {searcher.kind!r}; expected 'lexical' or 'vector'"
    )


# --- config load + resolve (§10) ---------------------------------------------------------------


class ConfigError(ValueError):
    """A malformed or incomplete config (missing key, unresolvable ``${VAR}``, unknown adapter,
    or a pipeline that violates the §10 field rules)."""


def _substitute_env(
    value: Any, *, key: str | None = None, refs: dict[str, str] | None = None
) -> Any:
    """Recursively replace whole-value ``${VAR}`` scalars with ``os.environ[VAR]`` (§10).

    A missing environment variable for a referenced placeholder is a clear error — secrets must be
    supplied at run time, never defaulted silently. ``key`` is the mapping key the scalar sits under
    (threaded on recursion) so the loader can enforce the secret rule: a scalar under a
    secret-named key (``_SECRET_KEY_RE``) MUST be a ``${VAR}`` placeholder, never a literal — else
    :class:`ConfigError` (fail-fast). While substituting, the ``${VAR}`` origin of each secret is
    recorded into ``refs`` (resolved value -> ``"${VAR}"``) so the manifest writer can emit the
    placeholder name, never the value.
    """
    if isinstance(value, str):
        is_secret = key is not None and _SECRET_KEY_RE.search(key) is not None
        match = _ENV_PLACEHOLDER.match(value)
        if match is None:
            if is_secret:
                raise ConfigError(
                    f"secret key {key!r} must be provided as a ${{VAR}} placeholder, "
                    "not a literal value"
                )
            return value
        var = match.group(1)
        if var not in os.environ:
            raise ConfigError(f"environment variable {var!r} referenced as ${{{var}}} is not set")
        resolved = os.environ[var]
        if is_secret and refs is not None:
            refs[resolved] = value  # value == "${VAR}"; resolved value -> its placeholder name
        return resolved
    if isinstance(value, Mapping):
        return {k: _substitute_env(v, key=k, refs=refs) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v, key=key, refs=refs) for v in value]
    return value


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise ConfigError(f"missing required key {key!r} in {where}")
    return mapping[key]


def load_config(path: str | Path) -> ResolvedConfig:
    """Load + resolve a YAML/JSON config file into a :class:`ResolvedConfig` (§10).

    ``${VAR}`` placeholders are substituted from the environment at load; a missing required key,
    unresolvable placeholder, or invalid pipeline raises :class:`ConfigError`.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)  # YAML is a JSON superset, so this also parses JSON configs.
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return resolve_config(raw)


def resolve_config(raw: Mapping[str, Any], *, timestamp: str | None = None) -> ResolvedConfig:
    """Resolve an already-parsed config mapping into a :class:`ResolvedConfig` (§10, §9.1).

    ``${VAR}`` placeholders are substituted here too (safe to call on a raw parsed mapping).
    ``timestamp`` defaults to a fresh UTC run stamp; the runner passes the run's single timestamp.
    """
    from datetime import datetime, timezone

    secret_env_refs: dict[str, str] = {}
    cfg = _substitute_env(dict(raw), refs=secret_env_refs)

    dataset = _require(cfg, "dataset", "config")
    indexer = _require(cfg, "indexer", "config")
    services = _resolve_services(_require(cfg, "services", "config"))
    baseline, variants = _resolve_pipelines(_require(cfg, "pipelines", "config"), services)
    # Resolve pipelines BEFORE stats so contrast/fdr validation sees the known system ids.
    stats = _resolve_stats(
        _require(cfg, "stats", "config"),
        baseline_id=baseline.id,
        variant_ids=[v.id for v in variants],
    )

    raw_cache = cfg.get("cache") or {}  # block absent -> disabled (opt-in, no behavior change)
    cache = CacheCfg(
        enabled=bool(raw_cache.get("enabled", False)),
        dir=str(raw_cache.get("dir", ".cache")),
    )
    metrics = _resolve_metrics(cfg.get("metrics"))

    return ResolvedConfig(
        dataset=dataset,
        indexer=indexer,
        services=services,
        baseline=baseline,
        variants=variants,
        stats=stats,
        cutoff=int(_require(cfg, "cutoff", "config")),
        top_k=int(_require(cfg, "top_k", "config")),
        baseline_id=baseline.id,
        timestamp=timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        seed=int(stats.seed),
        cache=cache,
        metrics=metrics,
        secret_env_refs=secret_env_refs,
    )


#: Valid ``metrics.unjudged`` policies (§7). Exhaustive — anything else is a ConfigError.
_UNJUDGED_POLICIES = ("condensed", "irrelevant")


def _resolve_metrics(raw: Any) -> MetricsCfg:
    """Build :class:`MetricsCfg` from the §10 ``metrics`` block. Absent -> condensed / 0.5.

    ``unjudged`` is validated against :data:`_UNJUDGED_POLICIES` (exhaustive, fail-fast);
    ``relevance_threshold`` is parsed as a float. One policy governs every metric.
    """
    block = raw or {}
    if not isinstance(block, Mapping):
        raise ConfigError("'metrics' must be a mapping of {unjudged, relevance_threshold}")
    unjudged = str(block.get("unjudged", "condensed"))
    if unjudged not in _UNJUDGED_POLICIES:
        raise ConfigError(
            f"metrics.unjudged {unjudged!r} is not a valid policy; "
            f"expected one of {list(_UNJUDGED_POLICIES)}"
        )
    return MetricsCfg(
        unjudged=unjudged,
        relevance_threshold=float(block.get("relevance_threshold", 0.5)),
    )


def _resolve_services(entries: Any) -> Services:
    """Build the typed :class:`Services` registry from the §10 ``services`` list.

    Each entry is a single-key mapping (``embedder`` | ``reranker`` | ``searcher``). A vector
    searcher must reference an existing embedder; duplicate names and unknown entry kinds raise.
    """
    if not isinstance(entries, list):
        raise ConfigError("'services' must be a list of typed service entries")

    embedders: dict[str, EmbedderCfg] = {}
    rerankers: dict[str, RerankerCfg] = {}
    searchers: dict[str, SearcherCfg] = {}

    for entry in entries:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ConfigError(
                "each 'services' entry must be a single-key mapping "
                "(embedder | reranker | searcher)"
            )
        (kind, body), = entry.items()
        if not isinstance(body, Mapping):
            raise ConfigError(f"service entry {kind!r} must map to a mapping of settings")
        name = _require(body, "name", f"{kind} service")
        if kind == "embedder":
            _reject_duplicate(name, embedders, rerankers, searchers)
            embedders[name] = _resolve_embedder(body)
        elif kind == "reranker":
            _reject_duplicate(name, embedders, rerankers, searchers)
            provider = _require(body, "provider", f"reranker {name!r}")
            if provider not in _RERANKER_PROVIDERS:
                extra = " (openai has no reranker)" if provider == "openai" else ""
                raise ConfigError(
                    f"reranker {name!r}: unknown provider {provider!r}; "
                    f"expected one of {list(_RERANKER_PROVIDERS)}{extra}"
                )
            rerankers[name] = RerankerCfg(
                name=name,
                provider=provider,
                settings=dict(_require(body, "settings", f"reranker {name!r}")),
            )
        elif kind == "searcher":
            _reject_duplicate(name, embedders, rerankers, searchers)
            searchers[name] = _resolve_searcher(body, embedders)
        else:
            raise ConfigError(
                f"unknown service entry kind {kind!r}; expected embedder | reranker | searcher"
            )

    return Services(embedders=embedders, rerankers=rerankers, searchers=searchers)


def _reject_duplicate(
    name: str, *registries: Mapping[str, object]
) -> None:
    if any(name in registry for registry in registries):
        raise ConfigError(f"duplicate service name {name!r}")


def _resolve_embedder(body: Mapping[str, Any]) -> EmbedderCfg:
    name = body["name"]
    provider = _require(body, "provider", f"embedder {name!r}")
    if provider not in _EMBEDDER_PROVIDERS:
        raise ConfigError(
            f"embedder {name!r}: unknown provider {provider!r}; "
            f"expected one of {list(_EMBEDDER_PROVIDERS)}"
        )
    return EmbedderCfg(
        name=name,
        provider=provider,
        settings=dict(_require(body, "settings", f"embedder {name!r}")),
    )


def _resolve_searcher(
    body: Mapping[str, Any], embedders: Mapping[str, EmbedderCfg]
) -> SearcherCfg:
    name = body["name"]
    kind = _require(body, "kind", f"searcher {name!r}")
    if kind not in _SEARCHER_KINDS:
        raise ConfigError(
            f"searcher {name!r}: unknown kind {kind!r}; expected one of {list(_SEARCHER_KINDS)}"
        )
    embedder = body.get("embedder")
    if kind == "vector":
        if embedder is None:
            raise ConfigError(f"vector searcher {name!r} requires an 'embedder' reference")
        if embedder not in embedders:
            raise ConfigError(
                f"searcher {name!r} references unknown embedder {embedder!r}; "
                f"known: {sorted(embedders)}"
            )
    elif embedder is not None:
        raise ConfigError(f"lexical searcher {name!r} must not reference an 'embedder'")
    return SearcherCfg(name=name, provider=_require(body, "provider", f"searcher {name!r}"),
                       kind=kind, embedder=embedder)


def _resolve_pipelines(
    block: Mapping[str, Any], services: Services
) -> tuple[PipelineCfg, list[PipelineCfg]]:
    """Resolve the explicit ``pipelines`` block into (baseline, ordered variants) (§10).

    ``baseline`` is the reference; ``variants`` is a map of id -> pipeline spec. A variant id that
    duplicates the baseline id is an error. Insertion order of ``variants`` is preserved.
    """
    baseline_id = str(block.get("baseline_id", "baseline"))
    baseline_body = _require(block, "baseline", "pipelines")
    baseline = _resolve_pipeline(baseline_id, baseline_body, services)

    variants_block = _require(block, "variants", "pipelines")
    if not isinstance(variants_block, Mapping):
        raise ConfigError("'pipelines.variants' must be a map of id -> pipeline spec")

    variants: list[PipelineCfg] = []
    for variant_id, body in variants_block.items():
        if variant_id == baseline_id:
            raise ConfigError(f"variant id {variant_id!r} duplicates the baseline id")
        variants.append(_resolve_pipeline(str(variant_id), body, services))
    return baseline, variants


def _lookup(accessor: Any, name: str) -> Any:
    """Resolve a service reference, re-raising the registry's ValueError as a ConfigError (§10)."""
    try:
        return accessor(name)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _resolve_pipeline(
    pipeline_id: str, body: Any, services: Services
) -> PipelineCfg:
    """Validate + resolve one pipeline spec (§10 pipeline field rules). Exhaustive; raises clearly."""
    if not isinstance(body, Mapping):
        raise ConfigError(f"pipeline {pipeline_id!r} must be a mapping")

    has_single = "retriever" in body
    has_multi = "retrievers" in body
    if has_single == has_multi:
        raise ConfigError(
            f"pipeline {pipeline_id!r} must set exactly one of 'retriever' or 'retrievers'"
        )

    retrievers: tuple[str, ...]
    if has_single:
        retrievers = (str(body["retriever"]),)
    else:
        listed = body["retrievers"]
        if not isinstance(listed, list) or len(listed) < 2:
            raise ConfigError(
                f"pipeline {pipeline_id!r}: 'retrievers' must be a list of 2+ searcher names"
            )
        retrievers = tuple(str(name) for name in listed)

    # Every retriever must be a known searcher; a vector searcher must reference a known embedder.
    for name in retrievers:
        searcher = _lookup(services.searcher, name)
        if searcher.kind == "vector":
            assert searcher.embedder is not None  # validated at service load
            _lookup(services.embedder, searcher.embedder)

    fuser = _resolve_fuser(pipeline_id, body.get("fuser"), n_retrievers=len(retrievers))

    reranker_raw = body.get("reranker")
    window_raw = body.get("rerank_window_size")
    if (reranker_raw is None) != (window_raw is None):
        raise ConfigError(
            f"pipeline {pipeline_id!r}: 'reranker' and 'rerank_window_size' must be set together"
        )
    reranker: str | None = None
    rerank_window_size: int | None = None
    if reranker_raw is not None and window_raw is not None:
        reranker = str(reranker_raw)
        _lookup(services.reranker, reranker)  # must exist + be a reranker
        rerank_window_size = int(window_raw)

    return PipelineCfg(
        id=pipeline_id,
        retrievers=retrievers,
        fuser=fuser,
        reranker=reranker,
        rerank_window_size=rerank_window_size,
    )


def _resolve_fuser(
    pipeline_id: str, body: Any, *, n_retrievers: int
) -> FuserCfg | None:
    """Validate the fuser rule: required iff 2+ retrievers, forbidden with a single retriever (§10)."""
    if n_retrievers >= 2:
        if body is None:
            raise ConfigError(
                f"pipeline {pipeline_id!r} has {n_retrievers} retrievers and requires a 'fuser'"
            )
    elif body is not None:
        raise ConfigError(
            f"pipeline {pipeline_id!r} has a single retriever; 'fuser' is only allowed with 'retrievers'"
        )
    if body is None:
        return None
    if not isinstance(body, Mapping):
        raise ConfigError(f"pipeline {pipeline_id!r}: 'fuser' must be a mapping")
    fuser_type = _require(body, "type", f"pipeline {pipeline_id!r} fuser")
    if fuser_type not in _FUSER_TYPES:
        raise ConfigError(
            f"pipeline {pipeline_id!r}: unknown fuser type {fuser_type!r}; "
            f"expected one of {list(_FUSER_TYPES)}"
        )
    return FuserCfg(
        type=fuser_type,
        rank_constant=int(_require(body, "rank_constant", f"pipeline {pipeline_id!r} fuser")),
        window=int(_require(body, "window", f"pipeline {pipeline_id!r} fuser")),
    )


def _resolve_stats(
    raw: Mapping[str, Any],
    *,
    baseline_id: str,
    variant_ids: Sequence[str],
) -> StatsCfg:
    """Build :class:`StatsCfg` from the §10 ``stats`` block. ``ci_level`` is parsed, not a gate.

    ``contrasts`` (optional) is a list of ``{a, b, family}`` system-pair entries; when absent it is
    synthesized as every-variant-vs-baseline, all ``family: true`` (§10). ``fdr_metrics``
    (optional) restricts the FDR family. Both are validated at build time (fail fast):
    every contrast id must be a known system (``{baseline_id} ∪ variant_ids``) and every
    ``fdr_metrics`` entry must be a canonical metric, else :class:`ConfigError`.
    """
    system_ids = {baseline_id, *variant_ids}
    contrasts = _resolve_contrasts(raw.get("contrasts"), baseline_id, variant_ids, system_ids)
    fdr_metrics = _resolve_fdr_metrics(raw.get("fdr_metrics"))
    test = str(raw.get("test", "permutation"))
    # fail fast on a wilcoxon-only key the selected test does not consume, so the manifest
    # never implies a test that was not run. The keys stay valid under `test: wilcoxon`.
    if test != "wilcoxon":
        for key in ("wilcoxon_zero_method", "wilcoxon_correction"):
            if key in raw:
                raise ConfigError(
                    f"stats.{key} is set but stats.test={test!r} does not consume it; "
                    "remove it or set stats.test: wilcoxon"
                )
    return StatsCfg(
        bootstrap_B=int(raw.get("bootstrap_B", 10000)),
        ci_level=float(raw.get("ci_level", 0.95)),
        alpha=float(raw.get("alpha", 0.05)),
        correction=str(raw.get("correction", "bh")),
        test=test,
        wilcoxon_zero_method=str(raw.get("wilcoxon_zero_method", "wilcox")),
        wilcoxon_correction=bool(raw.get("wilcoxon_correction", True)),
        seed=int(_require(raw, "seed", "stats")),
        contrasts=contrasts,
        fdr_metrics=fdr_metrics,
    )


def _resolve_contrasts(
    raw: Any,
    baseline_id: str,
    variant_ids: Sequence[str],
    system_ids: set[str],
) -> tuple[Contrast, ...]:
    """Parse (or synthesize) ``stats.contrasts`` (§10), validating every id."""
    if raw is None:
        # Absent -> reproduce the old all-vs-baseline behavior (every variant vs the baseline).
        return tuple(Contrast(a=vid, b=baseline_id, family=True) for vid in variant_ids)
    if not isinstance(raw, list):
        raise ConfigError("'stats.contrasts' must be a list of {a, b, family} entries")
    contrasts: list[Contrast] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ConfigError("each 'stats.contrasts' entry must be a mapping with keys a, b, family")
        a = str(_require(entry, "a", "stats.contrasts entry"))
        b = str(_require(entry, "b", "stats.contrasts entry"))
        for system_id in (a, b):
            if system_id not in system_ids:
                raise ConfigError(
                    f"stats.contrasts references unknown system {system_id!r}; "
                    f"known: {sorted(system_ids)}"
                )
        contrasts.append(Contrast(a=a, b=b, family=bool(entry.get("family", True))))
    return tuple(contrasts)


def _resolve_fdr_metrics(raw: Any) -> tuple[str, ...]:
    """Parse ``stats.fdr_metrics`` (§10), validating each against CANONICAL_METRICS."""
    if raw is None:
        return DEFAULT_FDR_METRICS
    if not isinstance(raw, list):
        raise ConfigError("'stats.fdr_metrics' must be a list of metric names")
    metrics = tuple(str(m) for m in raw)
    for metric in metrics:
        if metric not in CANONICAL_METRICS:
            raise ConfigError(
                f"stats.fdr_metrics entry {metric!r} is not a canonical metric; "
                f"known: {list(CANONICAL_METRICS)}"
            )
    return metrics


# --- lazy adapter factories (§11, Phase 11) ----------------------------------------------------


def _resolve_target(target: str) -> Any:
    """Import + return a ``"module:attr"`` dotted target (the deferred live resolution, Phase 11)."""
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def load_dataset(dataset_cfg: Mapping[str, Any]) -> Any:
    """Dispatch ``dataset.name`` -> the dataset adapter, lazily imported (§11, Phase 11).

    Unknown name -> :class:`ConfigError` (exhaustive; no silent default).
    """
    name = _require(dataset_cfg, "name", "dataset")
    target = DATASET_TARGETS.get(name)
    if target is None:
        raise ConfigError(f"unknown dataset name {name!r}; known: {sorted(DATASET_TARGETS)}")
    return _resolve_target(target)(dataset_cfg)


def make_index_writer(indexer_cfg: Mapping[str, Any]) -> Any:
    """Dispatch ``indexer.provider`` -> the ``IndexWriter`` adapter, lazily imported (§3.5, §11).

    The returned writer (ES ``ESIndexWriter``) exposes ``sem_field_name`` / ``create_mapping`` /
    ``ensure_index`` / ``bulk_index`` + ``embed_batch_size``; the runner injects it into the domain
    ``indexing.Indexer`` (so the backend is swappable via config alone, §1.4(3)).
    """
    provider = _require(indexer_cfg, "provider", "indexer")
    target = INDEX_WRITER_TARGETS.get(provider)
    if target is None:
        raise ConfigError(
            f"unknown indexer provider {provider!r}; known: {sorted(INDEX_WRITER_TARGETS)}"
        )
    return _resolve_target(target)(indexer_cfg)


def open_cache(cache_cfg: CacheCfg) -> DiskCache | None:
    """Open the live persistent cache from ``cache_cfg`` (docs/architecture.md §5.5).

    Returns ``None`` ONLY when caching is disabled (the factories then skip wrapping entirely — a
    true bypass). When caching is ENABLED, an unopenable/corrupt ``inference.sqlite``
    (:class:`sqlite3.DatabaseError`) **fails fast** with a clear, actionable error: a cache the user
    turned on that cannot open is a hard error — never a silent degrade to a slower cacheless run
    that the user would only diagnose through the symptom (CLAUDE.md exception convention: an
    unexpected condition is raised with context, not suppressed).
    """
    if not cache_cfg.enabled:
        return None
    try:
        return DiskCache(cache_cfg.dir)
    except sqlite3.DatabaseError as exc:  # corrupt / unopenable DB — unexpected; surface it, never hide it
        raise RuntimeError(
            f"cache at {cache_cfg.dir!r} is enabled but unusable ({exc}); it is likely corrupt — "
            f"delete it (`rm -rf {cache_cfg.dir}`) and re-run, or set cache.enabled: false"
        ) from exc


def make_searchers(
    indexer_cfg: Mapping[str, Any],
    mapping: IndexMapping,
    services: Services,
    *,
    embedders: Mapping[str, Embedder],
    cache: DiskCache | None = None,
) -> dict[str, Searcher]:
    """Build the FULL configured leaf ``Searcher`` set once, keyed by service name (§4, §11).

    Translates every :class:`SearcherCfg` in ``services`` into a backend-agnostic
    ``(name, kind, embedder_id)`` spec (:func:`_searcher_spec`), then dispatches ``indexer.provider``
    -> the backend's lazily-imported ``build_searchers``. The resulting map is shared across all
    pipelines (a pipeline just selects its leaves by name in :func:`build_pipeline`). ``cache`` is
    threaded to the backend's ``build_searchers`` (which fetches the index fingerprint + builds each
    leaf's identity + wraps in ``CachingSearcher``); ``None`` bypasses wrapping (docs/architecture.md §5.5).
    """
    provider = _require(indexer_cfg, "provider", "indexer")
    target = SEARCHER_BUILDER_TARGETS.get(provider)
    if target is None:
        raise ConfigError(
            f"unknown indexer provider {provider!r}; known: {sorted(SEARCHER_BUILDER_TARGETS)}"
        )
    specs = [_searcher_spec(cfg, services) for cfg in services.searchers.values()]
    return _resolve_target(target)(indexer_cfg, mapping, specs, embedders=embedders, cache=cache)


def make_rerankers_bound(
    indexer_cfg: Mapping[str, Any],
    mapping: IndexMapping,
    services: Services,
    *,
    rerank_clients: Mapping[str, RerankClient],
) -> dict[str, Reranker]:
    """Build the FULL configured ``Reranker`` set once, keyed by service name (§4, §11).

    Dispatches ``indexer.provider`` -> the backend's lazily-imported ``build_rerankers``, binding each
    reranker service name to its provider ``RerankClient`` connector (in ``rerank_clients``). Shared
    across pipelines (a pipeline selects its reranker by name in :func:`build_pipeline`).
    """
    provider = _require(indexer_cfg, "provider", "indexer")
    target = RERANKER_BUILDER_TARGETS.get(provider)
    if target is None:
        raise ConfigError(
            f"unknown indexer provider {provider!r}; known: {sorted(RERANKER_BUILDER_TARGETS)}"
        )
    return _resolve_target(target)(
        indexer_cfg, mapping, list(services.rerankers), rerank_clients=rerank_clients
    )


def make_embedders(services: Services, *, cache: DiskCache | None = None) -> dict[str, Any]:
    """Instantiate every configured embedder connector, keyed by service name (§3.4).

    ``benchmark.embedding`` is resolved at CALL time (not imported at config-module import time, §11)
    so config validation stays offline. Returns ``{name: Embedder}``; the provider was validated
    against ``_EMBEDDER_PROVIDERS`` at config load. When ``cache`` is set each connector is wrapped in
    a :class:`~benchmark.common.cache.CachingEmbedder` (query/doc vectors memoized on disk);
    ``cache is None`` bypasses wrapping ENTIRELY (docs/architecture.md §5.5). ``provider``/``model_id``/
    ``endpoint``/``dims`` are passed EXPLICITLY from the ``EmbedderCfg`` (the ``Embedder`` Protocol
    exposes none of them) so the airtight key stays honest.
    """
    make = _resolve_target("benchmark.embedding:make_embedder")
    out: dict[str, Any] = {}
    for name, cfg in services.embedders.items():
        inner = make(cfg.name, cfg.provider, cfg.settings)  # unchanged
        if cache is None:
            out[name] = inner  # disable flag bypasses wrapping entirely
        else:
            dims = cfg.settings.get("dims")
            out[name] = CachingEmbedder(
                inner, cache,
                provider=cfg.provider,
                model_id=str(cfg.settings["model_id"]),  # present iff make(...) succeeded
                endpoint=cfg.settings.get("base_url"),  # None -> connector default (§6)
                dims=int(dims) if dims is not None else None,
            )
    return out


def make_rerankers(services: Services, *, cache: DiskCache | None = None) -> dict[str, Any]:
    """Instantiate every configured rerank connector, keyed by service name (§3.4/§5.4).

    Resolves ``benchmark.reranking`` at CALL time (§11). Returns ``{name: RerankClient}``; the
    provider was validated against ``_RERANKER_PROVIDERS`` at config load. When ``cache`` is set each
    connector is wrapped in a :class:`~benchmark.common.cache.CachingRerankClient` (per-``(query,
    doc)`` scores memoized); ``cache is None`` bypasses wrapping ENTIRELY (docs/architecture.md §5.5).
    """
    make = _resolve_target("benchmark.reranking:make_reranker")
    out: dict[str, Any] = {}
    for name, cfg in services.rerankers.items():
        inner = make(cfg.name, cfg.provider, cfg.settings)
        if cache is None:
            out[name] = inner
        else:
            out[name] = CachingRerankClient(
                inner, cache,
                provider=cfg.provider,
                model_id=str(cfg.settings["model_id"]),
                endpoint=cfg.settings.get("base_url"),
            )
    return out
