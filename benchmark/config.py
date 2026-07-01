"""Config load + resolve, ``${VAR}`` substitution, adapter factories (docs/experiment.md §10, §11). Phase 6.

Parses the explicit §10 YAML config (``dataset`` / ``services`` / ``indexer`` / ``pipelines`` /
``stats`` / ``cutoff`` / ``top_k``), substitutes whole-value ``${VAR}`` environment placeholders at
load (secrets never live in the file), validates it, and resolves it into a
:class:`benchmark.matrix.ResolvedConfig`.

- The ``services`` block builds the typed :class:`Services` registry (embedders/rerankers/searchers
  by name). Embedders flatten to embedding ``InferenceEndpoint``s (§3.4/§3.5); rerankers flatten to
  ``rerank`` endpoints (``task_settings.top_n``) the runner registers lazily at R0 and hands to the
  ES ``searcher_factory``.
- ``pipelines`` is FULLY EXPLICIT: ``baseline`` (the reference) plus a map of named ``variants``.
  There is NO matrix expansion and NO sweep. Each pipeline is validated (§10 pipeline field rules)
  and resolved into a :class:`PipelineCfg`.
- ``load_dataset`` / ``make_indexer`` / ``make_searcher_factory`` dispatch on ``dataset.name`` /
  ``indexer.provider`` to a dotted-path target. They do NOT import the adapter at this phase
  (offline); the live import + construct is deferred to Phase 11. An unknown name/provider raises.

Imports only ``benchmark.models`` / ``benchmark.matrix`` / ``benchmark.stats`` + pyyaml + stdlib —
never an adapter module at import time (§11).
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from benchmark.logging_setup import get_logger
from benchmark.matrix import (
    EmbedderCfg,
    FuserCfg,
    PipelineCfg,
    RerankerCfg,
    ResolvedConfig,
    SearcherCfg,
    Services,
)
from benchmark.models import InferenceTaskType
from benchmark.stats import StatsCfg

logger = get_logger(__name__)

#: ``${VAR}`` env placeholder (§10). Whole-value only — secrets are always their own scalar.
_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

#: Adapter dispatch: dataset.name / indexer.provider -> dotted "module:attr" target. Resolved
#: LAZILY (Phase 11) so this phase imports no adapter and stays offline (§11).
DATASET_TARGETS: Mapping[str, str] = {
    "wands": "benchmark.datasets.wands:WandsDataset",
}
INDEXER_TARGETS: Mapping[str, str] = {
    "elasticsearch": "benchmark.backends.elasticsearch:ElasticsearchBackend",
}
SEARCHER_FACTORY_TARGETS: Mapping[str, str] = {
    "elasticsearch": "benchmark.backends.elasticsearch:make_searcher_factory",
}

#: Valid searcher kinds (§10). Exhaustive — anything else is a ConfigError.
_SEARCHER_KINDS = ("lexical", "vector")
#: Valid fuser types (§10). Only RRF today; anything else is a ConfigError.
_FUSER_TYPES = ("rrf",)


class ConfigError(ValueError):
    """A malformed or incomplete config (missing key, unresolvable ``${VAR}``, unknown adapter,
    or a pipeline that violates the §10 field rules)."""


def _substitute_env(value: Any) -> Any:
    """Recursively replace whole-value ``${VAR}`` scalars with ``os.environ[VAR]`` (§10).

    A missing environment variable for a referenced placeholder is a clear error — secrets must be
    supplied at run time, never defaulted silently.
    """
    if isinstance(value, str):
        match = _ENV_PLACEHOLDER.match(value)
        if match is None:
            return value
        var = match.group(1)
        if var not in os.environ:
            raise ConfigError(f"environment variable {var!r} referenced as ${{{var}}} is not set")
        return os.environ[var]
    if isinstance(value, Mapping):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
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

    cfg = _substitute_env(dict(raw))

    dataset = _require(cfg, "dataset", "config")
    indexer = _require(cfg, "indexer", "config")
    services = _resolve_services(_require(cfg, "services", "config"))
    baseline, variants = _resolve_pipelines(_require(cfg, "pipelines", "config"), services)
    stats = _resolve_stats(_require(cfg, "stats", "config"))

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
            rerankers[name] = RerankerCfg(
                name=name,
                provider=_require(body, "provider", f"reranker {name!r}"),
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
    task_type_raw = body.get("task_type", "text_embedding")
    try:
        task_type = InferenceTaskType(task_type_raw)
    except ValueError:
        raise ConfigError(
            f"embedder {name!r}: unknown task_type {task_type_raw!r}; "
            f"expected one of {[t.value for t in InferenceTaskType if t is not InferenceTaskType.RERANK]}"
        )
    if task_type is InferenceTaskType.RERANK:
        raise ConfigError(f"embedder {name!r} must not use the 'rerank' task_type")
    return EmbedderCfg(
        name=name,
        provider=_require(body, "provider", f"embedder {name!r}"),
        task_type=task_type,
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


def _resolve_stats(raw: Mapping[str, Any]) -> StatsCfg:
    """Build :class:`StatsCfg` from the §10 ``stats`` block. ``ci_level`` is parsed, not a gate."""
    return StatsCfg(
        bootstrap_B=int(raw.get("bootstrap_B", 10000)),
        ci_level=float(raw.get("ci_level", 0.95)),
        alpha=float(raw.get("alpha", 0.05)),
        correction=str(raw.get("correction", "bh")),
        test=str(raw.get("test", "wilcoxon")),
        wilcoxon_zero_method=str(raw.get("wilcoxon_zero_method", "wilcox")),
        wilcoxon_correction=bool(raw.get("wilcoxon_correction", True)),
        seed=int(_require(raw, "seed", "stats")),
    )


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


def make_indexer(indexer_cfg: Mapping[str, Any]) -> Any:
    """Dispatch ``indexer.provider`` -> the ingest backend adapter, lazily imported (§11, Phase 11)."""
    provider = _require(indexer_cfg, "provider", "indexer")
    target = INDEXER_TARGETS.get(provider)
    if target is None:
        raise ConfigError(f"unknown indexer provider {provider!r}; known: {sorted(INDEXER_TARGETS)}")
    return _resolve_target(target)(indexer_cfg)


def make_searcher_factory(indexer_cfg: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
    """Dispatch ``indexer.provider`` -> the backend's ``searcher_factory`` builder (§4, §11, Phase 11)."""
    provider = _require(indexer_cfg, "provider", "indexer")
    target = SEARCHER_FACTORY_TARGETS.get(provider)
    if target is None:
        raise ConfigError(
            f"unknown indexer provider {provider!r}; known: {sorted(SEARCHER_FACTORY_TARGETS)}"
        )
    return _resolve_target(target)(indexer_cfg, *args, **kwargs)
