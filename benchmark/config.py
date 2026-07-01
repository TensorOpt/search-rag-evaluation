"""Config load + resolve, ``${VAR}`` substitution, adapter factories, ``ConfigInferenceModel`` (docs/experiment.md §10, §11). Phase 6.

Loads the §10 YAML/JSON config, substitutes ``${VAR}`` environment placeholders at load (secrets
never live in the file), and resolves it into a :class:`benchmark.matrix.ResolvedConfig`.

- :class:`ConfigInferenceModel` implements the ``EmbeddingModel`` descriptor (§3.4): it flattens
  to one embedding ``InferenceEndpoint`` via ``as_endpoint()``. Rerankers are ``InferenceEndpoint``s
  too (task_type ``rerank``, ``task_settings.top_n``), carried on ``ResolvedConfig`` for the runner
  to register lazily at R0 (§8.0) and hand to the ES ``searcher_factory`` to build an ``ESReranker``
  (``Reranker`` is a behavioral ABC realized by the backend, not a config descriptor).
- ``load_dataset`` / ``make_backend`` / ``make_searcher_factory`` dispatch on ``dataset.name`` /
  ``backend.kind`` to a dotted-path target. They do NOT import the adapter at this phase (offline);
  the live import + construct is deferred to Phase 11. An unknown name/kind raises (exhaustive).

Imports only ``benchmark.models`` / ``benchmark.matrix`` / ``benchmark.stats`` + pyyaml + stdlib —
never an adapter module at import time (§11).
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from benchmark.logging_setup import get_logger
from benchmark.matrix import EmbeddingModelCfg, RerankerCfg, ResolvedConfig
from benchmark.models import InferenceEndpoint, InferenceTaskType
from benchmark.stats import StatsCfg

logger = get_logger(__name__)

#: ``${VAR}`` env placeholder (§10). Whole-value only — secrets are always their own scalar.
_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

#: Adapter dispatch: dataset.name / backend.kind -> dotted "module:attr" target. Resolved LAZILY
#: (Phase 11) so this phase imports no adapter and stays offline (§11).
DATASET_TARGETS: Mapping[str, str] = {
    "wands": "benchmark.datasets.wands:WandsDataset",
}
BACKEND_TARGETS: Mapping[str, str] = {
    "elasticsearch": "benchmark.backends.elasticsearch:ElasticsearchBackend",
}
SEARCHER_FACTORY_TARGETS: Mapping[str, str] = {
    "elasticsearch": "benchmark.backends.elasticsearch:make_searcher_factory",
}


@dataclass(frozen=True)
class ConfigInferenceModel:
    """A config-declared embedding model implementing the ``EmbeddingModel`` descriptor (§3.4, §10).

    No per-service adapter class: the config builds the ``InferenceEndpoint`` directly and this
    descriptor flattens to it. The backend's ``register_inference`` is the single path that
    materializes it at ingest (§3.4).
    """

    inference_id: str
    task_type: InferenceTaskType
    service: str
    service_settings: Mapping[str, Any] = field(default_factory=dict)
    task_settings: Mapping[str, Any] = field(default_factory=dict)

    def as_endpoint(self) -> InferenceEndpoint:
        return InferenceEndpoint(
            inference_id=self.inference_id,
            task_type=self.task_type,
            service=self.service,
            service_settings=self.service_settings,
            task_settings=self.task_settings,
        )


class ConfigError(ValueError):
    """A malformed or incomplete config (missing key, unresolvable ``${VAR}``, unknown adapter)."""


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
    if key not in mapping:
        raise ConfigError(f"missing required key {key!r} in {where}")
    return mapping[key]


def load_config(path: str | Path) -> ResolvedConfig:
    """Load + resolve a YAML/JSON config file into a :class:`ResolvedConfig` (§10).

    ``${VAR}`` placeholders are substituted from the environment at load; a missing required key or
    unresolvable placeholder raises :class:`ConfigError`.
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
    backend = _require(cfg, "backend", "config")

    embedding_models = [
        EmbeddingModelCfg(inference_id=_require(m, "inference_id", "embedding_models entry"))
        for m in _require(cfg, "embedding_models", "config")
    ]
    rerankers = [
        RerankerCfg(inference_id=_require(r, "inference_id", "rerankers entry"))
        for r in _require(cfg, "rerankers", "config")
    ]
    reranker_endpoints = {
        ep.inference_id: ep
        for ep in (_build_reranker_endpoint(r) for r in cfg["rerankers"])
    }

    stats = _resolve_stats(_require(cfg, "stats", "config"))

    return ResolvedConfig(
        dataset=dataset,
        backend=backend,
        embedding_models=embedding_models,
        rerankers=rerankers,
        rrf_k_sweep=[int(k) for k in _require(cfg, "rrf_k_sweep", "config")],
        variants=list(_require(cfg, "variants", "config")),
        reranker_endpoints=reranker_endpoints,
        stats=stats,
        cutoff=int(_require(cfg, "cutoff", "config")),
        top_k=int(_require(backend, "top_k", "backend")),
        rank_window_size=int(_require(backend, "rank_window_size", "backend")),
        hybrid_rerank_k=_resolve_hybrid_rerank_k(_require(cfg, "hybrid_rerank_k", "config")),
        baseline_id="bm25",
        timestamp=timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        seed=int(stats.seed),
    )


def _resolve_hybrid_rerank_k(value: Any) -> int | str:
    """A concrete int (static, §10) or the literal ``"best_per_model"`` (two-pass, §8.0a)."""
    if isinstance(value, bool):  # bool is an int subclass; a mode flag is never a k.
        raise ConfigError(f"hybrid_rerank_k must be an int or 'best_per_model', got {value!r}")
    if isinstance(value, int):
        return value
    if value == "best_per_model":
        return "best_per_model"
    raise ConfigError(f"hybrid_rerank_k must be an int or 'best_per_model', got {value!r}")


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


def _build_reranker_endpoint(raw: Mapping[str, Any]) -> InferenceEndpoint:
    """Flatten a ``rerankers`` entry to a ``rerank`` ``InferenceEndpoint`` (§3.4, §5.3).

    ``top_n`` (the rank-window cap) is a ``task_settings`` key, asserted ``>= W`` at R0 (§5.3).
    """
    return InferenceEndpoint(
        inference_id=_require(raw, "inference_id", "rerankers entry"),
        task_type=InferenceTaskType.RERANK,
        service=_require(raw, "service", "rerankers entry"),
        service_settings=raw.get("service_settings", {}),
        task_settings=raw.get("task_settings", {}),
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
        raise ConfigError(
            f"unknown dataset name {name!r}; known: {sorted(DATASET_TARGETS)}"
        )
    return _resolve_target(target)(dataset_cfg)


def make_backend(backend_cfg: Mapping[str, Any]) -> Any:
    """Dispatch ``backend.kind`` -> the ingest backend adapter, lazily imported (§11, Phase 11)."""
    kind = _require(backend_cfg, "kind", "backend")
    target = BACKEND_TARGETS.get(kind)
    if target is None:
        raise ConfigError(
            f"unknown backend kind {kind!r}; known: {sorted(BACKEND_TARGETS)}"
        )
    return _resolve_target(target)(backend_cfg)


def make_searcher_factory(backend_cfg: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
    """Dispatch ``backend.kind`` -> the backend's ``searcher_factory`` builder (§4, §11, Phase 11)."""
    kind = _require(backend_cfg, "kind", "backend")
    target = SEARCHER_FACTORY_TARGETS.get(kind)
    if target is None:
        raise ConfigError(
            f"unknown backend kind {kind!r}; known: {sorted(SEARCHER_FACTORY_TARGETS)}"
        )
    return _resolve_target(target)(backend_cfg, *args, **kwargs)
