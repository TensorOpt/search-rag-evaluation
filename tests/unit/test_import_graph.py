"""Import-graph test — automates the §11 module-layout invariant (§1.4(3) Generality).

docs/architecture.md §11: "search, indexing, evaluation.metrics, evaluation.stats, runner, io_csv
import only common (models/protocols/ranking + the cross-cutting leaf logging_setup) — never
datasets/*, providers/*, embedding, or reranking." and config.py "imports search (the composers) +
evaluation.stats + common only, and still imports no adapter at import time (the factories resolve
dotted targets lazily)."

We assert this at IMPORT TIME, not by reading source: each pure module is imported in a FRESH
interpreter (``sys.executable -c``) and we inspect that interpreter's ``sys.modules`` for any
``benchmark.providers.*`` / ``benchmark.datasets.*`` / ``benchmark.embedding`` / ``benchmark.reranking``
key. A fresh subprocess is the robust probe — importing in-process would see modules other tests
already pulled in. If someone reintroduces a top-level adapter import in ``runner`` (e.g. ``from
benchmark.providers.elasticsearch import ESIndexWriter``), that key appears in the subprocess's
``sys.modules`` and the assertion fails.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Pure modules (§11): may pull in common (models/protocols/ranking/logging_setup) — NEVER an adapter.
_PURE_MODULES = (
    "benchmark.search",
    "benchmark.indexing",
    "benchmark.evaluation.metrics",
    "benchmark.evaluation.stats",
    "benchmark.runner",
    "benchmark.io_csv",
)

#: The adapter package prefixes §11 forbids at import time for the pure modules + config.
#: ``benchmark.providers`` (ES adapter + the inference-provider connectors, §3.4) and the
#: provider-dispatch factories ``benchmark.embedding``/``benchmark.reranking`` are adapters too — pure
#: modules reach them only through ``config``'s lazy factories, never a top-level import.
_ADAPTER_PREFIXES = (
    "benchmark.providers",
    "benchmark.datasets",
    "benchmark.embedding",
    "benchmark.reranking",
)

# Import ``module``, then print every sys.modules key under an adapter package, one per line.
_PROBE = (
    "import importlib, sys\n"
    "importlib.import_module({module!r})\n"
    "print('\\n'.join(k for k in sys.modules\n"
    "                 if any(k == p or k.startswith(p + '.') for p in {prefixes!r})))\n"
)


def _adapter_modules_after_importing(module: str) -> set[str]:
    """Fresh-interpreter import of ``module``; return the adapter sys.modules keys it pulled in."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, prefixes=_ADAPTER_PREFIXES)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in proc.stdout.splitlines() if line}


@pytest.mark.parametrize("module", _PURE_MODULES)
def test_pure_module_imports_no_adapter(module: str) -> None:
    """Importing a §11 pure module must pull in NO benchmark.backends.*/datasets.* (§1.4(3))."""
    pulled = _adapter_modules_after_importing(module)
    assert pulled == set(), (
        f"{module} pulled adapter modules at import time: {sorted(pulled)} — "
        "§11 forbids a top-level adapter import in a pure module"
    )


def test_config_imports_search_but_no_adapter() -> None:
    """config.py imports search (the one wiring edge) but NO adapter at import time (§11)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, benchmark.config\n"
            "print('benchmark.search' in sys.modules)\n"
            "print('\\n'.join(k for k in sys.modules\n"
            f"                 if any(k == p or k.startswith(p + '.') for p in {_ADAPTER_PREFIXES!r})))\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = proc.stdout.splitlines()
    assert lines[0] == "True", "config.py must import benchmark.search (the build_pipeline edge, §11)"
    pulled = {line for line in lines[1:] if line}
    assert pulled == set(), (
        f"config.py pulled adapter modules at import time: {sorted(pulled)} — "
        "the lazy factories must resolve dotted targets at CALL time (§11)"
    )
