# Refactor design — layered, domain-driven module reorganization

> **Historical design record; implemented.** Authoritative docs: docs/methodology.md + docs/architecture.md.

**Scope.** Pure module reorganization of `benchmark/` into the layered `a–g` structure. **No**
behavior change, **no** metrics/stats math change, **no** change to the frozen CSV / `run_config`
schemas. The offline test suite still passes; test files relocate to mirror the layout and update
imports + construction wiring, but their behavioral assertions do not change. This document is the
target the implementation phase realizes; `docs/experiment.md` §11 is updated to match at that time.

The one non-mechanical piece is the **clean-OOP seam redesign** the user asked for: delete
`_ESSearcherFactory` and the `ElasticsearchBackend` god-object; express indexing and search as clean
domain OOP (`Indexer`, `SearchPipeline`) composed from concrete pieces that live in `providers`.

---

## 1. Target package tree

```
benchmark/
  __init__.py
  common/                    # (g) shared bottom layer — abstractions, data, pure utilities, logging
    __init__.py
    models.py                #   frozen data models + enums (was models.py, verbatim)
    protocols.py             #   ABCs + Protocols: Searcher/Fuser/Reranker/Dataset; Embedder/RerankClient/IndexWriter
    ranking.py               #   pure windowed ranking primitives: fuse_rrf_local + rerank_local (merge of fusion.py + rerank.py)
    logging_setup.py         #   console+file logging (was logging_setup.py, verbatim)
  providers/                 # (f) concrete implementations, each depends ONLY on common
    __init__.py
    inference.py             #   HTTP connectors: OpenAI/Cohere/Voyage Embedders + Cohere/Voyage RerankClients (was providers.py, minus the dispatch tables)
    elasticsearch.py         #   ES pieces: LexicalSearcher, VectorSearch, ESReranker, ESIndexWriter + build_searchers/build_rerankers (was backends/elasticsearch.py, minus orchestration + minus _ESSearcherFactory)
  embedding.py               # (c) embedder factory: make_embedder + EMBEDDER_PROVIDERS (dispatch provider -> providers.inference class)
  reranking.py               # (d) rerank-client factory: make_reranker + RERANKER_PROVIDERS (dispatch provider -> providers.inference class)
  indexing.py                # (a) domain Indexer (build orchestration) + embed-at-ingest streaming (_embed_documents/_embed_batch)
  search.py                  # (b) domain composers: RRFFuser, HybridSearch, SearchPipeline (was pipeline.py, verbatim)
  evaluation/                # (e) scoring + statistics
    __init__.py
    metrics.py               #   Evaluator, Metrics, QrelIndex (was metrics.py, verbatim)
    stats.py                 #   Comparator, StatsCfg, ComparisonResult (was stats.py, verbatim)
  datasets/                  # data-source adapters (its own axis; NOT under providers — see §3.5)
    __init__.py
    wands.py                 #   WandsDataset (was datasets/wands.py; import paths updated)
  config.py                  # composition layer: config value types + loader + build_pipeline + lazy dotted factories
  runner.py                  # composition layer: ExperimentRunner (the single execution path)
  io_csv.py                  # composition layer: the four frozen artifact writers
scripts/                     # CLI entry points (unchanged; stay top-level — see §3.6)
  fetch_data.py  index.py  run.py  wait_for_es.py
```

**Layer roles (a–g) vs. the three top-level composition modules.** `a–g` are the *engine*: a strict
DAG. `config.py` / `runner.py` / `io_csv.py` are the *application/composition* layer that sits
**above** the engine and wires it together (they may import several engine layers). Keeping them out
of `a–g` is what lets the engine stay a clean acyclic graph — see §2 and §3.4.

---

## 2. Dependency graph (per-module import lists) — acyclic, matches the a–g "needs"

Import lists below are **import-time** `benchmark.*` edges only (stdlib / numpy / scipy / yaml /
elasticsearch omitted). "injected" = a runtime composition dependency satisfied by passing a
concrete instance in through a constructor/parameter, with **no import edge** (Dependency-Inversion:
the consumer imports the abstraction from `common`, never the concrete provider).

| Module (layer) | import-time `benchmark.*` edges | runtime-injected (no import) |
|---|---|---|
| `common.models` (g) | — | — |
| `common.protocols` (g) | `common.models` | — |
| `common.ranking` (g) | `common.models` | — |
| `common.logging_setup` (g) | — | — |
| `providers.inference` (f) | `common.logging_setup`, `common.protocols` | — |
| `providers.elasticsearch` (f) | `common.models`, `common.protocols`, `common.ranking`, `common.logging_setup` | `Embedder`, `RerankClient` |
| `embedding` (c) | `providers.inference`, `common.protocols` | — |
| `reranking` (d) | `providers.inference`, `common.protocols` | — |
| `indexing` (a) | `common.models`, `common.protocols`, `common.logging_setup` | `IndexWriter` (providers), `Embedder` (embedding) |
| `search` (b) | `common.models`, `common.protocols`, `common.ranking` | leaf `Searcher`s (providers), `Reranker` (providers) |
| `evaluation.metrics` (e) | `common.models` | — |
| `evaluation.stats` (e) | — (numpy/scipy only) | — |
| `config` (comp.) | `common.models`?†, `search`, `evaluation.stats` | datasets / providers / embedding / reranking (lazy dotted) |
| `runner` (comp.) | `config`, `indexing`, `evaluation.metrics`, `evaluation.stats`, `io_csv`, `common.models`, `common.logging_setup` | — |
| `io_csv` (comp.) | `config`, `evaluation.metrics`, `evaluation.stats`, `common.models` | — |

† `config` defines its own value types (see §3.4); it imports `common` only for shared enums/models
if needed. It imports `search` for the composers used by `build_pipeline`, and `evaluation.stats`
for `StatsCfg`.

### 2.1 Acyclicity

Topological order (a valid linearization; every edge points right-to-left, i.e. up the list):

```
common.models < common.{protocols, ranking, logging_setup}
              < providers.{inference, elasticsearch}
              < embedding, reranking
              < indexing, search, evaluation.{metrics, stats}
              < config
              < io_csv
              < runner
```

Proof sketch: `common` imports nothing under `benchmark` except within itself (`protocols`/`ranking`
→ `models`, no back-edge). `providers` import only `common`. `embedding`/`reranking` import
`providers` + `common` only. `indexing`/`search`/`evaluation` import `common` only at import time
(their "needs providers/embedding/reranking" are **injected**, no import edge). `config` imports
`search` + `evaluation.stats` (+ `common`) — none of which import `config`, so no cycle. `io_csv`
imports `config` + `evaluation` + `common`; `config` never imports `io_csv` (the runner does), so no
cycle. `runner` is the top; nothing imports it. **No edge points backward → acyclic.**

### 2.2 Matches the a–g "needs" list

- **a. indexing** needs embedding (c), providers (f), common (g): satisfied — imports `common`
  abstractions at import time; consumes an `IndexWriter` (providers) and `Embedder`s (built by the
  embedding factory) injected by the composition layer. The DIP injection is *how* the "needs" are
  met without a backward import edge or an import-graph violation.
- **b. search** needs embedding (c), reranking (d), providers (f), common (g): same shape — composes
  injected leaf `Searcher`s (providers, using embedding-built query embedders) and an injected
  `Reranker` (providers, using a reranking-built `RerankClient`); imports only `common`.
- **c. embedding** needs providers (f), common (g): direct import edges. ✓
- **d. reranking** needs providers (f), common (g): direct import edges. ✓
- **e. evaluation** needs common (g): `metrics` imports `common.models`; `stats` is self-contained
  (numpy/scipy) and needs nothing from `common`. ✓
- **f. providers**: concrete embedders/rerankers (`inference`) + concrete searchers/reranker/index
  writer (`elasticsearch`); depend **only** on `common`. ✓
- **g. common**: models, protocols/ABCs, pure ranking utilities, logging. Depends on nothing. ✓

---

## 3. The clean-OOP seams (replacing `_ESSearcherFactory` + `ElasticsearchBackend`)

### 3.1 Abstractions in `common.protocols`

Kept (verbatim behavior): `Searcher`, `Fuser`, `Reranker` (ABCs), `Dataset` (ABC), `Embedder`,
`RerankClient` (Protocols).

**Changed:**
- `SearchBackend` (ingest Protocol) → renamed **`IndexWriter`** and given the backend-specific bits
  the domain `Indexer` must delegate:
  ```python
  @runtime_checkable
  class IndexWriter(Protocol):
      embed_batch_size: int                                   # ingest buffering granularity (§3.5)
      def sem_field_name(self, embedder_id: str) -> str: ...  # backend-safe dense_vector field name
      def create_mapping(self, schema: FieldSchema, sem_fields: Mapping[str, str],
                         vector_dims: Mapping[str, int]) -> IndexMapping: ...
      def ensure_index(self, mapping: IndexMapping) -> None: ...
      def bulk_index(self, docs: Iterable[Document], *, mapping: IndexMapping) -> None: ...
  ```
- **Removed** `Indexer` (Protocol) — the domain `Indexer` is now a single concrete, backend-agnostic
  class (§3.2); there is exactly one implementation, so a Protocol is dead abstraction.
- **Removed** `SearcherFactory` (Protocol) — replaced by plain leaf-object maps (§3.3).

### 3.2 `indexing.Indexer` — the clean indexing seam (replaces `ESIndexer` orchestration)

The orchestration that lived in `ESIndexer.build` becomes a backend-agnostic domain object; the
ES-specific mapping/field-naming moves behind the `IndexWriter`.

```python
# benchmark/indexing.py
class Indexer:
    """Builds + populates an index: discover dims -> create mapping -> ensure_index ->
    stream corpus through the embedders -> bulk_index. Backend-agnostic (§3.5)."""
    def __init__(self, writer: IndexWriter, embedders: Sequence[Embedder]) -> None:
        self.writer = writer
        self.embedders = list(embedders)

    def build(self, dataset: Dataset) -> IndexMapping:
        sem_fields = {e.id: self.writer.sem_field_name(e.id) for e in self.embedders}
        vector_dims = {sem_fields[e.id]: e.dim for e in self.embedders}
        schema = dataset.field_schema()
        mapping = self.writer.create_mapping(schema, sem_fields, vector_dims)
        self.writer.ensure_index(mapping)
        enriched = _embed_documents(dataset.documents(), self.embedders, sem_fields,
                                    schema.search_text_field, self.writer.embed_batch_size)
        self.writer.bulk_index(enriched, mapping=mapping)
        return mapping
```

`_embed_documents` / `_embed_batch` (the lazy, bounded-buffer embed-at-ingest streaming) move here
from the ES adapter verbatim — they only touch `Document` + `Embedder`, so they are domain, not
backend, code. The `IndexMapping` shape and the resulting ES index are byte-for-byte identical:
`create_mapping` runs the same `_schema_to_mapping` + `_sem_field_name` logic, just now owned by the
writer.

**Who constructs it:** the composition layer (`ExperimentRunner.build_index`) injects the
provider-built `writer` and the embedding-factory-built `embedders`:
```python
writer     = config.make_index_writer(cfg.indexer)   # -> providers.elasticsearch.ESIndexWriter (lazy)
embedders  = config.make_embedders(cfg.services)      # {name: Embedder} (lazy -> embedding.make_embedder)
mapping    = Indexer(writer, list(embedders.values())).build(dataset)
```

### 3.3 `providers.elasticsearch` — concrete pieces + leaf builders (replaces `_ESSearcherFactory`)

Concrete `Searcher`/`Reranker`/writer classes stay here (unchanged bodies): `LexicalSearcher`,
`VectorSearch`, `ESReranker`. `ElasticsearchBackend` becomes **`ESIndexWriter`** (same ingest
methods, plus `sem_field_name`/`create_mapping` moved off the free functions onto it).

`_ESSearcherFactory` + `make_searcher_factory` are **deleted**. In their place, two module-level
builder functions turn the config `Services` (passed as plain tuples, so `providers` never imports
`config`) into flat leaf/reranker maps, doing the exhaustive `kind` dispatch that used to live in
`_build_leaf`:

```python
# benchmark/providers/elasticsearch.py
SearcherSpec = tuple[str, str, str | None]   # (name, kind, embedder_id-or-None)

def build_searchers(indexer_cfg: Mapping[str, Any], mapping: IndexMapping,
                    specs: Sequence[SearcherSpec], *,
                    embedders: Mapping[str, Embedder]) -> dict[str, Searcher]:
    client = _make_client(indexer_cfg); index = mapping.index_name
    out: dict[str, Searcher] = {}
    for name, kind, embedder_id in specs:
        if kind == "lexical":
            out[name] = LexicalSearcher(client, index, [mapping.search_text_field], ...)
        elif kind == "vector":
            out[name] = VectorSearch(client, index, mapping.sem_field(embedder_id),
                                     embedders[embedder_id], ...)
        else:
            raise ValueError(f"searcher {name!r}: unknown kind {kind!r}")
    return out

def build_rerankers(indexer_cfg, mapping, names: Sequence[str], *,
                    rerank_clients: Mapping[str, RerankClient]) -> dict[str, Reranker]:
    client = _make_client(indexer_cfg); index = mapping.index_name
    return {n: ESReranker(client, index, mapping.search_text_field, rerank_clients[n]) for n in names}
```

### 3.4 `build_pipeline` — pure composition (no factory object)

With the leaves pre-built, `build_pipeline` reduces to composing the object graph over plain
`Searcher`/`Reranker` instances — this *is* the "clean OOP built from provider pieces" the brief
asks for. It stays in `config.py` (see rationale below), imports the `search` composers, and no
longer references any factory or adapter:

```python
# benchmark/config.py
def build_pipeline(pcfg: PipelineCfg, searchers: Mapping[str, Searcher],
                   rerankers: Mapping[str, Reranker]) -> SearchPipeline:
    leaves = [searchers[name] for name in pcfg.retrievers]
    if pcfg.fuser is not None:
        if pcfg.fuser.type != "rrf": raise ValueError(...)          # exhaustive, unchanged
        retriever: Searcher = HybridSearch(retrievers=leaves,
            fuser=RRFFuser(rank_constant=pcfg.fuser.rank_constant),
            retrieval_window_size=pcfg.fuser.window)
    else:
        if len(leaves) != 1: raise ValueError(...)
        (retriever,) = leaves
    if pcfg.reranker is not None:
        return SearchPipeline(retriever=retriever, reranker=rerankers[pcfg.reranker],
                              rerank_window_size=pcfg.rerank_window_size)
    return SearchPipeline(retriever=retriever)
```

**Why `build_pipeline` (and the config value types) stay in `config.py`, not in `search`.** Moving
them into `search` would make `search` depend on `PipelineCfg` — but `config` already imports
`search` for the composers, so `search`→`config` would create a cycle. Keeping the value types +
`build_pipeline` in `config` preserves the established one-way wiring edge `config`→`search`
(exactly the resolution the current §11/§4 already documents). `config` is the composition layer, so
it is allowed to import engine layers. **Resolution of the "config placement" question: `config.py`
stays a single top-level composition module holding the config value types + loader +
`build_pipeline` + the lazy factories.**

### 3.5 Config's lazy dotted factories (the DIP + import-graph seam)

`config.py` keeps resolving adapters lazily by dotted `"module:attr"` target at **call** time so it
imports no adapter at import time. The target tables change to the new layout:

```python
DATASET_TARGETS         = {"wands":         "benchmark.datasets.wands:WandsDataset"}
INDEX_WRITER_TARGETS    = {"elasticsearch": "benchmark.providers.elasticsearch:ESIndexWriter"}
SEARCHER_BUILDER_TARGETS= {"elasticsearch": "benchmark.providers.elasticsearch:build_searchers"}
RERANKER_BUILDER_TARGETS= {"elasticsearch": "benchmark.providers.elasticsearch:build_rerankers"}
# embedder/reranker connectors resolve the domain factories:
#   make_embedders -> "benchmark.embedding:make_embedder"
#   make_rerankers -> "benchmark.reranking:make_reranker"
```

The old `INDEXER_TARGETS` (`ElasticsearchBackend`) and `INDEX_BUILDER_TARGETS` (`ESIndexer`) **merge
into** `INDEX_WRITER_TARGETS`; the domain `Indexer` is constructed by the runner (not a factory).
`make_indexer` + `make_index_builder` collapse into **`make_index_writer`**; `make_searcher_factory`
becomes **`make_searchers`** + **`make_rerankers_bound`** (thin wrappers that translate `Services` →
the tuple specs above and resolve the builder target).

**Dataset placement (open question resolved).** `datasets/` **stays its own top-level subpackage**,
peer to `providers/` — not merged under it. Rationale: a dataset is an *input-source* adapter
(parses files → `Query`/`Document`/`Qrel`), an axis orthogonal to the retrieval/inference providers;
`docs/experiment.md` §12 treats "add a dataset" and "add a backend" as independent extension points,
and the import-graph test already tracks `benchmark.datasets` as a separate adapter prefix. Folding
it under `providers` would blur two unrelated swap axes for no gain.

### 3.6 `scripts/` placement (open question resolved)

`scripts/` stays top-level, unchanged. They are CLI entry points (the composition *entrypoints*),
call `setup_logging` + `benchmark.runner`, and are not part of the engine. Only their imports of
`ESIndexer`/backend symbols change (they import none directly today except via `runner`), so
`scripts/index.py` reads `writer.client.count(...)` on the returned `ESIndexWriter` instead of the
old backend. No structural move.

### 3.7 `fusion.py` / `rerank.py` placement (open question resolved)

Both are pure, stdlib-only windowed primitives over `ScoredDoc`. `fuse_rrf_local` is consumed by the
`search` domain (`RRFFuser`); `rerank_local` is consumed by a `provider` (`ESReranker`). Because a
**provider** consumes `rerank_local`, it cannot live in a domain layer (that would make
`providers`→domain, a backward edge). The shared bottom (`common`) is the only home reachable by
both `search` and `providers` without a cycle. **Resolution: merge both into `common/ranking.py`**
(two ~40-line files → one), matching the user's "utilities" bullet for `common`.

---

## 4. File-by-file migration map

| Old path | New path | Notes |
|---|---|---|
| `benchmark/models.py` | `benchmark/common/models.py` | verbatim |
| `benchmark/protocols.py` | `benchmark/common/protocols.py` | `SearchBackend`→`IndexWriter` (+ `sem_field_name`/`create_mapping`/`embed_batch_size`); **remove** `Indexer` Protocol + `SearcherFactory` Protocol; rest verbatim |
| `benchmark/fusion.py` | `benchmark/common/ranking.py` | **merge** with rerank.py; `fuse_rrf_local` verbatim |
| `benchmark/rerank.py` | `benchmark/common/ranking.py` | **merge**; `rerank_local` verbatim |
| `benchmark/logging_setup.py` | `benchmark/common/logging_setup.py` | verbatim |
| `benchmark/providers.py` | `benchmark/providers/inference.py` | connectors verbatim; `_EMBEDDER_CLASSES`/`_RERANKER_CLASSES` dispatch tables + `make_embedder`/`make_reranker` + `EMBEDDER_PROVIDERS`/`RERANKER_PROVIDERS` **move out** to `embedding.py`/`reranking.py` |
| `benchmark/backends/elasticsearch.py` | `benchmark/providers/elasticsearch.py` | **split** (see below) |
| — | `benchmark/embedding.py` | **new** (c): `make_embedder`, `_EMBEDDER_CLASSES`, `EMBEDDER_PROVIDERS` (from providers.py) |
| — | `benchmark/reranking.py` | **new** (d): `make_reranker`, `_RERANKER_CLASSES`, `RERANKER_PROVIDERS` (from providers.py) |
| — | `benchmark/indexing.py` | **new** (a): `Indexer` + `_embed_documents`/`_embed_batch` (from ES `ESIndexer.build` + `_embed_*`) |
| `benchmark/pipeline.py` | `benchmark/search.py` | `RRFFuser`/`HybridSearch`/`SearchPipeline` verbatim; import of `fuse_rrf_local` now from `common.ranking` |
| `benchmark/metrics.py` | `benchmark/evaluation/metrics.py` | verbatim |
| `benchmark/stats.py` | `benchmark/evaluation/stats.py` | verbatim |
| `benchmark/datasets/wands.py` | `benchmark/datasets/wands.py` | import paths → `benchmark.common.*` |
| `benchmark/config.py` | `benchmark/config.py` | target tables updated (§3.5); `build_pipeline` rewritten to compose from leaf maps (§3.4); `make_indexer`+`make_index_builder`→`make_index_writer`; `make_searcher_factory`→`make_searchers`+`make_rerankers_bound`; value types unchanged |
| `benchmark/runner.py` | `benchmark/runner.py` | `build_index` constructs `indexing.Indexer(writer, embedders)`; `run` builds leaf/reranker maps via `config.make_searchers`/`make_rerankers_bound` and calls the new `build_pipeline`; imports updated |
| `benchmark/io_csv.py` | `benchmark/io_csv.py` | imports → `evaluation.metrics`/`evaluation.stats`; writers verbatim |
| `benchmark/backends/__init__.py` | *(deleted)* | package removed |
| `scripts/*.py` | `scripts/*.py` | unchanged except `index.py` reads `writer.client.count` (was `backend.client.count`) |

**Splitting `backends/elasticsearch.py`:**
- → `providers/elasticsearch.py`: `_make_client`, `_hits_to_scored`, `_search`, `_msearch`,
  `LexicalSearcher`, `VectorSearch`, `ESReranker`, `_schema_to_mapping`, `_sem_field_name`, the
  ingest methods (renamed onto `ESIndexWriter`), and the **new** `build_searchers`/`build_rerankers`.
- → `indexing.py`: `_embed_documents`, `_embed_batch`, and the orchestration body of `ESIndexer.build`
  (now `Indexer.build`).
- **Deleted:** `_ESSearcherFactory`, `make_searcher_factory`, and the `ESIndexer` class shell.

**Test migration (mirror layout; behavioral assertions unchanged):**
- `tests/unit/test_models.py`→`tests/unit/common/test_models.py`; likewise `protocols`, and
  `test_fusion.py`+`test_rerank.py`→`tests/unit/common/test_ranking.py` (import path only).
- `test_providers.py`→`tests/unit/providers/test_inference.py`; `test_es_backend.py`→
  `tests/unit/providers/test_elasticsearch.py` — its `make_searcher_factory`/`factory.lexical/…`
  cases become `build_searchers`/`build_rerankers` cases (same `isinstance`/field/embedder
  assertions); its `ESIndexer().build` cases split into `ESIndexWriter` mapping/sem-field unit tests.
- `test_pipeline.py`→`test_search.py`; `test_metrics.py`/`test_stats.py`→`tests/unit/evaluation/`.
- `test_build_pipeline.py`: the `FakeFactory` is replaced by fake `{name: Searcher}` /
  `{name: Reranker}` maps passed straight to `build_pipeline`; graph-shape assertions unchanged.
- `test_runner.py`: `FakeFactory`→fake searcher/reranker maps (monkeypatch `config.make_searchers`/
  `make_rerankers_bound`); `ESIndexer()`→`indexing.Indexer(writer, embedders)` with the fake writer.
- `test_config.py`: `INDEXER_TARGETS`→`INDEX_WRITER_TARGETS`; `make_indexer`/`make_searcher_factory`
  unknown-provider cases → `make_index_writer`/`make_searchers`.
- `test_import_graph.py`: `_PURE_MODULES` = `search, indexing, evaluation.metrics, evaluation.stats,
  runner, io_csv` (+ the `config` test); `_ADAPTER_PREFIXES` = `benchmark.providers`,
  `benchmark.datasets`, `benchmark.embedding`, `benchmark.reranking`.

---

## 5. How the frozen schemas + import-graph guarantee are preserved

**Frozen CSV / `run_config` schemas (§9).** `io_csv.py` moves nothing about its output: the four
headers, field order, `PipelineCfg.id`-based filenames, `repr`-float formatting, empty-cell rules,
and `run_config_{ts}.json = dataclasses.asdict(ResolvedConfig)` (sorted, `default=str`) are
unchanged. The values feeding them are untouched: `Metrics.as_dict()` keys, `ComparisonResult`
fields, and the `ResolvedConfig` field set are all verbatim (metrics/stats modules move byte-for-
byte; the config value types are unchanged). The comparator's family-wide FDR and bootstrap seeding
are in `evaluation/stats.py` verbatim. **No artifact byte changes.**

**Import-graph guarantee (§11 rule + `test_import_graph.py`).** The rule generalizes cleanly: the
protected set (domain + composition modules) imports only `common` abstractions at import time and
reaches every concrete adapter through `config`'s lazy dotted-target factories. `config` imports
`search` (pure composers) + `evaluation.stats` + `common` — no adapter — at import time; the
embedding/reranking factories (which *do* import `providers.inference`) are themselves resolved only
via `config`'s lazy targets, so importing `config`/`runner`/`io_csv`/`search`/`indexing`/`evaluation`
never pulls in `providers`, `datasets`, `embedding`, or `reranking`. The subprocess-`sys.modules`
probe test is retained with the adapter-prefix set above, so any accidental top-level adapter import
fails CI exactly as today. Generality (§1.4(3)) holds: swapping the backend is still a config-only
edit (new `ESIndexWriter`-equivalent + `build_searchers`/`build_rerankers` + target-table rows).

---

## 6. Draft §11 text (to drop into `docs/experiment.md` during implementation)

> ## 11. Module / Package Layout
>
> ```
> benchmark/
>   common/              # (g) shared bottom layer, depends on nothing
>     models.py          #   Query, Document, Qrel, ScoredDoc, RankedResult, FieldSchema, IndexMapping, enums
>     protocols.py       #   Searcher/Fuser/Reranker + Dataset ABCs; Embedder, RerankClient, IndexWriter Protocols
>     ranking.py         #   fuse_rrf_local + rerank_local (pure windowed ranking primitives)
>     logging_setup.py   #   console + file logging (logs/run_{timestamp}.log); use instead of print()
>   providers/           # (f) concrete adapters, depend ONLY on common
>     inference.py       #   OpenAI/Cohere/Voyage Embedders + Cohere/Voyage RerankClients (stdlib HTTP)
>     elasticsearch.py   #   LexicalSearcher, VectorSearch, ESReranker, ESIndexWriter, build_searchers/build_rerankers
>   embedding.py         # (c) make_embedder + EMBEDDER_PROVIDERS (dispatch provider -> providers.inference)
>   reranking.py         # (d) make_reranker + RERANKER_PROVIDERS (dispatch provider -> providers.inference)
>   indexing.py          # (a) Indexer (build orchestration) + embed-at-ingest streaming
>   search.py            # (b) RRFFuser, HybridSearch, SearchPipeline (the composers)
>   evaluation/          # (e) scoring + statistics
>     metrics.py         #   Evaluator, Metrics, QrelIndex
>     stats.py           #   Comparator, StatsCfg, ComparisonResult (bootstrap CI, Wilcoxon/permutation, FDR/BH-BY)
>   datasets/wands.py    #   WandsDataset (implements Dataset; label->gain; search_text concat)
>   config.py            #   config value types + YAML load/resolve + build_pipeline + lazy dotted adapter factories
>   runner.py            #   ExperimentRunner (the single execution path, §8.0)
>   io_csv.py            #   write_results_csv / write_metrics_csv / write_comparison_csv / write_run_config
> ```
>
> **Layers.** `a–g` form a strict acyclic engine: `common` (g) ← `providers` (f) ← `embedding`(c)/
> `reranking`(d) ← `indexing`(a)/`search`(b)/`evaluation`(e). Domain layers (a/b/e) import only
> `common` abstractions at import time; they consume concrete `providers` pieces (index writer, leaf
> searchers, reranker) and embedders/rerank-clients **injected** at runtime — Dependency Inversion is
> what makes "indexing/search need providers" hold without a backward import edge. `config`, `runner`,
> `io_csv` are the composition layer above the engine: they wire it together and own the lazy
> dotted-target factories, so no engine module names an adapter.
>
> **The clean-OOP seams.** Indexing is a backend-agnostic `indexing.Indexer(writer, embedders)` whose
> `build()` discovers dims → asks the `IndexWriter` for the `IndexMapping` → `ensure_index` → streams
> the corpus through the embedders → `bulk_index`. Search is composed by `build_pipeline`
> (`config.py`) from plain leaf `Searcher`s + a `Reranker` that the ES adapter's `build_searchers`/
> `build_rerankers` mint from the resolved `Services` + `IndexMapping`. There is no
> `SearcherFactory`/`_ESSearcherFactory` and no `ElasticsearchBackend` god-object; the ES pieces are
> ordinary provider classes.
>
> **Import-graph rule (enforced by `test_import_graph.py`).** `search`, `indexing`,
> `evaluation.metrics`, `evaluation.stats`, `runner`, `io_csv`, and `config` import **no**
> `benchmark.providers.*` / `benchmark.datasets.*` / `benchmark.embedding` / `benchmark.reranking` at
> import time; `config` imports `search` (composers) + `evaluation.stats` (`StatsCfg`) only. The lazy
> factories resolve dotted targets at call time. This is success criterion §1.4(3).

(§4's `build_pipeline` note and §12's "add a backend/dataset/embedder/reranker" bullets get their
symbol names updated to `IndexWriter` / `build_searchers` / `build_rerankers` / `ESIndexWriter`;
no semantic change.)

---

## 7. Risks / trade-offs / ambiguities to confirm

1. **Test churn beyond simple relocation (medium).** The factory removal reshapes several tests
   (`test_es_backend`, `test_build_pipeline`, `test_runner`, `test_config`, `test_import_graph`).
   Behavioral assertions (index mapping body, sem-field names, embed-at-ingest call order, graph
   shape, `isinstance`, math, artifact bytes) are preserved, but the *construction/wiring* lines
   change. This is the price of the requested seam redesign; it is the one place "tests move to
   mirror the layout" also touches how objects are built, not just import paths. **Flag for the
   user:** confirm this is acceptable (it is unavoidable if `_ESSearcherFactory` is truly deleted).

2. **`build_searchers`/`build_rerankers` open the ES client per call.** `_ESSearcherFactory` held one
   client for lexical+vector+reranker; the two free functions each call `_make_client`. Behavior is
   identical (each `Searcher` already held its own client reference), but to avoid two client objects
   per run the runner can build searchers and rerankers in one pass, or the two functions can share a
   client via a tiny internal `_open(indexer_cfg)`. Minor; no correctness impact. **Alternative:**
   keep a single public `ESSearchProvider` object (client + `lexical/vector/reranker` builder
   methods) — that is cleaner OOP but is closer to the factory shape the user rejected, so the free
   functions are the default. Confirm the preference.

3. **`embedding.py` / `reranking.py` are thin (~15–20 lines each).** They exist to give layers (c)
   and (d) a home and to make the `provider`-dispatch an explicit domain seam (as the brief's a–g
   intends). A leaner alternative is to leave `make_embedder`/`make_reranker` in
   `providers/inference.py` and treat (c)/(d) as "the Embedder/RerankClient protocols in common." The
   design keeps them as separate modules because the brief lists c/d as first-class layers that
   *depend on* providers; collapsing them would erase that edge. Cheap to reverse if the user prefers
   fewer files.

4. **`indexing`/`search` "need providers" only via injection, not an import edge.** This is the
   correct DIP resolution of the brief's own stated tension, and it is what keeps the graph acyclic
   and the import-graph test green. If a reviewer expects a literal `import` edge from `indexing` →
   `providers`, that expectation cannot be met without violating the import-graph guarantee; the
   injection model is the intended reading of "composes provider pieces."

5. **`common.ranking` co-locates a search primitive and a rerank primitive.** Justified because a
   provider (`ESReranker`) consumes `rerank_local`, forcing it below the domain layer; `common` is
   the only cycle-free home. If the user would rather keep two files, `common/fusion.py` +
   `common/rerank.py` is equivalent (no dependency difference) — the merge is a laziness call, not a
   correctness one.
