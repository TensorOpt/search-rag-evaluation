# Search-Relevance Benchmark — Experimental Design

> Status: draft v4 (review round 2 revisions) · Owner: TensorOpt · License: MIT
> Scope of this document: the *design* of a reproducible search-relevance benchmark harness. It defines objectives, abstractions, data flow, and methodology. It is **not** the implementation — it pins interface boundaries and sequencing so implementation is mechanical.

---

## 1. Objective, Scope, and Success Criteria

### 1.1 Objective
Build a **reproducible search-relevance benchmark harness** that measures, for a fixed dataset, how much each of several retrieval strategies improves relevance over a **BM25 baseline**. The first concrete instantiation is:

- **Dataset:** WANDS (Wayfair ANnotation Dataset for Search).
- **Backend:** ElasticSearch as a **plain vector + BM25 index**, **minimum supported version 8.15** (also runs on 8.18+ and 9.x). ES is **not** an inference gateway — the **harness owns all inference**: it embeds the corpus and each query itself via provider connectors (Cohere / Voyage / OpenAI, §3.4) and stores the vectors in ES `dense_vector` fields. BM25 is a `match` query; semantic retrieval is an ES `knn` query over a `dense_vector` field (§5.3). The **8.15 floor is a soft convenience, not load-bearing**: `dense_vector` + `knn` have been available in ES since well before 8.15, so any modern ES will do — we keep the `>= 8.15` pin only to avoid churn and to match the shipped `elasticsearch>=8.15,<9` client, not because any 8.15-only feature is required (the old `semantic_text` / `_inference` path that needed 8.15 is gone).
- **Baseline ranker:** BM25.

### 1.2 Variants under test
Each variant is scored **against the BM25 baseline**. These are the six *conceptual* retrieval
shapes; they are realized as **explicit named pipelines** the user writes in the config (§10) — there
is **no matrix expansion and no sweep**. A user who wants two embedding models or two RRF `k` values
writes two named pipelines by hand.

| # | Strategy | One-line description |
|---|----------|----------------------|
| 0 | `bm25` (baseline) | Lexical BM25 over text fields. |
| 1 | `semantic` | Dense/sparse vector retrieval, over a chosen embedder. |
| 2 | `hybrid` | RRF fusion of BM25 + semantic at a chosen `rank_constant`. |
| 3 | `bm25_rerank` | BM25 candidates → rerank. |
| 4 | `semantic_rerank` | Semantic candidates → rerank. |
| 5 | `hybrid_rerank` | RRF(BM25, semantic) → rerank. |

### 1.3 Scope (in / out)
- **In:** offline ranking quality on a static qrel set; provider-backed embedding + rerank connectors; explicit config-driven named pipelines; statistical comparison vs baseline; reproducible artifacts.
- **Out:** online A/B testing, latency/throughput SLAs, query rewriting, learning-to-rank training, click models. (Latency *may* be logged as a secondary observation but is not a success criterion.)

### 1.4 Success criteria
1. **Correctness:** all three CSV artifact types are produced with the exact schemas in §9, for every variant in the matrix; the statistics follow one coherent multiple-comparison regime (FDR) (§8.3).
2. **Reproducibility:** a single config + captured seed reproduces identical metrics and statistics (modulo backend nondeterminism, pinned per §9.1). Pipelines are fully explicit in the config, so the set of runs is exactly what the file declares — there is no expansion or data-dependent selection to reproduce.
3. **Generality:** swapping WANDS→another dataset, or ES→another backend, requires only a new adapter + config — **no edits to pipeline, evaluator, or stats code** (verified by §11 checklists). Edge cases that only a different dataset can trigger (e.g. all-zero or empty paired sets, §8.1) have defined, dataset-independent behavior.
4. **DRY:** every named pipeline shares **one** pipeline implementation and **one** execution path; they differ only by configuration (verified by code inspection — pipelines are config entries, not modules). See §4 and §8.

---

## 2. Conceptual Model & Glossary

```mermaid
flowchart LR
  DS[Dataset] -->|documents + field_schema| IDX[Indexer]
  IDX --> BK[(IndexWriter / Index)]
  DS -->|queries| HARNESS[ExperimentRunner]
  CFG[Config: explicit named pipelines] --> HARNESS
  HARNESS -->|SearchPipeline object graph per named pipeline| PIPE[SearchPipeline]
  PIPE -->|search per query| BK
  PIPE --> RUN[Run: ranked results -> result CSV]
  DS -->|qrels| EVAL[Evaluator]
  RUN --> EVAL
  EVAL --> M[per-query metrics CSV]
  M --> CMP[Comparator vs baseline]
  CMP --> C[comparison CSV]
```

| Term | Definition |
|------|------------|
| **Query** | A search request: `query_id`, `text`, optional `class`. |
| **Document** | A retrievable item: `doc_id` + a typed field bag. For WANDS, a product. |
| **Qrel** | A graded judgement `(query_id, doc_id) → gain` (a float). WANDS: `Exact=1.0`, `Partial=0.5`, `Irrelevant=0.0`. |
| **Run** | The ranked output of one variant over **all** queries: ordered `(query_id, doc_id, score, position)`. |
| **Variant** | A named pipeline declared explicitly in the config (§10) — one `pipelines.variants` entry, run and compared against the baseline. No matrix expansion. |
| **Searcher** | Anything that turns a query into a ranked list: `search(query, *, top_k) -> [ScoredDoc]`. Leaf retrievers, `HybridSearch`, and the top-level `SearchPipeline` are all `Searcher`s (§3.3). |
| **Fuser** | Combines several ranked lists into one, client-side: `fuse(result_lists, *, rank_window_size)`. `RRFFuser` wraps `fuse_rrf_local` (§3.7). |
| **Reranker** | Behavioral: rescores + reorders a candidate list for a query, client-side: `rerank(query, candidates) -> [ScoredDoc]` (§3.4). |
| **Metric** | A per-query scalar over a run given qrels: `avg_relevance`, `ndcg@10`, `recall@10`, `precision@10`. |
| **Baseline** | The reference variant (`bm25`) all comparisons subtract from. |
| **Connector** | A direct client to an inference provider (embedding or rerank). The harness owns inference (§3.4): an `Embedder` (Cohere/Voyage/OpenAI) turns text into dense vectors; a `RerankClient` (Cohere/Voyage) scores candidate docs. Realized in `benchmark.providers.inference`; configured by a `provider` + `settings` (`api_key`, `model_id`, reranker `top_n`, …). |
| **CI (here)** | A per-comparison percentile bootstrap interval reported as **effect-size context only** — *not* a significance gate (§8.2/§8.3). |

---

## 3. Core Abstractions

The harness is built around small Python ABCs / `Protocol`s that pin the seams where **datasets**, **backends**, and **models** plug in. There are two kinds of seam:

- **Behavioral ABCs for retrieval** — `Searcher`, `Fuser`, `Reranker` (§3.3/§3.4). Everything that produces a ranked list is a `Searcher`; a variant is an **object graph** of these (a natural OOP composite that mirrors a real search pipeline). Fusion is **client-side** (`RRFFuser` over materialized result lists); reranking is a client-side `rerank()` pass.
- **The `Dataset` ABC** (§3.2) — the base every dataset adapter derives from; it carries two shared concrete helpers (`build_search_text`, `map_label`) over the four abstract methods.
- **Structural Protocols** — `Embedder` and `RerankClient` (the provider connectors, §3.4, realized in `benchmark.providers.inference`) and `IndexWriter` (the index-writer/ingest seam the domain `indexing.Indexer.build` delegates to, §3.5). The domain `Indexer` itself is a single concrete backend-agnostic class (§3.5), so it is not a Protocol.

Concrete adapters (WANDS, ElasticSearch) implement these and live behind the boundary; the composers, evaluator, and comparator depend **only** on the abstractions.

```mermaid
flowchart TB
  subgraph "Retrieval seams (behavioral ABCs)"
    SRC[Searcher]
    FUS[Fuser]
    RRK[Reranker]
    RRF[RRFFuser -.-> FUS]
    HYB[HybridSearch -.-> SRC]
    SP[SearchPipeline -.-> SRC]
  end
  subgraph "Ingest seams (structural Protocols)"
    DSP[Dataset]
    EMP[Embedder / RerankClient - provider connectors]
    IWP[IndexWriter - sem_field_name/create_mapping/ensure_index/bulk_index]
  end
  subgraph "Domain OOP"
    IDXR[indexing.Indexer - backend-agnostic]
  end
  subgraph "Adapters (concrete, today)"
    WANDS[WandsDataset] -.implements.-> DSP
    ESL[LexicalSearcher] -.implements.-> SRC
    ESV[VectorSearch] -.implements.-> SRC
    ESR[ESReranker] -.implements.-> RRK
    ESIW[ESIndexWriter] -.implements.-> IWP
  end
  IDXR -.delegates to.-> IWP
```

> Note: `Embedder` and `RerankClient` are **provider connectors** (§3.4), realized in `benchmark.providers.inference` — the harness calls Cohere / Voyage / OpenAI directly, ES runs no inference. The indexer embeds the corpus with each `Embedder` and writes `dense_vector` fields; there is **no** `register_inference` and **no** `semantic_text`. `Reranker` is **behavioral** (§3.4) — a concrete reranker (e.g. ES `ESReranker`) fetches candidate doc-text and scores it via a `RerankClient` inside `rerank()`.

### 3.1 Data models (plain frozen dataclasses)

```python
@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    query_class: str | None = None

@dataclass(frozen=True)
class Document:
    doc_id: str
    fields: Mapping[str, Any]              # backend-agnostic field bag

@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    gain: float                            # graded relevance; WANDS: Exact=1.0/Partial=0.5/Irrelevant=0.0

@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str
    score: float

@dataclass(frozen=True)
class RankedResult:                        # one query's ranked list
    query_id: str
    docs: Sequence[ScoredDoc]              # ordered by position; docs[0] is rank 1
```

`position` in the result CSV is **derived** as the 1-based index into `docs` at write time (§9). It is not stored on `ScoredDoc`, so it cannot drift from the ordering.

### 3.2 Dataset

`Dataset` is an **ABC** (`abc.ABC`, in `protocols.py`) — the single, format-agnostic base every dataset adapter derives from. A concrete adapter (`WandsDataset`; future Amazon ESCI, BEIR) implements the four abstract methods and owns its own **file parsing**, **label→gain mapping**, and **field roles**; it sets `name`/`version` in `__init__`.

```python
class Dataset(ABC):
    name: str        # config-dispatch name, set by the subclass (e.g. "wands")
    version: str     # dataset version string, set by the subclass (e.g. "2022.0")

    @abstractmethod
    def queries(self) -> Iterable[Query]: ...
    @abstractmethod
    def documents(self) -> Iterable[Document]: ...      # streamed for large corpora
    @abstractmethod
    def qrels(self) -> Iterable[Qrel]: ...
    @abstractmethod
    def field_schema(self) -> "FieldSchema": ...        # declares field roles (§5)

    # --- concrete shared helpers (the reason this is an ABC, not a Protocol) ---
    @staticmethod
    def build_search_text(field_values: Mapping[str, Any], schema: FieldSchema) -> str:
        """§5.1: join every BM25- and SEMANTIC_SOURCE-role field value, in schema order,
        by "\n". A missing search-text key raises (never silently emits empty)."""
    @staticmethod
    def map_label(label: str, mapping: Mapping[str, float]) -> float:
        """Map a string label to a float gain via `mapping`; exhaustive, raises ValueError
        on an unknown label (no silent default). BEIR-style numeric qrels skip this and set
        gain = float(rel) directly."""
```

`field_schema()` is the seam that lets the indexer build a backend mapping without knowing about WANDS. The label→gain mapping is the dataset adapter's responsibility and is applied while emitting `qrels()` (so the rest of the harness only ever sees float gains). `queries`/`documents`/`qrels`/`field_schema` + `Qrel(gain: float)` describe any graded-relevance IR dataset regardless of on-disk format (TSV, parquet, JSONL); **nothing dataset-specific leaks into the base**. The two concrete helpers are the only shared machinery: `build_search_text` (the §5.1 concatenation, reused verbatim by every adapter) and `map_label` (a convenience string-label→gain mapper for WANDS/ESCI). File-format handling stays per-adapter — TSV/parquet/JSONL differ too much to share.

```python
class FieldRole(StrEnum):
    ID = "id"                      # unique doc identifier -> backend doc _id; not ranked
    BM25 = "bm25"                  # text field concatenated into search_text for lexical (BM25) matching
    SEMANTIC_SOURCE = "semantic_source"  # text field concatenated into search_text, which is embedded (semantic)
    NUMERIC = "numeric"            # numeric field stored for filtering/faceting/analysis; not text-ranked
    STORED = "stored"              # kept for retrieval/display/debug only; never ranked

@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: FieldRole

@dataclass(frozen=True)
class FieldSchema:
    fields: Sequence[FieldSpec]
    # search_text_field: the canonical text field the dataset adapter builds by
    # CONCATENATING every BM25- and SEMANTIC_SOURCE-role field (in schema order,
    # joined by newlines). It is used as BOTH the BM25 target AND the semantic
    # source, so every variant ranks the SAME input text (fair comparison). See §5.1.
    search_text_field: str = "search_text"
    rerank_field: str = "search_text"      # field text passed to the reranker
```

**What the roles mean.** Each dataset column is tagged with one `FieldRole` so the indexer knows how to map it, without hard-coding WANDS. The two *text* roles both feed the single canonical `search_text` field — because that field is simultaneously the BM25 target and the semantic-embedding source, a field marked `BM25` or `SEMANTIC_SOURCE` becomes searchable both lexically and semantically. `ID` → the backend doc id; `NUMERIC` → stored/filterable numbers; `STORED` → carried along for display/debug but never ranked.

**Worked example — WANDS `field_schema()`:**

```python
FieldSchema(
    fields=[
        FieldSpec("product_id",          FieldRole.ID),
        FieldSpec("product_name",        FieldRole.SEMANTIC_SOURCE),
        FieldSpec("product_description",  FieldRole.SEMANTIC_SOURCE),
        FieldSpec("product_features",     FieldRole.BM25),
        FieldSpec("product_class",        FieldRole.BM25),
        FieldSpec("category hierarchy",   FieldRole.STORED),   # facet; NOT in search_text (§5.1)
        FieldSpec("rating_count",         FieldRole.NUMERIC),
        FieldSpec("average_rating",       FieldRole.NUMERIC),
        FieldSpec("review_count",         FieldRole.NUMERIC),
    ],
    search_text_field="search_text",   # = "\n".join(product_name, product_description,
                                        #              product_features, product_class)
    rerank_field="search_text",
)
```

Here `product_id` becomes the doc `_id`; the four text fields are concatenated (newline-joined) into `search_text`, which is what BM25 matches on *and* what each embedding model embeds; the numeric fields are stored for analysis but never ranked. Swapping in a different dataset means emitting a different `FieldSchema` — the indexer and pipeline are unchanged.

### 3.3 Retrieval seams (`Searcher` / `Fuser` / `Reranker`) + the ingest seam

Retrieval is a **composite of behavioral ABCs**. Everything that turns a query into a ranked list is a `Searcher`; composition mirrors a real search pipeline (leaf retrievers → optional client-side fusion → optional client-side rerank). Fusion runs **client-side over materialized result lists** — a `Searcher` returns concrete `ScoredDoc`s, a `Fuser` merges lists, a `Reranker` rescores + reorders a candidate list.

```python
class Searcher(ABC):
    @abstractmethod
    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        """Return up to top_k docs ranked best-first (score desc, tie-break doc_id, §9.1)."""

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        """Search several queries at once; results ALIGNED to queries by index. CONCRETE default
        loops search (correct — one round trip per query) so any Searcher (e.g. a fake) works.
        Efficient backends OVERRIDE it: ES leaf searchers via the Multi-Search API (_msearch),
        the composers (HybridSearch/SearchPipeline) to propagate batching to their leaves."""
        return [self.search(q, top_k=top_k) for q in queries]

class Fuser(ABC):
    @abstractmethod
    def fuse(self, result_lists: Sequence[Sequence[ScoredDoc]], *,
             rank_window_size: int) -> list[ScoredDoc]:
        """Fuse several ranked lists over a fixed window into one ranked list."""

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        """Return candidates reordered best-first by the model's relevance scores."""
```

> **Design note — why a composite of `Searcher`s.** A variant is a natural object graph: `bm25` is a leaf `Searcher`; `hybrid` is a `HybridSearch(Searcher)` holding several leaf `Searcher`s + a `Fuser`; every variant is wrapped in a top-level `SearchPipeline(Searcher)` that optionally applies a `Reranker` (§3.6). No declarative spec layer, no `capabilities()` branching, no server-side-vs-fallback fork — fusion and rerank are **always client-side**, so any backend that can produce ranked leaf lists gets hybrid + rerank for free with identical `rank_window_size` semantics.

**The ingest seam.** Writing the index still needs a wire-format-aware seam. `IndexWriter` is now
exactly that — the index-writer the domain `indexing.Indexer.build` (§3.5) delegates all
backend-specific work to; retrieval methods are gone. It carries the backend-safe field-naming +
mapping bits (`sem_field_name`/`create_mapping`) that used to be ES free functions, plus the ingest
buffering granularity (`embed_batch_size`).

```python
@runtime_checkable
class IndexWriter(Protocol):
    # index-writer / ingest seam (retrieval moved to Searcher/Fuser/Reranker)
    # ES is a plain index writer: the harness embeds the corpus client-side and hands
    # documents whose field bag already carries the dense_vector values — no inference here.
    embed_batch_size: int                                    # ingest buffering granularity (§3.5)
    def sem_field_name(self, embedder_id: str) -> str: ...   # backend-safe dense_vector field name
    def create_mapping(self, schema: FieldSchema, sem_fields: Mapping[str, str],
                       vector_dims: Mapping[str, int]) -> "IndexMapping": ...
    def ensure_index(self, mapping: "IndexMapping") -> None: ...
    def bulk_index(self, docs: Iterable[Document], *, mapping: "IndexMapping") -> None: ...
```

> **Query binding is internal to each `Searcher`.** A concrete `Searcher.search(query, top_k)` receives the query string directly and issues its own backend request. For ES this is load-bearing for both the vector searcher and the reranker: `VectorSearch` embeds `query` client-side into a `knn` query vector, and `ESReranker.rerank(query, candidates)` threads `query` into the provider `RerankClient` call over the candidate doc-text it fetches by id (§5.3).

> **Why there is no `bm25` capability flag.** With no `capabilities()` seam, backends no longer advertise features. Lexical **BM25 is not optional** — it is the baseline (§1.2), realized as a concrete `LexicalSearcher`. A pure vector index (FAISS/Qdrant) that cannot score lexically is the one case where a `bm25` graph cannot be built; that is **deferred** (§13) — the day such a backend is added, the matrix skips the `bm25`/`hybrid`/`bm25_rerank` variants (a matrix concern, not a backend flag).

### 3.4 Provider connectors (`Embedder` / `RerankClient`) & Reranker (behavioral)

```python
class Embedder(Protocol):                  # provider connector — text -> dense vectors
    id: str                                # config service name (== sem-field naming key, §3.5)
    @property
    def dim(self) -> int: ...              # output dimensionality (probed once, or settings.dims)
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

class RerankClient(Protocol):              # provider connector — scores candidate docs for a query
    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]: ...
    # one score per document, ALIGNED 1:1 to `documents` (higher = more relevant)

class Reranker(ABC):                       # BEHAVIORAL — rescores at query time (backend seam)
    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]: ...
```

The harness **owns inference** (§1.1): ES is a plain index, so embeddings and reranking are computed by **provider connectors** the harness calls directly — realized in `benchmark.providers.inference` and typed by two structural `Protocol`s:

- **`Embedder`** — a dense-embedding connector. `embed_documents` embeds the corpus at ingest (§3.5) into `dense_vector` fields; `embed_queries` embeds each query at search time so `VectorSearch` can run ES `knn` (§5.3). `id` is the config service name (the sem-field naming key, §3.5); `dim` is the output dimensionality the `dense_vector` mapping needs before ingest — taken from `settings.dims` when given (move-with-certainty) else discovered once by embedding a probe text (the authoritative source is the provider). Shipped: `OpenAIEmbedder`, `CohereEmbedder`, `VoyageEmbedder` — Cohere/Voyage carry a document-vs-query `input_type`, OpenAI has none.
- **`RerankClient`** — a rerank connector: `rerank_scores(query, documents)` returns one relevance score per document, **aligned 1:1 to `documents`** (higher = more relevant), realigning the provider's relevance-ordered response back to input order by `index`. Shipped: `CohereReranker`, `VoyageReranker`. **OpenAI has no reranker** — a reranker configured with `provider: openai` is rejected both at config load (§10) and by `make_reranker` (§5.4).

An **`EmbedderCfg` / `RerankerCfg`** service entry (§10) carries only `name`, `provider`, and a `settings` block (`api_key`, `model_id`, optional `rate_limit.requests_per_minute`, `batch_size`, `dims`, …). The runner instantiates the connectors lazily via `config.make_embedders` / `make_rerankers` (§11); a `provider` outside the shipped set raises at config load. There is **no `InferenceEndpoint`, no ES `_inference` task type, and no `register_inference`** — that machinery is gone.

`Reranker` is **behavioral** (§3.3): a concrete reranker (ES `ESReranker`) is constructed from a `RerankClient` + a doc-text lookup and, inside `rerank(query, candidates)`, fetches the candidate doc-text by id and scores it via the connector (through the `rerank_local` helper, §3.7).

> **Reranker `top_n` (verified, load-bearing).** A reranker's rank-window cap `top_n` is a plain key in the reranker service's `settings` block (no `service_settings`/`task_settings` split — that was an ES `_inference` wire concern, now gone). The runner reads `settings["top_n"]` for the `W <= top_n` assertion (§5.3 / §8 R0): it is the number of candidates the provider is asked to score per request.

### 3.5 Indexer

```python
@dataclass(frozen=True)
class IndexMapping:
    index_name: str
    search_text_field: str                 # canonical text field BM25 queries hit
    sem_fields: Mapping[str, str]          # embedder_id -> dense_vector field name
    backend_mapping: Mapping[str, Any]     # backend-native field map (ES mapping body)
    def sem_field(self, embedder_id: str) -> str:
        return self.sem_fields[embedder_id]

class Indexer:                                 # benchmark/indexing.py — concrete, backend-agnostic
    def __init__(self, writer: IndexWriter, embedders: Sequence[Embedder]) -> None: ...
    def build(self, dataset: Dataset) -> "IndexMapping": ...
```

`Indexer` is a single concrete domain object (not a Protocol — there is exactly one implementation).
Its `build` discovers dims → asks the injected `IndexWriter` for the `IndexMapping`
(`create_mapping`) → `ensure_index` → streams the corpus through the embedders (the
`_embed_documents`/`_embed_batch` bounded-buffer streaming lives here, as domain code that only
touches `Document` + `Embedder`) → `bulk_index`. All backend-specific mapping / field-naming is
delegated to the `IndexWriter` (ES: `ESIndexWriter`), so the same `Indexer` drives any backend.

**What `IndexMapping` is for.** It is the value returned by `Indexer.build(...)` and is the **single source of truth for the concrete field names each leaf `Searcher` must query**. The composers are dataset- and backend-agnostic, so nothing upstream knows that ES named the `dense_vector` field for embedder `cohere` `"sem__cohere"`; `IndexMapping` hands it those names. the ES adapter's `build_searchers` (§3.3, called before `build_pipeline`) reads exactly two things from it — `mapping.search_text_field` (the lexical target) and `mapping.sem_field(embedder)` (the `dense_vector` field for a given embedder) — to build the leaf `Searcher`s without re-deriving any backend-specific naming. `backend_mapping` is the raw ES mapping body used to create the index (§5.2); `index_name` is where documents land.

**Worked example** (the WANDS index built for three embedders):

```python
IndexMapping(
    index_name="wands_bench",
    search_text_field="search_text",                 # bm25(fields=["search_text"])
    sem_fields={                                      # embedder id -> dense_vector field
        "cohere": "sem__cohere",
        "voyage": "sem__voyage",
        "openai": "sem__openai",
    },
    backend_mapping={"mappings": {"properties": { ... }}},  # the §5.2 ES mapping body
)
# mapping.sem_field("cohere")    -> "sem__cohere"     # knn(field="sem__cohere") for the cohere variant
# mapping.search_text_field      -> "search_text"     # bm25 target for every variant
```

So the `semantic` variant for embedder `cohere` uses `VectorSearch(field=mapping.sem_field("cohere"), embedder_id="cohere")`, and the `bm25` baseline uses `LexicalSearcher(fields=[mapping.search_text_field])` — same composers, names supplied by the mapping (§4).

**Lifecycle (strict order — no endpoint registration; the harness embeds the corpus client-side):**
1. **Discover each embedder's output `dim`.** For each `Embedder`, read `settings.dims` or probe the provider once (§3.4) — the `dense_vector` mapping needs `dims` before the index is created. Nothing is registered server-side (ES runs no inference).
2. **Translate schema → `IndexMapping`.** From `dataset.field_schema()`: the canonical `search_text` field → a plain `text` field (the BM25 target); **one `dense_vector` field per embedder** (`dims` = that embedder's dim, `index: true`, `similarity: cosine`); numeric → `float`; stored → `keyword`; ids → doc `_id`. There is no `copy_to` and no `semantic_text`. See §5.2.
3. **Embed the corpus and stream it through `bulk_index`.** The indexer streams `dataset.documents()` through the embedders — a bounded buffer embeds each batch's `search_text` with every `Embedder` and attaches the vectors under each `dense_vector` field — so the corpus is embedded **client-side** and written already-vectorized, never fully materialized (43K WANDS / 1M ESCI docs stream through). `bulk_index` **streams + batches** via `elasticsearch.helpers.streaming_bulk(chunk_size=...)`; `_id = product_id` (idempotent); `raise_on_error=True` so any failed item surfaces a `BulkIndexError` (not swallowed); the index is refreshed once at the end. A provider failure surfaces as a `ProviderError` while the generator is consumed (§3.4/§5.4).
4. **Return `IndexMapping`** (index name, `search_text` field name, per-embedder `sem_field` names) so the pipeline can name fields without re-deriving them.

The indexer is **dataset- and model-agnostic**: everything specific arrives via `field_schema()` + the `Embedder` connectors.

### 3.6 The composers (`RRFFuser` / `HybridSearch` / `SearchPipeline`)

Three backend-agnostic composers (in `search.py`) wire leaf `Searcher`s into the six variants. They import only `common.models`/`common.protocols`/`common.ranking` + stdlib — no adapters, no numpy.

```python
class RRFFuser(Fuser):
    def __init__(self, *, rank_constant: int): ...
    def fuse(self, result_lists, *, rank_window_size):
        return fuse_rrf_local(result_lists, rank_constant=self.rank_constant,
                              rank_window_size=rank_window_size)          # client-side (§3.7)

class HybridSearch(Searcher):
    def __init__(self, *, retrievers: Sequence[Searcher], fuser: Fuser,
                 retrieval_window_size: int): ...
    def search(self, query, *, top_k):
        lists = [r.search(query, top_k=self.retrieval_window_size) for r in self.retrievers]
        return self.fuser.fuse(lists, rank_window_size=self.retrieval_window_size)[:top_k]
    def bulk_search(self, queries, *, top_k):                    # propagate batching to the leaves
        per_r = [r.bulk_search(queries, top_k=self.retrieval_window_size)  # each leaf ONCE (_msearch)
                 for r in self.retrievers]
        return [self.fuser.fuse([per_r[j][i] for j in range(len(self.retrievers))],
                                rank_window_size=self.retrieval_window_size)[:top_k]
                for i in range(len(queries))]                    # fuse per query, aligned

class SearchPipeline(Searcher):
    def __init__(self, *, retriever: Searcher, reranker: Reranker | None = None,
                 rerank_window_size: int | None = None):
        # reranker set  -> rerank_window_size REQUIRED
        # reranker None -> rerank_window_size MUST be None
        # otherwise ValueError (exhaustive, no silent default)
        ...
    def search(self, query, *, top_k):
        if self.reranker is None:
            return self.retriever.search(query, top_k=top_k)
        candidates = self.retriever.search(query, top_k=self.rerank_window_size)
        return self.reranker.rerank(query, candidates)[:top_k]
    def bulk_search(self, queries, *, top_k):                    # retrieval batches; rerank per query
        if self.reranker is None:
            return self.retriever.bulk_search(queries, top_k=top_k)
        cands = self.retriever.bulk_search(queries, top_k=self.rerank_window_size)
        return [self.reranker.rerank(q, c)[:top_k] for q, c in zip(queries, cands)]
```

**The six strategies as object graphs** (built by `build_pipeline` in Phase 6, §4):

```python
bm25            = SearchPipeline(retriever=LexicalSearcher(...))
semantic        = SearchPipeline(retriever=VectorSearch(...))
hybrid          = SearchPipeline(retriever=HybridSearch(
                      retrievers=[LexicalSearcher(...), VectorSearch(...)],
                      fuser=RRFFuser(rank_constant=k), retrieval_window_size=W))
bm25_rerank     = SearchPipeline(retriever=LexicalSearcher(...), reranker=r, rerank_window_size=W)
semantic_rerank = SearchPipeline(retriever=VectorSearch(...),    reranker=r, rerank_window_size=W)
hybrid_rerank   = SearchPipeline(retriever=HybridSearch([Lexical, Vector], RRFFuser(k), W),
                                 reranker=r, rerank_window_size=W)
```

```mermaid
flowchart LR
  L[LexicalSearcher] --> H[HybridSearch]
  V[VectorSearch] --> H
  H -->|RRFFuser client-side| SP[SearchPipeline]
  SP -->|reranker? rerank at rerank_window_size| OUT[ranked list]
```

`SearchPipeline` retrieves **`rerank_window_size` candidates** when a reranker is present (the candidate depth fed to rerank), reranks, then truncates to `top_k`; with no reranker it is a pass-through retrieving directly at `top_k`. There is exactly **one** `SearchPipeline` class; all six variants are object graphs of the same composers (§4), searched via the single `pipeline.search(query, top_k)` path (§8).

**`bulk_search` propagates batching to the leaves.** Both composers **override** `Searcher.bulk_search` (§3.3) so the runner can search the whole frozen query set with far fewer round trips (§8.0): `HybridSearch.bulk_search` calls each retriever's `bulk_search` **once** (so ES leaves batch via `_msearch`, §5.3) then **fuses per query** and truncates; `SearchPipeline.bulk_search` batches retrieval via `retriever.bulk_search` then **reranks per query** (the provider rerank is per-query — **bulk rerank is not batched; a future optimization**, §5.3/§13). Both return a `list[list[ScoredDoc]]` aligned to `queries` by index. A leaf that does not override `bulk_search` (e.g. a fake) still works via the ABC's default per-query loop.

### 3.7 Client-side fusion & rerank helpers

Fusion and rerank are **always client-side**, over materialized result lists — there is no server-side-vs-fallback split and no `capabilities()` branching. `RRFFuser` (§3.6) wraps `fuse_rrf_local`; a concrete `Reranker` (e.g. `ESReranker`) uses `rerank_local` to score + reorder client-side. Both helpers take `rank_window_size` so the window semantics are explicit and backend-independent:

```python
def fuse_rrf_local(lists: Sequence[Sequence[ScoredDoc]], *,
                   rank_constant: int, rank_window_size: int) -> list[ScoredDoc]:
    """Truncate each input list to its top rank_window_size BEFORE fusing, then
    RRF: score(d) = Σ 1/(rank_constant + rank_d_in_truncated_list), rank 1-based.
    Returns merged list sorted by fused score desc, tie-break doc_id."""

def rerank_local(query: Query, candidates: Sequence[ScoredDoc], *,
                 rank_window_size: int,
                 doc_text: Callable[[str], str],
                 score_fn: Callable[[Query, Sequence[str]], Sequence[float]]) -> list[ScoredDoc]:
    """Take only the top rank_window_size candidates, score them via
    score_fn(query, [doc_text(doc_id) for each]) -> one relevance score per doc
    text (higher = more relevant), return re-sorted by model score; candidates
    beyond the window keep their input order appended after the reranked head.
    Scoring is backend-specific, so the caller supplies score_fn: a concrete
    Reranker wraps its inference call into score_fn and passes the doc-text lookup."""
```

`RRFFuser.fuse` is a one-line delegation to `fuse_rrf_local`. `rerank_local` is the helper a concrete `Reranker.rerank` uses: it fetches candidate doc-text by id (`doc_text`), calls its inference endpoint (`score_fn`), and returns the windowed reorder. Because fusion/rerank are client-side, any backend that can produce ranked leaf lists composes into `hybrid` / `*_rerank` with identical `rank_window_size` semantics — no forking.

---

## 4. Variants as Object Compositions (DRY)

Every variant is a **`SearchPipeline` object graph** built from the same composers (§3.6). No variant has bespoke code; the config (§10) declares each pipeline explicitly as a `PipelineCfg`, and `build_pipeline` assembles the graph. There is **no matrix expansion** — the pipelines run are exactly the ones written in the config.

> `build_pipeline` lives in **`config.py`** (§11 — the config layer holds the config value types + the pipeline-assembly) and is built in **Phase 6**, not with the composers: it maps a `PipelineCfg` → a `SearchPipeline`, so keeping it in `config.py` (which already imports `search` for the composers) avoids a `search`→`config` forward dependency. `search.py` (Phase 5) defines only the composers (`RRFFuser`/`HybridSearch`/`SearchPipeline`). The backend-specific leaf `Searcher`s / `Reranker` (the ES `LexicalSearcher`/`VectorSearch`/`ESReranker`, Phase 9/10) are pre-built by the adapter's `build_searchers`/`build_rerankers` free functions from the resolved `Services` + `IndexMapping`; `build_pipeline` then just composes the object graph over those plain `{name: Searcher}` / `{name: Reranker}` maps, so it stays backend-agnostic (no adapter import). There is no `SearcherFactory` — `_ESSearcherFactory`/`make_searcher_factory` are gone. The six strategies below are the conceptual shapes; each is an *example a user writes* as a named pipeline, not something auto-expanded.

| Strategy | retriever graph | reranker |
|----------|-----------------|----------|
| `bm25` (baseline) | `LexicalSearcher(search_text)` | — |
| `semantic` | `VectorSearch(sem_field[embedder])` | — |
| `hybrid` | `HybridSearch([Lexical, Vector], RRFFuser(k), W)` | — |
| `bm25_rerank` | `LexicalSearcher` | `ESReranker(r), rerank_window_size=W` |
| `semantic_rerank` | `VectorSearch(sem_field[embedder])` | `ESReranker(r), rerank_window_size=W` |
| `hybrid_rerank` | `HybridSearch([Lexical, Vector], RRFFuser(k), W)` | `ESReranker(r), rerank_window_size=W` |

```python
def build_pipeline(pcfg: PipelineCfg, searchers: Mapping[str, Searcher],
                   rerankers: Mapping[str, Reranker]) -> SearchPipeline:
    leaves = [searchers[name] for name in pcfg.retrievers]   # leaves pre-built by build_searchers

    if pcfg.fuser is not None:                         # 2+ retrievers require a fuser (§10)
        retriever: Searcher = HybridSearch(retrievers=leaves,
                                           fuser=RRFFuser(rank_constant=pcfg.fuser.rank_constant),
                                           retrieval_window_size=pcfg.fuser.window)
    else:
        (retriever,) = leaves                          # exactly one leaf when not fusing

    if pcfg.reranker is not None:
        return SearchPipeline(retriever=retriever,
                              reranker=rerankers[pcfg.reranker],
                              rerank_window_size=pcfg.rerank_window_size)
    return SearchPipeline(retriever=retriever)
```

> The leaf `Searcher`s and `Reranker`s are minted up front by the ES adapter's
> `build_searchers(indexer_cfg, mapping, specs, *, embedders)` / `build_rerankers(indexer_cfg,
> mapping, names, *, rerank_clients)` free functions (§3.3) — they own the `kind` dispatch
> (`lexical` → `LexicalSearcher(fields=[mapping.search_text_field])`, `vector` →
> `VectorSearch(field=mapping.sem_field(embedder))`) and the `ESReranker` construction that the old
> factory did. `build_pipeline` then only composes the object graph over the resulting plain maps.

> All six shapes reuse the *same* composition; they differ only in how many retrievers a pipeline lists, `RRFFuser`'s `rank_constant`, and whether a reranker is set. Adding "semantic+rerank" costs zero new pipeline code — only a named `pipelines.variants` entry. `pcfg.fuser.rank_constant` is a concrete integer read straight from the config — `build_pipeline` never performs any selection.

> The reranker's field argument is `mapping.search_text_field` — `IndexMapping` (§3.5) carries only `search_text_field`/`sem_fields`, and §5.3 fixes `search_text` as the ES rerank field (`FieldSchema.rerank_field` also defaults to it). If a dataset ever needs a distinct rerank field, add `rerank_field` to `IndexMapping` and read it here.

---

## 5. ElasticSearch Mapping & Indexing Plan

### 5.1 Field roles (from `Dataset.field_schema()`)
For WANDS `product.csv`:

| Field | Role | ES mapping |
|-------|------|-----------|
| `product_id` | id | doc `_id` (`keyword`) |
| `product_name` | bm25 + semantic_source | feeds `search_text` |
| `product_description` | bm25 + semantic_source | feeds `search_text` |
| `product_features` | bm25 | feeds `search_text` |
| `product_class` | bm25 | feeds `search_text` |
| category hierarchy | stored (facet) | `keyword` — kept for faceting; **not** in `search_text` |
| `rating_count`, `average_rating`, `review_count` | numeric (stored) | `integer`/`float` |

A canonical **`search_text`** field is built by concatenating the values of every `BM25`- and `SEMANTIC_SOURCE`-role field (§3.2) — for WANDS: `product_name`, `product_description`, `product_features`, `product_class` — **in schema order, joined by newlines (`"\n"`)**. It is **both** the BM25 target and the semantic source — so every variant ranks the same input text (fair comparison; isolates the ranker, not the field selection). The dataset adapter performs this concatenation when emitting each `Document`'s field bag, via the shared `Dataset.build_search_text(field_values, schema)` helper (§3.2) — the rule is factored onto the ABC so every adapter reuses it.

### 5.2 One `dense_vector` field per embedder (verified shape)
With ElasticSearch there is exactly **ONE index** (`indexer.index`, e.g. `wands_bench`) — **not** one index per embedder. Inside that single index live a single **`search_text`** field (the BM25 target, §5.1) **plus one `dense_vector` field per embedder** that a **vector searcher** references. So in the §10 config, `semantic_co` (and any second `semantic_*`) are **fields in the same index, not separate indices**: each is a `dense_vector` field the harness populates by embedding the doc's `search_text` with that embedder's connector **at ingest** and writing the resulting vector into the field (§3.5). ES computes **no** embeddings — there is no `semantic_text` field and no `copy_to`; the vectors arrive already-computed in each document's `_source`.

**Where and which (how the indexer is driven).** The indexer learns **WHERE** to build from `indexer.{provider, index, settings}` (§10). It learns **WHICH** embedders to build `dense_vector` fields for from the `Embedder`s passed to the `indexing.Indexer(writer, embedders)` constructor — the runner instantiates one per configured `embedder` service (§11). The dataset's `FieldSchema` (§3.2) says **which columns feed `search_text`**. This single-`indexer`-block model works because ES is one store that holds BM25 + all vector fields together; a per-store model (§12/§13) is only needed for a pure vector store.

```jsonc
// mapping (the harness embeds the corpus client-side; ES only stores + searches the vectors):
"mappings": {
  "properties": {
    "search_text": { "type": "text" },
    "sem__cohere": { "type": "dense_vector", "dims": 1024, "index": true, "similarity": "cosine" },
    "sem__voyage": { "type": "dense_vector", "dims": 1024, "index": true, "similarity": "cosine" },
    "sem__openai": { "type": "dense_vector", "dims": 1536, "index": true, "similarity": "cosine" }
  }
}
```

Each `dense_vector` field is `index: true` with `similarity: cosine` (cosine suits the normalized embeddings these providers emit); its `dims` is that embedder's output dimensionality (probed once, or `settings.dims`, §3.4). Adding an embedding model = add one `embedder` service + a `vector` searcher + one `dense_vector` field + reindex (the new field must be embedded for the whole corpus). A full reindex is the clean path for an existing corpus and is recorded in run metadata.

**Vector field naming (dot-free).** The `dense_vector` field name for an embedder is `"sem__"` + the embedder `id` with every non-alphanumeric run replaced by `_` (`ESIndexWriter.sem_field_name` builds it; e.g. `voyage-3.5` → `sem__voyage_3_5`). ES field names cannot contain `.` (dots denote subfields), so the sanitization is load-bearing, not cosmetic. `IndexMapping.sem_fields` maps each embedder `id` → its `dense_vector` field name so `sem_field(embedder_id)` resolves the name the vector searcher queries.

> **No ES ML-memory constraint.** Because embeddings and reranking run in the **provider** (not on an ES ML node), the old single-node ML-memory ceiling — where a co-deployed embedder + reranker could exhaust the ML budget and return HTTP 429 "Could not start deployment" — **no longer applies**. ES deploys no model; it only stores and searches vectors. The capacity concern shifts to the **provider's rate limits and cost** (§13), handled by the connector's `RateLimiter` + retry/backoff on 429 (§3.4/§5.4).

**Bulk ingest is streamed + chunked.** `ESIndexWriter.bulk_index` writes via `elasticsearch.helpers.streaming_bulk(client, actions, chunk_size=...)` over a **lazy** actions generator (each `{"_op_type": "index", "_index": …, "_id": doc.doc_id, "_source": dict(doc.fields)}`), so the corpus streams through in fixed-size chunks and is never fully materialized — required for 43K-doc (WANDS) / ~1M-doc (ESCI) corpora that would break a single `bulk()` body. `chunk_size` is a module constant (the ES helpers default, 500) overridable via `indexer.settings.bulk_chunk_size`. `raise_on_error=True` so a failed item surfaces a `BulkIndexError` (**errors surface, not swallowed**); the index is refreshed once at the end; empty input is a logged no-op.

### 5.3 ES `Searcher` / `Reranker` implementations
Each retriever is realized as its **own ES query** and returns a materialized `list[ScoredDoc]`; hybrid fusion and rerank happen **client-side** in the harness (§3.6/§3.7). There are no nested `rrf` / `text_similarity_reranker` retriever trees, and no `server_side` capability. (Change from the v4 draft: ES server-side `rrf` and `text_similarity_reranker` are dropped in favor of client-side composition; one-round-trip server-side fusion is noted as a deferred performance optimization in §13.)

- **`LexicalSearcher.search(query, top_k)`** → a `match` query, returns the top-`top_k` docs:
  ```jsonc
  { "query": { "match": { "search_text": "$Q" } }, "size": $top_k }
  ```
- **`LexicalSearcher.bulk_search(queries, top_k)`** (and `VectorSearch.bulk_search`, Phase 10) → the whole query set via the ES **Multi-Search API** (`_msearch`), **chunked** into groups of `msearch_chunk_size` (a module constant, default 100, overridable via `indexer.settings.msearch_chunk_size`): per chunk the payload alternates a per-search header `{}` then the body, and `response["responses"]` is parsed **in order** into an ALIGNED `list[list[ScoredDoc]]`. A shared `_msearch(client, index, bodies, *, chunk_size)` helper does the chunking + parsing (reused by `VectorSearch`); each per-search response is checked for an `"error"` key and **raises** if present (**not** silently emptied). Hits map to `ScoredDoc` through the same `_hits_to_scored` helper as `search`, so the **client-side (score desc, doc_id asc) tie-break (§9.1)** is identical. This cuts the ~480 (WANDS) / ~48K (ESCI) per-query round trips to a handful of `_msearch` calls (§8.0).
  > **Bulk rerank is NOT batched (future optimization).** `SearchPipeline.bulk_search` batches only *retrieval* via `_msearch`; the provider rerank call is per-query, so `ESReranker.rerank` is still invoked once per query. Batching rerank is deferred (§13).
- **`VectorSearch.search(query, top_k)`** → embed the query client-side (`query_embedder.embed_queries([query])[0]`), then an ES `knn` query over that embedder's `dense_vector` field:
  ```jsonc
  { "knn": { "field": "sem__$m", "query_vector": [ ... ], "k": $top_k,
             "num_candidates": max($top_k, num_candidates) }, "size": $top_k }
  ```
  `num_candidates` (the per-shard ANN candidate pool, default 100, overridable via `indexer.settings.knn_num_candidates`) is floored at `top_k`. `VectorSearch.bulk_search` embeds the whole query set through the connector (batched) and issues one `knn` body per query via the shared `_msearch`. The old `semantic`-query / `semantic_text` path is gone — ES no longer embeds the query.
- **hybrid** → `HybridSearch` (§3.6) queries `LexicalSearcher` and `VectorSearch` each at `retrieval_window_size` (W) and fuses their two result lists with `RRFFuser` (`fuse_rrf_local`, §3.7). No `rrf` retriever is sent to ES.
- **`ESReranker.rerank(query, candidates)`** → fetch the candidates' `search_text` by id (one `mget(index, ids=[...], source=[search_text])`; a **not-found** candidate or one missing the field **raises**, never silently drops), call the provider `RerankClient` with `query` + the candidate doc-texts, and reorder via `rerank_local` (§3.7). The rerank window is the whole candidate list (`rank_window_size = len(candidates)`) — `SearchPipeline` already retrieved exactly `rerank_window_size` candidates (§3.6). An empty candidate list short-circuits to `[]` (no `mget`, no provider call).

  The connector's `rerank_scores(query, doc_texts)` (§5.4) returns one relevance score per candidate doc-text **aligned to input order** (higher = more relevant; a cross-encoder score that **may be negative**), having realigned the provider's relevance-ordered response back by `index`. `ESReranker` hands that aligned score list to `rerank_local` as `score_fn`, which re-sorts the candidates by model score desc (tie-break `doc_id`, §9.1). The query passed through is the query **text** (`str`) — `rerank_local`'s `query` is opaque, only forwarded to `score_fn`.

`rank_window_size` (W) is the candidate depth fed to client-side fusion (`retrieval_window_size`) and rerank (`rerank_window_size`); fixed per matrix and recorded.

> **Constraints (verified, encoded as assertions at run start):**
> - `ESReranker` window `W <= top_n`, where **`top_n` is read from the reranker service's `settings["top_n"]`** (§3.4) — the number of candidates the provider is asked to score per request. The runner asserts `W <= top_n` before running any rerank variant (§8 step R0); a missing `top_n` or `W > top_n` raises.
> - The rerank doc text must be a real stored field; we use `search_text`.

### 5.4 Provider connectors & failure model
Embeddings and reranking are computed by direct **provider connectors** in `benchmark.providers.inference` (§3.4) — the harness calls Cohere / Voyage / OpenAI over stdlib `urllib.request` + `json` (zero new dependencies). One shared `_post_json` (bearer auth) + a `RateLimiter` back every connector:

- **Embedders** (`OpenAIEmbedder` / `CohereEmbedder` / `VoyageEmbedder`): `embed_documents` / `embed_queries` sub-chunk arbitrary input to a per-provider `batch_size` (OpenAI 2048 / Cohere 96 / Voyage 128, overridable via `settings.batch_size`) so a 43K-doc corpus never exceeds a provider's per-request cap. Cohere/Voyage send a document-vs-query `input_type`; OpenAI has none. `dim` is `settings.dims` when given, else discovered once via a probe embed. A provider returning the wrong number of vectors raises (never pads).
- **Rerankers** (`CohereReranker` / `VoyageReranker`): `rerank_scores(query, documents)` requests a score for **every** document (`top_n` / `top_k` = `len(documents)`) and realigns the provider's relevance-ordered response back to input order by `index`; a missing index (a truncated response) raises rather than defaulting to `0`. **OpenAI has no reranker** — `make_reranker("…", "openai", …)` raises (§3.4).

**Failure model (errors never swallowed).** `_post_json` retries on a retryable HTTP status (429 + transient 5xx) or a connection-level error, with exponential backoff honoring `Retry-After`, up to `max_retries`. A non-retryable status (401/403/400) raises `ProviderError` immediately with the raw response body; an exhausted retry budget raises with the last body. `ProviderError` carries the provider, HTTP status, URL, and raw body so the provider's own error payload is inspectable. `RateLimiter` enforces a minimum interval from `settings.rate_limit.requests_per_minute` (serial request spacing; a no-op when unset).

---

## 6. End-to-End Data Flow (no gaps)

```mermaid
flowchart TD
  A[1. Dataset.load -> queries, documents, qrels, field_schema] --> B[2. Indexer.build -> embed corpus via connectors, ensure_index, bulk_index -> IndexMapping]
  B --> C[3. Read explicit pipelines from config: baseline + named variants, baseline first]
  C --> D{for each named pipeline}
  D --> E[3a. R0 if reranker: assert rerank_window_size <= reranker settings.top_n]
  E --> F[3b. build_pipeline -> SearchPipeline graph; pipeline.search per query over ALL queries]
  F --> G[4. write result_variant_ts.csv: query_id,product_id,score,position]
  G --> H[5. Evaluator.score per query -> write metrics_variant_ts.csv]
  H --> I[(per-query metric vectors held in memory keyed by query_id)]
  I --> J[after all pipelines done]
  J --> K[6. Comparator: ONE family-wide compare of ALL variant pipelines vs baseline over SAME query set, FDR/BH across the family -> split rows per variant -> comparison_bm25_variant_ts.csv each]
  K --> L[7. write run_config_ts.json]
```

Concrete materialization rules:
- **Result CSV** (step 4): for each `RankedResult`, write one row per `ScoredDoc`, `position = 1-based index`, `score = ScoredDoc.score`. At most `top_k` rows per query.
- **Per-query metrics** (step 5): the Evaluator joins each `RankedResult` to qrels by `query_id` (qrels indexed once into `dict[query_id, dict[doc_id, gain]]`), computes the four metrics (§7), writes one row per query, **and** returns the per-query vectors in memory keyed by `query_id` for the Comparator (so metrics are computed once, not re-derived from CSV).
- **Comparison** (step 6): runs only after all runs' metric vectors exist; a **single** `Comparator.compare(baseline_maps, variant_maps)` call pairs each variant against the baseline on the **identical query set** (§8.1) and applies the **FDR (BH/BY) correction family-wide** across all `(variant, metric)` tests (§8.3), then the returned rows (each carrying both raw and FDR-adjusted significance) are grouped by variant and written one `comparison_bm25_{variant}_{ts}.csv` each.

The baseline (`bm25`) is always materialized first so every later comparison has its paired reference in memory.

---

## 7. Metrics

All metrics are per query at cutoff **k=10**, then aggregated (mean across queries) for reporting; the per-query `Metrics` are retained for §8 statistics. Let the query's ranked returned list be `d_1..d_n` (position 1 = top). For each returned doc, `gain(d)` is the **qrel gain if a judgement exists** (a float graded per dataset — WANDS: `{0, 0.5, 1}` for Irrelevant/Partial/Exact) or **`MISSING` (`NaN`) if no qrel entry exists** for that `(query, doc)` pair.

**Condensed-list evaluation (Sakai).** A `MISSING` judgement is **not** irrelevant — it is **skipped**. Form the **condensed list**: the ranked returned docs with the `MISSING` ones dropped, judged docs kept in original rank order. Metrics are computed over the **condensed top-10** = the first `min(10, #judged-in-list)` **judged** docs. Because missing docs are dropped, the condensed top-10 **may reach past original rank 10** to fill up to 10 judged docs. Let `g_1..g_m` be the condensed-top-10 gains in condensed rank order, `m = n_scored`, all judged.

> **CRUCIAL — judged-irrelevant vs missing:** a **judged-irrelevant** doc (`gain == 0.0`, present in qrels) is **KEPT** in the condensed list — it counts toward `n_scored`, contributes `2^0 − 1 = 0` to DCG, and is not relevant. Only a doc with **no qrel entry** is `MISSING` and skipped. "Labeled irrelevant" (gain 0.0) and "no judgement" are distinct.

Two per-query counts are recorded:
- **`n_scored`** = size of the condensed top-10 (the judged docs the metrics were computed over, `<= 10`) — "total number this was calculated from".
- **`n_missing`** = number of `MISSING` docs skipped while scanning the ranked list to collect that condensed top-10 (count of `NaN`-gain docs in the scanned prefix: from rank 1 up to and including the 10th judged doc, or the whole returned list if fewer than 10 judged docs exist) — "number where the judgement was missing".

- **avg_relevance** (per query): mean graded gain over the condensed top-10:
  `avg_relevance = (1/m) · Σ_{i=1..m} g_i`. **`NaN` if `m == 0`.**

- **ndcg@10** (graded gains):
  `DCG@10 = Σ_{i=1..m} (2^{g_i} − 1) / log2(i+1)` using **condensed positions** `1..m`.
  Let `g_(1) >= g_(2) >= …` be this query's **judged** gains sorted descending (the ideal ordering, over **all** the query's qrels — unaffected by skipping). Then
  `IDCG@10 = Σ_{i=1..min(10, #judged)} (2^{g_(i)} − 1) / log2(i+1)`.
  IDCG is explicitly **truncated to the top 10** of the ideal ordering (not summed over all judged gains), so queries with more than 10 relevant docs are not deflated.
  `nDCG@10 = DCG@10 / IDCG@10`, defined `0` if `IDCG@10 = 0`. **`NaN` if `m == 0`.**

- **Binary relevance threshold:** a doc is **relevant** iff `gain >= 0.5` (`Partial` or `Exact`). (Threshold tracks the grade set: with WANDS grades `{0, 0.5, 1}`, `0.5` keeps "Partial or Exact" relevant, as before the regrade from `{0,1,2}`.)

- **precision@10:** `|relevant ∩ condensed-top-10| / m` — the **denominator is `m = n_scored`, NOT 10**. **`NaN` if `m == 0`.**

- **recall@10:** `|relevant ∩ condensed-top-10| / R`, where `R = #relevant judged docs for that query` over all of `label.csv` (relevant iff `gain >= 0.5`; `R` uses qrels directly and is unaffected by skipping). If `R = 0`, recall is **`NaN`**.

**Per-query NaN summary (each metric independent):** `avg_relevance`/`ndcg@10`/`precision@10` are `NaN` when `n_scored == 0`; `recall@10` is `NaN` when `R == 0`. **Any** of the four metrics may be `NaN` for a given query. A `NaN` metric excludes that query from that metric's aggregation and deltas (§8.1). On disk any `NaN` metric cell is written as an **empty field** (§9) — the comparator never re-parses the CSV; it excludes by the in-memory `NaN` (§8.1), so the empty-vs-NaN distinction is a pure serialization choice with no decision impact. The per-query `n_scored` and `n_missing` are recorded (§9).

> **Missing-judgement policy:** WANDS qrels are pooled. A **missing** judgement (no qrel entry) is **NOT treated as irrelevant** — it is **skipped** via **condensed-list evaluation** (Sakai): only **judged-irrelevant** docs (`gain == 0.0`, actually labeled) count as a `0`. Per-query `n_scored` (judged docs scored) and `n_missing` (missing docs skipped) are recorded in the metrics CSV (§9) so the missing-judgement ratio can be inspected and revisited (§13).

---

## 8. Single Execution Path (DRY) & Statistics

### 8.0 One runner, config-only differences
```python
class ExperimentRunner:
    def run(self, cfg: ResolvedConfig) -> None:
        dataset = load_dataset(cfg.dataset)
        writer = make_index_writer(cfg.indexer)      # IndexWriter ingest seam (§3.3), lazily built
        embedders = make_embedders(cfg.services)     # {name: Embedder connector} (§3.4), lazily built
        # eval:run does NOT (re)index — it REQUIRES an index already built by eval:index. Verify it
        # exists and holds the WHOLE corpus (doc count == dataset), else raise IndexNotReadyError so a
        # missing/partial/stale index never silently skews the metrics. mapping() is query-only:
        # the leaf searchers' field names, with NO dim probe and NO (re)indexing.
        mapping = Indexer(writer, list(embedders.values())).mapping(dataset)
        indexed, expected = writer.doc_count(), sum(1 for _ in dataset.documents())
        if indexed is None or indexed != expected:   # index absent, or partially built
            raise IndexNotReadyError(...)             # -> (re)build it with eval:index first
        rerankers = make_rerankers(cfg.services)     # {name: RerankClient connector} (§3.4/§5.4)
        searchers = make_searchers(cfg.indexer, cfg.services, mapping, embedders=embedders)   # ES build_searchers -> {name: Searcher}
        reranker_objs = make_rerankers_bound(cfg.indexer, cfg.services, mapping, rerank_clients=rerankers)  # ES build_rerankers -> {name: Reranker}
        queries = list(dataset.queries())            # frozen, shared query set
        qrels   = QrelIndex(dataset.qrels())

        # The pipelines are exactly what the config declares — baseline first, then the named
        # variants in config order (§10). No expansion, no sweep, no selection phase.
        per_query: dict[str, dict[str, Metrics]] = {}

        def run_one(pcfg: PipelineCfg) -> None:
            if pcfg.reranker:                        # R0: the W <= top_n cap only — no endpoint registration
                top_n = cfg.services.reranker(pcfg.reranker).settings["top_n"]   # a plain settings key (§5.4)
                assert pcfg.rerank_window_size <= top_n   # W <= top_n (§5.4)
            pipeline = build_pipeline(pcfg, searchers, reranker_objs)   # a SearchPipeline graph (§4)
            # Batch the frozen query set through one pipeline.bulk_search — retrieval leaves batch
            # via _msearch (§5.3) instead of one round trip per query; result[i] aligns to queries[i].
            query_texts = [q.text for q in queries]
            ranked = pipeline.bulk_search(query_texts, top_k=cfg.top_k)
            results = [
                RankedResult(q.query_id, docs) for q, docs in zip(queries, ranked)
            ]
            write_result_csv(pcfg, results, cfg.timestamp)
            metrics = Evaluator(qrels).score_run(results)   # per-query vectors
            write_metrics_csv(pcfg, metrics, cfg.timestamp)
            per_query[pcfg.id] = metrics

        for pcfg in cfg.pipelines():                 # baseline first, then variants
            run_one(pcfg)

        # Comparator pass — ONE family-wide call so the FDR correction (§8.3) is applied across the run.
        # Collect the baseline's per-query metric maps and ALL variant pipelines' maps, call compare()
        # ONCE (it pairs each variant vs the baseline, per metric, computes the RAW per-test
        # significance, and FDR-corrects the whole family of (variant, metric) tests together — each
        # returned ComparisonResult carries both the raw and the FDR-adjusted significance), then split
        # the returned rows by variant and write one comparison_bm25_{variant}_{ts}.csv each.
        # Metrics.as_dict() supplies the plain {metric: value} maps the comparator consumes
        # (§11 import rule: stats sees maps, not Metrics).
        baseline_maps = {q: m.as_dict() for q, m in per_query[cfg.baseline_id].items()}
        variant_maps = {
            vid: {q: m.as_dict() for q, m in metrics.items()}
            for vid, metrics in per_query.items() if vid != cfg.baseline_id
        }
        rows = Comparator(cfg.stats).compare(baseline_maps, variant_maps)   # family-wide FDR inside
        for vid in variant_maps:
            vrows = [r for r in rows if r.variant == vid]
            write_comparison_csv(cfg.baseline_id, vid, vrows, cfg.timestamp)
        write_run_config(cfg)
```
Every pipeline — baseline included — traverses the **identical** `run_one` code path; only the `PipelineCfg` differs. This is the DRY guarantee, verifiable by inspection: the runner is a flat loop over the explicit config pipelines with no expansion or selection phase.

**Build vs. run are separate steps.** `eval:index` (`scripts/index.py`) builds/populates the index via `Indexer.build` (embed the corpus → `dense_vector`). `eval:run` **does not index** — its prelude only *verifies* a pre-built index (the doc-count check above) and then queries it. So building is done once; re-running the eval reuses the index and makes no document-embedding calls (only per-query embeddings for vector retrieval). Changing the dataset or an embedder means rebuilding with `eval:index` before the next `eval:run` — otherwise the count check passes on a stale index (a known limitation: counts match but content changed). `IndexNotReadyError` (missing or partial index) exits `eval:run` non-zero with a message pointing at `eval:index`.

### 8.1 Pairing, point estimate, and empty/degenerate paired sets
Comparisons are **paired by `query_id`** between a variant and the baseline (`bm25`). For metric `m` and query `q`: `δ_q = m_variant(q) − m_baseline(q)`.

- The paired set is the queries present for **both** runs (always identical — the same frozen `queries` list drives every run, §8.0).
- **Per-metric `NaN` exclusion (general rule).** For **each** metric independently, the paired set is further restricted to queries whose in-memory `Metrics` value **for that metric** is **not `NaN`** in **either** run. Because any metric can now be `NaN` (`avg_relevance`/`ndcg@10`/`precision@10` when `n_scored == 0`; `recall@10` when `R == 0`, §7), this replaces the old recall-only restriction — recall is simply the `R == 0` case of the general rule. The `NaN` mask is identical across variants by construction (`R` and `n_scored` come from the shared qrels/query set, §7). The comparator detects exclusions by the in-memory `NaN`, never by re-reading the CSV, so per-metric deltas remain comparable and aligned with the on-disk empty cell.
- **delta** = mean over the (possibly restricted) paired set of `δ_q`.

**Degenerate paired sets (dataset-general, defined for *every* metric).** Because the harness sells dataset-agnosticism (§1.4(3)), a swapped-in dataset can produce a paired set that is empty or all-zero for *any* metric — not just recall. The comparator handles two degenerate cases uniformly, before any bootstrap/test call, mirroring the all-zero short-circuit of §8.2:

| Case | Trigger | Comparator output for that `(variant, metric)` row |
|------|---------|----------------------------------------------------|
| **Empty paired set** | 0 paired queries (the metric is `NaN` on every query for a swapped dataset — e.g. recall with `R=0` everywhere, or `avg_relevance`/`ndcg`/`precision` with `n_scored=0` everywhere) | `delta` = empty, `delta_ci_lo`/`delta_ci_high` = empty, `p_value = 1.0`, `significant_raw = false`, `p_value_adjusted = 1.0`, `significant = false`, and a recorded note `note=empty_paired_set`; **excluded from the FDR family size `m`** |
| **All-zero deltas** | ≥1 paired query but every `δ_q == 0` | `delta = 0.0`, `delta_ci_lo = delta_ci_high = 0.0`, `p_value = 1.0`, `significant_raw = false`, `p_value_adjusted = 1.0`, `significant = false`, note `note=all_zero_delta` (§8.2); **excluded from the FDR family size `m`** |

In both cases the comparator never calls scipy/the bootstrap, so `mean of empty set` and a bootstrap over zero indices are never evaluated. The note is recorded in `run_config_*.json` per affected `(variant, metric)`. For WANDS the empty case does not arise, but the behavior is defined so a different dataset cannot produce an undefined metric or crash.

### 8.2 Effect-size CI & p-value (seeded, reproducible)
The CI and the significance decision are deliberately kept in **distinct, clearly-labeled roles**: the CI is per-comparison effect-size context, and `significant` is the **FDR-controlled decision** (§8.3). They are **not** two gates and **may disagree** — see §8.3 for why that is correct under a step-up FDR procedure.

- **delta_ci_lo / delta_ci_high** = **per-comparison percentile bootstrap CI at the fixed unadjusted level** (`2.5` / `97.5` percentiles, i.e. a nominal 95% interval): resample the **paired query indices** with replacement `B = 10,000` times using a seeded `numpy.random.default_rng(seed)`, recompute mean `δ` each time, and take the 2.5 / 97.5 percentiles. Resampling **query indices** (not metrics independently) preserves pairing. This interval is reported **purely as effect-size / uncertainty context for a single comparison**; it is **not** multiplicity-adjusted and is **not** a significance gate. The level (2.5/97.5) is fixed and recorded in run metadata (§9.1).
- **p_value** = **Wilcoxon signed-rank test** (two-sided) on the paired `δ_q` (primary; no normality assumption), with explicitly pinned handling of zeros and ties (heavy in nDCG/recall/precision, where many baseline/variant rankings coincide):
  - `zero_method="wilcox"` (drop zero-deltas before ranking) and `correction=True` (continuity correction), both recorded in run metadata so the p-value is reproducible regardless of scipy default drift.
  - **All-zero deltas** (every `δ_q == 0`, the test is undefined): the harness short-circuits to `p_value = 1.0` and `significant = false` rather than calling scipy (see §8.1 table).
  - Because sparse-delta metrics stress Wilcoxon's zero/tie handling, the **seeded paired-permutation test** (sign-flip permutation on the paired deltas, same seeded `rng`) is offered as a configurable primary; when selected it uses the same all-zero short-circuit.

### 8.3 Significance & multiple comparisons (single, coherent FDR regime)
- Family = all `(variant × metric)` tests **within one run**. Control the **False Discovery Rate (FDR)** — the expected proportion of false discoveries *among the rejections* — with **Benjamini-Hochberg (BH)** at FDR level `q = α = 0.05` by default, applied to the **raw** p-values. Degenerate rows (`empty_paired_set` / `all_zero_delta`, §8.1) are assigned `significant_raw=false`, `p_value=1.0`, `p_value_adjusted=1.0`, `significant=false` and are **NOT** counted in the family size `m` — `m` is the number of *real* tests (those that produced a p-value from scipy/the permutation test).
- **`alpha` is both thresholds.** The single configured `alpha = 0.05` serves as **both** the raw per-test threshold (for `significant_raw`) **and** the FDR target level `q` (for the BH/BY step-up). There is one number, used in two clearly-labeled roles.
- **Decision rule (the FDR gate).** Order the family's raw p-values ascending `p_(1) <= … <= p_(m)`. The BH step-up finds the **largest `k`** with `p_(k) <= (k/m)·α` and **rejects all hypotheses with rank `<= k`**. Equivalently — and this is what the harness emits — BH defines **adjusted p-values (q-values)** `q_(k) = min_{j >= k} ( m·p_(j) / j )`, monotone non-decreasing in rank and clamped to `<= 1`; `significant = (p_value_adjusted <= α)` reproduces the step-up rejection set exactly. Unlike Holm (which defines no per-test adjusted value), **BH q-values are well-defined and ARE reported** in the comparison CSV.
- **Two significance flags are emitted, in distinct roles.** `significant_raw = (p_value <= α)` is the **uncorrected per-test decision**, computed independently of the family (it does not depend on `m` or the other tests). `significant = (p_value_adjusted <= α)` is the **FDR decision** over the family. Both are written to the CSV so a reader can see the uncorrected discovery and its post-correction fate; because BH is more powerful than Holm/FWER, a test that is `significant_raw` may or may not survive FDR correction, and both outcomes are reported honestly.
- **Why FDR, not FWER/Holm, is the right regime here.** This is an **exploratory** analysis: the goal is **discovering the best pipeline** among many correlated retrieval configurations and inspecting hybrid-retrieval failure modes — not a confirmatory or clinical test. FDR (control the expected *proportion* of false discoveries among rejections) beats FWER/Holm (control the probability of *any* false rejection) for three reasons:
  1. **Holm/FWER is overly strict for many correlated hypotheses** — it sacrifices power and likely **hides true metric improvements**, causing us to miss the best retrieval configuration.
  2. **The tests are highly correlated** (e.g. `hybrid+rerank` is inherently correlated with `hybrid` without rerank; two hybrid pipelines at different RRF `k` are correlated). **Benjamini-Hochberg controls FDR under independence AND positive regression dependence (PRDS)**, which positively-correlated retrieval configs plausibly satisfy — it handles correlation with far more power than Bonferroni-style FWER.
  3. **The cost of a false positive here is low and asymmetric:** it means provisionally selecting a slightly-suboptimal tuning parameter, which is caught in later A/B testing — not a life-or-death error the Holm-Bonferroni regime is designed for. Missing a real improvement (a false negative) is the more costly error for discovery, so we prefer FDR's power.
- **Configurable arbitrary-dependence option.** The default is **BH** (`correction: bh`). **Benjamini-Yekutieli (BY)** (`correction: by`) is the conservative alternative that is **valid under arbitrary dependence** (it costs a `log`-factor of power via the `c(m) = Σ_{i=1..m} 1/i` scaling); offer it via config for when PRDS is doubted. Any `correction` other than `bh`/`by` raises `NotImplementedError`.
- **The CI is *not* a second gate, by design — and may disagree with `significant`.** The reported CI (§8.2) is a per-comparison, unadjusted 2.5/97.5 bootstrap interval used only for effect-size context. Under a **step-up FDR procedure** there is **no simple per-test alpha** that yields a matching interval: a test's rejection depends on the *whole ordered family*, so its unadjusted CI can exclude 0 while it is not FDR-significant (or vice versa) at no shared confidence level tied to the family decision. **Therefore the CI and the `significant` flag are not guaranteed to agree, and disagreement is normal and expected.** We do not attempt to reconcile them; a matching FDR-adjusted-interval regime remains **deferred (§13)**. The CSV documents this: `p_value` is raw, `significant_raw` is the raw per-test decision, `p_value_adjusted` is the BH q-value, `significant` is the FDR decision, and the CI is unadjusted effect-size context.
- The **raw** (uncorrected) `p_value` **and** the FDR-adjusted `p_value_adjusted` (q-value) are both written to the CSV, alongside `significant_raw` and `significant`. Correction method (`bh`/`by`), family size `m`, and `α` (as the raw threshold AND the FDR level `q`) are recorded in run metadata.

---

## 9. Output Artifacts, Naming, Reproducibility

`{timestamp}` = UTC `YYYYMMDDTHHMMSSZ` of run start (single value for the whole run). `{variant}` = the pipeline's name from config (its `pipelines.variants` map key, e.g. `hybrid_e5_k60`). `{baseline}` = the baseline pipeline's id (`bm25`/`baseline`). All CSVs UTF-8, comma-separated, header present. **Field names and order are fixed:**

**`result_{variant}_{timestamp}.csv`**
```
query_id,product_id,score,position
```
One row per returned doc; `position` 1-based ascending; ≤ top_k rows per query.

**`metrics_{variant}_{timestamp}.csv`**
```
query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_scored,n_missing
```
One row per query. `n_scored` and `n_missing` are **non-negative integers, ALWAYS present** (never empty): the condensed-list counts of §7 (`n_scored` = judged docs scored, `n_missing` = missing docs skipped). Any of the **four metric** cells (`avg_relevance`, `ndcg@10`, `recall@10`, `precision@10`) is written as an **empty field** (two adjacent commas, no quoting) when its in-memory `Metrics` value is `NaN` (§7): `avg_relevance`/`ndcg@10`/`precision@10` empty when `n_scored == 0`, `recall@10` empty when `R == 0`. This empty↔`NaN` mapping is fixed so a reader never guesses; consumers must treat an empty metric cell as "excluded", and the comparator does this from the in-memory `NaN`, not by re-parsing this file (§8.1).

**`comparison_{baseline}_{variant}_{timestamp}.csv`**
```
variant,metric,delta,delta_ci_lo,delta_ci_high,p_value,significant_raw,p_value_adjusted,significant
```
One row per metric ∈ {`avg_relevance`,`ndcg@10`,`recall@10`,`precision@10`}; `significant` ∈ {`true`,`false`} and `significant_raw` ∈ {`true`,`false`}. `delta_ci_lo/high` are the **per-comparison unadjusted 2.5/97.5 bootstrap interval (effect-size context only, §8.2)**; `p_value` is the **raw** (uncorrected) Wilcoxon (or permutation) p; `significant_raw` is the **uncorrected per-test decision** (`p_value <= α`); `p_value_adjusted` is the **BH (or BY) FDR-adjusted p-value (q-value)** over the family; `significant` is the **FDR decision** (`p_value_adjusted <= α`, §8.3). The CI is in a different role from the significance flags and **may disagree** with them (§8.3). For a degenerate paired set, `delta` and the CI cells are written **empty** (empty paired set) or `0.0` (all-zero deltas) per the §8.1 table, with `p_value=1.0`, `significant_raw=false`, `p_value_adjusted=1.0`, `significant=false`.

### 9.1 Reproducibility
- **Config capture:** the fully-resolved config (the resolved **services** registry — embedders/rerankers/searchers by name — and the resolved **pipelines**: the baseline plus every named variant with its retrievers/fuser/reranker/window; bootstrap B, the fixed CI level 2.5/97.5, `α` — recorded as **both** the raw per-test threshold **and** the FDR level `q`, family size m, correction method (`bh` or `by`), test + its zero/tie params, any degenerate-paired-set notes, dataset version, ES + endpoint versions, cutoff, seed) is serialized to `run_config_{timestamp}.json` alongside the CSVs. Under BH/BY the harness records/emits FDR-adjusted p-values (q-values) per test in the comparison CSV (§9), so — unlike Holm — the adjusted significance is fully materialized.
- **Seeds:** one master seed feeds the bootstrap and any permutation test; recorded in config. Given the seed, stats are deterministic. There is no data-dependent pipeline selection, so the set of runs depends only on the config file.
- **Determinism caveats:** ES scoring ties and approximate-kNN introduce nondeterminism. Mitigations: stable tie-break on `doc_id` in each `Searcher.search` (score desc, doc_id asc, §9.1); idempotent indexing (`_id = product_id`); recorded ES/endpoint versions.

---

## 10. Explicit Config

A single YAML/JSON config declares everything as **explicit, named building blocks** — no axes, no
expander, no sweep. The user reads the config top to bottom and sees exactly which pipelines run.
The structure is: `dataset` / `services` (named embedders, rerankers, searchers) / `indexer` /
`pipelines` (one `baseline` + a map of named `variants`) / `stats` / `cutoff` / `top_k`.

```yaml
dataset:
  name: wands
  path: ./dataset/wands
services:                       # named, typed, reusable building blocks
  # Embedders/rerankers are PROVIDER CONNECTORS (benchmark/providers.py): the harness calls Cohere /
  # Voyage / OpenAI directly — ES runs no inference (§1.1/§3.4). `provider` selects the connector;
  # `model_id`/`api_key`/`rate_limit`/`dims`(optional) live in `settings`. OpenAI has NO reranker.
  - embedder: { name: cohere, provider: cohere, settings: { api_key: ${COHERE_KEY}, model_id: embed-english-v3.0 } }
  - reranker: { name: co-rr,  provider: cohere, settings: { api_key: ${COHERE_KEY}, model_id: rerank-v3.5, top_n: 100 } }
  # - embedder: { name: voyage, provider: voyage, settings: { api_key: ${VOYAGE_KEY}, model_id: voyage-3.5 } }
  # - embedder: { name: openai, provider: openai, settings: { api_key: ${OPENAI_KEY}, model_id: text-embedding-3-small } }
  - searcher: { name: bm25,        provider: elasticsearch, kind: lexical }
  - searcher: { name: semantic_co, provider: elasticsearch, kind: vector, embedder: cohere }
# ONE ES index for everything: a single search_text (BM25) `text` field + one `dense_vector` field per
# embedder referenced by a vector searcher above (§5.2). semantic_co is a FIELD in this one index, not
# a separate index; the harness embeds each doc's search_text with the connector and stores the vector.
indexer:
  provider: elasticsearch
  index: wands_bench
  settings: { url: ${ES_URL} }
pipelines:
  baseline:                      # the reference every variant is compared against
    retriever: bm25
  variants:                      # each is one explicit run; the map key is its id
    semantic_co:   { retriever: semantic_co }
    hybrid_co_k60:
      retrievers: [bm25, semantic_co]
      fuser: { type: rrf, rank_constant: 60, window: 100 }
    bm25_rerank:
      retriever: bm25
      reranker: co-rr
      rerank_window_size: 100
    hybrid_co_rerank:
      retrievers: [bm25, semantic_co]
      fuser: { type: rrf, rank_constant: 60, window: 100 }
      reranker: co-rr
      rerank_window_size: 100
stats:
  test: wilcoxon
  correction: bh                 # Benjamini-Hochberg FDR (§8.3, default); by = Benjamini-Yekutieli
  alpha: 0.05                    # BOTH the raw per-test threshold AND the FDR target level q
  bootstrap_B: 10000
  ci_level: 0.95                 # UNADJUSTED per-comparison effect-size CI (§8.2); NOT a gate
  seed: 1234
cutoff: 10                       # metrics @10
top_k: 100                       # results retrieved per query
```

**Services (`${VAR}` env placeholders resolved at load, secrets never in the file):**
- **`embedder`** — a named embedding **provider connector** (§3.4). `provider` selects the connector
  (`cohere` | `voyage` | `openai`); `settings` carries connector knobs (`model_id`, `api_key`,
  optional `rate_limit.requests_per_minute`, `batch_size`, `dims`, …). The runner instantiates it
  lazily via `make_embedders`; the harness embeds the corpus into a `dense_vector` field (§3.5) — no
  ES `_inference` endpoint, no `embedding_type`. An unknown `provider` raises at config load.
- **`reranker`** — a named rerank **provider connector** (§3.4; `cohere` | `voyage` — **OpenAI has no
  reranker**, rejected at load). `top_n` (the rank-window cap) is a plain `settings` key — the number
  of candidates the provider scores per request; the runner reads `settings["top_n"]` for the
  `W <= top_n` assertion at R0 (§5.3/§8.0).
- **`searcher`** — a named leaf retriever. `kind` is `lexical` or `vector`; a `vector` searcher
  references an `embedder` by name.

**`indexer`** — a **single** block naming ONE ES index (`indexer.index`) and how to reach it
(`provider`, `settings.url`). With ES this one index holds everything (§5.2): a single `search_text`
BM25 field **plus one `dense_vector` field per embedder that a `vector` searcher references** — the
indexer builds a field for each `Embedder` the runner passes to the `indexing.Indexer` constructor (one per configured
`embedder` service), and reaches ES from this block. It does **not** index one-per-embedder. (A pure
vector store would need a per-store indexing model instead — deferred, §12/§13.)

**Pipeline field rules (validated at load; a violation raises a clear `ConfigError`, exhaustive — no silent default):**
- Exactly **one of `retriever`** (a single searcher name) **XOR `retrievers`** (a list of 2+ searcher names).
- `retrievers` (2+) **requires a `fuser`**; a `fuser` is only allowed with `retrievers`.
  `fuser: { type: rrf, rank_constant: <int>, window: <int> }` — `type` is exhaustive (only `rrf`
  today; anything else raises).
- `reranker` (a reranker service name) **requires `rerank_window_size`**, and vice-versa.
- Every referenced service name must **exist and be the right type**; a vector searcher must
  reference an existing embedder.
- `pipelines.baseline` is the reference; `pipelines.variants` is a map of `id -> pipeline spec`. The
  run ids are the map keys (baseline id = `baseline`, configurable via `baseline_id`). A variant id
  that duplicates the baseline id is an error.

Each named pipeline → a `SearchPipeline` object graph (§4, `build_pipeline`) → one `result_*` +
`metrics_*` + `comparison_bm25_*` triple. Each reranker's `top_n` must be `>= rerank_window_size`
(asserted at R0, §8.0 / §5.3). **There is no matrix expansion and no k-sweep** — the pipelines run
are exactly those listed. An optional sweep/expansion helper that would *generate* many pipelines
from axes is a possible future convenience, deliberately omitted here for config legibility (§13).

---

## 11. Module / Package Layout

```
benchmark/
  common/              # (g) shared bottom layer, depends on nothing
    models.py          #   Query, Document, Qrel, ScoredDoc, RankedResult, FieldSchema, IndexMapping, enums
    protocols.py       #   Searcher/Fuser/Reranker + Dataset ABCs; Embedder, RerankClient, IndexWriter Protocols
    ranking.py         #   fuse_rrf_local + rerank_local (pure windowed ranking primitives)
    logging_setup.py   #   console + file logging (logs/run_{timestamp}.log); use instead of print()
  providers/           # (f) concrete adapters, depend ONLY on common
    inference.py       #   OpenAI/Cohere/Voyage Embedders + Cohere/Voyage RerankClients (stdlib HTTP)
    elasticsearch.py   #   LexicalSearcher, VectorSearch, ESReranker, ESIndexWriter, build_searchers/build_rerankers
  embedding.py         # (c) make_embedder + EMBEDDER_PROVIDERS (dispatch provider -> providers.inference)
  reranking.py         # (d) make_reranker + RERANKER_PROVIDERS (dispatch provider -> providers.inference)
  indexing.py          # (a) Indexer (build orchestration) + embed-at-ingest streaming
  search.py            # (b) RRFFuser, HybridSearch, SearchPipeline (the composers)
  evaluation/          # (e) scoring + statistics
    metrics.py         #   Evaluator, Metrics, QrelIndex
    stats.py           #   Comparator, StatsCfg, ComparisonResult (bootstrap CI, Wilcoxon/permutation, FDR/BH-BY)
  datasets/wands.py    #   WandsDataset (implements Dataset; label->gain; search_text concat)
  config.py            #   config value types + YAML load/resolve + build_pipeline + lazy dotted adapter factories
  runner.py            #   ExperimentRunner (the single execution path, §8.0)
  io_csv.py            #   write_result_csv / write_metrics_csv / write_comparison_csv / write_run_config
docs/experiment.md
dataset/wands/         # query.csv, product.csv, label.csv (gitignored)
```

**Layers.** `a–g` form a strict acyclic engine: `common` (g) ← `providers` (f) ← `embedding`(c)/
`reranking`(d) ← `indexing`(a)/`search`(b)/`evaluation`(e). Domain layers (a/b/e) import only
`common` abstractions at import time; they consume concrete `providers` pieces (index writer, leaf
searchers, reranker) and embedders/rerank-clients **injected** at runtime — Dependency Inversion is
what makes "indexing/search need providers" hold without a backward import edge. `config`, `runner`,
`io_csv` are the composition layer above the engine: they wire it together and own the lazy
dotted-target factories, so no engine module names an adapter.

**The clean-OOP seams.** Indexing is a backend-agnostic `indexing.Indexer(writer, embedders)` whose
`build()` discovers dims → asks the `IndexWriter` for the `IndexMapping` → `ensure_index` → streams
the corpus through the embedders → `bulk_index`. Search is composed by `build_pipeline`
(`config.py`) from plain leaf `Searcher`s + a `Reranker` that the ES adapter's `build_searchers`/
`build_rerankers` mint from the resolved `Services` + `IndexMapping`. There is no
`SearcherFactory`/`_ESSearcherFactory` and no `ElasticsearchBackend` god-object; the ES pieces are
ordinary provider classes.

**Import-graph rule (enforced by `test_import_graph.py`).** `search`, `indexing`,
`evaluation.metrics`, `evaluation.stats`, `runner`, `io_csv`, and `config` import **no**
`benchmark.providers.*` / `benchmark.datasets.*` / `benchmark.embedding` / `benchmark.reranking` at
import time; `config` imports `search` (composers) + `evaluation.stats` (`StatsCfg`) only. The lazy
factories resolve dotted targets at call time. This is success criterion §1.4(3). The
degenerate-paired-set handling (§8.1) lives entirely in `evaluation/stats.py` and is dataset-agnostic
(operates on paired delta arrays only).

---

## 12. Extension Guide

Each extension is an *adapter + config* change; pipeline, metrics, stats, runner stay untouched.

**Add a dataset:** **derive from the `Dataset` ABC** (§3.2) in `datasets/`, implement the four abstract methods (`queries`/`documents`/`qrels`/`field_schema`), set `self.name`/`self.version` in `__init__`, register the adapter in `config.py`'s `DATASET_TARGETS`, and set `dataset.name` in the config. The adapter owns three things: **file parsing** (TSV/parquet/JSONL — per-adapter, nothing shared), **label→gain** (a string mapping via `map_label`, or a numeric passthrough), and **field roles** (its own `field_schema`). Reuse the ABC's `build_search_text(row_fields, schema)` to build the canonical `search_text` — do not re-implement the §5.1 concatenation. The stats layer already defines empty/all-zero paired-set behavior for every metric (§8.1), so a new dataset cannot produce an undefined metric or crash even if some metric is degenerate on every query.

The abstract interface is deliberately format-agnostic; the intended next targets confirm it:

- **WANDS** (shipped): TSV files; string labels `Exact`/`Partial`/`Irrelevant` → `1.0`/`0.5`/`0.0` via `map_label(label, {"Exact":1.0,"Partial":0.5,"Irrelevant":0.0})`.
- **Amazon ESCI:** parquet/CSV; string labels `E`/`S`/`C`/`I` (Exact/Substitute/Complement/Irrelevant) → a per-dataset gain mapping via `map_label` (e.g. `{"E":1.0,"S":0.5,"C":0.25,"I":0.0}` — the exact grades are a modeling choice); product `title`/`description`/`bullets`/`brand` as text fields in its own `field_schema` (title/description/bullets `SEMANTIC_SOURCE` or `BM25`, brand likely `STORED`).
- **BEIR:** JSONL `corpus`/`queries` + a **numeric** qrels TSV → **no `map_label`**; set `gain = float(rel)` directly. Generic `title`+`text` fields, both marked `SEMANTIC_SOURCE` in `field_schema` so `build_search_text` concatenates them into `search_text`.

None of these touch the pipeline/indexer/evaluator/stats — each is an adapter + config change, and `build_search_text` is reused across all three.

**Add a backend (Vespa, OpenSearch, Qdrant, FAISS, …):** in `providers/`, implement (a) an
`IndexWriter` (§3.3) — `ensure_index`, `bulk_index`, plus the backend-safe `sem_field_name` /
`create_mapping` and the `embed_batch_size` buffering knob — that writes the harness-computed
vectors; and (b) the leaf builders `build_searchers` / `build_rerankers` (§3.3): free functions that
turn the resolved `Services` + `IndexMapping` into `{name: Searcher}` / `{name: Reranker}` maps,
doing the exhaustive `kind` dispatch to concrete leaf `Searcher`s (lexical + vector) and a `Reranker`.
The domain `indexing.Indexer` is **backend-agnostic and shared** — you do **not** write a per-backend
`Indexer` or a factory; there is no `SearcherFactory`/`_ESSearcherFactory` and no "backend"
god-object. Register the three in `config.py`'s target tables (`INDEX_WRITER_TARGETS`,
`SEARCHER_BUILDER_TARGETS`, `RERANKER_BUILDER_TARGETS`) and set `indexer.provider`. `HybridSearch` +
`RRFFuser` + `SearchPipeline` (client-side, §3.6/§3.7) then compose those leaves into
`hybrid`/`*_rerank` with no new code, reproducing ES's `rank_window_size` semantics; the `Reranker`
uses the `rerank_local` helper (§3.7) over a `RerankClient`.

> **Multiple vector stores (e.g. Qdrant) — a per-store indexing model, deferred (§13).** The single-`indexer`-block, one-ES-index model (§5.2) works because ES is one store holding BM25 + every `dense_vector` field together. A **pure vector store** (Qdrant, FAISS) has **no BM25** and needs **one collection per embedding** (different embedders have different vector dims), so that model does not carry over. The good news: the **client-side `Searcher`/`Fuser` composition already fuses across stores** (each leaf `Searcher` just returns a ranked list), so mixing an ES lexical leaf with a Qdrant vector leaf is a clean future extension — what it needs is a **per-store indexing model** (each searcher/embedder carrying its own store/collection settings) rather than the single `indexer` block that indexes everything at once (true only for ES). There is **no Qdrant adapter today (YAGNI)**; this is deferred (§13).

**Add an embedder:** add an `embedder` service (`provider` ∈ Cohere / Voyage / OpenAI — all three connectors are shipped, §3.4) and a `vector` `searcher` that references it → the indexer adds one `dense_vector` field and the harness embeds the corpus into it at reindex. No code change unless the provider is new — then add one `Embedder` connector in `benchmark.providers.inference` and its dispatch entry in `benchmark.embedding` (`EMBEDDER_PROVIDERS`). Reference the new searcher from whatever named pipelines you want it in.

**Add a reranker:** add a `reranker` service (`provider` ∈ Cohere / Voyage — OpenAI has no reranker, §3.4; with `settings.top_n >= rerank_window_size`) → reference it from the pipelines that should rerank. No code change unless the provider is new — then add one `RerankClient` connector in `benchmark.providers.inference` and its dispatch entry in `benchmark.reranking` (`RERANKER_PROVIDERS`).

---

## 13. Open Questions / Deferred
- **Optional sweep / expansion helper (deliberately omitted):** pipelines are **fully explicit** named config entries (§10) — chosen for legibility, so the config is readable at a glance and the runner is a flat loop with no data-dependent selection. A convenience helper that *generates* many pipelines from axes (e.g. embedding_models × RRF-k) could be added later as pure config sugar that emits explicit `PipelineCfg`s before the run; it is intentionally not shipped, to keep the config transparent and reproducible. If added, any data-dependent auto-selection (e.g. "best k per model" on the eval set) would reintroduce selection-on-the-evaluation-set bias and must be treated as exploratory.
- **Matching FDR-adjusted interval regime:** **BH-FDR is now the DEFAULT decision** (§8.3), with an unadjusted descriptive CI (§8.2) that may disagree with the flag. What remains **deferred** is a **matching FDR-adjusted confidence-interval regime** (or a max-statistic simultaneous confidence band) so the reported interval and the FDR decision coincide *by construction* rather than living in separate roles — and, possibly, **switching BY to the default** if PRDS proves doubtful for these correlated retrieval configs.
- **Server-side fusion / rerank as a performance optimization (deferred):** fusion and rerank run **client-side** (§3.6/§3.7) — chosen for simplicity and generality (one composite model, any backend that returns ranked leaf lists composes). ES *can* fuse (`rrf`) and rerank (`text_similarity_reranker`) server-side in a single round trip, which would cut per-query latency; adopting that as an optional fast path (an alternate `Searcher` that emits the nested retriever tree, selected by config) is deferred as a performance optimization. It is out of scope for the v1 relevance-quality success criteria.
- **Multiple vector stores / per-store indexing model (deferred, no adapter today):** the single-`indexer`-block model indexes everything into ONE ES index (§5.2) because ES holds BM25 + every `dense_vector` field together. A **pure vector store** (Qdrant/FAISS) has no BM25 and needs **one collection per embedding** (differing vector dims), so the single-index model does not carry over. Since the client-side `Searcher`/`Fuser` composition **already fuses across stores** (§12), the clean extension is a **per-store indexing model** — each searcher/embedder carrying its own store/collection settings — rather than one `indexer` block. Deferred; **no Qdrant adapter today (YAGNI)**, no consumer.
- **Lexical-less backend (no `bm25` graph):** BM25 is realized as a concrete `LexicalSearcher` (§3.3), so every in-scope backend provides it. If a lexical-less backend (e.g. a pure vector index like FAISS/Qdrant) is added, simply omit the lexical `searcher` service and any pipeline that references it (a config concern — there is no backend capability flag). Deferred until such a backend exists (no consumer today).
- **Provider rate limits & cost (operational):** embeddings and reranking now hit an external provider (§5.4), so throughput is bounded by the provider's rate limits and each call costs money. The connector's `RateLimiter` (`settings.rate_limit.requests_per_minute`) + retry/backoff on 429 handle request spacing; sizing a run (batch sizes, RPM, budget) and a cost/latency-per-provider table are operational concerns, not correctness. (The old ES ML-node memory ceiling for a co-deployed embedder + reranker no longer applies — ES deploys no model, §5.2.)
- Latency/cost per inference endpoint as a secondary table (out of scope for v1 success criteria).
- Pooling-depth / missing-judgement sensitivity analysis (current default: **missing judgements are skipped via condensed-list evaluation**, §7 — NOT scored as gain 0; only judged-irrelevant docs count as a 0).
- **Missing-judgement ratio → LLM-as-a-judge backfill.** After the first full-pass run, inspect the aggregate missing-judgement ratio `n_missing / (n_scored + n_missing)` (summed over queries, from the §9 metrics CSVs). If judgements are missing **> 10%** of the time, run a separate measurement: compute **Cohen's kappa** agreement between the original WANDS labels and an **LLM-as-a-judge** evaluator on the judged pairs; then **backfill** the missing judgements with the LLM judge into `dataset/wands/label_augmented.csv` and re-run against the augmented labels.
- `search_text` concatenation vs per-field semantic embedding (chunking behavior of long concatenations on small-context embedding models).
- Whether the `Reranker` should score field snippets vs the whole `search_text` for long documents.
