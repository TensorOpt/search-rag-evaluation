"""WandsDataset adapter: label->gain, search_text concat (docs/experiment.md §3.2, §5.1, §7). Phase 8.

Adapter deriving from the ``Dataset`` ABC (§3.2): reads the WANDS ``query.csv`` / ``product.csv`` /
``label.csv`` (all **tab-separated** despite the ``.csv`` extension — product descriptions contain
commas, so a comma parser would corrupt them) and yields ``Query`` / ``Document`` / ``Qrel``. The
label->gain mapping (§7, via ``Dataset.map_label``) and the ``search_text`` concatenation (§5.1, via
``Dataset.build_search_text``) are applied here so the rest of the harness only ever sees float gains
and a single canonical text field.

Imports ``benchmark.models`` + ``benchmark.protocols`` (the ``Dataset`` ABC) + stdlib
(``csv``/``pathlib``); no backend/pipeline (§11).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    Qrel,
    Query,
)
from benchmark.protocols import Dataset

#: label.csv string label -> float relevance gain (§7). Exhaustive; unknown -> ValueError.
WANDS_GAINS: Mapping[str, float] = {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0}

#: product.csv text columns kept as-is in the field bag.
_TEXT_COLUMNS = (
    "product_name",
    "product_class",
    "category hierarchy",
    "product_description",
    "product_features",
)
#: product.csv numeric columns and their parsers.
_NUMERIC_COLUMNS: Mapping[str, type[int] | type[float]] = {
    "rating_count": int,
    "average_rating": float,
    "review_count": int,
}

#: WANDS field roles (§5.1). category hierarchy is a STORED facet, NOT in search_text.
_FIELD_SCHEMA = FieldSchema(
    fields=[
        FieldSpec("product_id", FieldRole.ID),
        FieldSpec("product_name", FieldRole.SEMANTIC_SOURCE),
        FieldSpec("product_description", FieldRole.SEMANTIC_SOURCE),
        FieldSpec("product_features", FieldRole.BM25),
        FieldSpec("product_class", FieldRole.BM25),
        FieldSpec("category hierarchy", FieldRole.STORED),
        FieldSpec("rating_count", FieldRole.NUMERIC),
        FieldSpec("average_rating", FieldRole.NUMERIC),
        FieldSpec("review_count", FieldRole.NUMERIC),
    ],
    search_text_field="search_text",
    rerank_field="search_text",
)


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield tab-separated rows as dicts (files are TSV despite the .csv name)."""
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


class WandsDataset(Dataset):
    """WANDS dataset adapter deriving from the ``Dataset`` ABC (§3.2)."""

    def __init__(self, dataset_cfg: Mapping[str, Any]) -> None:
        if "path" not in dataset_cfg:
            raise ValueError("wands dataset config requires a 'path'")
        self._dir = Path(dataset_cfg["path"])
        self.name = "wands"
        self.version = str(dataset_cfg.get("version", "2022.0"))

    def queries(self) -> Iterable[Query]:
        for row in _read_rows(self._dir / "query.csv"):
            yield Query(
                query_id=row["query_id"],
                text=row["query"],
                query_class=row["query_class"],
            )

    def documents(self) -> Iterator[Document]:
        """Stream one ``Document`` per product row; fields carry the computed ``search_text`` (§5.1)."""
        for row in _read_rows(self._dir / "product.csv"):
            fields: dict[str, Any] = {name: row[name] for name in _TEXT_COLUMNS}
            for name, parse in _NUMERIC_COLUMNS.items():
                fields[name] = parse(row[name])
            fields["search_text"] = self.build_search_text(row, _FIELD_SCHEMA)
            yield Document(doc_id=row["product_id"], fields=fields)

    def qrels(self) -> Iterable[Qrel]:
        for row in _read_rows(self._dir / "label.csv"):
            yield Qrel(
                query_id=row["query_id"],
                doc_id=row["product_id"],
                gain=self.map_label(row["label"], WANDS_GAINS),
            )

    def field_schema(self) -> FieldSchema:
        return _FIELD_SCHEMA
