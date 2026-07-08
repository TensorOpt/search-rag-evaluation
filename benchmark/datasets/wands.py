"""WandsDataset adapter: label->gain, search_text concat (docs/architecture.md §3.2, §5.1 + docs/methodology.md §7). Phase 8.

Adapter deriving from the ``Dataset`` ABC (§3.2): reads the WANDS ``query.csv`` / ``product.csv`` /
``label.csv`` (all **tab-separated** despite the ``.csv`` extension — product descriptions contain
commas, so a comma parser would corrupt them) and yields ``Query`` / ``Document`` / ``Qrel``. The
label->gain mapping (§7, via ``Dataset.map_label``) and the ``search_text`` concatenation (§5.1, via
``Dataset.build_search_text``) are applied here so the rest of the harness only ever sees float gains
and a single canonical text field.

Imports ``benchmark.common.models`` + ``benchmark.common.protocols`` (the ``Dataset`` ABC) + stdlib
(``csv``/``pathlib``); no backend/search (§11).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from benchmark.common.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    Qrel,
    Query,
)
from benchmark.common.protocols import Dataset

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


def _parse_numeric(raw: str | None, numeric_type: type[int] | type[float]) -> int | float | None:
    """Parse a WANDS numeric cell; ``None`` for an empty cell (the field is then omitted).

    The full WANDS ``product.csv`` formats integer counts as floats (``"15.0"``) and leaves many
    cells empty, so a bare ``int("15.0")`` fails: parse via ``float`` first, then narrow ``int``
    columns. Exhaustive on the declared type — an unsupported type raises rather than defaulting.
    """
    text = (raw or "").strip()
    if not text:
        return None  # missing numeric (empty cell) — omit the field; numerics are stored, never ranked
    number = float(text)
    if numeric_type is int:
        return int(number)
    if numeric_type is float:
        return number
    raise ValueError(f"unsupported numeric type {numeric_type!r} for a WANDS column")


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
            for name, numeric_type in _NUMERIC_COLUMNS.items():
                value = _parse_numeric(row.get(name), numeric_type)
                if value is not None:  # empty cell -> omit the field entirely (§3.2 stored numerics)
                    fields[name] = value
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

    def gain_mapping(self) -> Mapping[str, float]:
        """The WANDS label->gain map applied to ``qrels()`` (§7): Exact/Partial/Irrelevant."""
        return dict(WANDS_GAINS)
