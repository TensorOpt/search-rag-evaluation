"""Phase 8 unit tests for benchmark.datasets.wands (docs/experiment.md §3.2, §5.1, §7)."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from benchmark.datasets.wands import WANDS_GAINS, WandsDataset
from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    Qrel,
    Query,
)
from benchmark.protocols import Dataset

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "wands_sample"


@pytest.fixture
def dataset() -> WandsDataset:
    return WandsDataset({"path": str(FIXTURE)})


def test_name_and_default_version(dataset: WandsDataset) -> None:
    assert dataset.name == "wands"
    assert dataset.version == "2022.0"
    assert WandsDataset({"path": str(FIXTURE), "version": "1.0"}).version == "1.0"


def test_path_required() -> None:
    with pytest.raises(ValueError):
        WandsDataset({})


def test_queries_roundtrip(dataset: WandsDataset) -> None:
    queries = list(dataset.queries())
    assert queries[0] == Query(query_id="0", text="salon chair", query_class="Furniture")
    assert queries[1] == Query(
        query_id="1", text="solid wood platform bed", query_class="Beds"
    )
    assert len(queries) == 3


def test_documents_roundtrip_and_numeric_parsing(dataset: WandsDataset) -> None:
    docs = {d.doc_id: d for d in dataset.documents()}
    salon = docs["100"]
    assert isinstance(salon, Document)
    assert salon.fields["product_name"] == "Salon Styling Chair"
    assert salon.fields["category hierarchy"] == "Furniture / Salon"
    # numeric columns parsed to int/float; text kept str.
    assert salon.fields["rating_count"] == 12 and isinstance(salon.fields["rating_count"], int)
    assert salon.fields["review_count"] == 9 and isinstance(salon.fields["review_count"], int)
    assert salon.fields["average_rating"] == pytest.approx(4.5)
    assert isinstance(salon.fields["average_rating"], float)


def test_documents_is_a_streaming_generator(dataset: WandsDataset) -> None:
    stream = dataset.documents()
    assert isinstance(stream, types.GeneratorType)


def test_search_text_concat_with_comma_in_description(dataset: WandsDataset) -> None:
    docs = {d.doc_id: d for d in dataset.documents()}
    salon = docs["100"]
    # §5.1 order: name, description, features, class — joined by newlines.
    expected = "\n".join(
        [
            "Salon Styling Chair",
            "Adjustable, hydraulic chair, with a chrome base",  # comma survives TSV parse
            "height adjustable | chrome base",
            "Salon Chairs",
        ]
    )
    assert salon.fields["search_text"] == expected
    # The comma-containing description was not split into extra columns.
    assert salon.fields["product_description"] == "Adjustable, hydraulic chair, with a chrome base"


def test_qrels_label_to_gain(dataset: WandsDataset) -> None:
    qrels = list(dataset.qrels())
    by_pair = {(q.query_id, q.doc_id): q for q in qrels}
    assert by_pair[("0", "100")] == Qrel(query_id="0", doc_id="100", gain=1.0)  # Exact
    assert by_pair[("0", "102")].gain == pytest.approx(0.5)  # Partial
    assert by_pair[("0", "101")].gain == pytest.approx(0.0)  # Irrelevant
    assert by_pair[("1", "101")].gain == pytest.approx(1.0)
    # gains are floats.
    assert all(isinstance(q.gain, float) for q in qrels)


def test_qrels_leading_id_column_handled(dataset: WandsDataset) -> None:
    # The 'id' column is ignored; pairing is by query_id/product_id.
    assert len(list(dataset.qrels())) == 4


def test_unknown_label_raises(tmp_path: Path) -> None:
    (tmp_path / "label.csv").write_text(
        "id\tquery_id\tproduct_id\tlabel\n0\t0\t100\tBogus\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown label"):
        list(WandsDataset({"path": str(tmp_path)}).qrels())


def test_wands_derives_from_dataset_abc(dataset: WandsDataset) -> None:
    assert issubclass(WandsDataset, Dataset)
    assert isinstance(dataset, Dataset)


def test_dataset_abc_is_not_instantiable() -> None:
    with pytest.raises(TypeError):
        Dataset()  # type: ignore[abstract]


def test_build_search_text_joins_text_roles_in_schema_order() -> None:
    schema = FieldSchema(
        fields=[
            FieldSpec("doc_id", FieldRole.ID),
            FieldSpec("title", FieldRole.SEMANTIC_SOURCE),
            FieldSpec("brand", FieldRole.STORED),  # skipped: not a text role
            FieldSpec("body", FieldRole.BM25),
            FieldSpec("price", FieldRole.NUMERIC),  # skipped
        ]
    )
    values = {"doc_id": "d1", "title": "Chair", "brand": "Acme", "body": "wood seat", "price": 9}
    assert Dataset.build_search_text(values, schema) == "Chair\nwood seat"


def test_build_search_text_missing_key_raises() -> None:
    schema = FieldSchema(fields=[FieldSpec("title", FieldRole.BM25)])
    with pytest.raises(KeyError):
        Dataset.build_search_text({}, schema)


def test_map_label_wands_gains() -> None:
    assert Dataset.map_label("Exact", WANDS_GAINS) == pytest.approx(1.0)
    assert Dataset.map_label("Partial", WANDS_GAINS) == pytest.approx(0.5)
    assert Dataset.map_label("Irrelevant", WANDS_GAINS) == pytest.approx(0.0)
    assert all(
        isinstance(Dataset.map_label(label, WANDS_GAINS), float) for label in WANDS_GAINS
    )


def test_map_label_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        Dataset.map_label("Bogus", WANDS_GAINS)


def test_field_schema_roles(dataset: WandsDataset) -> None:
    schema = dataset.field_schema()
    roles = {spec.name: spec.role for spec in schema.fields}
    assert roles["product_id"] == FieldRole.ID
    assert roles["product_name"] == FieldRole.SEMANTIC_SOURCE
    assert roles["product_description"] == FieldRole.SEMANTIC_SOURCE
    assert roles["product_features"] == FieldRole.BM25
    assert roles["product_class"] == FieldRole.BM25
    assert roles["category hierarchy"] == FieldRole.STORED
    assert roles["rating_count"] == FieldRole.NUMERIC
    assert roles["average_rating"] == FieldRole.NUMERIC
    assert roles["review_count"] == FieldRole.NUMERIC
    assert schema.search_text_field == schema.rerank_field == "search_text"
