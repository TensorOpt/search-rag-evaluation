"""Phase 8 unit tests for benchmark.datasets.wands (docs/architecture.md §3.2, §5.1 + docs/methodology.md §7)."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from benchmark.datasets.wands import WANDS_GAINS, WandsDataset
from benchmark.common.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    Qrel,
    Query,
)
from benchmark.common.protocols import Dataset

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


def test_documents_numeric_float_formatted_and_empty_cells(tmp_path: Path) -> None:
    # Full WANDS formats integer counts as floats ("15.0") and leaves many numeric cells empty;
    # both must parse (int columns narrowed via float; empty cells omitted, not 0/None).
    header = (
        "product_id\tproduct_name\tproduct_class\tcategory hierarchy\t"
        "product_description\tproduct_features\trating_count\taverage_rating\treview_count"
    )
    rows = [
        "1\tChair\tSeating\tFurniture\tA chair\tcomfy\t15.0\t4.5\t12.0",  # float-formatted ints
        "2\tTable\tSurfaces\tFurniture\tA table\tsturdy\t\t\t",  # all-empty numeric cells
    ]
    (tmp_path / "product.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    docs = {d.doc_id: d for d in WandsDataset({"path": str(tmp_path)}).documents()}

    chair = docs["1"]
    assert chair.fields["rating_count"] == 15 and isinstance(chair.fields["rating_count"], int)
    assert chair.fields["review_count"] == 12 and isinstance(chair.fields["review_count"], int)
    assert chair.fields["average_rating"] == pytest.approx(4.5)
    # Empty numeric cells are omitted entirely (not stored as 0 or None).
    table = docs["2"]
    assert "rating_count" not in table.fields
    assert "average_rating" not in table.fields
    assert "review_count" not in table.fields


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
