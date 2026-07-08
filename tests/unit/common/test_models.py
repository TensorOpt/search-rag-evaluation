"""Phase 1 unit tests for benchmark.models (docs/architecture.md §3.1-§3.5)."""

from __future__ import annotations

import dataclasses

import pytest

from benchmark.common.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    IndexMapping,
    Qrel,
    Query,
    RankedResult,
    ScoredDoc,
)


# --- construction --------------------------------------------------------------


def test_query_construction_and_default():
    q = Query(query_id="q1", text="couch")
    assert (q.query_id, q.text, q.query_class) == ("q1", "couch", None)
    assert Query(query_id="q1", text="couch", query_class="furniture").query_class == "furniture"


def test_document_construction():
    d = Document(doc_id="p1", fields={"search_text": "blue sofa"})
    assert d.doc_id == "p1"
    assert d.fields["search_text"] == "blue sofa"


def test_qrel_construction():
    r = Qrel(query_id="q1", doc_id="p1", gain=0.5)
    assert (r.query_id, r.doc_id, r.gain) == ("q1", "p1", 0.5)


def test_scoreddoc_construction():
    s = ScoredDoc(doc_id="p1", score=1.5)
    assert (s.doc_id, s.score) == ("p1", 1.5)


def test_rankedresult_construction_order():
    docs = [ScoredDoc("p1", 3.0), ScoredDoc("p2", 2.0)]
    rr = RankedResult(query_id="q1", docs=docs)
    assert rr.docs[0].doc_id == "p1"  # docs[0] is rank 1


def test_fieldspec_and_schema_construction():
    fs = FieldSpec(name="product_name", role=FieldRole.BM25)
    schema = FieldSchema(fields=[fs])
    assert schema.fields[0].role is FieldRole.BM25


def test_indexmapping_construction():
    m = IndexMapping(
        index_name="wands_bench",
        search_text_field="search_text",
        sem_fields={"e5-small": "sem__e5_small"},
        backend_mapping={},
    )
    assert m.index_name == "wands_bench"
    assert m.search_text_field == "search_text"


# --- immutability (frozen) -----------------------------------------------------


@pytest.mark.parametrize(
    ("instance", "attr", "value"),
    [
        (Query("q1", "t"), "text", "x"),
        (Document("p1", {}), "doc_id", "p2"),
        (Qrel("q1", "p1", 1.0), "gain", 2.0),
        (ScoredDoc("p1", 1.0), "score", 2.0),
        (RankedResult("q1", []), "query_id", "q2"),
        (FieldSpec("n", FieldRole.ID), "name", "m"),
        (FieldSchema([]), "search_text_field", "other"),
        (IndexMapping("i", "search_text", {}, {}), "index_name", "j"),
    ],
)
def test_frozen_instances_reject_mutation(instance, attr, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, attr, value)


# --- field-name / default / enum contracts ------------------------------------


def test_scoreddoc_has_no_position_field():
    field_names = {f.name for f in dataclasses.fields(ScoredDoc)}
    assert field_names == {"doc_id", "score"}
    assert "position" not in field_names
    assert not hasattr(ScoredDoc("p1", 1.0), "position")


def test_fieldschema_defaults_search_text():
    schema = FieldSchema(fields=[])
    assert schema.search_text_field == "search_text"
    assert schema.rerank_field == "search_text"
    assert schema.search_text_field == schema.rerank_field == "search_text"


def test_field_role_values():
    assert FieldRole.ID == "id"
    assert FieldRole.BM25 == "bm25"
    assert FieldRole.SEMANTIC_SOURCE == "semantic_source"
    assert FieldRole.NUMERIC == "numeric"
    assert FieldRole.STORED == "stored"


# --- IndexMapping.sem_field ----------------------------------------------------


def test_indexmapping_sem_field_resolves():
    m = IndexMapping(
        index_name="wands_bench",
        search_text_field="search_text",
        sem_fields={"e5-small": "sem__e5_small", "elser": "sem__elser"},
        backend_mapping={},
    )
    assert m.sem_field("e5-small") == "sem__e5_small"
    assert m.sem_field("elser") == "sem__elser"


def test_indexmapping_sem_field_unknown_id_raises_keyerror():
    m = IndexMapping(
        index_name="wands_bench",
        search_text_field="search_text",
        sem_fields={"e5-small": "sem__e5_small"},
        backend_mapping={},
    )
    with pytest.raises(KeyError):
        m.sem_field("does-not-exist")
