import pytest
from scripts.prepare_datasets import _normalize_quad_list


def test_normalize_quads_basic():
    raw = [
        {"Aspect": "food", "Category": "FOOD#QUALITY", "Opinion": "good", "VA": "7.0#7.1"}
    ]

    out = _normalize_quad_list(raw)

    assert len(out) == 1
    assert out[0]["Aspect"] == "food"
    assert out[0]["Category"] == "FOOD#QUALITY"
    assert out[0]["Opinion"] == "good"
    assert out[0]["VA"] == "7.0#7.1"


def test_normalize_quads_missing_fields():
    raw = [
        {"Aspect": "service"},  # only Aspect given
        {},                      # empty dict
    ]

    out = _normalize_quad_list(raw)

    assert len(out) == 2
    assert out[0]["Category"] == "NULL"
    assert out[1]["Aspect"] == "NULL"
    assert out[1]["VA"] == "NULL"


def test_normalize_quads_non_dict_entries_are_skipped():
    raw = [
        {"Aspect": "food"},
        "bad_entry",
        123,
        None,
        ["something"],
    ]

    out = _normalize_quad_list(raw)

    # Only the first one is valid
    assert len(out) == 1
    assert out[0]["Aspect"] == "food"


def test_normalize_quads_weird_va_types():
    raw = [
        {"Aspect": "food", "Category": "FOOD#QUALITY", "Opinion": "fine", "VA": ["a", "b"]},
        {"Aspect": "service", "VA": {"x": 1}},
    ]

    out = _normalize_quad_list(raw)

    assert len(out) == 2
    assert out[0]["VA"] == "['a', 'b']"
    assert out[1]["VA"] == "{'x': 1}"


def test_normalize_quads_invalid_input_returns_empty():
    assert _normalize_quad_list(None) == []
    assert _normalize_quad_list("not a list") == []
    assert _normalize_quad_list(12345) == []
