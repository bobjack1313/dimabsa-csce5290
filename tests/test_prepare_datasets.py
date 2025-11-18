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


import pytest

from scripts.prepare_datasets import (
    normalize_task1_record,
    normalize_task2_record,
    _normalize_quad_list,
)


# New normalize Helper
def quad(aspect="A", category="C", opinion="O", va="5.00#5.00"):
    return {"Aspect": aspect, "Category": category, "Opinion": opinion, "VA": va}


def test_task1_single_quad_extracts_va_and_aspect():
    rec = {
        "ID": "x1",
        "Text": "hello world",
        "Quadruplet": [quad(aspect="battery", va="7.12#6.44")]
    }

    out = normalize_task1_record(rec)

    assert out["ID"] == "x1"
    assert out["Text"] == "hello world"

    # Aspect list
    assert out["Aspect"] == ["battery"]

    # VA_gold extraction
    assert out["VA_gold"]["V"] == pytest.approx(7.12)
    assert out["VA_gold"]["A"] == pytest.approx(6.44)

    # Quadruplet unchanged
    assert out["Quadruplet"][0]["VA"] == "7.12#6.44"


def test_task1_multiple_quads_uses_first_va_only():
    rec = {
        "ID": "x2",
        "Text": "two aspects",
        "Quadruplet": [
            quad(aspect="screen", va="4.00#5.00"),
            quad(aspect="keyboard", va="9.00#9.00")
        ],
    }

    out = normalize_task1_record(rec)

    # Only first quad VA_used
    assert out["VA_gold"]["V"] == pytest.approx(4.00)
    assert out["VA_gold"]["A"] == pytest.approx(5.00)

    # Aspect list contains all non-NULL aspects
    assert out["Aspect"] == ["screen", "keyboard"]


def test_task1_handles_null_aspect_and_missing_va_safely():
    rec = {
        "ID": "x3",
        "Text": "nullish",
        "Quadruplet": [
            {"Aspect": "NULL", "Category": "C", "Opinion": "O", "VA": "NULL"}
        ]
    }

    out = normalize_task1_record(rec)

    # Aspect list should exclude NULL aspects
    assert out["Aspect"] == []

    # Missing VA → None values
    assert out["VA_gold"]["V"] is None
    assert out["VA_gold"]["A"] is None


def test_task1_no_quadruplet_gives_empty_aspect_and_none_va():
    rec = {
        "ID": "x4",
        "Text": "no quad",
        "Quadruplet": []
    }

    out = normalize_task1_record(rec)

    assert out["Aspect"] == []
    assert out["VA_gold"]["V"] is None
    assert out["VA_gold"]["A"] is None


def test_task2_basic_quad_normalization():
    rec = {
        "ID": "t1",
        "Text": "sample",
        "Quadruplet": [quad("battery", "HARDWARE#GENERAL", "bad", "3.50#7.25")]
    }

    out = normalize_task2_record(rec)

    assert out["ID"] == "t1"
    assert out["Text"] == "sample"
    assert len(out["Quadruplet"]) == 1

    q = out["Quadruplet"][0]
    assert q["Aspect"] == "battery"
    assert q["Category"] == "HARDWARE#GENERAL"
    assert q["Opinion"] == "bad"
    assert q["VA"] == "3.50#7.25"


def test_task2_null_fields_are_preserved_as_strings():
    rec = {
        "ID": "t2",
        "Text": "sample",
        "Quadruplet": [
            {"Aspect": None, "Category": None, "Opinion": None, "VA": None}
        ]
    }

    out = normalize_task2_record(rec)
    q = out["Quadruplet"][0]

    assert q["Aspect"] == "None"  # converted to string
    assert q["Category"] == "None"
    assert q["Opinion"] == "None"
    assert q["VA"] == "None"


def test_task2_handles_missing_quadruplet_gracefully():
    rec = {
        "ID": "t3",
        "Text": "missing",
        # no Quadruplet key at all
    }

    out = normalize_task2_record(rec)
    assert out["Quadruplet"] == []


def test__normalize_quad_list_basic_behavior():
    raw = [
        {"Aspect": "screen", "Category": "DISPLAY#GENERAL", "Opinion": "great", "VA": "8.00#8.00"},
        {"Aspect": "battery"}  # missing keys → filled with NULL
    ]

    out = _normalize_quad_list(raw)
    assert len(out) == 2

    assert out[0]["Aspect"] == "screen"
    assert out[0]["Category"] == "DISPLAY#GENERAL"
    assert out[0]["Opinion"] == "great"
    assert out[0]["VA"] == "8.00#8.00"

    assert out[1]["Aspect"] == "battery"
    assert out[1]["Category"] == "NULL"
    assert out[1]["Opinion"] == "NULL"
    assert out[1]["VA"] == "NULL"
