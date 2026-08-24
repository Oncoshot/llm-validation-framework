"""`llmvalidate.cells` — the sentinels, list cells and facet columns a scored table uses.

These conventions are what `validate` already implements internally; the tests here pin the
public helpers against that behaviour, so a caller writing or reading a cell agrees with the
scorer rather than re-deriving the rules from prose.
"""

import pytest

from llmvalidate import (
    CODE_SUFFIX,
    FACET_SUFFIXES,
    NO_FINDING,
    VALUE_SUFFIX,
    facet_columns,
    format_list_cell,
    is_no_finding,
    is_unlabelled,
    parse_list_cell,
    split_facet,
)
from llmvalidate.validation import is_expected_undefined, is_scalar_empty


# --- Sentinels: "says nothing" and "says nothing was found" are different -----

@pytest.mark.parametrize("value", [None, "", float("nan")])
def test_unlabelled_values(value):
    assert is_unlabelled(value)
    assert not is_no_finding(value)


@pytest.mark.parametrize("value", [NO_FINDING, "-", [], list()])
def test_no_finding_values(value):
    assert is_no_finding(value)
    assert not is_unlabelled(value)


@pytest.mark.parametrize("value", ["Lung", "0", 0, 0.0, False, ["EGFR"], [NO_FINDING], " - ", "--"])
def test_ordinary_values_are_neither(value):
    # `['-']` is a list holding a phantom element, not a no-finding list — the distinction
    # that costs precision when a caller gets it wrong. `" - "` is a formatting problem, and
    # is reported as an ordinary value rather than silently accepted.
    assert not is_unlabelled(value)
    assert not is_no_finding(value)


def test_the_predicates_agree_with_the_scorers_own_view():
    # The helpers exist to state what `validate` already does; if they drifted from it, a
    # caller validating its data would disagree with the grade.
    for value in (None, "", NO_FINDING, float("nan"), "Lung", 0, 0.0, [], ["EGFR"]):
        assert is_expected_undefined(value) == is_unlabelled(value)
        expected_empty = is_unlabelled(value) or (not isinstance(value, list) and value == NO_FINDING)
        assert is_scalar_empty(value) == expected_empty


# --- List cells ---------------------------------------------------------------

@pytest.mark.parametrize("cell,expected", [
    ("['EGFR', 'KRAS']", ["EGFR", "KRAS"]),
    ('["EGFR", "KRAS"]', ["EGFR", "KRAS"]),
    ("[EGFR, KRAS]", ["EGFR", "KRAS"]),        # the unquoted form flatten() emits
    ("['EGFR']", ["EGFR"]),
    ("[]", []),
    (NO_FINDING, []),                          # no finding, either spelling
    ("", []),
    ("   ", []),
    (None, []),
    ("EGFR", ["EGFR"]),                        # a single element written without brackets
    ("  EGFR  ", ["EGFR"]),
    ("['EGFR', ", ["['EGFR',"]),               # malformed: degrades, never raises
    ("[EGFR, ]", ["EGFR"]),                    # dequoted, trailing separator
    ("['Tumor, benign']", ["Tumor, benign"]),  # a comma inside a *quoted* element survives
    (["EGFR", "KRAS"], ["EGFR", "KRAS"]),      # already parsed
    ([], []),
    (14, ["14"]),
])
def test_parse_list_cell(cell, expected):
    assert parse_list_cell(cell) == expected


def test_parse_list_cell_handles_nan_like_the_scorer():
    assert parse_list_cell(float("nan")) == []


def test_parse_list_cell_preserves_order_and_duplicates():
    # Set-wise *scoring* ignores both, but parsing must not decide that for the caller —
    # a caller counting duplicates or reporting the first element needs the cell as written.
    assert parse_list_cell("['KRAS', 'EGFR', 'KRAS']") == ["KRAS", "EGFR", "KRAS"]


@pytest.mark.parametrize("items,expected", [
    (["EGFR", "KRAS"], "['EGFR', 'KRAS']"),
    (["EGFR"], "['EGFR']"),
    ([], NO_FINDING),
    (None, NO_FINDING),
    ((), NO_FINDING),
    (("EGFR",), "['EGFR']"),
    ([14, 7], "['14', '7']"),                  # cells are text
    (NO_FINDING, NO_FINDING),                  # already a cell: unchanged
    ("['EGFR']", "['EGFR']"),
])
def test_format_list_cell(items, expected):
    assert format_list_cell(items) == expected


@pytest.mark.parametrize("items", [["EGFR", "KRAS"], ["EGFR"], [], ["a b", "c,d"]])
def test_the_list_cell_round_trip(items):
    assert parse_list_cell(format_list_cell(items)) == items


# --- Facet columns ------------------------------------------------------------

def test_facet_columns_are_value_then_code():
    # Order matters at every call site that unpacks the pair, so it is pinned here.
    assert facet_columns("Primary Histology") == ("Primary Histology-value", "Primary Histology-code")
    assert FACET_SUFFIXES == (VALUE_SUFFIX, CODE_SUFFIX) == ("-value", "-code")


@pytest.mark.parametrize("column,expected", [
    ("Primary Histology-value", ("Primary Histology", "value")),
    ("Primary Histology-code", ("Primary Histology", "code")),
    ("Primary Site", ("Primary Site", None)),          # free-form: no facets
    ("Medications-code", ("Medications", "code")),
    ("-value", ("-value", None)),                      # nothing left to be a field name
    ("", ("", None)),
    ("ICD-10-code", ("ICD-10", "code")),               # a field name containing a dash
    ("ICD-10", ("ICD-10", None)),
])
def test_split_facet(column, expected):
    assert split_facet(column) == expected


def test_facet_columns_and_split_facet_are_inverses():
    for field in ("Primary Histology", "Medications", "ICD-10", "a-b-c"):
        value_column, code_column = facet_columns(field)
        assert split_facet(value_column) == (field, "value")
        assert split_facet(code_column) == (field, "code")


def test_flatten_uses_these_names():
    # The convention has one owner: what `facet_columns` builds is what a coded field
    # actually flattens to.
    from llmvalidate.structured import StructuredField, StructuredGroup, StructuredResult
    from llmvalidate.utils import flatten_structured_result

    flat = flatten_structured_result(StructuredResult(groups=[
        StructuredGroup(group_name="Diagnosis", fields=[
            StructuredField(name="Primary Histology", value="Adenocarcinoma", code="8140/3"),
            StructuredField(name="Primary Site", value="Lung"),
        ]),
    ]))
    value_column, code_column = facet_columns("Primary Histology")
    assert flat[value_column] == "Adenocarcinoma"
    assert flat[code_column] == "8140/3"
    assert flat["Primary Site"] == "Lung"          # free-form stays a single column
