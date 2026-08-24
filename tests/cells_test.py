"""`llmvalidate.cells` — the sentinels, list cells and facet columns a scored table uses.

These conventions are what `validate` already implements internally; the tests here pin the
public helpers against that behaviour, so a caller writing or reading a cell agrees with the
scorer rather than re-deriving the rules from prose.
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from llmvalidate import (
    CODE_SUFFIX,
    FACET_SUFFIXES,
    NO_FINDING,
    VALUE_SUFFIX,
    canonical_date_cell,
    facet_columns,
    format_list_cell,
    is_canonical_date_cell,
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


# Every shape "no value at all" arrives in. A cell out of a frame or a parser is under no
# obligation to be a built-in: `numpy.float64` happens to subclass `float` but
# `numpy.float32` and `Decimal("nan")` do not, and `pd.NA` cannot even be compared — it used
# to raise `TypeError` here, and stringified to a phantom `['<NA>']` element in a list cell.
MISSING = [
    pytest.param(None, id="None"),
    pytest.param("", id="empty"),
    pytest.param(float("nan"), id="float-nan"),
    pytest.param(np.float64("nan"), id="np.float64-nan"),
    pytest.param(np.float32("nan"), id="np.float32-nan"),
    pytest.param(np.nan, id="np.nan"),
    pytest.param(Decimal("nan"), id="Decimal-nan"),
    pytest.param(pd.NA, id="pd.NA"),
    pytest.param(pd.NaT, id="pd.NaT"),
]


@pytest.mark.parametrize("value", MISSING)
def test_every_missing_shape_is_unlabelled(value):
    assert is_unlabelled(value) is True
    # Missing is not an assertion that nothing was found — nothing was asserted at all.
    assert is_no_finding(value) is False
    # ...and in a list cell it is None: neither elements, nor the no-finding answer, nor
    # one element spelling "nan".
    assert parse_list_cell(value) is None


@pytest.mark.parametrize("value", MISSING + [
    pytest.param("Lung", id="text"), pytest.param(NO_FINDING, id="sentinel"),
    pytest.param(np.float64(1.5), id="np.float64"), pytest.param([], id="empty-list"),
])
def test_the_predicates_return_real_bools(value):
    # A numpy comparison yields `numpy.bool_`, and `pd.NA == x` yields `pd.NA` — neither is
    # usable by a caller that writes `if is_no_finding(cell):`.
    assert type(is_unlabelled(value)) is bool
    assert type(is_no_finding(value)) is bool


def test_values_that_cannot_be_compared_are_not_missing():
    # An array is not a scalar cell; the guard exists so a caller gets False instead of
    # "truth value of an array is ambiguous".
    assert is_unlabelled(np.array([1.0, 2.0])) is False
    assert is_no_finding(np.array([1.0, 2.0])) is False


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
    ("", None),                                # not labelled: says nothing at all
    ("   ", None),
    (None, None),
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


def test_parse_list_cell_keeps_the_sentinel_distinction():
    # The module's central claim has to survive the parse. Handing `[]` back for an
    # unlabelled cell would put an out-of-scope row into the scorer's denominator, and
    # `is_no_finding([])` would then report an answer nobody gave.
    assert parse_list_cell(float("nan")) is None
    assert parse_list_cell("") is None
    assert parse_list_cell(NO_FINDING) == []        # a real, graded "nothing to find"
    assert parse_list_cell("[]") == []
    assert is_no_finding(parse_list_cell(NO_FINDING)) is True
    assert is_unlabelled(parse_list_cell(None)) is True


def test_parse_list_cell_preserves_order_and_duplicates():
    # Set-wise *scoring* ignores both, but parsing must not decide that for the caller —
    # a caller counting duplicates or reporting the first element needs the cell as written.
    assert parse_list_cell("['KRAS', 'EGFR', 'KRAS']") == ["KRAS", "EGFR", "KRAS"]


@pytest.mark.parametrize("items,expected", [
    (["EGFR", "KRAS"], "['EGFR', 'KRAS']"),
    (["EGFR"], "['EGFR']"),
    ([], NO_FINDING),                          # no finding
    (None, ""),                                # nothing to say -> the unlabelled cell
    ((), NO_FINDING),
    (("EGFR",), "['EGFR']"),
    ([14, 7], "['14', '7']"),                  # cells are text
    (NO_FINDING, NO_FINDING),                  # already a cell: unchanged
    ("['EGFR']", "['EGFR']"),
])
def test_format_list_cell(items, expected):
    assert format_list_cell(items) == expected


@pytest.mark.parametrize("items", [["EGFR", "KRAS"], ["EGFR"], [], ["a b", "c,d"], None])
def test_the_list_cell_round_trip(items):
    # Exact inverses, distinction included: None -> "" -> None, [] -> "-" -> [].
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


# --- Date cells: the two conventions together --------------------------------
# `to_canonical_date` returns None for "no usable date" so a caller can decide what that
# means in a cell. These are that decision, and the reason a consumer no longer has to make
# it for itself (the Optimization harness had this layer written twice, once per side).

DAY, MONTH = "YYYY-MM-DD", "YYYY-MM"


@pytest.mark.parametrize("cell,mask,dayFirst,expected", [
    # a real value renders at the mask — the rules are `to_canonical_date`'s
    ("26/11/2024", DAY, True, "2024-11-26"),
    ("dx 26 Nov 2024", DAY, True, "2024-11-26"),
    ("2024-11-26 09:30", DAY, True, "2024-11-26"),
    ("reported 11/26/2024", MONTH, False, "2024-11"),
    ("2024-11-26", MONTH, True, "2024-11"),          # the mask truncates
    ("Nov 2024", DAY, True, "2024-11"),              # coarser stays coarse
    # a no-finding cell is an answer, and stays one
    (NO_FINDING, DAY, True, NO_FINDING),
    # text that holds no readable date is "we looked, there is nothing"
    ("not stated", DAY, True, NO_FINDING),
    ("26 Nov", DAY, True, NO_FINDING),               # no year
    ("26/11/2024", DAY, False, NO_FINDING),          # unreadable under this reading
    ("2024-02-30", DAY, True, NO_FINDING),           # impossible date
])
def test_canonical_date_cell(cell, mask, dayFirst, expected):
    assert canonical_date_cell(cell, mask, dayFirst=dayFirst) == expected


@pytest.mark.parametrize("cell", MISSING)
def test_an_unlabelled_cell_survives_canonicalising(cell):
    # The distinction this module exists for: "says nothing" must not become "says nothing
    # was found", or a scorer gains an out-of-scope row in its denominator.
    out = canonical_date_cell(cell, DAY)
    assert is_unlabelled(out), out
    assert not is_no_finding(out)


@pytest.mark.parametrize("cell,mask,dayFirst", [
    ("26/11/2024", DAY, True), ("Nov 2024", DAY, True), ("reported 11/26/2024", MONTH, False),
    ("not stated", DAY, True), (NO_FINDING, DAY, True), ("", DAY, True), (None, DAY, True),
    (float("nan"), MONTH, True), ("2024-11-26", MONTH, True),
])
def test_canonicalising_yields_something_the_column_may_hold(cell, mask, dayFirst):
    # The invariant a producer leans on: canonicalise anything, and the cell is acceptable.
    assert is_canonical_date_cell(canonical_date_cell(cell, mask, dayFirst=dayFirst),
                                  mask, dayFirst=dayFirst)


@pytest.mark.parametrize("cell,mask,dayFirst", [
    ("2024-11-26", DAY, True),
    ("2024-11", MONTH, True),
    ("2024-11", DAY, True),          # coarser than the mask is clean, and scores wrong
    (NO_FINDING, DAY, True),
    ("", DAY, True),
    (None, DAY, True),
    (float("nan"), DAY, True),
])
def test_cells_a_date_column_may_hold(cell, mask, dayFirst):
    assert is_canonical_date_cell(cell, mask, dayFirst=dayFirst)


@pytest.mark.parametrize("cell,mask,dayFirst", [
    ("26/11/2024", DAY, True),       # readable, not canonical
    ("2024-11-26 09:30", DAY, True), # carries a time
    ("2024-11-26", MONTH, True),     # finer than the mask
    ("2024-1-5", DAY, True),         # not zero-padded
    ("????-11-26", DAY, True),       # unknown year
    ("2024-02-30", DAY, True),       # impossible day
    ("dx 2024-11-26", DAY, True),    # a date plus other text
])
def test_cells_a_date_column_may_not_hold(cell, mask, dayFirst):
    assert not is_canonical_date_cell(cell, mask, dayFirst=dayFirst)


def test_the_reading_is_declared_by_the_caller_not_guessed():
    # Why `dayFirst` travels with the schema in a caller that has date columns: the same
    # text is two different dates, and one reading may make it no date at all.
    assert canonical_date_cell("05/01/2023", DAY, dayFirst=True) == "2023-01-05"
    assert canonical_date_cell("05/01/2023", DAY, dayFirst=False) == "2023-05-01"
    assert canonical_date_cell("11/26/2024", DAY, dayFirst=False) == "2024-11-26"
    assert canonical_date_cell("11/26/2024", DAY, dayFirst=True) == NO_FINDING
