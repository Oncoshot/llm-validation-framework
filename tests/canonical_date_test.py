"""`to_canonical_date` / `is_canonical_date` — a date field's one fixed shape.

`to_sortable_date` returns whatever precision the source stated; these two put a mask on
top, for a field that is *defined* to hold a date. The rules under test: the mask truncates
but never pads, a time is dropped, a year-less date is not a date, and only a value this
module would itself produce counts as canonical.
"""

import pytest

from llmvalidate import DATE_MASKS, is_canonical_date, to_canonical_date

DAY = "YYYY-MM-DD"
MONTH = "YYYY-MM"
YEAR = "YYYY"


@pytest.mark.parametrize("raw,mask,expected,dayFirst", [
    # --- the dirty forms a document actually carries ---
    ("26/11/2024", DAY, "2024-11-26", True),
    ("26.11.2024", DAY, "2024-11-26", True),
    ("26 Nov 2024", DAY, "2024-11-26", True),
    ("dx 26/11/2024 confirmed", DAY, "2024-11-26", True),   # left-most date in the text
    ("11/26/2024", DAY, "2024-11-26", False),               # month-first source
    # --- already canonical: unchanged, so the transform is idempotent ---
    ("2024-11-26", DAY, "2024-11-26", True),
    ("2024-11", MONTH, "2024-11", True),
    ("2024", YEAR, "2024", True),
    # --- a date field stores a date: the time goes ---
    ("2024-11-26 09:30", DAY, "2024-11-26", True),
    ("2024-11-26T09:30:07", DAY, "2024-11-26", True),
    ("2011.04.03 08:09", DAY, "2011-04-03", True),
    # --- the mask truncates ---
    ("reported 11/26/2024", MONTH, "2024-11", False),
    ("2024-11-26", MONTH, "2024-11", True),
    ("2024-11-26 09:30", YEAR, "2024", True),
    # --- ...but never pads: a coarser source stays coarse ---
    ("Nov 2024", DAY, "2024-11", True),
    ("March 2021", DAY, "2021-03", True),
    ("2024", MONTH, "2024", True),
    # --- dayFirst decides an ambiguous numeric date, as in to_sortable_date ---
    ("05/01/2023", DAY, "2023-01-05", True),
    ("05/01/2023", DAY, "2023-05-01", False),
    ("12/5/2025", MONTH, "2025-12", False),
    ("12/5/2025", MONTH, "2025-05", True),
])
def test_to_canonical_date(raw, mask, expected, dayFirst):
    assert to_canonical_date(raw, mask, dayFirst) == expected


@pytest.mark.parametrize("raw,mask,dayFirst", [
    ("26 Nov", DAY, True),            # no year: `????-11-26` is not a date here
    ("Nov", MONTH, True),
    ("26/11/2024", DAY, False),       # only a date under the *other* reading
    ("2024-02-30", DAY, True),        # impossible calendar dates
    ("2024-13-01", DAY, True),
    ("not stated", DAY, True),
    ("", DAY, True),
    ("-", DAY, True),                 # a sentinel is the caller's business, not a date
    (None, DAY, True),
    (42, DAY, True),
])
def test_no_usable_date_returns_none(raw, mask, dayFirst):
    # None rather than a sentinel: "-" / "" / None mean different things to different
    # callers, so mapping onto one of them is the caller's decision.
    assert to_canonical_date(raw, mask, dayFirst) is None


def test_the_default_mask_is_day_precision():
    assert to_canonical_date("26/11/2024") == "2024-11-26"
    assert is_canonical_date("2024-11-26")


def test_unknown_mask_fails_loudly():
    with pytest.raises(ValueError) as excinfo:
        to_canonical_date("2024-11-26", "DD/MM/YYYY")
    assert "DD/MM/YYYY" in str(excinfo.value)
    assert all(mask in str(excinfo.value) for mask in DATE_MASKS)


@pytest.mark.parametrize("value,mask", [
    ("2024-11-26", DAY),
    ("2024-11", MONTH),
    ("2024", YEAR),
    ("2024-11", DAY),        # coarser than the mask: the source stated only a month
    ("2024", DAY),
    ("2024-02-29", DAY),     # a real leap day
    (" 2024-11-26 ", DAY),   # surrounding whitespace is tolerated
])
def test_canonical_values_are_recognised(value, mask):
    assert is_canonical_date(value, mask)


@pytest.mark.parametrize("value,mask", [
    ("26/11/2024", DAY),        # readable, but not canonical
    ("26 Nov 2024", DAY),
    ("2024-11-26 09:30", DAY),  # carries a time
    ("2024-11-26", MONTH),      # finer than the mask: the field holds a month
    ("2024-11", YEAR),
    ("2024-1-5", DAY),          # not zero-padded
    ("2024/11/26", DAY),        # right order, wrong separator
    ("????-11-26", DAY),        # unknown year
    ("2024-13", MONTH),         # impossible month
    ("2024-02-30", DAY),        # impossible day
    ("dx 2024-11-26", DAY),     # a canonical date plus other text is not a date cell
    ("", DAY),
    ("-", DAY),
    (None, DAY),
    (20241126, DAY),
])
def test_non_canonical_values_are_rejected(value, mask):
    assert not is_canonical_date(value, mask)


@pytest.mark.parametrize("mask", list(DATE_MASKS))
@pytest.mark.parametrize("raw", ["26/11/2024", "26 Nov 2024", "2024-11-26 09:30", "Nov 2024",
                                 "2024", "2024-11-26"])
def test_output_is_canonical_and_stable(raw, mask):
    # The invariant a producer leans on: canonicalise once and the value is accepted;
    # canonicalise again and nothing moves.
    once = to_canonical_date(raw, mask)
    assert is_canonical_date(once, mask)
    assert to_canonical_date(once, mask) == once


@pytest.mark.parametrize("mask", list(DATE_MASKS))
def test_a_canonical_value_reads_the_same_under_either_reading(mask):
    # Bounds what dayFirst is for: it decides how a *source* is read, never how a canonical
    # value is judged — which is why a canonical column can be validated without knowing
    # which convention produced it.
    for value in ("2024-11-26", "2024-11", "2024"):
        expected = is_canonical_date(value, mask, True)
        assert is_canonical_date(value, mask, False) == expected
