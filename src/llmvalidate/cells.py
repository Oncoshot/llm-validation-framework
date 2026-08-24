"""The value-level conventions of a scored table: sentinels, list cells, facet columns.

`validate` compares a label column against its `Res: ` column cell by cell, and three
conventions decide what those cells may say. All three are defined by this package but were,
until now, only *implemented* inside it — so every producer and consumer of a scored table
(dataset builders, extraction pipelines, harnesses that serve or submit predictions)
re-derived them from the docs, usually as a string literal or an f-string. This module states
them once:

* **Sentinels.** An empty cell means *not labelled* — the row simply says nothing about this
  field, and scoring skips it. `"-"` means *no information*: a real answer, asserting the
  source stated nothing to find, and graded like any other. Conflating the two is the single
  most common mistake against this contract, hence `is_unlabelled` and `is_no_finding` rather
  than one "is it empty" helper.
* **List cells.** A `list` field's cell is a `literal_eval`-able list repr (`"['EGFR',
  'KRAS']"`), with `"-"` for no finding. `parse_list_cell` / `format_list_cell` are that
  round-trip.
* **Facet columns.** A field carrying a `code` flattens to two scored columns, `<field>-value`
  and `<field>-code` (see `utils.flatten_structured_result`). `facet_columns` builds the pair
  and `split_facet` takes it apart.

Std-lib only, and eagerly importable: reading or writing a cell must not cost the caller a
pandas import.

The predicates here read a cell **exactly as written**: `" - "` is not the sentinel. That is
on purpose — a caller is usually checking what a file or a submission *contains*, and a cell
padded with spaces is a formatting problem worth seeing rather than papering over. `validate`
is more forgiving at compare time (it strips through `normalize` before matching), as is
`is_canonical_date`, whose whole job is to tell a date from a rewrite of one.
"""

import math
from ast import literal_eval
from typing import Any

# The "no information" sentinel: the source was read and stated nothing to find. A real,
# graded answer — not the absence of one.
NO_FINDING = "-"

# The two scored columns a coded field flattens into, in the order `facet_columns` returns
# them (value first, matching `flatten_structured_result`'s output order).
VALUE_SUFFIX = "-value"
CODE_SUFFIX = "-code"
FACET_SUFFIXES = (VALUE_SUFFIX, CODE_SUFFIX)


# --- Sentinels ---------------------------------------------------------------

def is_unlabelled(value: Any) -> bool:
    """True when a cell says nothing about its field — `None`, `""`, or NaN.

    Such a cell is **not** a prediction of "nothing to find": scoring treats it as out of
    scope for that row (partial labelling), so it lands in neither the numerator nor the
    denominator. Contrast `is_no_finding`.
    """
    if value is None or value == "":
        return True
    return isinstance(value, float) and math.isnan(value)


def is_no_finding(value: Any) -> bool:
    """True when a cell asserts "the source states nothing to find" — `"-"`, or `[]` for a list.

    A graded answer: predicting it against a real label costs recall, and predicting a value
    against it costs precision. `["-"]` is **not** this — it is a list holding one phantom
    element, which scores as a spurious hit; the empty list is how a list field says nothing.
    """
    if isinstance(value, list):
        return not value
    return value == NO_FINDING


# --- List cells --------------------------------------------------------------

def parse_list_cell(cell: Any) -> list:
    """The elements of a `list` field's cell, in order.

    A list repr parses to its elements; `"-"` and an empty cell are no finding, so `[]`. A
    string that is not a list repr is treated as a single element — a one-element cell
    written without brackets is far more likely than a caller wanting a crash. A malformed
    repr (`"['a', "`) likewise degrades to one element rather than raising, matching how
    `validate` leaves an unparsable cell alone instead of failing the run.

    Bracketed-but-unquoted cells (`"[EGFR, KRAS]"`) are split on commas. That form is not a
    Python repr and `literal_eval` rejects it, but it is what this package itself writes:
    `flatten_structured_result` strips the quotes out of a stringified list by default. An
    element containing a comma is therefore only safe in the *quoted* form — in the
    unquoted one the separator is genuinely ambiguous, and splitting is the best available
    reading.

    Already a list? Returned as a list of itself, so this is safe to apply twice.
    """
    if isinstance(cell, list):
        return list(cell)
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    text = str(cell).strip()
    if text in ("", NO_FINDING):
        return []
    if text.startswith("["):
        try:
            parsed = literal_eval(text)
        except (ValueError, SyntaxError):
            if text.endswith("]"):  # bracketed but unquoted, e.g. flatten()'s "[A, B]"
                inner = text[1:-1].strip()
                return [part.strip() for part in inner.split(",") if part.strip()] if inner else []
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed]
        return [str(parsed).strip()]
    return [text]


def format_list_cell(items: Any) -> str:
    """A `list` field's cell for `items`: `"-"` when there are none, else a list repr.

    The inverse of `parse_list_cell`, and the form a dataset or export should write so the
    scorer reads back the elements it was given.
    """
    if items is None:
        return NO_FINDING
    if isinstance(items, str):
        return items.strip() or NO_FINDING
    elements = [str(x) for x in items]
    return repr(elements) if elements else NO_FINDING


# --- Facet columns -----------------------------------------------------------

def facet_columns(field: str) -> tuple[str, str]:
    """The `(value_column, code_column)` pair a coded field is scored as.

    Mind the order — **value first**, as in `flatten_structured_result`'s output and the
    `FACET_SUFFIXES` tuple. A field with no `code` is a single column named `field` and has
    no facets at all.
    """
    return f"{field}{VALUE_SUFFIX}", f"{field}{CODE_SUFFIX}"


def split_facet(column: str) -> tuple[str, str | None]:
    """A scored column split into `(field, facet)`, where facet is `"value"`, `"code"`, or None.

    `"Primary Histology-code"` -> `("Primary Histology", "code")`; a free-form column comes
    back unchanged with a facet of None, so a caller can group facets under their logical
    field without special-casing.
    """
    for suffix in FACET_SUFFIXES:
        if column.endswith(suffix) and len(column) > len(suffix):
            return column[: -len(suffix)], suffix.lstrip("-")
    return column, None
