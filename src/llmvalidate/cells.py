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
  'KRAS']"`), with `"-"` for no finding and an empty cell for not labelled.
  `parse_list_cell` / `format_list_cell` are that round-trip, and they carry the sentinel
  distinction through it rather than flattening it: an unlabelled cell parses to `None`,
  a no-finding one to `[]`.
* **Facet columns.** A field carrying a `code` flattens to two scored columns, `<field>-value`
  and `<field>-code` (see `utils.flatten_structured_result`). `facet_columns` builds the pair
  and `split_facet` takes it apart.
* **Date cells.** A column defined to hold a date holds one canonical rendering of one — and
  the sentinels. `canonical_date_cell` / `is_canonical_date_cell` are the two conventions
  together: `sortable_date`'s mask rules for a real value, and the rules above for a cell
  that says nothing or says "nothing to find".

Std-lib only, and eagerly importable: reading or writing a cell must not cost the caller a
pandas import.

Missing values are accepted in every shape they arrive in — `None`, NaN (of any float
type), `pd.NA`, `pd.NaT` — because a cell that came out of a frame or a parser is under no
obligation to be a built-in, and `pd.NA` cannot even be compared without raising.

Beyond that, the predicates here read a cell **exactly as written**: `" - "` is not the
sentinel. That is
on purpose — a caller is usually checking what a file or a submission *contains*, and a cell
padded with spaces is a formatting problem worth seeing rather than papering over. `validate`
is more forgiving at compare time (it strips through `normalize` before matching), as is
`is_canonical_date`, whose whole job is to tell a date from a rewrite of one.
"""

import math
from ast import literal_eval
from typing import Any

from .sortable_date import is_canonical_date, to_canonical_date

# The "no information" sentinel: the source was read and stated nothing to find. A real,
# graded answer — not the absence of one.
NO_FINDING = "-"

# pandas' own missing-value singletons, recognised by type name so this module stays
# std-lib only. `pd.NA` earns the special case twice over: it is not a float, so no NaN
# test finds it, and it cannot be compared either — `pd.NA == ""` is `pd.NA`, whose truth
# value raises.
_MISSING_TYPE_NAMES = frozenset({"NAType", "NaTType"})

# The two scored columns a coded field flattens into, in the order `facet_columns` returns
# them (value first, matching `flatten_structured_result`'s output order).
VALUE_SUFFIX = "-value"
CODE_SUFFIX = "-code"
FACET_SUFFIXES = (VALUE_SUFFIX, CODE_SUFFIX)


# --- Sentinels ---------------------------------------------------------------

def _is_missing_scalar(value: Any) -> bool:
    """True for every shape "no value at all" arrives in.

    `None`, any NaN, and pandas' `pd.NA` / `pd.NaT`. NaN is tested by conversion rather
    than by `isinstance(value, float)`: `numpy.float64` happens to subclass `float`, but
    `numpy.float32` and `Decimal("nan")` do not, and a cell that came out of a frame or a
    parser is under no obligation to be a built-in.
    """
    if value is None:
        return True
    if type(value).__name__ in _MISSING_TYPE_NAMES:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _equals(value: Any, other: str) -> bool:
    """`value == other` as a real `bool`, False when the two cannot be compared.

    Guards two things a raw `==` would do to a caller: returning a `numpy.bool_` from a
    function annotated `-> bool`, and raising on a value whose comparison is not a boolean
    at all (an array, or `pd.NA`).
    """
    try:
        return bool(value == other)
    except (TypeError, ValueError):
        return False


def is_unlabelled(value: Any) -> bool:
    """True when a cell says nothing about its field — `None`, `""`, NaN, `pd.NA`, `pd.NaT`.

    Such a cell is **not** a prediction of "nothing to find": scoring treats it as out of
    scope for that row (partial labelling), so it lands in neither the numerator nor the
    denominator. Contrast `is_no_finding`.
    """
    return _is_missing_scalar(value) or _equals(value, "")


def is_no_finding(value: Any) -> bool:
    """True when a cell asserts "the source states nothing to find" — `"-"`, or `[]` for a list.

    A graded answer: predicting it against a real label costs recall, and predicting a value
    against it costs precision. `["-"]` is **not** this — it is a list holding one phantom
    element, which scores as a spurious hit; the empty list is how a list field says nothing.
    A missing value is not this either: nothing was asserted at all (see `is_unlabelled`).
    """
    if isinstance(value, list):
        return not value
    if _is_missing_scalar(value):
        return False
    return _equals(value, NO_FINDING)


# --- List cells --------------------------------------------------------------

def parse_list_cell(cell: Any) -> list | None:
    """The elements of a `list` field's cell, in order — or **None** when it is unlabelled.

    The sentinel distinction survives the parse, because collapsing it here would quietly
    undo the point of this module: `"-"` and `[]` are the graded no-finding answer and come
    back as `[]`, while a cell that says nothing at all (empty, NaN, `pd.NA`) comes back as
    None. Hand `[]` to the scorer for an unlabelled row and you add an out-of-scope row to
    the denominator.

    A list repr parses to its elements. A string that is not a list repr is treated as a
    single element — a one-element cell written without brackets is far more likely than a
    caller wanting a crash. A malformed repr (`"['a', "`) likewise degrades to one element
    rather than raising, matching how `validate` leaves an unparsable cell alone instead of
    failing the run.

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
    if _is_missing_scalar(cell):
        return None   # says nothing — neither elements nor a no-finding answer
    text = str(cell).strip()
    if text == "":
        return None            # an all-whitespace cell says nothing either
    if text == NO_FINDING:
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

    The exact inverse of `parse_list_cell`, distinction included: `[]` is the no-finding
    answer `"-"`, and None — nothing to say about this field — is the empty cell. The form a
    dataset or export should write so the scorer reads back what it was given.
    """
    if items is None:
        return ""
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


# --- Date cells ---------------------------------------------------------------

def canonical_date_cell(cell: Any, mask: str = "YYYY-MM-DD", dayFirst: bool = True) -> Any:
    """`cell`'s canonical date at `mask`, with the sentinels preserved.

    The cell-level counterpart to `sortable_date.to_canonical_date`, which returns None for
    "no usable date" precisely so a caller can decide what that means in a *cell*. Here:

    * an **unlabelled** cell (empty, NaN, `pd.NA`) is returned unchanged — it says nothing
      about the field, and turning that into an answer would put an out-of-scope row into a
      scorer's denominator;
    * a **no-finding** cell (`"-"`) stays `NO_FINDING`;
    * text that holds **no readable date** becomes `NO_FINDING` — the honest reading of "we
      looked and there is nothing here";
    * anything else is rendered at the mask (see `to_canonical_date` for the rules: the mask
      truncates but never pads, a time is dropped, a year-less date is not a date).

    Whatever this returns satisfies `is_canonical_date_cell` for the same `(mask, dayFirst)`,
    so a producer can put dirty text straight through it.

    One caution for **label** builders: text holding no readable date is indistinguishable
    here from a cell that recorded nothing, since both come back as `NO_FINDING`. Compare the
    result against what the source was meant to say and fail the build when they disagree —
    silently mislabelling a row is worse than not building.
    """
    if is_unlabelled(cell):
        return cell
    if is_no_finding(cell):
        return NO_FINDING
    rendered = to_canonical_date(cell, mask, dayFirst=dayFirst)
    return rendered if rendered is not None else NO_FINDING


def is_canonical_date_cell(cell: Any, mask: str = "YYYY-MM-DD", dayFirst: bool = True) -> bool:
    """True when a column defined to hold a `mask` date may hold `cell` as it stands.

    That is: a sentinel — unlabelled or no-finding — or a canonical date at the mask's
    precision **or coarser** (`is_canonical_date`). A test of *cleanliness*, not of
    correctness: a coarser value is clean and will simply score wrong against a finer label.
    Finer than the mask is not clean, and neither is anything `dayFirst` cannot read.
    """
    if is_unlabelled(cell) or is_no_finding(cell):
        return True
    return is_canonical_date(cell, mask, dayFirst=dayFirst)
