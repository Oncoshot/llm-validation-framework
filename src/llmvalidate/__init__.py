# Managed by python-semantic-release — do not edit by hand; the release workflow rewrites
# this line. Set to 0.7.0 once by hand to clear the drift from ONC-12379, during which the
# release never reached this file and it stayed at 0.0.0 across every published version.
__version__ = "1.1.0"

# `structured` is pydantic-only and cheap, so it is imported eagerly: consumers that
# only need the extraction contract (StructuredResult / StructuredField) get it without
# pulling in the pandas/numpy scoring stack.
from .structured import StructuredResult, StructuredGroup, StructuredField

# `sortable_date` is std-lib only (`re`, `datetime`), so it is likewise eager: normalising a
# date string must not cost the caller a pandas import. Keep it dependency-free.
from .sortable_date import DATE_MASKS, is_canonical_date, to_canonical_date, to_sortable_date

# `cells` states the value-level conventions of a scored table — the no-finding sentinel,
# list cells, and the `-value`/`-code` facet columns. Std-lib only for the same reason:
# reading or writing a cell must not pull in the scoring stack.
from .cells import (
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

__all__ = [
    "validate",
    "bootstrap_CI",
    "StructuredResult",
    "StructuredGroup",
    "StructuredField",
    # dates
    "to_sortable_date",
    "to_canonical_date",
    "is_canonical_date",
    "DATE_MASKS",
    # cell conventions
    "NO_FINDING",
    "is_no_finding",
    "is_unlabelled",
    "parse_list_cell",
    "format_list_cell",
    "facet_columns",
    "split_facet",
    "FACET_SUFFIXES",
    "VALUE_SUFFIX",
    "CODE_SUFFIX",
]


def __getattr__(name: str):
    """Lazily expose the pandas-backed scorer (PEP 562).

    `validation` imports pandas/numpy/tqdm, so importing this package — or the
    lightweight `llmvalidate.structured` types — must not import it eagerly. `validate`
    and `bootstrap_CI` are resolved on first access, loading `validation` (and pandas)
    only then, and cached into the module namespace so later access skips this hook.
    `from llmvalidate import validate`, `from llmvalidate.validation import validate`,
    and `import llmvalidate.validation` all keep working. See ONC-12308.
    """
    if name in ("validate", "bootstrap_CI"):
        from . import validation
        attr = getattr(validation, name)
        globals()[name] = attr  # cache so repeated access doesn't re-enter __getattr__
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exposed entry points in `dir()` so introspection (and IDEs)
    still list `validate` / `bootstrap_CI` before first access — PEP 562 `__getattr__`
    alone would hide them, unlike the previous eager imports."""
    return sorted(set(__all__) | set(globals()))
