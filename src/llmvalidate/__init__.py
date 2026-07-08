__version__ = "0.0.0"

# `structured` is pydantic-only and cheap, so it is imported eagerly: consumers that
# only need the extraction contract (StructuredResult / StructuredField) get it without
# pulling in the pandas/numpy scoring stack.
from .structured import StructuredResult, StructuredGroup, StructuredField

__all__ = [
    "validate",
    "bootstrap_CI",
    "StructuredResult",
    "StructuredGroup",
    "StructuredField",
]


def __getattr__(name: str):
    """Lazily expose the pandas-backed scorer (PEP 562).

    `validation` imports pandas/numpy/tqdm, so importing this package — or the
    lightweight `llmvalidate.structured` types — must not import it eagerly. `validate`
    and `bootstrap_CI` are resolved on first access, loading `validation` (and pandas)
    only then. `from llmvalidate import validate`, `from llmvalidate.validation import
    validate`, and `import llmvalidate.validation` all keep working. See ONC-12308.
    """
    if name in ("validate", "bootstrap_CI"):
        from . import validation
        return getattr(validation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
