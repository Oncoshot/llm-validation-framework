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
