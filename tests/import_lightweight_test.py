"""ONC-12308: importing the package or its structured types must not load pandas.

Each check runs in a *fresh* interpreter (pandas may already be in this process's
`sys.modules` via other tests), with `src/` on `PYTHONPATH` so it exercises the source
tree regardless of how the package is installed.
"""

import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _run(code: str) -> str:
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _pandas_loaded_expr() -> str:
    return "any(m == 'pandas' or m.startswith('pandas.') for m in sys.modules)"


def test_importing_structured_types_does_not_load_pandas():
    out = _run(
        "import sys\n"
        "import llmvalidate\n"
        "from llmvalidate import StructuredResult, StructuredGroup, StructuredField\n"
        "from llmvalidate.structured import StructuredField as _SF\n"
        f"print({_pandas_loaded_expr()})\n"
    )
    assert out == "False", f"pandas was imported eagerly by the lightweight path: {out}"


def test_importing_sortable_date_does_not_load_pandas():
    # `to_sortable_date` is std-lib only; reaching it from either the package namespace or
    # the submodule must stay on the lightweight path.
    out = _run(
        "import sys\n"
        "from llmvalidate import to_sortable_date\n"
        "from llmvalidate.sortable_date import to_sortable_date as _t\n"
        "assert to_sortable_date is _t\n"
        "assert to_sortable_date('5 Jan 2020') == '2020-01-05'\n"
        f"print({_pandas_loaded_expr()})\n"
    )
    assert out == "False", f"pandas was imported by the sortable_date path: {out}"


def test_importing_the_cell_conventions_does_not_load_pandas():
    # `cells` and the date masks are std-lib only: reading or writing a cell, or
    # canonicalising a date, must not drag the scoring stack in.
    out = _run(
        "import sys\n"
        "from llmvalidate import NO_FINDING, is_no_finding, is_unlabelled\n"
        "from llmvalidate import parse_list_cell, format_list_cell, facet_columns, split_facet\n"
        "from llmvalidate import to_canonical_date, is_canonical_date, DATE_MASKS\n"
        "from llmvalidate import canonical_date_cell, is_canonical_date_cell, is_date_mask\n"
        "assert canonical_date_cell('dx 26/11/2024') == '2024-11-26'\n"
        "assert is_canonical_date_cell('-') and is_date_mask('YYYY-MM')\n"
        "assert parse_list_cell(format_list_cell(['EGFR'])) == ['EGFR']\n"
        "assert facet_columns('X') == ('X-value', 'X-code')\n"
        "assert to_canonical_date('26/11/2024', 'YYYY-MM') == '2024-11'\n"
        "assert is_no_finding(NO_FINDING) and is_unlabelled('')\n"
        f"print({_pandas_loaded_expr()})\n"
    )
    assert out == "False", f"pandas was imported by the cells/date path: {out}"


def test_scorer_is_accessible_and_loads_lazily():
    # Accessing validate/bootstrap_CI works and is what pulls pandas in (lazily).
    out = _run(
        "import sys\n"
        "import llmvalidate\n"
        "assert callable(llmvalidate.validate)\n"
        "assert callable(llmvalidate.bootstrap_CI)\n"
        "from llmvalidate.validation import validate as _v\n"  # direct path still works
        f"print({_pandas_loaded_expr()})\n"
    )
    assert out == "True", f"expected pandas loaded once the scorer is used, got: {out}"


def test_entry_points_listed_in_dir_without_loading_pandas():
    # dir()/introspection must show validate & bootstrap_CI even before first access,
    # and merely listing them must not drag in pandas.
    out = _run(
        "import sys, llmvalidate\n"
        "names = dir(llmvalidate)\n"
        "assert 'validate' in names and 'bootstrap_CI' in names, names\n"
        f"print({_pandas_loaded_expr()})\n"
    )
    assert out == "False", f"dir() should not load pandas: {out}"


def test_resolved_scorer_is_cached_on_the_module():
    # After first access the attribute is cached into the module namespace (so repeated
    # access skips __getattr__) and identity is stable.
    out = _run(
        "import llmvalidate\n"
        "v1 = llmvalidate.validate\n"
        "assert 'validate' in vars(llmvalidate)\n"
        "assert llmvalidate.validate is v1\n"
        "print('ok')\n"
    )
    assert out == "ok"


def test_unknown_attribute_still_raises():
    out = _run(
        "import llmvalidate\n"
        "try:\n"
        "    llmvalidate.nope\n"
        "    print('no-error')\n"
        "except AttributeError:\n"
        "    print('AttributeError')\n"
    )
    assert out == "AttributeError"
