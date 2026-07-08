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
