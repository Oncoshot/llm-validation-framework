import math
import pandas as pd
import numpy as np
import pytest
import llmvalidate.validation as v


def test_fully_labelled_binary_field_metrics_through_validate():
    # Regression for ONC-12248: a fully-labelled (no NaN) binary field used to
    # silently produce all-zero confusion counts because validate()->convert_lists
    # re-infers the all-bool column to numpy bool, whose scalars fail `is True`.
    src = pd.DataFrame({
        "Has metastasis":      pd.Series([True, False, True, False], dtype=object),
        "Res: Has metastasis": pd.Series([True, False, True, True],  dtype=object),
    })

    _, metrics = v.validate(
        src, ["Has metastasis"], structure_callback=None, output_folder=None
    )

    row = metrics[metrics["field"] == "Has metastasis"].iloc[0]

    assert row["TP"] == 2
    assert row["FP"] == 1
    assert row["FN"] == 0
    assert row["TN"] == 1

    assert row["precision (micro)"] == pytest.approx(2 / 3)
    assert row["recall (micro)"] == pytest.approx(1.0)
    assert row["F1 score (micro)"] == pytest.approx(0.8)


def test_compare_results_binary_accepts_numpy_bool():
    # numpy.bool_ inputs must score identically to native Python bools.
    assert v.compare_results_binary(np.True_, np.True_) == {"TP": 1, "TN": 0, "FP": 0, "FN": 0}
    assert v.compare_results_binary(np.False_, np.False_) == {"TP": 0, "TN": 1, "FP": 0, "FN": 0}
    assert v.compare_results_binary(np.True_, np.False_) == {"TP": 0, "TN": 0, "FP": 0, "FN": 1}
    assert v.compare_results_binary(np.False_, np.True_) == {"TP": 0, "TN": 0, "FP": 1, "FN": 0}
    # native bools still behave exactly as before
    assert v.compare_results_binary(True, True) == {"TP": 1, "TN": 0, "FP": 0, "FN": 0}
