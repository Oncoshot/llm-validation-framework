import math
import pandas as pd
import pytest
import src.validation as v
pd.options.display.width = 0

def get_test_df(addconfidence):
    flag =        [True,  True,  False, False, True,  False, True, True]
    res_flag =    [True,  False, True,  False, False, False, True, False]

    fruits = [
        ['apple', 'banana'],
        ['apple'],
        '-',
        ['cherry'],
        [],
        ['apple'],
        ['apple', 'banana'],
        None
    ]
    res_fruits = [
        ['apple', 'cherry'],
        ['apple', 'banana'],
        [],
        ['cherry'],
        ['apple'],
        '-',
        ['banana'],
        ['apple']
    ]
    res_fruits_confidence = [
        'High',
        'Low',
        'High',
        None,
        'NA',
        'Low',
        'High',
        'High'
    ]

    color = [
        'red',
        'blue',
        '-',
        'green',
        '-',
        '-',
        '4',
        None
    ]
    res_color = [
        'red',
        'green',
        'yellow',
        '-',
        '',
        '-',
        4,
        'red'
    ]
    res_color_confidence = [
        'High',
        'Low',
        '-',
        'High',
        '',
        'High',
        'Low',
        'High'
    ]

    orphan = ['x','y','z','w','-','',None, None]

    df = pd.DataFrame({
        'flag': flag,
        'Res: flag': res_flag,
        'fruits': fruits,
        'Res: fruits': res_fruits,
        'orphan': orphan,
        'color': color,
        'Res: color': res_color
    })

    if addconfidence:
        # Insert after 'Res: fruits'
        pos_fruits = df.columns.get_loc('Res: fruits')
        df.insert(pos_fruits + 1, 'Res: fruits confidence', res_fruits_confidence)
        # Insert after 'Res: color'
        pos_color = df.columns.get_loc('Res: color')  # recompute after previous insert
        df.insert(pos_color + 1, 'Res: color confidence', res_color_confidence)

    return df

def _is_none_or_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def test_compare_results_all_mixed_fields():
    df = get_test_df(False)

    res_df = v.compare_results_all(df, ['flag', 'fruits', 'color'])

    # ---- Binary field assertions (flag) ----
    # Row 0: TP
    assert res_df.loc[0, 'TP: flag'] == 1
    # Row 1: FN
    assert res_df.loc[1, 'FN: flag'] == 1
    assert res_df.loc[2, 'FP: flag'] == 1
    assert res_df.loc[3, 'TN: flag'] == 1

    # ---- List field assertions (fruits) ----
    # Row 0 mixed
    assert res_df.loc[0, 'Cor: fruits'] == 1
    assert res_df.loc[0, 'Mis: fruits'] == 1
    assert res_df.loc[0, 'Spu: fruits'] == 1
    assert res_df.loc[0, 'Precision: fruits'] == pytest.approx(0.5)
    assert res_df.loc[0, 'Recall: fruits'] == pytest.approx(0.5)
    assert res_df.loc[0, 'F1 score: fruits'] == pytest.approx(0.5)

    # Row 1: one correct + one spurious
    assert res_df.loc[1, 'Cor: fruits'] == 1
    assert res_df.loc[1, 'Spu: fruits'] == 1
    assert res_df.loc[1, 'Precision: fruits'] == pytest.approx(0.5)
    assert res_df.loc[1, 'Recall: fruits'] == pytest.approx(1.0)

    # Row 2: expected '-' vs [] => zeros, metrics NaN
    assert res_df.loc[2, 'Cor: fruits'] == 0
    assert math.isnan(res_df.loc[2, 'Precision: fruits'])

    # Row 3: perfect
    assert res_df.loc[3, 'Cor: fruits'] == 1
    assert res_df.loc[3, 'Precision: fruits'] == pytest.approx(1.0)

    # Row 4: expected empty list, actual has item -> spurious
    assert res_df.loc[4, 'Spu: fruits'] == 1
    assert res_df.loc[4, 'Mis: fruits'] == 0

    # Row 5: expected ['apple'], actual '-' (empty) -> missing
    assert res_df.loc[5, 'Mis: fruits'] == 1
    assert res_df.loc[5, 'Spu: fruits'] == 0

    # ---- Scalar non-binary field assertions (color) ----
    assert res_df.loc[0, 'Cor: color'] == 1          # correct
    assert res_df.loc[1, 'Inc: color'] == 1          # incorrect
    assert res_df.loc[2, 'Spu: color'] == 1          # spurious
    assert res_df.loc[3, 'Mis: color'] == 1          # missing
    # Rows 4 & 5: both sides empty label cases ('-' and ''), treated as labeled empty -> zeros + NaN metrics
    assert res_df.loc[4, 'Cor: color'] == 0
    assert res_df.loc[5, 'Cor: color'] == 0
    # Row 6: numeric string vs number -> match
    assert res_df.loc[6, 'Cor: color'] == 1
    assert res_df.loc[6, 'Inc: color'] == 0

    # Ensure orphan column passed through unchanged
    assert 'orphan' in res_df.columns

    expected_columns = [
        'TP: flag','TN: flag','FP: flag','FN: flag',
        'Cor: fruits','Mis: fruits','Spu: fruits',
        'Precision: fruits','Recall: fruits','F1 score: fruits',
        'Cor: color','Inc: color','Mis: color','Spu: color'
    ]
    for col in expected_columns:
        assert col in res_df.columns, f"Missing column {col} in compare_results_all output"

