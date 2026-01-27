import pandas as pd
import pytest
import validation as v
from utils import convert_lists
from structured import StructuredResult, StructuredField, StructuredGroup 
from utils import flatten_structured_result
pd.options.display.width = 0

_CASES = [
    { 
        "labels": {
            "flag": True,
            "fruits": ["apple", "banana"],
            "color": "red", 
            "orphan": "x",
            "raw_text": "case 0",
        }, 
        "structured_result": StructuredResult(
            groups= [
                StructuredGroup(
                    group_name="group1",
                    fields=[
                        StructuredField(name="flag", value=True, justification="", confidence=""),
                        StructuredField(name="fruits", value=["apple", "cherry"], justification="", confidence="High"),
                    ],
                ),
                StructuredGroup(
                    group_name="group two",
                    fields=[
                        StructuredField(name="color", value="Red", justification="", confidence="High"),
                    ],
                ),
            ],
        )},
    {
        "labels": {
            "flag": True,
            "fruits": ["Apple"],
            "color": "blue",
            "orphan": "y",
            "raw_text": "case 1",
        },
        "structured_result": StructuredResult(
            groups= [
            StructuredGroup(
                group_name="group1",
                fields=[
                    StructuredField(name="flag", value=False, justification="", confidence=""),
                    StructuredField(name="fruits", value=["apple", "banana"], justification="", confidence="Low"),
                ],
            ),
            StructuredGroup(
                group_name="group two",
                fields=[
                    StructuredField(name="color", value="green", justification="", confidence="Low"),
                ],
            ),
        ]),
    },
    {
        "labels": {
            "flag": False,
            "fruits": "-",
            "color": "-",
            "orphan": "z",
            "raw_text": "case 2",
        },
        "structured_result": StructuredResult(
            groups= [
            StructuredGroup(
                group_name="group1",
                fields=[
                    StructuredField(name="flag", value=True, justification="", confidence=""),
                    StructuredField(name="fruits", value=[], justification="", confidence="High"),
                ],
            ),
            StructuredGroup(
                group_name="group two",
                fields=[
                    StructuredField(name="color", value="yellow", justification="", confidence="-"),
                ],
            ),
        ]),
    },
    {
        "labels": {
            "flag": False,
            "fruits": ["cherry"],
            "color": "green",
            "orphan": "w",
            "raw_text": "case 3",
        },
        "structured_result": StructuredResult(
            groups= [
            StructuredGroup(
                group_name="group1",
                fields=[
                    StructuredField(name="flag", value=False, justification="", confidence=""),
                    StructuredField(name="fruits", value=["cherry"], justification="", confidence=None),
                ],
            ),
            StructuredGroup(
                group_name="group two",
                fields=[
                    StructuredField(name="color", value="-", justification="", confidence="High"),
                ],
            ),
        ]),
    },
    {
        "labels": {
            "flag": True,
            "fruits": [],
            "color": "-",
            "orphan": "-",
            "raw_text": "case 4",
        },
        "structured_result": StructuredResult(
            groups= [
            StructuredGroup(
                group_name="group1",
                fields=[
                    StructuredField(name="flag", value=False, justification="", confidence=""),
                    StructuredField(name="fruits", value=["apple"], justification="", confidence="NA"),
                ],
            ),
            StructuredGroup(
                group_name="group two",
                fields=[
                    StructuredField(name="color", value="", justification="", confidence=""),
                ],
            ),
        ]),
    },
    {
        "labels": {
            "flag": False,
            "fruits": ["apple"],
            "color": "-",
            "orphan": "",
            "raw_text": "case 5",
        },
        "structured_result": StructuredResult(
            groups= [
            StructuredGroup(
                group_name="group1",
                fields=[
                    StructuredField(name="flag", value=False, justification="", confidence=""),
                    StructuredField(name="fruits", value="-", justification="", confidence="Low"),
                ],
            ),
            StructuredGroup(
                group_name="group two",
                fields=[
                    StructuredField(name="color", value="-", justification="", confidence="High"),
                ],
            ),
        ]),
    },
    {
        "labels": {
            "flag": True,
            "fruits": ["apple", "banana"],
            "color": "4",
            "orphan": None,
            "raw_text": "case 6",
        },
        "structured_result": StructuredResult(
            groups= [
            StructuredGroup(
                group_name="group1",
                fields=[
                    StructuredField(name="flag", value=True, justification="", confidence=""),
                    StructuredField(name="fruits", value=["banana"], justification="", confidence="High"),
                ],
            ),
            StructuredGroup(
                group_name="group two",
                fields=[
                    StructuredField(name="color", value=4, justification="", confidence="Low"),
                ],
            ),
        ]),
    },
    {
        "labels": {
            "flag": True,
            "fruits": None,
            "color": None,
            "orphan": None,
            "raw_text": "case 7",
        },
        "structured_result": StructuredResult(
            groups= [
            StructuredGroup(
                group_name="group1",
                fields=[
                    StructuredField(name="flag", value=False, justification="", confidence=""),
                    StructuredField(name="fruits", value=["apple"], justification="", confidence="High"),
                ],
            ),
            StructuredGroup(
                group_name="group two",
                fields=[
                    StructuredField(name="color", value="red", justification="", confidence="High"),
                ],
            ),
        ]),
    },
]

def _structure_callback(row, i, raw_text_column_name):
    """
    Return the pre-built StructuredResult (list[StructuredGroup]) for case i.
    """
    res = _CASES[i]["structured_result"]
    tokens = {}

    res = flatten_structured_result(res)

    return res, {}

def _metric(metrics_df, field, name, confidence_level='Overall'):
    """Helper function to get metric value for a specific field and confidence level"""
    return metrics_df.loc[(metrics_df.field == field) & (metrics_df.confidence == confidence_level), name].iloc[0]

def test_validate_basic_no_confidence():
    source_df = pd.DataFrame([c["labels"] for c in _CASES])

    res_df, metrics_df = v.validate(
        source_df=source_df,
        fields=['flag', 'fruits', 'color'],
        structure_callback=_structure_callback,
        output_folder=None,
        raw_text_column_name='raw_text'
    )

    # --- Column order validation ---
    expected_columns = [
        # original non-label columns
        'orphan', 'raw_text',
        # flag group (binary)
        'flag', 'Res: flag', 'Res: flag confidence', 'Res: flag justification',
        'TP: flag', 'FP: flag', 'FN: flag', 'TN: flag',
        # fruits group (non-binary)
        'fruits', 'Res: fruits', 'Res: fruits confidence', 'Res: fruits justification',
        'Cor: fruits', 'Mis: fruits', 'Spu: fruits',
        'Cor: fruits items', 'Mis: fruits items', 'Spu: fruits items',
        'Precision: fruits', 'Recall: fruits', 'F1 score: fruits', 'F2 score: fruits',
        # color group (non-binary)
        'color', 'Res: color', 'Res: color confidence', 'Res: color justification',
        'Cor: color', 'Inc: color', 'Mis: color', 'Spu: color', 
        'Cor: color items', 'Inc: color items', 'Mis: color items', 'Spu: color items', 
        # system columns
        'Sys: from cache', 'Sys: exception', 'Sys: time taken'
    ]
    assert list(res_df.columns) == expected_columns

    # --- Spot check result dataframe ---
    assert res_df.loc[0, 'TP: flag'] == 1
    assert res_df.loc[1, 'FN: flag'] == 1
    assert res_df.loc[2, 'FP: flag'] == 1
    assert res_df.loc[3, 'TN: flag'] == 1

    assert res_df.loc[0, 'Cor: fruits'] == 1
    assert res_df.loc[0, 'Mis: fruits'] == 1
    assert res_df.loc[0, 'Spu: fruits'] == 1
    assert res_df.loc[0, 'Precision: fruits'] == pytest.approx(0.5)

    assert res_df.loc[0, 'Cor: color'] == 1
    assert res_df.loc[1, 'Inc: color'] == 1
    assert res_df.loc[2, 'Spu: color'] == 1
    assert res_df.loc[3, 'Mis: color'] == 1

    # --- Verify that confidence column is present in metrics ---
    assert 'confidence' in metrics_df.columns

    # --- Metrics checks for Overall confidence level ---
    assert _metric(metrics_df, 'flag', 'total cases') == 8
    assert _metric(metrics_df, 'flag', 'positive cases') == 8
    assert _metric(metrics_df, 'flag', 'TP') == 2
    assert _metric(metrics_df, 'flag', 'FP') == 1
    assert _metric(metrics_df, 'flag', 'FN') == 3
    assert _metric(metrics_df, 'flag', 'TN') == 2
    assert _metric(metrics_df, 'flag', 'precision (micro)') == pytest.approx(2/3)
    assert _metric(metrics_df, 'flag', 'recall (micro)') == pytest.approx(2/5)
    assert _metric(metrics_df, 'flag', 'F1 score (micro)') == pytest.approx(0.5)
    assert _metric(metrics_df, 'flag', 'accuracy (micro)') == pytest.approx(0.5)

    assert _metric(metrics_df, 'fruits', 'total cases') == 7
    assert _metric(metrics_df, 'fruits', 'positive cases') == 5
    assert _metric(metrics_df, 'fruits', 'cor') == 4
    assert _metric(metrics_df, 'fruits', 'mis') == 3
    assert _metric(metrics_df, 'fruits', 'spu') == 3
    assert _metric(metrics_df, 'fruits', 'precision (micro)') == pytest.approx(4/7)
    assert _metric(metrics_df, 'fruits', 'recall (micro)') == pytest.approx(4/7)
    assert _metric(metrics_df, 'fruits', 'F1 score (micro)') == pytest.approx(4/7)

    assert _metric(metrics_df, 'color', 'total cases') == 7
    assert _metric(metrics_df, 'color', 'positive cases') == 4
    assert _metric(metrics_df, 'color', 'cor') == 2
    assert _metric(metrics_df, 'color', 'inc') == 1
    assert _metric(metrics_df, 'color', 'mis') == 1
    assert _metric(metrics_df, 'color', 'spu') == 1
    assert _metric(metrics_df, 'color', 'precision (micro)') == pytest.approx(0.5)
    assert _metric(metrics_df, 'color', 'recall (micro)') == pytest.approx(0.5)
    assert _metric(metrics_df, 'color', 'F1 score (micro)') == pytest.approx(0.5)

    assert metrics_df.loc[metrics_df.field == 'exceptions', 'positive cases'].iloc[0] == 0

def test_process_all_with_cases():
    """
    Verifies process_all independently:
    - Flattens structured_result into 'Res:' columns
    - Creates confidence/justification columns only when confidence is not None
    - Preserves list / scalar / numeric types
    - No caching triggered (unique raw_text per row)
    - No exceptions raised
    - Missing confidence (None) results in NaN for that row
    - Column ordering groups label + result + confidence + justification
    """
    source_df = pd.DataFrame([c["labels"] for c in _CASES])

    res_df = v.process_all(
        source_df,
        _structure_callback,
        raw_text_column_name='raw_text',
        max_workers=1,
        use_threads=True
    )

    # --- Column order validation (process_all has no metrics yet) ---
    expected_columns = [
        'orphan', 'raw_text',
        'flag', 'Res: flag', 'Res: flag confidence', 'Res: flag justification',
        'fruits', 'Res: fruits', 'Res: fruits confidence', 'Res: fruits justification',
        'color', 'Res: color', 'Res: color confidence', 'Res: color justification',
        'Sys: from cache', 'Sys: exception', 'Sys: time taken'
    ]
    assert list(res_df.columns) == expected_columns

    # Basic shape
    assert len(res_df) == len(_CASES)

    # Columns existence
    for base in ['flag', 'fruits', 'color']:
        assert f"Res: {base}" in res_df.columns

    # Confidence columns
    assert "Res: flag confidence" in res_df.columns
    assert "Res: fruits confidence" in res_df.columns
    assert "Res: color confidence" in res_df.columns

    # Justification columns
    for base in ['flag', 'fruits', 'color']:
        assert f"Res: {base} justification" in res_df.columns

    # Row-wise value checks
    for i, case in enumerate(_CASES):
        flat = flatten_structured_result(case["structured_result"])
        assert res_df.loc[i, "Res: flag"] == flat["flag"]
        assert res_df.loc[i, "Res: fruits"] == flat["fruits"]
        assert res_df.loc[i, "Res: color"] == flat["color"]

    # Numeric preservation
    assert res_df.loc[6, "Res: color"] == 4

    # fruits confidence None -> NaN
    assert pd.isna(res_df.loc[3, "Res: fruits confidence"])

    # Other fruits confidence values
    for i in [0,1,2,4,5,6,7]:
        flat = flatten_structured_result(_CASES[i]["structured_result"])
        assert res_df.loc[i, "Res: fruits confidence"] == flat["fruits confidence"]

    # No caching
    assert not res_df["Sys: from cache"].any()

    # No exceptions
    assert res_df["Sys: exception"].isna().all()

    # Time taken column present and non-negative
    assert "Sys: time taken" in res_df.columns
    assert (res_df["Sys: time taken"] >= 0).all()


def test_empty_list():
    """
    Mimic the real flow:
      - labels (expected) come in as "[]" like from CSV, then convert_lists -> []
      - structure_callback builds StructuredResult with value=[] (a real list)
      - flatten_structured_result runs inside the callback and (bug) turns [] -> "[]"
    The test FAILS if the (actual) result is still a string, and also verifies no Spurious/Missing.
    """

    # Labels simulate CSV input: "[]" -> convert_lists(source_df) will turn it into []
    source_df = pd.DataFrame([{"empty_list": "[]", "raw_text": "case_empty"}])
    source_df = convert_lists(source_df)

    # Use a callback that mirrors production behavior:
    # build a StructuredResult with value=[], then flatten (this is where bug happens).
    def cb_realistic(row, i, raw_text_column_name):
        sr = StructuredResult(
            groups=[
                StructuredGroup(
                    group_name="g",
                    fields=[
                        # LLM return a string empty list "[]"
                        StructuredField(name="empty_list", value="[]", justification="", confidence="High"),
                    ],
                )
            ]
        )
        # This uses your actual flattening logic (which currently can turn [] -> "[]")
        flat = flatten_structured_result(sr)
        return flat, {}

    # Run full validate pipeline (convert_lists -> process_all -> compare_results_all -> metrics)
    res_df, _ = v.validate(
        source_df=source_df,
        fields=["empty_list"],
        structure_callback=cb_realistic,
        output_folder=None,
        raw_text_column_name="raw_text",
    )

    human_label = res_df.at[0, "empty_list"]        # after convert_lists: should be []
    llm_output   = res_df.at[0, "Res: empty_list"]   # should ALSO be a real list [], not "[]"

    # 1) Catch the bug: actual MUST be a list (will FAIL if flatten turned [] into "[]")
    assert isinstance(human_label, list), f"Human Label not normalized to list: {human_label!r} ({type(human_label).__name__})"
    assert isinstance(llm_output, list),   f"LLM output not normalized to list (flatten may have stringified it): {llm_output!r} ({type(llm_output).__name__})"

    # 2) Make sure not mark as  spurious
    assert res_df.at[0, "Spu: empty_list"] == 0, "Spurious flagged for empty list"
    assert res_df.at[0, "Mis: empty_list"] == 0, "Missing flagged for empty list"

def test_validate_with_none_structure_callback():
    """
    Test validate function when structure_callback is None.
    This case assumes that source_df already contains 'Res: ' columns.
    """
    # First, create a DataFrame with labels
    source_df = pd.DataFrame([c["labels"] for c in _CASES])
    
    # Run the normal validation to get the results with 'Res: ' columns
    res_df_with_callback, _ = v.validate(
        source_df=source_df,
        fields=['flag', 'fruits', 'color'],
        structure_callback=_structure_callback,
        output_folder=None,
        raw_text_column_name='raw_text'
    )
    
    # Extract only the original labels and the 'Res: ' columns to simulate 
    # a DataFrame that already has results
    source_with_results = source_df.copy()
    
    # Add the 'Res: ' columns from the previous run
    res_columns = [col for col in res_df_with_callback.columns if col.startswith('Res: ')]
    for col in res_columns:
        source_with_results[col] = res_df_with_callback[col]
    
    # Now test validate with structure_callback=None
    res_df, metrics_df = v.validate(
        source_df=source_with_results,
        fields=['flag', 'fruits', 'color'],
        structure_callback=None,  # This is the key test case
        output_folder=None,
        raw_text_column_name='raw_text'
    )

    # --- Column order validation ---
    expected_columns = [
        # original non-label columns
        'orphan', 'raw_text',
        # flag group (binary)
        'flag', 'Res: flag', 'Res: flag confidence', 'Res: flag justification',
        'TP: flag', 'FP: flag', 'FN: flag', 'TN: flag',
        # fruits group (non-binary)
        'fruits', 'Res: fruits', 'Res: fruits confidence', 'Res: fruits justification',
        'Cor: fruits', 'Mis: fruits', 'Spu: fruits',
        'Cor: fruits items', 'Mis: fruits items', 'Spu: fruits items',
        'Precision: fruits', 'Recall: fruits', 'F1 score: fruits', 'F2 score: fruits',
        # color group (non-binary)
        'color', 'Res: color', 'Res: color confidence', 'Res: color justification',
        'Cor: color', 'Inc: color', 'Mis: color', 'Spu: color', 
        'Cor: color items', 'Inc: color items', 'Mis: color items', 'Spu: color items'
    ]
    assert list(res_df.columns) == expected_columns

    # --- Verify that results are identical to when using structure_callback ---
    # The comparison and metrics should be the same since we're using the same 'Res: ' columns
    
    # Spot check result dataframe - should match test_validate_basic_no_confidence
    assert res_df.loc[0, 'TP: flag'] == 1
    assert res_df.loc[1, 'FN: flag'] == 1
    assert res_df.loc[2, 'FP: flag'] == 1
    assert res_df.loc[3, 'TN: flag'] == 1

    assert res_df.loc[0, 'Cor: fruits'] == 1
    assert res_df.loc[0, 'Mis: fruits'] == 1
    assert res_df.loc[0, 'Spu: fruits'] == 1
    assert res_df.loc[0, 'Precision: fruits'] == pytest.approx(0.5)

    assert res_df.loc[0, 'Cor: color'] == 1
    assert res_df.loc[1, 'Inc: color'] == 1
    assert res_df.loc[2, 'Spu: color'] == 1
    assert res_df.loc[3, 'Mis: color'] == 1

    # --- Verify that confidence column is present in metrics ---
    assert 'confidence' in metrics_df.columns

    # --- Metrics checks - should match test_validate_basic_no_confidence ---
    assert _metric(metrics_df, 'flag', 'total cases') == 8
    assert _metric(metrics_df, 'flag', 'positive cases') == 8
    assert _metric(metrics_df, 'flag', 'TP') == 2
    assert _metric(metrics_df, 'flag', 'FP') == 1
    assert _metric(metrics_df, 'flag', 'FN') == 3
    assert _metric(metrics_df, 'flag', 'TN') == 2
    assert _metric(metrics_df, 'flag', 'precision (micro)') == pytest.approx(2/3)
    assert _metric(metrics_df, 'flag', 'recall (micro)') == pytest.approx(2/5)
    assert _metric(metrics_df, 'flag', 'F1 score (micro)') == pytest.approx(0.5)
    assert _metric(metrics_df, 'flag', 'accuracy (micro)') == pytest.approx(0.5)

    assert _metric(metrics_df, 'fruits', 'total cases') == 7
    assert _metric(metrics_df, 'fruits', 'positive cases') == 5
    assert _metric(metrics_df, 'fruits', 'cor') == 4
    assert _metric(metrics_df, 'fruits', 'mis') == 3
    assert _metric(metrics_df, 'fruits', 'spu') == 3
    assert _metric(metrics_df, 'fruits', 'precision (micro)') == pytest.approx(4/7)
    assert _metric(metrics_df, 'fruits', 'recall (micro)') == pytest.approx(4/7)
    assert _metric(metrics_df, 'fruits', 'F1 score (micro)') == pytest.approx(4/7)

    assert _metric(metrics_df, 'color', 'total cases') == 7
    assert _metric(metrics_df, 'color', 'positive cases') == 4
    assert _metric(metrics_df, 'color', 'cor') == 2
    assert _metric(metrics_df, 'color', 'inc') == 1
    assert _metric(metrics_df, 'color', 'mis') == 1
    assert _metric(metrics_df, 'color', 'spu') == 1
    assert _metric(metrics_df, 'color', 'precision (micro)') == pytest.approx(0.5)
    assert _metric(metrics_df, 'color', 'recall (micro)') == pytest.approx(0.5)
    assert _metric(metrics_df, 'color', 'F1 score (micro)') == pytest.approx(0.5)

    # --- Verify that no system columns from process_all are present ---
    # Since structure_callback was None, process_all was not called, so these shouldn't exist
    assert 'Sys: from cache' not in res_df.columns
    assert 'Sys: exception' not in res_df.columns  
    assert 'Sys: time taken' not in res_df.columns
     
def test_metrics_with_confidence_levels():
    """
    Test that get_metrics generates metrics for all available confidence levels and overall.
    """
    source_df = pd.DataFrame([c["labels"] for c in _CASES])

    res_df, metrics_df = v.validate(
        source_df=source_df,
        fields=['flag', 'fruits', 'color'],
        structure_callback=_structure_callback,
        output_folder=None,
        raw_text_column_name='raw_text'
    )

    # --- Verify that confidence column is present ---
    assert 'confidence' in metrics_df.columns

    # --- Verify that Overall confidence level is present for all fields ---
    assert len(metrics_df[metrics_df.confidence == 'Overall']) > 0
    
    # Check that each field has an Overall entry
    for field in ['flag', 'fruits', 'color']:
        overall_rows = metrics_df[(metrics_df.field == field) & (metrics_df.confidence == 'Overall')]
        assert len(overall_rows) == 1, f"Field {field} should have exactly one Overall entry"

    # --- Verify specific confidence levels are present if they exist in the data ---
    # Our test data includes 'High', 'Low', etc. confidence levels
    confidence_levels = metrics_df['confidence'].dropna().unique()
    assert 'Overall' in confidence_levels
    
    # Check for specific confidence levels that should be in our test data
    expected_confidence_levels = ['High', 'Low']
    for conf_level in expected_confidence_levels:
        if conf_level in confidence_levels:
            # Verify that at least some fields have metrics for this confidence level
            conf_rows = metrics_df[metrics_df.confidence == conf_level]
            assert len(conf_rows) > 0, f"Should have at least some rows for confidence level {conf_level}"

def test_reorder_result_columns():
    """Test that _reorder_result_columns groups columns with the same base name together."""
    
    # Create test data with columns that should be grouped
    test_columns = [
        'Patient ID',
        'Document ID', 
        'Document Text',
        'Document Date',
        'First Primary Diagnosis Matches Label',
        'First Primary Diagnosis Correct?',
        'First Primary Diagnosis Comment',
        'First Primary Histology Matches Label',
        'First Primary Histology Correct?',
        'First Primary Histology Comment',
        'Treatment Drugs Matches Label',
        'Differences Count Treatment Drugs - Treatment Drugs Label',
        'Treatment Drugs Correct?',
        'Treatment Drugs Comment',
        'First Primary Diagnosis',
        'Res: First Primary Diagnosis',
        'Res: First Primary Diagnosis confidence',
        'Res: First Primary Diagnosis justification',
        'Cor: First Primary Diagnosis',
        'Inc: First Primary Diagnosis',
        'Mis: First Primary Diagnosis',
        'Spu: First Primary Diagnosis',
        'Cor: First Primary Diagnosis items',
        'Inc: First Primary Diagnosis items',
        'Mis: First Primary Diagnosis items',
        'Spu: First Primary Diagnosis items',
        'First Primary Histology',
        'Res: First Primary Histology',
        'Res: First Primary Histology confidence',
        'Res: First Primary Histology justification',
        'Cor: First Primary Histology',
        'Inc: First Primary Histology',
        'Mis: First Primary Histology',
        'Spu: First Primary Histology',
        'Cor: First Primary Histology items',
        'Inc: First Primary Histology items',
        'Mis: First Primary Histology items',
        'Spu: First Primary Histology items',
        'Treatment Drugs',
        'Res: Treatment Drugs',
        'Cor: Treatment Drugs',
        'Mis: Treatment Drugs',
        'Spu: Treatment Drugs',
        'Cor: Treatment Drugs items',
        'Mis: Treatment Drugs items',
        'Spu: Treatment Drugs items',
        'Precision: Treatment Drugs',
        'Recall: Treatment Drugs',
        'F1 score: Treatment Drugs',
        'F2 score: Treatment Drugs',
        'Sys: exception',
        'Sys: time taken'
    ]
    
    # Create DataFrame with test data
    test_data = {col: [0] for col in test_columns}
    df = pd.DataFrame(test_data)
    
    # Apply the reordering function
    reordered_df = v._reorder_result_columns(df)
    reordered_columns = list(reordered_df.columns)
    
    # Expected order: ungrouped columns first, then grouped by base name
    expected_order = [
        # Ungrouped columns (don't start with any base name)
        'Patient ID',
        'Document ID', 
        'Document Text',
        'Document Date',
        'Differences Count Treatment Drugs - Treatment Drugs Label',
        
        # First Primary Diagnosis group
        'First Primary Diagnosis Matches Label',
        'First Primary Diagnosis Correct?',
        'First Primary Diagnosis Comment',
        'First Primary Diagnosis',
        'Res: First Primary Diagnosis',
        'Res: First Primary Diagnosis confidence',
        'Res: First Primary Diagnosis justification',
        'Cor: First Primary Diagnosis',
        'Inc: First Primary Diagnosis',
        'Mis: First Primary Diagnosis',
        'Spu: First Primary Diagnosis',
        'Cor: First Primary Diagnosis items',
        'Inc: First Primary Diagnosis items',
        'Mis: First Primary Diagnosis items',
        'Spu: First Primary Diagnosis items',
        
        # First Primary Histology group
        'First Primary Histology Matches Label',
        'First Primary Histology Correct?',
        'First Primary Histology Comment',
        'First Primary Histology',
        'Res: First Primary Histology',
        'Res: First Primary Histology confidence',
        'Res: First Primary Histology justification',
        'Cor: First Primary Histology',
        'Inc: First Primary Histology',
        'Mis: First Primary Histology',
        'Spu: First Primary Histology',
        'Cor: First Primary Histology items',
        'Inc: First Primary Histology items',
        'Mis: First Primary Histology items',
        'Spu: First Primary Histology items',
        
        # Treatment Drugs group
        'Treatment Drugs Matches Label',
        'Treatment Drugs Correct?',
        'Treatment Drugs Comment',
        'Treatment Drugs',
        'Res: Treatment Drugs',
        'Cor: Treatment Drugs',
        'Mis: Treatment Drugs',
        'Spu: Treatment Drugs',
        'Cor: Treatment Drugs items',
        'Mis: Treatment Drugs items',
        'Spu: Treatment Drugs items',
        'Precision: Treatment Drugs',
        'Recall: Treatment Drugs',
        'F1 score: Treatment Drugs',
        'F2 score: Treatment Drugs',
        
        # System columns
        'Sys: exception',
        'Sys: time taken'
    ]
    
    # Verify the order matches expectation
    assert reordered_columns == expected_order, f"Column order mismatch.\nExpected: {expected_order}\nActual: {reordered_columns}"
    
    # Verify that related columns are grouped together
    def find_column_index(col_name):
        return reordered_columns.index(col_name)
    
    # Test First Primary Diagnosis group
    diagnosis_base_idx = find_column_index('First Primary Diagnosis')
    diagnosis_matches_idx = find_column_index('First Primary Diagnosis Matches Label')
    diagnosis_correct_idx = find_column_index('First Primary Diagnosis Correct?')
    diagnosis_comment_idx = find_column_index('First Primary Diagnosis Comment')
    
    # All related columns should come before the base column
    assert diagnosis_matches_idx < diagnosis_base_idx, "Related columns should come before base column"
    assert diagnosis_correct_idx < diagnosis_base_idx, "Related columns should come before base column"
    assert diagnosis_comment_idx < diagnosis_base_idx, "Related columns should come before base column"
    
    # Related columns should be consecutive
    assert diagnosis_correct_idx == diagnosis_matches_idx + 1, "Related columns should be consecutive"
    assert diagnosis_comment_idx == diagnosis_correct_idx + 1, "Related columns should be consecutive"
    
    # Test Treatment Drugs group
    drugs_base_idx = find_column_index('Treatment Drugs')
    drugs_matches_idx = find_column_index('Treatment Drugs Matches Label')
    drugs_correct_idx = find_column_index('Treatment Drugs Correct?')
    drugs_comment_idx = find_column_index('Treatment Drugs Comment')
    
    # All related columns should come before the base column
    assert drugs_matches_idx < drugs_base_idx, "Related columns should come before base column"
    assert drugs_correct_idx < drugs_base_idx, "Related columns should come before base column"
    assert drugs_comment_idx < drugs_base_idx, "Related columns should come before base column"
    
    # Related columns should be consecutive
    assert drugs_correct_idx == drugs_matches_idx + 1, "Related columns should be consecutive"
    assert drugs_comment_idx == drugs_correct_idx + 1, "Related columns should be consecutive"
    
    print("✓ All tests passed! Column reordering works correctly.")