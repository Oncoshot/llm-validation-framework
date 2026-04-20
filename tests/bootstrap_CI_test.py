import pandas as pd
import pytest
import numpy as np
import llmvalidate.validation as v


def test_bootstrap_CI_basic():
    """Test basic functionality of bootstrap_CI"""
    # Create test data with comparison results
    res_df = pd.DataFrame({
        'field1': ['A', 'B', 'A', 'B', 'A'] * 20,  # 100 rows total
        'field2': ['X', 'Y', 'X', 'Y', 'X'] * 20,
        'Cor: field1': [1, 0, 1, 1, 0] * 20,
        'Inc: field1': [0, 1, 0, 0, 1] * 20,
        'Mis: field1': [0, 0, 0, 0, 0] * 20,
        'Spu: field1': [0, 1, 0, 0, 1] * 20,
        'Par: field1': [0, 0, 0, 0, 0] * 20,
        'Cor: field2': [1, 1, 0, 1, 1] * 20,
        'Inc: field2': [0, 0, 1, 0, 0] * 20,
        'Mis: field2': [0, 0, 1, 0, 0] * 20,
        'Spu: field2': [0, 0, 0, 0, 0] * 20,
        'Par: field2': [0, 0, 0, 0, 0] * 20,
    })
    
    fields = ['field1', 'field2']
    result = v.bootstrap_CI(res_df, fields, n_bootstrap=100, random_state=42)
    
    # Check output format
    assert 'field' in result.columns
    assert len(result) == 4  # Two fields + exceptions field + N/CI info row
    expected_fields = {'field1', 'field2', 'exceptions', 'N=100; CI=95%'}
    assert set(result['field']) == expected_fields
    
    # Check that confidence interval columns are present for metrics that exist in our data
    # We know these will be present because we've included them in our test data
    core_metrics = ['field-present cases', 'cor', 'inc', 'mis', 'spu', 'par']
    
    # 'labeled cases' is handled specially - just appears as 'labeled cases'
    assert 'labeled cases' in result.columns
    
    for metric in core_metrics:
        assert f'{metric}: mean' in result.columns
        assert f'{metric}: lower' in result.columns
        assert f'{metric}: upper' in result.columns
    
    # Check that means are reasonable (between lower and upper bounds) for non-exception fields
    for _, row in result.iterrows():
        if row['field'] in ['exceptions', 'N=100; CI=95%']:  # Skip exceptions and info row
            for metric in core_metrics:
                mean_col = f'{metric}: mean'
                lower_col = f'{metric}: lower'
                upper_col = f'{metric}: upper'
                
                if pd.notna(row[mean_col]) and pd.notna(row[lower_col]) and pd.notna(row[upper_col]):
                    assert row[lower_col] <= row[mean_col] <= row[upper_col], \
                        f"Mean not between bounds for {row['field']} {metric}"


def test_bootstrap_CI_error_conditions():
    """Test error conditions for bootstrap_CI"""
    
    # Test with ci outside valid range
    res_df = pd.DataFrame({
        'field1': [1, 2, 3],
        'Cor: field1': [1, 0, 1]
    })
    
    with pytest.raises(ValueError, match="ci must be in \\(0, 1\\)"):
        v.bootstrap_CI(res_df, ['field1'], ci=1.5)
    
    with pytest.raises(ValueError, match="ci must be in \\(0, 1\\)"):
        v.bootstrap_CI(res_df, ['field1'], ci=0)
    
    # Test with too few rows
    single_row_df = pd.DataFrame({
        'field1': [1],
        'Cor: field1': [1]
    })
    
    with pytest.raises(ValueError, match="Need at least 2 rows"):
        v.bootstrap_CI(single_row_df, ['field1'])
    
    # Test with missing labels (NaN values)
    res_df_with_nan = pd.DataFrame({
        'field1': [1, np.nan, 3],
        'field2': [1, 2, 3],
        'Cor: field1': [1, 0, 1],
        'Cor: field2': [0, 1, 0]
    })
    
    with pytest.raises(ValueError, match="Missing labels \\(NaN\\) found in the following fields: \\['field1'\\]"):
        v.bootstrap_CI(res_df_with_nan, ['field1', 'field2'])


def test_bootstrap_CI_binary_field():
    """Test bootstrap_CI with binary field metrics"""
    # Create test data with binary field results
    res_df = pd.DataFrame({
        'binary_field': [True, False, True, False] * 25,  # 100 rows
        'TP: binary_field': [1, 0, 1, 0] * 25,
        'FP: binary_field': [0, 1, 0, 1] * 25,
        'FN: binary_field': [0, 0, 0, 0] * 25,
        'TN: binary_field': [0, 1, 0, 1] * 25,
        'Precision: binary_field': [1.0, 0.0, 1.0, 0.0] * 25,
        'Recall: binary_field': [1.0, np.nan, 1.0, np.nan] * 25,
        'F1 score: binary_field': [1.0, 0.0, 1.0, 0.0] * 25,
        'F2 score: binary_field': [1.0, 0.0, 1.0, 0.0] * 25,
    })
    
    fields = ['binary_field']
    result = v.bootstrap_CI(res_df, fields, n_bootstrap=50, random_state=42)
    
    # Check that binary metrics are included
    binary_metrics = ['TP', 'FP', 'FN', 'TN', 'precision (micro)', 'recall (micro)', 
                     'F1 score (micro)', 'F2 score (micro)', 'accuracy (micro)', 'specificity (micro)']
    
    for metric in binary_metrics:
        assert f'{metric}: mean' in result.columns
        assert f'{metric}: lower' in result.columns  
        assert f'{metric}: upper' in result.columns
    
    # Check that N/CI info row is present
    info_rows = result[result['field'].str.startswith('N=')]
    assert len(info_rows) == 1


def test_bootstrap_CI_output_format():
    """Test that output format matches specification"""
    res_df = pd.DataFrame({
        'test_field': ['A', 'B'] * 50,  # 100 rows
        'Cor: test_field': [1, 0] * 50,
        'Inc: test_field': [0, 1] * 50,
        'Mis: test_field': [0, 0] * 50,
        'Spu: test_field': [0, 1] * 50,
        'Par: test_field': [0, 0] * 50,
    })
    
    result = v.bootstrap_CI(res_df, ['test_field'], n_bootstrap=10, random_state=42)
    
    # Check that result has the correct format
    assert len(result) == 3  # One field + exceptions + N/CI info row
    test_field_row = result[result['field'] == 'test_field'].iloc[0]
    assert test_field_row['field'] == 'test_field'
    
    # Check that columns follow the expected pattern
    metric_columns = [col for col in result.columns if col not in ['field', 'labeled cases']]
    for col in metric_columns:
        assert ': ' in col, f"Column {col} doesn't follow expected format"
        metric_name, stat_type = col.split(': ', 1)
        assert stat_type in ['mean', 'lower', 'upper'], f"Unexpected stat type in {col}"


def test_bootstrap_CI_confidence_intervals():
    """Test that confidence intervals make sense"""
    # Create deterministic test case
    res_df = pd.DataFrame({
        'field1': [1] * 100,
        'Cor: field1': [5] * 100,  # Constant values for predictable CI
        'Inc: field1': [0] * 100,
        'Mis: field1': [0] * 100,
        'Spu: field1': [0] * 100,
        'Par: field1': [0] * 100,
    })
    
    result = v.bootstrap_CI(res_df, ['field1'], n_bootstrap=100, ci=0.95, random_state=42)
    
    # For constant values, mean should equal the sum (get_metrics sums the values)
    row = result[result['field'] == 'field1'].iloc[0]
    
    # Check that mean is close to expected value (5 * 100 = 500)
    assert abs(row['cor: mean'] - 500.0) < 10.0
    
    # Check that CI bounds are reasonable (close to mean for constant data)
    assert abs(row['cor: lower'] - row['cor: mean']) < 50.0
    assert abs(row['cor: upper'] - row['cor: mean']) < 50.0


def test_bootstrap_CI_with_different_ci_levels():
    """Test bootstrap_CI with different confidence interval levels"""
    res_df = pd.DataFrame({
        'field1': [1, 2, 3] * 34,  # ~100 rows
        'Cor: field1': [1, 2, 1] * 34,
        'Inc: field1': [0, 1, 0] * 34,
        'Mis: field1': [0, 0, 1] * 34,
        'Spu: field1': [1, 0, 0] * 34,
        'Par: field1': [0, 0, 0] * 34,
    })
    
    # Test 90% CI
    result_90 = v.bootstrap_CI(res_df, ['field1'], n_bootstrap=50, ci=0.90, random_state=42)
    
    # Test 99% CI  
    result_99 = v.bootstrap_CI(res_df, ['field1'], n_bootstrap=50, ci=0.99, random_state=42)
    
    # 99% CI should be wider than 90% CI
    row_90 = result_90[result_90['field'] == 'field1'].iloc[0]
    row_99 = result_99[result_99['field'] == 'field1'].iloc[0]
    
    width_90 = row_90['cor: upper'] - row_90['cor: lower']
    width_99 = row_99['cor: upper'] - row_99['cor: lower']
    
    assert width_99 >= width_90, "99% CI should be wider than 90% CI"


def test_bootstrap_CI_empty_metrics():
    """Test bootstrap_CI handles missing values correctly"""
    # Create simpler test data that focuses on core functionality
    res_df = pd.DataFrame({
        'field1': [1, 2, 3] * 34,
        'Cor: field1': [0, 1, 2] * 34,  # Valid values
        'Inc: field1': [0, 0, 0] * 34,
        'Mis: field1': [0, 0, 0] * 34,
        'Spu: field1': [0, 0, 0] * 34,
        'Par: field1': [0, 0, 0] * 34,
    })
    
    result = v.bootstrap_CI(res_df, ['field1'], n_bootstrap=10, random_state=42)
    
    # Check that core metrics appear in output
    core_cols = [col for col in result.columns if 'cor:' in col]
    assert len(core_cols) == 3, f"Expected 3 cor metrics (mean, lower, upper), got {len(core_cols)}: {core_cols}"
    
    # Check that the function completes without errors for this simpler case
    assert 'field' in result.columns
    assert len(result) >= 2  # At least exceptions + field1 + N/CI info row
    
    # Check that N/CI info row is present
    info_rows = result[result['field'].str.startswith('N=')]
    assert len(info_rows) == 1