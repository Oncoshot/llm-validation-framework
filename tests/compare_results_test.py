from numpy import nan
import validation as v
import pytest
import math

@pytest.mark.parametrize("expected, actual, parents, expected_output", [
    # scalar values
    #   expected is not labeled so it does not make sense to compare results
    (None, "", {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (None, None, {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (math.nan, None, {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (math.nan, "", {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    ("", "4", {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (None, "4", {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    ("", "", {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    #   expected is "No information"
    ("-", "", {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ("-", "-", {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ("-", "4", {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [4.0], 'Partial': []}),
    #   both values non-empty
    ("female", "female", {}, {'Correct': ["female"], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ("Female", "female", {}, {'Correct': ["female"], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    (1, 1, {}, {'Correct': [1], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    (4, 4.0, {}, {'Correct': [4.0], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    (4, 4.01, {}, {'Correct': [], 'Incorrect': [4.01], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ("4", 4, {}, {'Correct': [4.0], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ("4", 4.01, {}, {'Correct': [], 'Incorrect': [4.01], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ("4", "4.0", {}, {'Correct': [4.0], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ('apple', 'tree', { 'Apple': 'Tree' }, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': ['apple']}),
    ('tree', 'apple', { 'Apple': 'Tree' }, {'Correct': [], 'Incorrect': ['apple'], 'Missing': [], 'Spurious': [], 'Partial': []}),
    #   expected is not empty but actual is empty
    ("female", nan, {}, {'Correct': [], 'Incorrect': [], 'Missing': ["female"], 'Spurious': [], 'Partial': []}),
    ("female", None, {}, {'Correct': [], 'Incorrect': [], 'Missing': ["female"], 'Spurious': [], 'Partial': []}),
    ("female", "", {}, {'Correct': [], 'Incorrect': [], 'Missing': ["female"], 'Spurious': [], 'Partial': []}),
    (4, nan, {}, {'Correct': [], 'Incorrect': [], 'Missing': [4], 'Spurious': [], 'Partial': []}),
    (4, None, {}, {'Correct': [], 'Incorrect': [], 'Missing': [4], 'Spurious': [], 'Partial': []}),
    (4, "", {}, {'Correct': [], 'Incorrect': [], 'Missing': [4], 'Spurious': [], 'Partial': []}),

    # list values
    #   expected is not labeled so it does not make sense to compare results
    (None, [], {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (None, None, {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (None, ["4"], {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (math.nan, None, {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (math.nan, [], {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    (math.nan, ["4"], {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    ("", "", {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    ("", ["4"], {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    ("", [], {}, {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}),
    #   both values are empty
    #       both lists are empty
    ([], [], {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    #       empty list and empty value
    ([], "", {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ("-", "", {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    #   both values non-empty
    ([4], [4], {}, {'Correct': [4.0], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    ([4, 2, 1], [4, 3, 1], {}, {'Correct': [1.0, 4.0], 'Incorrect': [], 'Missing': [2.0], 'Spurious': [3.0], 'Partial': []}),
    (4, [4], {}, {'Correct': [4.0], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    (4, [4.01], {}, {'Correct': [], 'Incorrect': [], 'Missing': [4.0], 'Spurious': [4.01], 'Partial': []}),
    ("BRCA1", ["BRCA1"], {}, {'Correct': ["brca1"], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    (['apple', 'Banana', 'cherry'], ['banana', 'Cherry', 'Date'], {}, {'Correct': ["banana", "cherry"], 'Incorrect': [], 'Missing': ["apple"], 'Spurious': ["date"], 'Partial': []}),
    (['apple', 'banana'], ['Apple', 'Banana', 'cherry'], {}, {'Correct': ["apple", "banana"], 'Incorrect': [], 'Missing': [], 'Spurious': ["cherry"], 'Partial': []}),
    (['apple', 'banana', 'egg'], ['tree', 'Banana', 'cherry'], { 'Apple': 'Tree' }, {'Correct': ["banana"], 'Incorrect': [], 'Missing': ["egg"], 'Spurious': ["cherry"], 'Partial': ["apple"]}),
    (['tree', 'Banana', 'cherry'], ['apple', 'banana', 'egg'], { 'Apple': 'Tree' }, {'Correct': ["banana"], 'Incorrect': ["apple"], 'Missing': ["cherry"], 'Spurious': ["egg"], 'Partial': []}),
    #   (different case)
    (["BRCA1"], ["BrCA1"], {}, {'Correct': ["brca1"], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    #   (different case and order)
    (['ER Negative', 'PR Negative', 'Her2 Negative'], ["PR Negative", "ER negative", "Her2 Negative"], {}, {'Correct': ["er negative", "her2 negative", "pr negative"], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    #   (duplicates)
    (['Lung Cancer', "Pancreatic Cancer"], ['Lung Cancer', "Pancreatic Cancer",  "Pancreatic Cancer"], {}, {'Correct': ["lung cancer", "pancreatic cancer"], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    (['Lung Cancer', "Pancreatic Cancer",  "Pancreatic Cancer"], ['Lung Cancer', "Pancreatic Cancer"], {}, {'Correct': ["lung cancer", "pancreatic cancer"], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}),
    #   one value is empty while another one is non-empty
    ([], [4], {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [4.0], 'Partial': []}),
    ("-", [4], {}, {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [4.0], 'Partial': []}),
    ("[]", [], {}, {'Correct': [], 'Incorrect': [], 'Missing': ["[]"], 'Spurious': [], 'Partial': []}),
    ([4], "", {}, {'Correct': [], 'Incorrect': [], 'Missing': [4.0], 'Spurious': [], 'Partial': []}),
    ([4], [], {}, {'Correct': [], 'Incorrect': [], 'Missing': [4.0], 'Spurious': [], 'Partial': []}),
    ([4], math.nan, {}, {'Correct': [], 'Incorrect': [], 'Missing': [4.0], 'Spurious': [], 'Partial': []}),
    ([4], None, {}, {'Correct': [], 'Incorrect': [], 'Missing': [4.0], 'Spurious': [], 'Partial': []}),
])
def test_compare_results(expected, actual, parents, expected_output):
    result = v.compare_results(expected, actual, parents)
    
    # For list comparisons, we need to sort the lists to compare them properly since order doesn't matter
    if result['Correct'] is not None:
        for key in result:
            if isinstance(result[key], list):
                result[key] = sorted(result[key])
    
    if expected_output['Correct'] is not None:
        for key in expected_output:
            if isinstance(expected_output[key], list):
                expected_output[key] = sorted(expected_output[key])
    
    assert result == expected_output
