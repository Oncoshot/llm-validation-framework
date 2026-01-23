from datetime import datetime
import math
from ast import literal_eval
import string
import pandas as pd
import numpy as np
import time
import os
import concurrent.futures as cf
from tqdm import tqdm
from shared.utils import convert_lists, flatten_structured_result

def compare_results_binary(expected, actual):
    """Compares boolean labels and returns confusion matrix counts."""

    if (is_expected_undefined(expected)):
        return {'TP': None, 'TN': None, 'FP': None, 'FN': None}

    TP = 1 if expected is True and actual is True else 0 
    FN = 1 if expected is True and actual is not True else 0 
    TN = 1 if expected is False and actual is False else 0
    FP = 1 if expected is False and actual is not False else 0

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

def calculate_binary_metrics(TP, FP, FN, TN):
    """Computes precision, recall, F1, F2, accuracy and specificity from confusion matrix counts."""
    
    # arithmetic operations with None cause exception
    # arithmetic operations with NaN result in NaN without exception
    # change None with NaN, to make sure there is no exception, 
    # but non-numbers (None and NaN) are handled properly
    # we dont want to receive a number (even 0) if there is non-number operand
    if TP is None:
        TP = np.nan

    if FP is None:
        FP = np.nan

    if FN is None:
        FN = np.nan

    if TN is None:
        TN = np.nan

    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall    = TP / (TP + FN) if (TP + FN) > 0 else np.nan

    # F-beta score calculation: F_beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    f2_score = 5 * precision * recall / (4 * precision + recall) if (precision + recall) > 0 else np.nan
    
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan  # TN rate

    return precision, recall, f1_score, f2_score, accuracy, specificity

def compare_results_all(df, fields, parents={}, comparison_callback=None, raw_text_column_name: str = "raw_text"):
    """ For each case from df calls and for each field from 'fields' compares results with 'Res: '+field
        and adds columns 'Inc: '+field, 'Cor: '+field, 'Mis: '+field, 'Spu: '+field, 'Par: '+field
        Input
            'df' - pandas dataframe with input, labeled data and results in 'Res: ' columns
            'fields' - fields to compare results
            'parents' - A dictionary where key is a child and value is a parent for partial match
            - **comparison_callback (callable, optional)**: 
                A callback function that will be applied to each row to generate comparison results.

        Returns pandas data frame same as df but with added columns:
            'Cor: ' for Correct
            'Inc: ' for Incorrect
            'Mis: ' for Missing
            'Spu: ' for Spurious
            'Par: ' for Partial
    """

    fields = [f for f in fields if f in df] #ignore fields which are not in columns (they may be added later by comparison_callback)

    # Determine which fields are binary by looking at the unique non-empty expected values.
    binary_fields = {}
    for field in fields:
        binary_fields[field] = pd.api.types.is_bool_dtype(df[field])

    # Create a list to store modified rows
    modified_rows = []
    # Iterate over the rows of the DataFrame
    for i, row in df.iterrows():

        for field in fields:
            expected = row[field]

            if 'Res: ' + field not in row:
                continue

            actual = row['Res: ' + field]

            if binary_fields[field]: 
                # For binary fields, use the binary comparison function.
                res = compare_results_binary(expected, actual)
                res_items = {}
            else:
                res_items = compare_results(expected, actual, parents)

                # Convert lists to counts, preserving None values
                res = {key: len(value) if value is not None else None for key, value in res_items.items()}

                if isinstance(expected, list) or isinstance(actual, list):
                    # если сравниваем списки то вычисляем метрики на уровне отдельного кейса (для скаляров не имеет смысла)
                    TN_case = None # TN для списков не определён

                    precision, recall, f1_score, f2_score, specificity = calculate_metrics(
                        res['Correct'], res['Partial'], res['Incorrect'], res['Spurious'], res['Missing'], TN_case
                    )

                    row['Precision: ' + field] = precision
                    row['Recall: ' + field] = recall
                    row['F1 score: ' + field] = f1_score
                    row['F2 score: ' + field] = f2_score

                    if not parents:
                        #we dont want useless columns if no parents anyway
                        del res['Incorrect'] #incorrect appear only if parent mismatch (only for lists)
                        del res_items['Incorrect']

                if not parents:
                    #we dont want useless columns if no parents anyway
                    del res['Partial']      #partial appear only if parent match
                    del res_items['Partial']

            # Iterate over the key-value pairs in the result dictionary
            for key, value in res.items():
                # Create a new column in the DataFrame using the key
                # Assign the corresponding value to the column
                row[key[:3] + ': ' + field] = value

            # Iterate over the key-value pairs in the result items dictionary
            for key, value in res_items.items():
                # Create a new column in the DataFrame using the key
                # Assign the corresponding value to the column
                row[key[:3] + ': ' + field + ' items'] = value

        if comparison_callback:
            comparison_callback(row, i, raw_text_column_name)

        # Append the modified row to the list
        modified_rows.append(row)

    # Recreate the DataFrame with the updated rows
    res_df = pd.DataFrame(modified_rows)
    res_df.index = df.index

    res_df = _reorder_result_columns(res_df)

    return res_df

def calculate_metrics(cor, par, inc, spu, mis, TN):
    """
    Computes precision, recall, and F1 score.
    """
    # https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/#:~:text=When%20you%20train%20a%20NER,to%20tune%20a%20NER%20system.
    partial_weight = 0.5

    # arithmetic operations with None cause exception
    # arithmetic operations with NaN result in NaN without exception
    # change None with NaN, to make sure there is no exception, 
    # but non-numbers (None and NaN) are handled properly
    # we dont want to receive a number (even 0) if there is non-number operand
    if cor is None:
        cor = np.nan

    if par is None:
        par = np.nan

    if inc is None:
        inc = np.nan

    if spu is None:
        spu = np.nan

    if mis is None:
        mis = np.nan

    if TN is None:
        TN = np.nan  

    # Ensure denominator is not zero before division
    denominator_precision = cor + inc + par + spu
    if denominator_precision == 0:
        precision = np.nan
    else:
        # Of all values extracted, how many were right?
        precision = (cor + partial_weight * par) / denominator_precision  # TP / (TP + FP)

    denominator_recall = cor + inc + par + mis
    if denominator_recall == 0:
        recall = np.nan
    else:
        # Of all cases with information about value, how many did we extract?
        recall = (cor + partial_weight * par) / denominator_recall  # TP / (TP + FN)

    if precision == 0 or recall == 0:
        f1_score = f2_score = 0  # if any zero then f1 = 0
    else:
        # F-beta score calculation: F_beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
        f2_score = 5 * precision * recall / (4 * precision + recall) if (precision + recall) > 0 else np.nan

    denom_specificity = TN + spu
    if denom_specificity == 0:
        specificity = np.nan
    else:
        # Of all cases with no information about value, how many were correctly left empty?
        specificity = TN / denom_specificity # TN / (TN + FP)

    return precision, recall, f1_score, f2_score, specificity

def get_metrics(res_df, fields):
    """ For results dataframe generated by validate
        produces dataframe with metrics for all available confidence levels and overall
    """
    metrics_list = []  # Accumulate rows as dictionaries instead of modifying DataFrame

    total_cases = len(res_df.index)
    
    # Handle case where 'Sys: exception' column doesn't exist
    if 'Sys: exception' in res_df.columns:
        exceptions_no = sum(res_df['Sys: exception'].notna())
        clean_df = res_df[res_df['Sys: exception'].isna()]
    else:
        exceptions_no = None
        clean_df = res_df

    # Append the first row for exceptions
    metrics_list.append({
        'field': 'exceptions',
        'confidence': None,
        'total cases': total_cases,
        'positive cases': exceptions_no,
        'TP': None, 'TN': None, 'FP': None, 'FN': None,
        'precision (micro)': None, 'recall (micro)': None, 'F1 score (micro)': None, 'F2 score (micro)': None, 'accuracy (micro)': None, 'specificity (micro)': None,
        'cor': None, 'inc': None, 'mis': None, 'spu': None, 'par': None,
        'precision (macro)': None, 'recall (macro)': None, 'F1 score (macro)': None, 'F2 score (macro)': None
    })

    # Find all unique confidence levels across all fields
    all_confidence_levels = set()
    for field in fields:
        confidence_col = f'Res: {field} confidence'
        if confidence_col in clean_df.columns:
            confidence_values = clean_df[confidence_col].dropna().unique()
            all_confidence_levels.update(confidence_values)
    
    # Convert to sorted list for consistent ordering
    confidence_levels = ['Overall'] + sorted(list(all_confidence_levels))

    for field in fields:
        confidence_field_name = f'Res: {field} confidence'
        has_confidence = confidence_field_name in clean_df.columns
        
        for confidence_level in confidence_levels:
            # Filter data based on confidence level
            if confidence_level == 'Overall':
                field_df = clean_df
            else:
                if not has_confidence:
                    continue  # Skip this confidence level if field doesn't have confidence data
                field_df = clean_df[clean_df[confidence_field_name] == confidence_level]
            
            if field_df.empty and confidence_level != 'Overall':
                continue  # Skip empty confidence groups

            total_cases = field_df[field].count()
            positive_cases = field_df[field].apply(lambda x: not (is_scalar_empty(x) or x == [])).sum()

            TP = TN = FP = FN = \
            cor = inc = mis = spu = par = \
            precision_micro = recall_micro = f1_score_micro = f2_score_micro = accuracy_micro = specificity_micro = \
            precision_macro = recall_macro = f1_score_macro = f2_score_macro = None

            # Check if binary columns exist for this field:
            if "TP: " + field in field_df.columns:

                # --- BINARY FIELD METRICS ---

                TP = field_df["TP: " + field].sum()
                TN = field_df["TN: " + field].sum() if 'TN: ' + field in field_df else None
                FP = field_df["FP: " + field].sum()
                FN = field_df["FN: " + field].sum()

                precision_micro, recall_micro, f1_score_micro, f2_score_micro, accuracy_micro, specificity_micro = calculate_binary_metrics(TP, FP, FN, TN)

                if 'Precision: ' + field in field_df:
                    precision_macro = field_df['Precision: ' + field].mean()
                    recall_macro = field_df['Recall: ' + field].mean()
                    f1_score_macro = field_df['F1 score: ' + field].mean()
                    f2_score_macro = field_df['F2 score: ' + field].mean()
            else:

                # --- NON-BINARY FIELD METRICS --

                cor = field_df['Cor: ' + field].sum()
                inc = field_df['Inc: ' + field].sum() if 'Inc: ' + field in field_df else None
                mis = field_df['Mis: ' + field].sum()
                spu = field_df['Spu: ' + field].sum()
                par = field_df['Par: ' + field].sum() if 'Par: ' + field in field_df else None

                if 'Precision: ' + field in field_df:
                    # if Precision present it means that it is a list field
                    precision_macro = field_df['Precision: ' + field].mean()
                    recall_macro = field_df['Recall: ' + field].mean()
                    f1_score_macro = field_df['F1 score: ' + field].mean()
                    f2_score_macro = field_df['F2 score: ' + field].mean()

                    # TN is not defined for list values (e.g. label ['apple'] and prediction ['orange', 'banana'] will produce SPU=2)
                    TN_field = None
                else:
                    # scalar field
                    TN_field = total_cases - positive_cases - spu
                
                precision_micro, recall_micro, f1_score_micro, f2_score_micro, specificity_micro = calculate_metrics(
                    cor, par or 0, inc or 0, spu, mis, TN_field
                )

            # Append computed row as a dictionary
            metrics_list.append({
                'field': field,
                'confidence': confidence_level,
                'total cases': total_cases,
                'positive cases': positive_cases,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                'precision (micro)': precision_micro,
                'recall (micro)': recall_micro,
                'F1 score (micro)': f1_score_micro,
                'F2 score (micro)': f2_score_micro,
                'accuracy (micro)': accuracy_micro,
                'specificity (micro)': specificity_micro,
                'cor': cor,
                'inc': inc,
                'mis': mis,
                'spu': spu,
                'par': par,
                'precision (macro)': precision_macro,
                'recall (macro)': recall_macro,
                'F1 score (macro)': f1_score_macro,
                'F2 score (macro)': f2_score_macro
            })

    # Convert the list to a DataFrame at the end
    metrics = pd.DataFrame(metrics_list)

    metrics.dropna(axis=1, how="all", inplace=True)

    return metrics

def compare_results(expected, actual, parents={}):
    """
    Compares expected results with actual results, ignoring duplicates.

    If the inputs are scalar values (int, float, string), the function checks for equality,
    considering case-insensitivity for strings.

    If the inputs are lists, the function computes lists of 'Correct', 'Missing', and 'Spurious' elements without considering duplicates:
    - 'Correct': List of unique elements that are present in both lists (considering case-insensitive for strings).
    - 'Incorrect': For scalar values: list containing actual value if expected not equals to actual; For lists: list of actual values that are children of expected
    - 'Missing': List of unique elements that are in the expected list but not in the actual list.
    - 'Spurious': List of unique elements that are in the actual list but not in the expected list.
    - 'Partial': List of unique elements from expected which parents are present in actual lists.

    strings will try to parse into float: "4" equals to 4

    Args:
    expected (int, float, str, list): The expected result.
    actual (int, float, str, list): The actual result.
    parents : A dictionary where key is a child and value is a parent for partial match

    Returns:
    dict: A dictionary with keys 'Correct', 'Incorrect', 'Missing', 'Spurious', and 'Partial', mapping to their respective lists.

    How to distinguish empty results from not labeled results:
        SCALAR:
            if labeled value is empty then row was not labeled at all
            if label value is "-" then row was labeled as "No information"
            any other value means particular label value
        LISTS:
            empty - not labeled
            '-' OR [] - No information
            [item1, item2,..] - particular label values
        Labeled cases - number of rows with label
        Empty labels - number of rows marked as "No information"


        SCALAR
        label       result       Cor   Inc   Spu   Mis   Par
        a           a            [a]
        a           b                  [b]
        TNBC        BC                                   [BC]
        BC          TNBC               [TNBC]
        a                                          [a]
        -           a                        [a]
        -                        []    []    []    []    []
                    a            None  None  None  None  None

        LISTS
        label       result       Cor   Inc   Spu   Mis   Par
        [a]         [a]          [a]
        [a]         [b]                      [b]   [a]
        [TNBC]      [BC]                                 [BC]
        [BC]        [TNBC]             [TNBC]
        [a]         []                             [a]
        [] or -     [a]                      [a]
        [] or -     []           []    []    []    []    []
                    [a]          None  None  None  None  None
    """

    # Convert dictionary keys to casefold for case-insensitive matching
    parents = {k.casefold(): v.casefold() for k, v in parents.items()}

    output = {'Correct': [], 'Incorrect': [], 'Missing': [], 'Spurious': [], 'Partial': []}

    if isinstance(expected, list) or isinstance(actual, list):
        if (is_expected_undefined(expected)):
            return {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}

        # Convert all elements to lowercase strings for case-insensitive comparison
        # and remove duplicates by converting to sets.
        expected_set = normalize_list(expected)
        expected_parents_set = {parents[s] for s in expected_set if s in parents}
        
        actual_set = normalize_list(actual)
        actual_parents_set = {parents[s] for s in actual_set if s in parents}

        output['Correct'] = list(expected_set & actual_set)  # matching exactly

        for missing in expected_set - actual_set - actual_parents_set:
            if missing in parents and parents[missing] in actual_set:
                output['Partial'].append(missing)  # actual set has parent value therefore Partial
            else:
                output['Missing'].append(missing)

        for spurious in actual_set - expected_set - expected_parents_set:
            if spurious in parents and parents[spurious] in expected_set:
                output['Incorrect'].append(spurious)  # expected set has parent value therefore Incorrect
            else:
                output['Spurious'].append(spurious)

    else: #both expected and actual are not lists

        expected = normalize(expected)
        actual = normalize(actual)

        if (is_expected_undefined(expected)):
            return {'Correct': None, 'Incorrect': None, 'Missing': None, 'Spurious': None, 'Partial': None}

        if is_scalar_empty(expected) and is_scalar_empty(actual):
            return output

        if expected == actual:
            output['Correct'] = [expected]
        elif is_scalar_empty(expected) and not is_scalar_empty(actual):
            output['Spurious'] = [actual]
        elif not is_scalar_empty(expected) and is_scalar_empty(actual):
            output['Missing'] = [expected]
        elif expected in parents and parents[expected] == actual:
            output['Partial'] = [expected]  # actual value is a parent of expected
        else:
            output['Incorrect'] = [actual]

    return output

def normalize(x):
    if isinstance(x, str):
        x = x.strip()  # Remove leading and trailing whitespace
        try:
            return float(x)
        except:
            return x.casefold()
    else:
        return x

def normalize_list(values):
    if isinstance(values, list):
        result = set(normalize(x) for x in values)
    else:
        if not is_scalar_empty(values):
            result = {normalize(values)}
        else:
            result = set()
    return result

def is_scalar_empty(value):
    if value in [None, "", "-"]:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False

def is_expected_undefined(value):
    if value in [None, ""]:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False

def _single_row_worker(row_tuple):
    """Runs in a separate process / thread.
       Accepts (i, row_dict, raw_text_column_name, callback) because
       top‑level, pickle‑friendly functions are required for ProcessPools.
    """
    i, row_in, raw_text_column_name, callback = row_tuple
    row_out   = {}                # what we will send back

    # local per‑worker cache – fast, zero IPC cost
    if "_CACHE" not in _single_row_worker.__dict__:
        _single_row_worker._CACHE = {}
    cache = _single_row_worker._CACHE

    row_start_time = time.time()  # Start the timer for each row
    try:
        cache_key = row_in[raw_text_column_name]
        cache_key = "" if pd.isna(cache_key) else str(cache_key).strip(string.whitespace + string.punctuation).lower()

        # Check if result is already cached
        if cache_key in cache:
            from_cache = True
            res, tokens_usage = cache[cache_key]
        else:
            res, tokens_usage = callback(row_in, i, raw_text_column_name)  # structure information
            from_cache = False
            cache[cache_key] = (res, {})  # Store result in cache

        # Iterate over the key-value pairs in the result dictionary
        for key, value in res.items():
            # Create a new column in the DataFrame using the key
            # Assign the corresponding value to the column
            row_out['Res: ' + key] = value

        row_out['Sys: from cache'] = from_cache

        # Iterate over the key-value pairs in the tokens dictionary
        for token_key, token_value in tokens_usage.items():
            # Assign the corresponding value to the column
            row_out['Sys: ' + token_key] = token_value


        row_out["Sys: exception"]  = np.nan          # keep column dtype homogeneous

    except Exception as e:
        row_out['Sys: exception'] = e

    # Calculate the time taken to process the row
    row_end_time = time.time()  # End the timer for each row
    row_elapsed_time = row_end_time - row_start_time
    # Assign the time taken to a new column in the DataFrame
    row_out['Sys: time taken'] = row_elapsed_time

    row_out["_orig_index"]     = i                  # re‑index later
    return row_out

def _reorder_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column order:
      1. Columns that don't belong to any base group
      2. For each label base (a column that has a matching 'Res: {base}') in original order:
           All columns starting with that base name, then Res:, confidence, justification, metrics
      3. Remaining Res: columns (rare edge cases)
      4. System columns (Sys:)
      5. Any stragglers (safety)
    Metrics auto-detected (binary vs non-binary).
    """
    if df.empty:
        return df

    cols = list(df.columns)

    # Label base columns: those with a corresponding Res: column
    label_bases = [c for c in cols if f'Res: {c}' in cols]

    system_cols = [c for c in cols if c.startswith('Sys: ')]
    res_cols_all = [c for c in cols if c.startswith('Res: ')]

    metric_prefixes_binary_counts = ["TP: ", "FP: ", "FN: ", "TN: "]
    metric_prefixes_non_binary_counts = ["Cor: ", "Inc: ", "Mis: ", "Spu: ", "Par: "]
    metric_prefixes = ["Precision: ", "Recall: ", "Accuracy: ", "F1 score: ", "F2 score: ", "Specificity: "]

    metric_prefixes_all = metric_prefixes_binary_counts + metric_prefixes_non_binary_counts + metric_prefixes

    # Identify metric columns so they are not treated as original non-label columns
    metric_cols = {c for c in cols if any(c.startswith(p) for p in metric_prefixes_all)}

    used = set()
    new_order = []

    # Helper function to find columns that start with a base name
    def find_columns_starting_with_base(base_name):
        """Find all columns that start with the base name (but are not the base itself or Res:/Sys:/metric columns)"""
        related_cols = []
        for c in cols:
            if c == base_name:
                continue
            if c.startswith('Res: ') or c.startswith('Sys: '):
                continue
            if c in metric_cols:
                continue
            if c.startswith(base_name):
                related_cols.append(c)
        return related_cols

    # 1. Columns that don't belong to any base group
    for c in cols:
        if c in label_bases:
            continue
        if c.startswith('Res: ') or c.startswith('Sys: '):
            continue
        if c in metric_cols:
            continue
        
        # Check if this column belongs to any base group
        belongs_to_base = False
        for base in label_bases:
            if c.startswith(base) and c != base:
                belongs_to_base = True
                break
        
        if not belongs_to_base:
            new_order.append(c)
            used.add(c)

    # 2. Grouped label sections with all related columns
    for base in label_bases:
        # First add all columns that start with this base name (excluding the base itself and special columns)
        related_cols = find_columns_starting_with_base(base)
        for related_col in related_cols:
            if related_col not in used:
                new_order.append(related_col)
                used.add(related_col)

        # Then add the base column itself
        if base not in used:
            new_order.append(base)
            used.add(base)

        # Then add Res: column
        core_col = f'Res: {base}'
        if core_col in cols and core_col not in used:
            new_order.append(core_col)
            used.add(core_col)

        # confidence / justification
        conf_col = f'Res: {base} confidence'
        just_col = f'Res: {base} justification'
        if conf_col in cols and conf_col not in used:
            new_order.append(conf_col)
            used.add(conf_col)
        if just_col in cols and just_col not in used:
            new_order.append(just_col)
            used.add(just_col)

        # Metrics (binary first else non-binary set)
        if (f'TP: {base}') in cols:  # binary metrics
            for p in (metric_prefixes_binary_counts + metric_prefixes):
                mc = p + base
                if mc in cols and mc not in used:
                    new_order.append(mc)
                    used.add(mc)
        else:
            if any((p + base) in cols for p in metric_prefixes_non_binary_counts):
                for p in metric_prefixes_non_binary_counts:
                    mc = p + base
                    if mc in cols and mc not in used:
                        new_order.append(mc)
                        used.add(mc)

            if any((p + base + ' items') in cols for p in metric_prefixes_non_binary_counts):
                for p in metric_prefixes_non_binary_counts:
                    mc = p + base + ' items'
                    if mc in cols and mc not in used:
                        new_order.append(mc)
                        used.add(mc)

            if any((p + base) in cols for p in metric_prefixes):
                for p in metric_prefixes:
                    mc = p + base
                    if mc in cols and mc not in used:
                        new_order.append(mc)
                        used.add(mc)

    # 3. Remaining Res: columns
    for c in res_cols_all:
        if c not in used:
            new_order.append(c)
            used.add(c)

    # 4. System columns
    for c in system_cols:
        if c not in used:
            new_order.append(c)
            used.add(c)

    # 5. Stragglers (safety)
    for c in cols:
        if c not in used:
            new_order.append(c)
            used.add(c)

    return df[new_order]

def process_all(df: pd.DataFrame,
                callback,
                raw_text_column_name: str = "raw_text",
                max_workers: int | None    = 1,
                use_threads: bool          = True):
    """ For each case from df calls 'callback' function and adds results
        Input
            'df' - pandas dataframe with input and labeled data
            'callback' - function which does structuring, receives two parameters:
                            row - row from 'df' with case data
                            i - index of the row

        Returns pandas data frame same as df but with added columns:
            'Res: + result_name'
            'Sys: exception' - exception information if there was an exception
            'Sys: time taken' - time taken in seconds

        Parallel version of ``process_all`` using the concurrent.futures API.
        ``use_threads=False`` → processes       (best for CPU work)  
        ``use_threads=True``  → ThreadPoolExec (best for I/O work)

        To enable parallelism set max_workers = None

        If use_threads = False dont forget to wrap outer code in 
            if __name__ == "__main__":  
                main()
    """

    # prepare data once to avoid pickling the whole DataFrame for every task
    tasks = [(i, row, raw_text_column_name, callback) for i, row in df.iterrows()]

    Executor = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor
    max_workers = max_workers or os.cpu_count()

    # On a ThreadPoolExecutor this will clear the shared in-process cache for all threads.
    # On a ProcessPoolExecutor, each child process starts with a fresh copy of the module (i.e. an empty cache) anyway
    _single_row_worker._CACHE = {} #reset cache

    results = []
    with Executor(max_workers=max_workers) as pool:
        # tqdm + as_completed gives a responsive progress bar
        for f in tqdm(cf.as_completed([pool.submit(_single_row_worker, t) for t in tasks]),
                      total=len(tasks),
                      desc="Processing rows"):
            results.append(f.result())

    # Re‑assemble – much faster than repeatedly appending rows
    res_df = pd.DataFrame(results).set_index("_orig_index")
    res_df = df.join(res_df, how="left")            # preserve original cols & order

    # Reorder for readability (no metrics yet)
    res_df = _reorder_result_columns(res_df)

    return res_df


def validate(source_df, fields, structure_callback, output_folder=None, drop_columns=[], 
             file_prefix = datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
             raw_text_column_name = 'raw_text',
             comparison_callback = None,
             max_workers: int | None = 1,
             use_threads: bool = True):
    """
    Performs validation by applying a structuring callback to an input dataset and evaluating performance.

    This function processes the given CSV file, applies the provided structuring logic, and compares the generated
    results against the provided labels. It returns the resulting DataFrames and generates two CSV files in the 
    specified output folder, each prefixed with the current timestamp:

    1. **results.csv**: Detailed case-by-case comparison of the generated values and the corresponding labels.
    2. **metrics.csv**: Aggregated performance metrics across all cases, broken down by each specified field.

    ### Parameters:
    - **source_df (Data Frame)**: 
        Dataframe with raw texts and labels. Must have unique index 'Case No'.
    
    - **fields (list of str, optional)**: 
        List of column names representing the label fields to be evaluated.
        If None and structure_callback is also None, fields will be automatically inferred 
        from columns that have corresponding "Res: " columns.

    - **structure_callback (callable, optional)**: 
        A callback function that will be applied to each row to generate structured predictions.
        If None, assumes that source_df already contains "Res: " columns with results.

    - **comparison_callback (callable, optional)**: 
        A callback function that will be applied to each row to generate comparison results.

    - **output_folder (str, optional)**: 
        Directory where the output CSV files will be saved. 
        If not provided, results will only be returned as DataFrames.

    - **drop_columns (list of str, optional)**: 
        List of columns to exclude from the output dataset. 
        For example, this can be used to omit raw text fields for privacy or file size reduction.
        Or to remove irrelevant output fields (they will have 'Res: prefix')

    ### Returns:
    - **tuple (pd.DataFrame, pd.DataFrame)**: 
        A pair of DataFrames corresponding to:
        - The **results** DataFrame with row-level details.
        - The **metrics** DataFrame with aggregated performance statistics.

    ### Notes:
    - The output CSV files are saved with filenames prefixed by the current timestamp to avoid overwriting previous runs.
    - The source df should have index with unique values for proper processing.
    """

    if source_df.index.has_duplicates:
        raise ValueError("Please remove duplicate values in index column")

    # Infer fields from columns if both structure_callback and fields are None
    if structure_callback is None and fields is None:
        # Find columns that have corresponding "Res: " columns
        res_columns = [col for col in source_df.columns if col.startswith('Res: ')]
        # Extract the field names by removing the "Res: " prefix
        inferred_fields = []
        for res_col in res_columns:
            field_name = res_col[5:]  # Remove "Res: " prefix
            # Only include if the base field column exists in the dataframe
            if field_name in source_df.columns:
                inferred_fields.append(field_name)
        
        if not inferred_fields:
            raise ValueError("Cannot infer fields: no columns found with both base field and corresponding 'Res: ' columns")
        
        fields = inferred_fields

    # Validate that fields parameter is provided when structure_callback is not None
    if structure_callback is not None and fields is None:
        raise ValueError("fields parameter is required when structure_callback is provided")

    # Validate that all specified fields exist in source_df
    missing_fields = [field for field in fields if field not in source_df.columns]
    if missing_fields:
        raise ValueError(f"The following fields are missing from source_df: {missing_fields}")

    # If structure_callback is None, validate that corresponding "Res: " columns exist
    if structure_callback is None:
        missing_res_columns = [f"Res: {field}" for field in fields if f"Res: {field}" not in source_df.columns]
        if missing_res_columns:
            raise ValueError(f"When structure_callback is None, the following result columns must be present in source_df: {missing_res_columns}")

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    source_df = convert_lists(source_df)  #it reads lists and strings from csv, so we need to parse them and make lists

    if structure_callback is not None:
        ###################################
        # Get function result for each row and save into new DataFrame
        res_df = process_all(source_df, structure_callback, raw_text_column_name, max_workers, use_threads)
    else:
        # Use source_df as res_df when structure_callback is None (results already present)
        res_df = source_df.copy()

    res_df = convert_lists(res_df) 
    res_df.drop(columns=drop_columns, inplace=True)

    # Save results
    if output_folder:
        res_df.to_csv(os.path.join(output_folder, f"{file_prefix} results.csv"))
    
    ###################################
    # Analyse the results and calculate the metrics
    res_df = compare_results_all(res_df, fields, parents={}, comparison_callback=comparison_callback, raw_text_column_name=raw_text_column_name)

    # Save results
    if output_folder:
        res_df.to_csv(os.path.join(output_folder, f"{file_prefix} results.csv"))

    ###################################
    # Calculate and display metrics for each field
    metrics_df = get_metrics(res_df, fields)

    # Save metrics
    if output_folder:
        metrics_df.to_csv(os.path.join(output_folder, f"{file_prefix} metrics.csv"), index=False)

    return res_df, metrics_df