from typing import Any, List, Dict, Optional, Union
from ast import literal_eval
from shared.structured import StructuredResult
import pandas as pd
import re
import json

def convert_lists(data):
    """ If an element of the DataFrame in any row and any column
        or a value in a dictionary is a string and starts with '[', 
        it will convert it into a list, unless it's already a list.
    """
    def convert_element(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return x
        if pd.notnull(x) and isinstance(x, str) and x.startswith('['):
            try:
                return literal_eval(x)
            except (ValueError, SyntaxError):
                return x
        return x

    if isinstance(data, pd.DataFrame):
        return data.map(convert_element)
    elif isinstance(data, dict):
        return {key: convert_element(value) for key, value in data.items()}
    else:
        # For other types, apply the conversion directly
        return convert_element(data)

def flatten_structured_result(structured_result: StructuredResult, remove_quotes: bool = True) -> Dict[str, Any]:
    """
    Flatten a StructuredResult into a flat dictionary ignoring groups.

    Output format (for each field with name N):
      N: <value>
      N confidence: <confidence>        (only if provided)
      N justification: <justification>  (only if provided)

    Later duplicates overwrite earlier ones (last one wins).

    Args:
    remove_quotes (bool): If True, remove double quotes inside stringified lists
                            (e.g., '["A", "B"]' -> '[A, B]').
    """
    flat: Dict[str, Any] = {}
    if not structured_result or not structured_result.groups:
        return flat

    for group in structured_result.groups:
        if not group or not group.fields:
            continue
        for field in group.fields:
            if not field or field.name is None:
                continue
            base_name = field.name.strip()
            if not base_name:
                continue

            # always define value before using it
            value = field.value

            if remove_quotes and isinstance(value, str) and re.match(r'^\[.*\]$', value.strip()):
                value = re.sub(r'"\s*([^"]*?)\s*"', r'\1', value)

            flat[base_name] = value

            if field.confidence is not None:
                flat[f"{base_name} confidence"] = field.confidence
            if field.justification is not None:
                flat[f"{base_name} justification"] = field.justification
    
    # Include version in flattened result if available
    if structured_result.version is not None:
        flat["version"] = structured_result.version
        flat["batch"] = structured_result.batch
        
    flat = convert_lists(flat)

    return flat


# Function to remove duplicates
def remove_duplicates(input_data: Union[str, List[Union[str, Dict]]]) -> Union[str, List[Union[str, Dict]]]:
    """
    Removes duplicates:
      - For semicolon-separated strings: returns a cleaned semicolon-separated string
      - For lists of strings: returns a list with duplicates removed
      - For lists of dicts: removes duplicate dicts based on all key-value pairs
    
    Deduplication is case-insensitive for strings, 
    and for dicts it uses a tuple of sorted key-value pairs as the uniqueness check.
    Preserves the first occurrence's casing/ordering.
    """
    # Case 1: semicolon-separated string
    if isinstance(input_data, str) and ";" in input_data:
        items = [item.strip() for item in input_data.split(";") if item.strip()]
        seen = {}
        for item in items:
            key = item.lower()
            if key not in seen:
                seen[key] = item
        return "; ".join(seen.values())

    # Case 2: list of strings
    elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
        seen = {}
        for item in input_data:
            key = item.strip().lower()
            if key not in seen:
                seen[key] = item.strip()
        return list(seen.values())

    # Case 3: list of dicts
    elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
        seen = set()
        unique_dicts = []
        for d in input_data:
            # Create a normalized tuple representation for deduplication
            key = tuple(sorted((k, str(v).lower()) for k, v in d.items()))
            if key not in seen:
                seen.add(key)
                unique_dicts.append(d)
        return unique_dicts

    # Fallback: return as-is
    return input_data

# select and rename field to build group
def select_rename_field(field_dict, new_name=None):
    return {
        "name": new_name if new_name else field_dict.get("Field Name", "-"),
        "value": convert_value_to_string(field_dict.get("Extracted Value", "-")),
        "justification": field_dict.get("Extraction Justification", "") + " " + field_dict.get('Overall Extraction Confidence Level Justification', "-"),
        "confidence": field_dict.get("Overall Extraction Confidence Level", "-")
    }

# separate each field from flat dictionary
def separate_field(dict_result: dict, field_name: str, cleaners: dict[str, callable] = None) -> dict:
    """Build base field dict, apply cleaners only on Extracted Value if provided for this field."""
    cleaners = cleaners or {}
    cleaner = cleaners.get(field_name)

    raw_value = dict_result.get(f"{field_name} Extracted Value", "-")
    cleaned_value = cleaner(raw_value) if cleaner else raw_value

    return {
        "Field Name": field_name,
        "Extracted Value": remove_duplicates(cleaned_value),
        "Extraction Justification": dict_result.get(f"{field_name} Extraction Justification", ""),
        "Input Ambiguity": dict_result.get(f"{field_name} Input Ambiguity", ""),
        "Model Ambiguity": dict_result.get(f"{field_name} Model Ambiguity", ""),
        "Output Ambiguity": dict_result.get(f"{field_name} Output Ambiguity", ""),
    }


# Serialization of final output
def convert_value_to_string(extracted_value):
    if isinstance(extracted_value, str):
        return extracted_value
    # If it's None, return as-is
    if extracted_value is None:
        return None
    try:
        # Try JSON serialization (works for lists, dicts, numbers, booleans, etc.)
        return json.dumps(extracted_value, ensure_ascii=False)
    except (TypeError, ValueError):
        # Fallback for unserializable objects
        return str(extracted_value)