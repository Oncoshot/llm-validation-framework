"""List cells end to end: what `flatten_structured_result` emits, and how `validate` reads it.

Regression cover for the quote-stripping bug. `flatten_structured_result` used to take a
`remove_quotes` flag (default **on**) that rewrote a stringified list `'["A", "B"]'` as
`'[A, B]'` — a form `literal_eval` cannot read back, so the cell reached the scorer as one
long string instead of a list. Measured consequences, all reproduced below:

| label                  | prediction         | F1 before | F1 now |
|------------------------|--------------------|-----------|--------|
| `['Drug A', 'Drug B']` | `[Drug A, Drug B]` | 0.000     | 1.000  |
| `[Drug A, Drug B]`     | `[Drug A]`         | 0.000     | 0.667  |
| `[Drug A, Drug B]`     | `[Drug B, Drug A]` | 0.000     | 1.000  |

The flag is gone. Nothing strips quotes any more, so `convert_lists` — which always parsed a
list repr, and now also recovers the elements of a quote-stripped one — sees the cell intact
and historical data still scores set-wise.
"""

import pandas as pd
import pytest

import llmvalidate.validation as v
from llmvalidate.structured import StructuredField, StructuredGroup, StructuredResult
from llmvalidate.utils import convert_lists, convert_value_to_string, flatten_structured_result

LABEL = "['Drug A', 'Drug B']"


def _flat(value, code=None) -> dict:
    return flatten_structured_result(StructuredResult(groups=[
        StructuredGroup(group_name="g", fields=[
            StructuredField(name="Medications", value=value, code=code)]),
    ]))


def _score(label, prediction, field="Medications"):
    """(micro F1, macro F1) for one case. Macro is empty unless the field scored set-wise."""
    df = pd.DataFrame({field: [label], f"Res: {field}": [prediction]}, index=["c1"])
    _, metrics = v.validate(df, [field], structure_callback=None, output_folder=None)
    row = metrics[metrics["field"] == field].iloc[0]
    return float(row["F1 score (micro)"]), row.get("F1 score (macro)", "")


# --- What flatten emits ------------------------------------------------------

def test_a_stringified_list_is_emitted_as_a_real_list():
    # `convert_value_to_string` is what the LLM path hands over for a multi-value field.
    as_string = convert_value_to_string(["Drug A", "Drug B"])
    assert as_string == '["Drug A", "Drug B"]'
    assert _flat(as_string)["Medications"] == ["Drug A", "Drug B"]


def test_a_real_list_is_emitted_unchanged():
    assert _flat(["Drug A", "Drug B"])["Medications"] == ["Drug A", "Drug B"]
    assert _flat([])["Medications"] == []


def test_a_coded_lists_facets_are_both_parsed():
    flat = _flat('["Osimertinib"]', code='["1721560"]')
    assert flat["Medications-value"] == ["Osimertinib"]
    assert flat["Medications-code"] == ["1721560"]


@pytest.mark.parametrize("value", ["Lung", "-", "", "14", "see note"])
def test_scalars_are_left_alone(value):
    assert _flat(value)["Medications"] == value


@pytest.mark.parametrize("value,expected", [
    ("[see note]", ["see note"]),
    ("[not stated]", ["not stated"]),
])
def test_brackets_mean_a_list_cell(value, expected):
    # The one behavioural consequence worth knowing: a bracket makes a cell a list, so
    # bracketed prose becomes a one-element list. It applies to labels and predictions
    # alike (both go through `convert_lists`), so the two sides still agree — but a scalar
    # is better written without brackets.
    assert _flat(value)["Medications"] == expected


def test_the_removed_flag_is_rejected_rather_than_ignored():
    # `remove_quotes` is gone, not deprecated — a caller still passing it hears about it
    # instead of silently getting different scoring.
    with pytest.raises(TypeError):
        flatten_structured_result(
            StructuredResult(groups=[StructuredGroup(group_name="g", fields=[
                StructuredField(name="Medications", value=["Drug A"])])]),
            remove_quotes=True,
        )


def test_nothing_emits_the_unreadable_form_any_more():
    # The invariant the bug broke: a flattened cell can be read back into what it came from.
    for value in (["Drug A", "Drug B"], '["Drug A", "Drug B"]', "[Drug A, Drug B]"):
        cell = _flat(value)["Medications"]
        assert convert_lists(cell) == ["Drug A", "Drug B"], value


# --- What validate reads -----------------------------------------------------

@pytest.mark.parametrize("label,prediction,expected_f1", [
    # Quoted on both sides: always worked, still works.
    (LABEL, "['Drug A', 'Drug B']", 1.0),
    (LABEL, "['Drug A']", pytest.approx(2 / 3)),
    (LABEL, "['Drug B', 'Drug A']", 1.0),
    # A quote-stripped prediction against a quoted label: scored 0.000 before.
    (LABEL, "[Drug A, Drug B]", 1.0),
    (LABEL, "[Drug B, Drug A]", 1.0),
    (LABEL, "[Drug A]", pytest.approx(2 / 3)),
    # Quote-stripped on both sides: used to be whole-string equality, so a half-right or
    # reordered list scored 0 and no macro metrics were reported at all.
    ("[Drug A, Drug B]", "[Drug A, Drug B]", 1.0),
    ("[Drug A, Drug B]", "[Drug B, Drug A]", 1.0),
    ("[Drug A, Drug B]", "[Drug A]", pytest.approx(2 / 3)),
])
def test_list_cells_are_scored_setwise_however_they_were_written(label, prediction, expected_f1):
    micro, macro = _score(label, prediction)
    assert micro == expected_f1
    assert macro != "", "a list field must report macro metrics (it was scored as a scalar)"


def test_a_wrong_element_still_costs():
    # Recovering the elements must not make everything match: the fix restores set-wise
    # scoring, it does not soften it.
    micro, _ = _score(LABEL, "[Drug A, Drug C]")
    assert micro == pytest.approx(0.5)


def test_no_finding_and_unlabelled_are_untouched():
    assert _score(LABEL, "-")[0] == 0.0          # nothing extracted against a real label
    assert _score("-", "[Drug A]")[0] == 0.0     # extracted against an explicit no-finding


def test_an_unterminated_cell_is_left_as_text():
    # Only a *bracketed and closed* cell is recovered; a truncated one stays a string rather
    # than being silently reinterpreted.
    assert convert_lists("['Drug A', ") == "['Drug A', "
