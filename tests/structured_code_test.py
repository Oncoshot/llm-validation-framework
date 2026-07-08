"""ONC-12309: optional `code` on StructuredField -> `<name>-value` / `<name>-code` facets,
scored against same-named labels, with a single `<name> confidence` binning both facets.
"""

import pandas as pd

import llmvalidate.validation as v
from llmvalidate.structured import StructuredField, StructuredGroup, StructuredResult
from llmvalidate.utils import flatten_structured_result


def _flat(field: StructuredField) -> dict:
    return flatten_structured_result(
        StructuredResult(groups=[StructuredGroup(group_name="g", fields=[field])])
    )


# --- flatten -----------------------------------------------------------------

def test_coded_field_flattens_to_value_and_code_facets():
    flat = _flat(StructuredField(name="Primary Histology", value="Adenocarcinoma",
                                 code="8140/3", confidence="high", justification="clear"))
    assert flat["Primary Histology-value"] == "Adenocarcinoma"
    assert flat["Primary Histology-code"] == "8140/3"
    assert "Primary Histology" not in flat  # no bare value key when coded
    # confidence / justification stay keyed by the logical field (single, not per-facet)
    assert flat["Primary Histology confidence"] == "high"
    assert flat["Primary Histology justification"] == "clear"
    assert "Primary Histology-value confidence" not in flat
    assert "Primary Histology-code confidence" not in flat


def test_uncoded_field_is_unchanged():
    flat = _flat(StructuredField(name="Primary Site", value="Lung", confidence="low"))
    assert flat["Primary Site"] == "Lung"
    assert flat["Primary Site confidence"] == "low"
    assert "Primary Site-value" not in flat and "Primary Site-code" not in flat


def test_coded_list_field_keeps_lists_on_both_facets():
    flat = _flat(StructuredField(name="Medications", value=["Osimertinib"], code=["1721560"]))
    assert flat["Medications-value"] == ["Osimertinib"]
    assert flat["Medications-code"] == ["1721560"]


# --- scoring: one confidence column bins both facets -------------------------

def test_single_confidence_column_bins_both_facets():
    # Labels + Res for both facets, and ONE Res: confidence keyed by the logical field.
    src = pd.DataFrame({
        "raw_text": ["a", "b", "c", "d"],
        "Primary Histology-value": ["Adenocarcinoma", "Carcinoma", "Adenocarcinoma", "Carcinoma"],
        "Primary Histology-code":  ["8140/3", "8010/3", "8140/3", "8010/3"],
        "Res: Primary Histology-value": ["Adenocarcinoma", "Carcinoma", "Carcinoma", "Adenocarcinoma"],
        "Res: Primary Histology-code":  ["8140/3", "8010/3", "8010/3", "8140/3"],
        "Res: Primary Histology confidence": ["high", "high", "low", "low"],
    })
    _, m = v.validate(
        src, ["Primary Histology-value", "Primary Histology-code"],
        structure_callback=None, output_folder=None,
    )

    def f1(field, conf):
        sub = m[(m["field"] == field) & (m["confidence"] == conf)]
        assert not sub.empty, f"no {conf} row for {field}"
        return float(sub.iloc[0]["F1 score (micro)"])

    # Rows a,b are high (both facets correct); c,d are low (both wrong) — and BOTH facets
    # get that breakdown from the single `Res: Primary Histology confidence` column.
    for field in ("Primary Histology-value", "Primary Histology-code"):
        assert f1(field, "high") == 1.0
        assert f1(field, "low") == 0.0


def test_non_facet_confidence_still_direct():
    # A plain (non-coded) field keeps using its own Res: <field> confidence column.
    src = pd.DataFrame({
        "raw_text": ["a", "b"],
        "Primary Site": ["Lung", "Breast"],
        "Res: Primary Site": ["Lung", "Colon"],
        "Res: Primary Site confidence": ["high", "low"],
    })
    _, m = v.validate(src, ["Primary Site"], structure_callback=None, output_folder=None)
    high = m[(m["field"] == "Primary Site") & (m["confidence"] == "high")].iloc[0]
    low = m[(m["field"] == "Primary Site") & (m["confidence"] == "low")].iloc[0]
    assert float(high["F1 score (micro)"]) == 1.0
    assert float(low["F1 score (micro)"]) == 0.0
