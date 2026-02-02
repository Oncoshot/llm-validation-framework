# LLM Validation Framework

Evaluate LLM-extracted structured data against labeled ground truth, compute per-row comparison results, and aggregate metrics by field and confidence. 

## Quick Start

- Requirements: Python 3.11+ (see [requirements.txt](requirements.txt)).
- Install:

```sh
pip install -r requirements.txt
```

- Run the sample pipeline:

```sh
python runme.py
```

This reads [samples.csv](samples.csv) and writes timestamped outputs to [validation_results](validation_results)/samples:
- results: `YYYY-MM-DD HH-MM-SS results.csv`
- metrics: `YYYY-MM-DD HH-MM-SS metrics.csv`

See a generated example: [validation_results/samples/2026-02-02 12-43-44 results.csv](validation_results/samples/2026-02-02%2012-43-44%20results.csv) and [validation_results/samples/2026-02-02 12-43-44 metrics.csv](validation_results/samples/2026-02-02%2012-43-44%20metrics.csv).

The sample runner is [runme.py](runme.py) and uses [src.validation.validate](src/validation.py) with `structure_callback=None` and `fields=None` to infer fields from existing `Res:` columns.

## Input Format

- Labels: base columns (e.g., “First Primary Diagnosis”, “Treatment Drugs”).
- Predictions: corresponding `Res: {Field}` columns.
- Lists can be stringified (e.g., "['XELOX']"); they are normalized by [src.utils.convert_lists](src/utils.py).
- Example file: [samples.csv](samples.csv).

When a `structure_callback` is provided, the framework flattens your structured output via [src.utils.flatten_structured_result](src/utils.py) into the same `Res:` column schema.

## Output

[src.validation.validate](src/validation.py) produces:

- Row-level results: counts and items for Correct, Incorrect, Missing, Spurious, plus per-row metrics for list fields. 
- Aggregated metrics per field and confidence.

Binary metrics use:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2PR / (P + R)
- F2 = 5PR / (4P + R)
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Specificity = TN / (TN + FP)

Non-binary per-field metrics use:
- Precision = cor / (cor + inc + spu)
- Recall = cor / (cor + inc + mis)
- F1, F2 as above with P = Precision and R = Recall.

Confidence breakdowns are generated for any present `Res: {Field} confidence` column and always include "Overall".

## Programmatic Usage

Two modes:

1) Results already present (CSV-style):

```python
import pandas as pd
from src.validation import validate

df = pd.read_csv("samples.csv", index_col="Patient ID")
res_df, metrics_df = validate(
	source_df=df,
	fields=None,                 # infer from Res: columns
	structure_callback=None,     # use existing Res: columns
	raw_text_column_name=None,
	output_folder="validation_results/samples"
)
```

2) Provide a structure callback that returns flattened fields:

```python
from src.structured import StructuredResult, StructuredGroup, StructuredField
from src.utils import flatten_structured_result
from src.validation import validate

def structure_callback(row, i, raw_text_column_name):
	sr = StructuredResult(
		groups=[
			StructuredGroup(
				group_name="group1",
				fields=[
					StructuredField(name="First Primary Diagnosis", value="Colorectal Cancer", confidence="High"),
					StructuredField(name="Treatment Drugs", value="['XELOX']", confidence="High"),
				],
			)
		]
	)
	flat = flatten_structured_result(sr)
	return flat, {}  # ({field: value, ...}, {optional token usage})

# Apply on a DataFrame with labels and raw_text (if using caching)
res_df, metrics_df = validate(
	source_df=df,
	fields=["First Primary Diagnosis","Treatment Drugs"],
	structure_callback=structure_callback,
	raw_text_column_name="raw_text",
	output_folder="validation_results/custom"
)
```

## Testing

- Run all tests:

```sh
pytest
```

Key tests:
- [tests/validate_test.py](tests/validate_test.py)
- [tests/compare_results_test.py](tests/compare_results_test.py)
- [tests/compare_results_all_test.py](tests/compare_results_all_test.py)

## Repository Layout

- Core: [src/validation.py](src/validation.py), [src/utils.py](src/utils.py), [src/structured.py](src/structured.py), [src/standardize.py](src/standardize.py)
- Sample runner: [runme.py](runme.py)
- Sample data: [samples.csv](samples.csv)
- Outputs: [validation_results](validation_results)
- Tests: [tests](tests)
- Config: [requirements.txt](requirements.txt)