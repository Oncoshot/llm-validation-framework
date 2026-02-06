# LLM Validation Framework

A comprehensive Python framework for evaluating LLM-extracted structured data against ground truth labels. Supports binary classification, scalar value extraction, and list field analysis with detailed performance metrics and confidence-based evaluation.

## ✨ Key Features

- **Multi-field validation** - Binary (True/False), scalar (single values), and list (multiple values) data types
- **Dual usage modes** - Validate pre-computed results OR run live LLM inference with validation  
- **Comprehensive metrics** - Precision, recall, F1/F2, accuracy with both micro and macro aggregation
- **Confidence analysis** - Automatic performance breakdown by confidence levels
- **Production ready** - Parallel processing, intelligent caching, detailed progress tracking

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt  # Python 3.11+ required
```

### Demo
```bash
python runme.py
```

Processes the included [samples.csv](samples.csv) (14 test cases covering all validation scenarios) and outputs timestamped results to `validation_results/samples/`:

- **[Results CSV](validation_results/samples/2026-02-06%2012-27-38%20results.csv)** - Row-by-row comparison with confusion matrix counts and item-level details   
- **[Metrics CSV](validation_results/samples/2026-02-06%2012-27-38%20metrics.csv)** - Aggregated performance statistics with confidence breakdowns

| Rows | Field Type | Test Scenarios |
|------|------------|----------------|
| **1-4** | Binary (`Has metastasis`) | True Positive, True Negative, False Positive, False Negative |
| **5-9** | Scalar (`Diagnosis`, `Histology`) | Correct, incorrect, missing, spurious, and empty extractions |
| **10-14** | List (`Treatment Drugs`, `Test Results`) | Perfect match, spurious items, missing items, correct empty, mixed results |

## 📊 Usage Modes

### Mode 1: Validate Existing Results
When you have LLM predictions in `Res: {Field Name}` columns:

```python
import pandas as pd
from src.validation import validate

df = pd.read_csv("data.csv", index_col="Patient ID")
# df must contain: "Field Name" and "Res: Field Name" columns

results_df, metrics_df = validate(
    source_df=df,
    fields=["Diagnosis", "Treatment"],  # or None for auto-detection
    structure_callback=None,
    output_folder="validation_results"
)
```

### Mode 2: Live LLM Inference + Validation

```python
from src.structured import StructuredResult, StructuredGroup, StructuredField
from src.utils import flatten_structured_result

def llm_callback(row, i, raw_text_column_name):
    raw_text = row[raw_text_column_name]
    # Your LLM inference logic here
    result = StructuredResult(
        groups=[StructuredGroup(
            group_name="medical",
            fields=[
                StructuredField(name="Diagnosis", value="Cancer", confidence="High"),
                StructuredField(name="Treatment", value=["Drug A"], confidence="Medium")
            ]
        )]
    )
    return flatten_structured_result(result), {}

results_df, metrics_df = validate(
    source_df=df,
    fields=["Diagnosis", "Treatment"],
    structure_callback=llm_callback,
    raw_text_column_name="medical_report",
    output_folder="validation_results",
    max_workers=4
)
```

## 📋 Input Data Requirements

### DataFrame Format
- **Unique index** - Each row must have a unique identifier (e.g., "Patient ID")
- **Label columns** - Ground truth values for each field you want to validate
- **Result columns** (Mode 1 only) - LLM predictions as `Res: {Field Name}` columns
- **Raw text column** (Mode 2 only) - Source text for LLM inference (e.g., "medical_report")

### Supported Field Types

| Type | Description | Label Examples | Result Examples |
|------|-------------|----------------|-----------------|
| **Binary** | True/False detection | `True`, `False` | `True`, `False` |
| **Scalar** | Single text/numeric value | `"Lung Cancer"` <br> `42` | `"Breast Cancer"` <br> `38` |
| **List** | Multiple values | `["Drug A", "Drug B"]` <br> `"['Item1', 'Item2']"` | `["Drug A"]` <br> `[]` |

### Special Value Handling
- **`"-"`** = Labeled as "No information is available in the source document"
- **`null/empty`** = Field not labeled/evaluated 
- **Lists** - Can be Python lists `["a", "b"]` or stringified `"['a', 'b']"` (auto-converted)

## 📈 Output Files

The framework generates two timestamped CSV files for each validation run:

### 1. Results CSV (`YYYY-MM-DD HH-MM-SS results.csv`)
**Row-level analysis** with detailed per-case metrics:

**Original Data:**
- All input columns (labels, raw text, etc.)
- `Res: {Field}` columns with LLM predictions 
- `Res: {Field} confidence` and `Res: {Field} justification` (if available)

**Binary Fields:**
- `TP/FP/FN/TN: {Field}` - Confusion matrix counts (1 or 0 per row)

**Non-Binary Fields:**  
- `Cor/Inc/Mis/Spu: {Field}` - Item counts per row
- `Cor/Inc/Mis/Spu: {Field} items` - Actual item lists
- `Precision/Recall/F1/F2: {Field}` - Per-row metrics (list fields only)

**System Columns:**
- `Sys: from cache` - Whether result was cached (speeds up duplicate text)
- `Sys: exception` - Error information if processing failed
- `Sys: time taken` - Processing time per row in seconds

### 2. Metrics CSV (`YYYY-MM-DD HH-MM-SS metrics.csv`)  
**Aggregated statistics** with confidence breakdowns:

**Core Information:**
- `field` - Field name being evaluated
- `confidence` - Confidence level ("Overall", "High", "Medium", "Low", etc.)  
- `labeled cases` - Total rows with ground truth labels
- `field-present cases` - Rows where document has information about the field (label is not '-')

**Binary Metrics:** `TP`, `TN`, `FP`, `FN`, `precision`, `recall`, `F1/F2`, `accuracy`, `specificity`

**Non-Binary Metrics:** `cor`, `inc`, `mis`, `spu`, `precision/recall/F1/F2 (micro)`, `precision/recall/F1/F2 (macro)`

## ⚡ Performance Metrics Explained
### Binary Classification Metrics

For fields with True/False values (e.g., "Has metastasis"):

#### Confusion Matrix Counts
| Count | Definition | Example |
|-------|------------|---------|
| **TP (True Positive)** | Correctly predicted positive | Label: `True`, Prediction: `True` → TP=1 |
| **TN (True Negative)** | Correctly predicted negative | Label: `False`, Prediction: `False` → TN=1 |
| **FP (False Positive)** | Incorrectly predicted positive | Label: `False`, Prediction: `True` → FP=1 |
| **FN (False Negative)** | Incorrectly predicted negative | Label: `True`, Prediction: `False` → FN=1 |

#### Binary Classification Formulas
| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | `TP / (TP + FP)` | Of all positive predictions, how many were correct? |
| **Recall** | `TP / (TP + FN)` | Of all actual positives, how many were found? |
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Overall percentage of correct predictions |
| **Specificity** | `TN / (TN + FP)` | Of all actual negatives, how many were correctly identified? |
### Structured Extraction Metrics

For scalar and list fields (e.g., "Diagnosis", "Treatment Drugs"):

#### Core Counts (Per Case Analysis)
| Count | Definition | Example |
|-------|------------|---------|
| **Correct (Cor)** | Items extracted correctly | Label: `["DrugA", "DrugB"]`, Prediction: `["DrugA"]` → Cor=1 |
| **Missing (Mis)** | Items present in label but not extracted | (Same example) → Mis=1 (DrugB missing) |
| **Spurious (Spu)** | Items extracted but not in label | Label: `["DrugA"]`, Prediction: `["DrugA", "DrugC"]` → Spu=1 |
| **Incorrect (Inc)** | Wrong values for scalar fields | Label: `"Cancer"`, Prediction: `"Diabetes"` → Inc=1 |

#### Structured Extraction Formulas

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | `Cor / (Cor + Spu + Inc)` | Of all extracted items, how many were correct? |
| **Recall** | `Cor / (Cor + Mis + Inc)` | Of all labeled items, how many were correctly extracted? |

**Note:** For scalar fields, Inc (incorrect) is used; for list fields, Inc is typically 0 since items are either correct, missing, or spurious.

The following formulas apply to both binary classification and structured extraction metrics:

| Metric | Formula | Meaning |
|--------|---------|--------|
| **F1 Score** | `2 × (P × R) / (P + R)` | Balanced harmonic mean of precision and recall |
| **F2 Score** | `5 × (P × R) / (4P + R)` | Recall-weighted F-score (emphasizes recall over precision) |

Where P = Precision and R = Recall (calculated differently for each metric type).

## 🛠️ Advanced Configuration

### Parallel Processing
```python
validate(
    source_df=df,
    fields=["diagnosis", "treatment"], 
    structure_callback=callback,
    max_workers=None,      # Auto-detect CPU count (or specify number)
    use_threads=True       # True for I/O-bound (LLM API calls), False for CPU-bound
)
```

### Performance Features
- **Automatic caching** - Identical raw text inputs are deduplicated and cached
- **Progress tracking** - Real-time progress bar for long-running validations  
- **Cache statistics** - Check `Sys: from cache` column in results to monitor cache hits

### Confidence Analysis  
The framework automatically detects `Res: {Field} confidence` columns and generates:
- Separate metrics for each unique confidence level found in your data
- Overall metrics aggregating across all confidence levels
- Useful for setting confidence thresholds and analyzing prediction reliability

## 🧪 Development & Testing

```bash
# Install development dependencies
pip install -r requirements.txt

# Run all tests
pytest  

# Run with coverage reporting
pytest --cov=src

# Run specific test modules
pytest tests/validate_test.py              # Core validation logic
pytest tests/compare_results_test.py       # Comparison algorithms  
pytest tests/compare_results_all_test.py   # End-to-end comparisons
```

## 📁 Project Structure

```
llm-validation-framework/
├── src/
│   ├── validation.py     # Main validation pipeline and metrics calculation
│   ├── structured.py     # Pydantic data models for LLM results
│   ├── utils.py         # Utility functions (list conversion, flattening)
│   └── standardize.py   # Data standardization helpers
├── tests/               # Comprehensive test suite
├── validation_results/  # Output directory (auto-created)
├── samples.csv         # Demo dataset with all validation scenarios  
├── runme.py           # Demo script
└── requirements.txt   # Dependencies (pandas, pydantic, tqdm, etc.)
```

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| **"Cannot infer fields"** | Ensure DataFrame has both `{Field}` and `Res: {Field}` columns when `structure_callback=None` |
| **"Missing fields"** | Verify `fields` parameter contains column names that exist in your DataFrame |
| **"Duplicate index"** | Use `df.reset_index(drop=True)` or ensure your DataFrame index has unique values |
| **Import/dependency errors** | Run `pip install -r requirements.txt` and verify Python 3.11+ |
| **Slow performance** | Enable parallel processing with `max_workers=None` and `use_threads=True` for LLM API calls |