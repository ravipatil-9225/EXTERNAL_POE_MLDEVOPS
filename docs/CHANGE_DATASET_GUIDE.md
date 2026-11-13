# Guide: Changing or Adding a New Dataset

This guide provides step-by-step instructions for running this ML DevOps pipeline on any new dataset without breaking existing tests or mixing experiment runs.

---

## 1. Which Files to Change vs Leave Alone

| File | Status | Action Required |
| :--- | :--- | :--- |
| **`src/config.py`** | **CHANGE** | Add or update the dataset configuration dictionary. |
| **`data/`** | **CHANGE** | Place your new dataset CSV into `data/domain/` or a dedicated folder. |
| **`tests/conftest.py`** | **OPTIONAL** | Only change if you want pytest to test the new dataset by default (`dataset_name = "domain"`). |
| **`src/load_data.py`** | **DO NOT TOUCH** | Reads everything dynamically from `config.py`. |
| **`src/train.py`** | **DO NOT TOUCH** | Trains RF / LogisticRegression / Ridge dynamically based on task type. |
| **`src/drift_report.py`** | **DO NOT TOUCH** | Auto-detects task type and produces matching drift/performance reports. |
| **`tests/test_data.py`** | **DO NOT TOUCH** | Tests generic data contracts (null checks, columns, disjoint indices). |
| **`tests/test_model.py`** | **DO NOT TOUCH** | Tests generic model invariants (length match, beats baseline). |

---

## 2. Exact Fields to Edit in `src/config.py`

Every dataset in `DATASETS` contains the following keys:

```python
"new_dataset": {
    "name": "new_dataset",                      # Unique key identifier
    "file_path": DATA_DIR / "domain" / "my_data.csv",  # Path to CSV
    "target_col": "target_column_name",        # Name of the column to predict
    "task_type": "classification",             # "classification" or "regression"
    "split_type": "random",                    # "random" or "time"
    "test_size": 0.2,                          # Test split ratio (e.g. 0.2 = 20%)
    "random_state": 42,                        # Reproducibility seed
    "drop_cols": ["id_column"],                # Columns to exclude from features
    "experiment_name": "new_dataset_exp",      # Distinct MLflow experiment name
    "reference_ratio": 0.7,                    # Evidently reference split (first 70%)
}
```

### Complete Filled Example (Insurance Charges Regression):
```python
"domain": {
    "name": "domain",
    "file_path": DATA_DIR / "domain" / "insurance.csv",
    "target_col": "charges",
    "task_type": "regression",
    "split_type": "random",
    "test_size": 0.2,
    "random_state": 42,
    "drop_cols": [],
    "experiment_name": "domain_insurance_regression",
    "reference_ratio": 0.7,
}
```

---

## 3. How to Choose Target Column and Task Type

- **Target Column**: Identify what the business problem wants to predict:
  - In credit card data $\rightarrow$ `Class` (0 = normal, 1 = fraud).
  - In customer marketing $\rightarrow$ `y` (yes / no).
  - In housing / insurance $\rightarrow$ `price` or `charges` (continuous dollar amount).
- **Task Type**:
  - If the target is discrete, categorical, or binary (e.g. `Species`, `disease present/absent`), set `"task_type": "classification"`.
  - If the target is a continuous numeric number (e.g. `charges`, `passengers`, `sales`), set `"task_type": "regression"`.

---

## 4. How to Choose Random Split vs Time Split

- **Use `"split_type": "random"`** when observations are independent and identically distributed (i.i.d.), such as medical patients, flower measurements, or loan applications.
- **Use `"split_type": "time"`** when observations have a temporal sequence (e.g. daily sales, hourly traffic, monthly passenger counts). 
  - *Why?* Shuffling time-series data leads to **lookahead bias** (training on future information to predict past events). A chronological split strictly trains on the past $[t_0 \dots t_k]$ and tests on the future $[t_{k+1} \dots t_N]$.

---

## 5. How to Update Tests for the New Dataset

In `tests/conftest.py`:
Change line 18:
```python
@pytest.fixture(scope="session")
def dataset_name():
    return "domain"  # Change from "irish" to "domain" or your new dataset key
```
Because `tests/test_data.py` and `tests/test_model.py` dynamically query `dataset_config["target_col"]` and `X_train.columns`, **no other test lines need to be modified!**

---

## 6. How to Prevent MLflow Runs from Mixing

Always give each dataset a distinct `"experiment_name"` in `src/config.py` (e.g., `irish_classification`, `domain_heart_disease`, `domain_insurance`).
- When `mlflow.set_experiment(experiment_name)` executes, MLflow creates an isolated namespace.
- In the MLflow UI, each experiment appears as a separate tab on the left navigation bar, keeping metrics and artifacts completely segregated.

---

## 7. Verification Checklist

1. [ ] **Step 1**: Place new CSV into `data/domain/<filename>.csv`.
2. [ ] **Step 2**: Update `domain` entry in `src/config.py`.
3. [ ] **Step 3**: Verify loading and missing value handling:
   ```powershell
   python -c "import sys; sys.path.insert(0, 'src'); from load_data import inspect_dataset; inspect_dataset('domain')"
   ```
4. [ ] **Step 4**: Run training suite and verify MLflow logging:
   ```powershell
   python -c "import sys; sys.path.insert(0, 'src'); from train import run_experiment_suite; run_experiment_suite('domain')"
   ```
5. [ ] **Step 5**: Run Evidently report generation:
   ```powershell
   python -c "import sys; sys.path.insert(0, 'src'); from drift_report import run_and_log_reports; run_and_log_reports('domain')"
   ```
6. [ ] **Step 6**: Confirm new HTML files exist in `reports/` (`domain_drift.html`, `domain_performance.html`).
7. [ ] **Step 7**: Launch `mlflow ui` and confirm the new experiment appears with 3 runs and logged artifacts.

---

## 8. Common Errors & Quick Fixes

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError` | Path in `config.py` does not match the actual file location. | Double check `file_path` in `config.py` and filename casing. |
| `KeyError: 'target'` | Column name in `target_col` does not match CSV header. | Run `python src/load_data.py` to inspect column headers and fix the spelling in `config.py`. |
| `ValueError: could not convert string to float` | Categorical feature passed directly into regression/classifier without encoding. | Use numeric columns or ensure categorical columns are dropped/encoded. |
| `UnicodeDecodeError` | CSV encoded with non-standard character set (e.g. latin1). | Add `encoding="latin1"` or `encoding="utf-8-sig"` to `pd.read_csv()` in `src/load_data.py`. |
| `Port 5000 already in use` | Another instance of MLflow is already running. | Run `mlflow ui --port 5001`. |

