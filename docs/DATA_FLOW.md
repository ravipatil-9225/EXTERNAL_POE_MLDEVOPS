# Data Flow Architecture Guide

This document explains end-to-end how data flows through the **ml_devops_poe** project, from raw CSV ingestion to automated testing, MLflow tracking, and Evidently drift monitoring.

---

## 1. Plain-English Step-by-Step Walkthrough

1. **Raw File Ingestion**:
   - The raw CSV dataset is read from disk (`data/irish/Iris.csv` or `data/domain/heart_disease.csv`).
   - The path, target column name, drop columns, and task type are retrieved strictly from `src/config.py`.

2. **Cleaning & Missing Value Imputation**:
   - Any identifier or extraneous columns specified in `config.py` (such as `Id`) are dropped.
   - Missing values are inspected: numerical columns are filled with their median value; categorical columns are filled with their mode.

3. **Train-Test Splitting**:
   - If `split_type="random"`, standard stratified random splitting is performed (`test_size=0.2`, `random_state=42`).
   - If `split_type="time"`, the dataset is partitioned chronologically without shuffling, preventing lookahead bias on temporal sequences.

4. **Model Training & Inference**:
   - A scikit-learn model (`RandomForestClassifier`, `LogisticRegression`, etc.) is fitted on the training split `(X_train, y_train)`.
   - Predictions `y_pred` are generated for `X_test`.

5. **MLflow Experiment Tracking**:
   - Training runs are grouped under the experiment name defined in `config.py`.
   - MLflow logs hyperparameters (depth, estimators, C), evaluation metrics (Accuracy, F1, RMSE), and the serialized model artifact alongside an input sample and schema signature.

6. **Automated Testing with pytest**:
   - Automated pytest fixtures load the clean splits and run 5 test assertions: verifying zero null values in target, verifying column presence, ensuring strictly disjoint train-test indices, and confirming the model's accuracy beats a naive majority-class baseline.

7. **Evidently Data Drift & Performance Analysis**:
   - **Data Drift**: Training data is divided into a reference baseline (first 70%) and a current observation window (last 30%). Two-sample Kolmogorov-Smirnov tests assess per-column distribution drift.
   - **Model Performance**: Evaluates predictions on reference (training) vs current (testing) splits.
   - HTML reports are saved locally to `reports/` and uploaded as artifacts into MLflow.

---

## 2. Mermaid Flowchart of Full Pipeline

```mermaid
flowchart TD
    A["Raw Dataset File (CSV)"] --> B["src/config.py\n(Central Settings)"]
    B --> C["src/load_data.py\n(Load, Drop Cols, Impute Nulls)"]
    C --> D{"Split Type?"}
    D -- "random" --> E["Random Train/Test Split\n(80% Train, 20% Test)"]
    D -- "time" --> F["Chronological Split\n(No Shuffling)"]
    
    E --> G["src/train.py\n(Fit Model: RF / LR)"]
    F --> G
    
    G --> H["MLflow Experiment Store\n(mlruns/)\n- Params, Metrics, Model Artifact"]
    
    E --> I["tests/ (pytest)\n- test_data.py (3 tests)\n- test_model.py (2 tests)"]
    I --> J{"All 5 Tests Pass?"}
    J -- Yes --> K["Quality Gate Passed"]
    J -- No --> L["Fail Build / Abort"]
    
    E --> M["src/drift_report.py\n(Evidently Reports)"]
    M --> N["Data Drift Report\n(Reference 70% vs Current 30%)"]
    M --> O["Performance Report\n(Train vs Test Predictions)"]
    
    N --> P["reports/*_drift.html"]
    O --> Q["reports/*_performance.html"]
    
    P --> H
    Q --> H
```

---

## 3. ASCII Flow Diagram (Text Editor Friendly)

```
+-------------------------------------------------------------------------+
|                         RAW DATASET FILE (.csv)                         |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  CENTRAL CONFIGURATION (src/config.py)                  |
|    - File path, target column, task type, split type, drop columns      |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                 LOAD & PREPROCESS (src/load_data.py)                    |
|    - Drop unwanted columns (e.g. Id)                                    |
|    - Impute missing values (median/mode)                                |
|    - Split: Random (classification) OR Time-series (chronological)      |
+-------------------------------------------------------------------------+
         |                                  |                    |
         v                                  v                    v
+-------------------+             +------------------+   +----------------+
|  MODEL TRAINING   |             |   PYTEST SUITE   |   |   EVIDENTLY    |
|   (src/train.py)  |             |     (tests/)     |   | (drift_report) |
+-------------------+             +------------------+   +----------------+
         |                                  |                    |
         | Model fit & evaluation           | 5 Quality gates    | 70/30 Split
         v                                  v                    v
+-------------------+             +------------------+   +----------------+
| MLFLOW TRACKING   |             | Pass / Fail      |   | HTML Reports   |
| - Params          |             | - Zero nulls     |   | - Drift HTML   |
| - Metrics (Acc/F1)|             | - Column check   |   | - Perf HTML    |
| - Model Artifact  |             | - Zero leakage   |   +----------------+
| - Report Artifacts|             | - Beat baseline  |           |
+-------------------+             +------------------+           |
         |                                                       |
         +<------------------------------------------------------+
         v
+-------------------------------------------------------------------------+
|                      MLFLOW USER INTERFACE (Web UI)                     |
|           Runs, Metrics Comparison, Artifacts, and HTML Reports         |
+-------------------------------------------------------------------------+
```

---

## 4. Mermaid Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Examiner
    participant Config as src/config.py
    participant Loader as src/load_data.py
    participant Trainer as src/train.py
    participant Tester as pytest (tests/)
    participant Monitor as src/drift_report.py
    participant MLflow as MLflow Store (mlruns/)

    User->>Loader: load_dataset(dataset_name)
    Loader->>Config: get_config(dataset_name)
    Config-->>Loader: Returns settings dictionary
    Loader-->>User: Returns (df, X_train, X_test, y_train, y_test)

    User->>Trainer: run_experiment_suite(dataset_name)
    Trainer->>Loader: load_dataset(dataset_name)
    Trainer->>MLflow: log_param(), log_metric(), log_model()
    MLflow-->>Trainer: Run logged (Run ID)

    User->>Tester: pytest -v
    Tester->>Loader: Requests fixture loaded_data
    Tester-->>User: All 5 assertions PASSED

    User->>Monitor: run_and_log_reports(dataset_name)
    Monitor->>Loader: load_dataset(dataset_name)
    Monitor->>Monitor: Evidently DataDriftPreset & ClassificationPreset
    Monitor->>MLflow: log_artifact(*_drift.html, *_performance.html)
    MLflow-->>User: Visual reports visible in MLflow UI
```

---

## 5. File Responsibilities, Inputs & Outputs

| File | Primary Responsibility | Input(s) | Output / Returns |
| :--- | :--- | :--- | :--- |
| `src/config.py` | Central configuration dictionary | Dataset key name (`'irish'`, `'domain'`) | Configuration dictionary (`dict`) |
| `src/load_data.py` | Reading, cleaning, imputing, splitting | Dataset key name (`str`) | `(df, X_train, X_test, y_train, y_test)` |
| `src/train.py` | Multi-run training & MLflow logging | Dataset name, model type, hyperparams | MLflow `run_id` (`str`), logs artifacts |
| `src/drift_report.py` | Drift & performance reporting | Dataset key name (`str`) | `(drift_html, perf_html)` logged to MLflow |
| `tests/conftest.py` | Shared test fixtures | Pytest test runner | Fixture objects (`loaded_data`, `trained_model`) |
| `tests/test_data.py` | Data integrity validation | Pytest fixtures | Boolean assertions (Zero nulls, column match) |
| `tests/test_model.py` | Inference & accuracy validation | Pytest fixtures | Boolean assertions (Length match, beats baseline) |

---

## 6. Output Storage Destinations

- **HTML Reports**: Saved to `ml_devops_poe/reports/` (`*_drift.html`, `*_performance.html`).
- **MLflow Tracking & Models**: Saved locally in `ml_devops_poe/mlruns/` (SQLite / file store with parameters, metrics, model artifacts, and copies of Evidently HTML files).
- **Console Terminal**: Displays inspection shapes, evaluation metrics, pytest pass/fail summaries, and drift score tables.

