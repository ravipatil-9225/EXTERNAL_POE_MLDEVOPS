# ML DevOps Laboratory Performance POE Project

A runnable, end-to-end Machine Learning DevOps demonstration project built for Laboratory Performance Evaluation (15 marks). It demonstrates three core pillars of modern ML engineering:
1. **Experiment Tracking & Model Registry** using **MLflow**
2. **Automated Testing & Data Validation** using **pytest**
3. **Data Drift & Model Performance Monitoring** using **Evidently**

The pipeline runs first on the benchmark **Irish (Iris)** flower dataset, and seamlessly generalizes to any **domain dataset** (Healthcare, Agriculture, Insurance, Banking, Time Series) through a single central configuration file.

---

## Tool Mapping Table

| Tool | Core ML DevOps Task | Role in this Project |
| :--- | :--- | :--- |
| **MLflow** | Experiment Tracking & Artifact Logging | Tracks hyperparameter combinations, logs evaluation metrics (Accuracy, F1, RMSE), saves serialized model artifacts with schemas and input examples, and logs Evidently HTML reports into an interactive UI. |
| **pytest** | Automated Testing & Quality Gates | Validates data integrity (zero nulls, non-empty features, column checks), verifies zero train-test data leakage, and ensures model performance beats a naive majority-class baseline before deployment. |
| **Evidently** | Data Drift & Model Monitoring | Generates visual HTML reports comparing reference (historical/training) vs current (production/testing) data distributions using two-sample statistical tests (e.g. Kolmogorov-Smirnov) and monitors degradation in classification/regression performance. |

---

## One-Paragraph Explanation of Each Tool's Role

- **MLflow**: In production machine learning, models are iterated continuously. Without structured tracking, hyperparameter choices, random seeds, and evaluation metrics are easily lost. MLflow automatically logs runs under named experiments, captures parameters and performance metrics, and packages trained model binaries with input-output signatures for auditability and reproducibility.
- **pytest**: In software engineering, code changes are protected by tests; in ML DevOps, test suites must additionally validate data assumptions and model inference behavior. pytest runs automated test fixtures that enforce data contracts (preventing silent schema corruption or null values), detect data leakage between splits, and ensure that a trained model never degrades below baseline acceptance thresholds.
- **Evidently**: Even if a deployed model is statistically sound at launch, real-world data inevitably drifts over time due to seasonal shifts, user behavior changes, or upstream data pipeline errors. Evidently quantifies statistical distance between reference and production distributions on a per-column basis, flags distribution drift, and produces interactive visual reports that pinpoint exactly which features degraded.

---

## Project Structure & File Guide

```
ml_devops_poe/
├── data/
│   ├── irish/Iris.csv              # Primary Irish benchmark dataset
│   └── domain/                     # Pre-packaged domain datasets (Healthcare, Agriculture, etc.)
├── src/
│   ├── config.py                   # Central configuration (paths, target, task type, split type)
│   ├── load_data.py                # Generic data loader, imputer, and random/time split logic
│   ├── train.py                    # Multi-run model trainer logging params, metrics, and models to MLflow
│   └── drift_report.py             # Generates Evidently drift and performance HTML reports and logs to MLflow
├── tests/
│   ├── conftest.py                 # Shared pytest fixtures for data and model
│   ├── test_data.py                # 3 data validation tests (no nulls, columns exist, zero index leakage)
│   └── test_model.py               # 2 model quality tests (shape match, beats majority baseline)
├── reports/                        # Saved Evidently visual HTML reports
├── mlruns/                         # Local MLflow file-based experiment tracking store
├── docs/
│   ├── DATA_FLOW.md                # Data movement walkthrough, Mermaid charts, and ASCII diagrams
│   └── CHANGE_DATASET_GUIDE.md     # Step-by-step instructions to switch to any new dataset
├── run_all.bat                     # 1-click Windows runner script
├── run_all.sh                      # 1-click Linux/Mac runner script
├── requirements.txt                # Pinned dependency versions
├── COMMANDS.txt                    # Plain text command cheat-sheet for lab viva
└── README.md                       # Project documentation
```

---

## Setup Instructions

### 1. Create and Activate Virtual Environment
**Windows (PowerShell / CMD):**
```powershell
cd ml_devops_poe
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux / macOS:**
```bash
cd ml_devops_poe
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Pinned Dependencies
```powershell
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option A: One-Click Full Execution
- **Windows**: `.\run_all.bat`
- **Linux/Mac**: `bash run_all.sh`

### Option B: Step-by-Step Execution
1. **Inspect Data Loading**:
   ```powershell
   python src/load_data.py
   ```
2. **Train Models and Log to MLflow**:
   ```powershell
   python src/train.py
   ```
3. **Run Automated pytest Suite**:
   ```powershell
   pytest -v
   ```
4. **Generate Evidently HTML Reports and Log to MLflow**:
   ```powershell
   python src/drift_report.py
   ```

---

## Opening the MLflow UI & Comparing Runs

1. Launch the UI server:
   ```powershell
   mlflow ui
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
3. **In the UI**:
   - On the left sidebar, click the experiment name (`irish_classification` or `domain_heart_disease`).
   - Check the boxes next to 2 or more runs, then click **Compare** at the top of the table.
   - Inspect the side-by-side metric table, scatter plots, and hyperparameter comparisons.
   - Click any individual run $\rightarrow$ scroll to **Artifacts** $\rightarrow$ view the logged `model` and the visual reports under `evidently_reports/`.

