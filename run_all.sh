#!/usr/bin/env bash
# ==============================================================================
# End-to-End Pipeline Execution Script for ML DevOps POE (Linux / macOS)
# Runs data inspection, training on both datasets, pytest tests,
# and Evidently drift + performance reports with MLflow artifact logging.
# ==============================================================================
set -e

echo "[1/5] Activating virtual environment..."
source .venv/bin/activate || source .venv/Scripts/activate

echo "[2/5] Inspecting and loading datasets..."
python src/load_data.py

echo "[3/5] Running MLflow training runs on Irish dataset..."
python src/train.py

echo "[3b/5] Running MLflow training runs on Domain dataset..."
python -c "import sys; sys.path.insert(0, 'src'); from train import run_experiment_suite; run_experiment_suite('domain')"

echo "[4/5] Executing pytest test suite..."
pytest -v

echo "[5/5] Generating Evidently HTML drift & performance reports and logging to MLflow..."
python src/drift_report.py
python -c "import sys; sys.path.insert(0, 'src'); from drift_report import run_and_log_reports; run_and_log_reports('domain')"

echo "=============================================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo " - MLflow experiments logged in local mlruns/"
echo " - Evidently HTML reports saved in reports/"
echo " - Launch the UI with: mlflow ui"
echo "=============================================================================="

