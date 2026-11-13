@echo off
REM ==============================================================================
REM End-to-End Pipeline Execution Script for ML DevOps POE (Windows)
REM Runs data inspection, training on both datasets, pytest tests,
REM and Evidently drift + performance reports with MLflow artifact logging.
REM ==============================================================================

echo [1/5] Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Error activating virtual environment!
    exit /b %ERRORLEVEL%
)

echo [2/5] Inspecting and loading datasets...
python src\load_data.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [3/5] Running MLflow training runs on Irish dataset...
python src\train.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [3b/5] Running MLflow training runs on Domain dataset...
python -c "import sys; sys.path.insert(0, 'src'); from train import run_experiment_suite; run_experiment_suite('domain')"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [4/5] Executing pytest test suite...
pytest -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo [5/5] Generating Evidently HTML drift and performance reports and logging to MLflow...
python src\drift_report.py
python -c "import sys; sys.path.insert(0, 'src'); from drift_report import run_and_log_reports; run_and_log_reports('domain')"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo ==============================================================================
echo PIPELINE COMPLETED SUCCESSFULLY!
echo  - MLflow experiments logged in local mlruns/
echo  - Evidently HTML reports saved in reports/
echo  - Launch the UI with: mlflow ui
echo ==============================================================================
