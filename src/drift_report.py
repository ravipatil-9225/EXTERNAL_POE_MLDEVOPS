"""Evidently data drift and model performance reporting module for ML DevOps POE project.
Generates:
1. Data Drift Report: compares reference dataset (first 70% of train) vs current dataset (last 30%).
2. Model Performance Report: evaluates model quality on reference (train) vs current (test) distributions.
Saves both reports as HTML in reports/ and logs them as artifacts to the MLflow experiment.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import mlflow
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# Robust Evidently import (supports Evidently 0.4+ / 0.7+)
try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, ClassificationPreset, RegressionPreset
except ImportError:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset, RegressionPreset

try:
    from config import get_config, REPORTS_DIR
    from load_data import load_dataset
except ImportError:
    from src.config import get_config, REPORTS_DIR
    from src.load_data import load_dataset


def generate_drift_report(dataset_name: str) -> Tuple[Path, Dict[str, Any]]:
    """Generate Evidently Data Drift Report for reference (70%) vs current (30%) splits.

    Args:
        dataset_name: 'irish' or 'domain'

    Returns:
        tuple: (output_html_path, drift_summary_dict)
    """
    cfg = get_config(dataset_name)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{dataset_name}_drift.html"

    # Load cleaned features and target
    _, X_train, _, _, _ = load_dataset(dataset_name)

    # Reference = first 70% of train, Current = last 30% of train
    ref_ratio = cfg.get("reference_ratio", 0.7)
    ref_split_idx = int(len(X_train) * ref_ratio)

    reference_df = X_train.iloc[:ref_split_idx].copy()
    current_df = X_train.iloc[ref_split_idx:].copy()

    print(f"\n[Evidently] Computing Data Drift for '{dataset_name}':")
    print(f"   Reference sample size : {len(reference_df)} rows ({ref_ratio * 100:.0f}%)")
    print(f"   Current sample size   : {len(current_df)} rows ({(1 - ref_ratio) * 100:.0f}%)")

    # Run Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    drift_report.save_html(str(report_path))

    # Parse key summary metrics
    report_dict = drift_report.as_dict()
    drift_result = report_dict["metrics"][0]["result"]

    drift_summary = {
        "dataset_drift_detected": bool(drift_result.get("dataset_drift", False)),
        "share_of_drifted_columns": float(drift_result.get("share_of_drifted_columns", 0.0)),
        "number_of_drifted_columns": int(drift_result.get("number_of_drifted_columns", 0)),
        "total_columns": int(drift_result.get("number_of_columns", len(X_train.columns))),
    }

    print(f"   Dataset Drift Detected: {drift_summary['dataset_drift_detected']}")
    print(f"   Drifted Columns Count : {drift_summary['number_of_drifted_columns']} / {drift_summary['total_columns']}")
    print(f"   Share of Drifted Cols : {drift_summary['share_of_drifted_columns']:.2%}")
    print(f"   Saved Report HTML to  : {report_path}")

    # Inspect column-level p-values
    drift_by_columns = drift_result.get("drift_by_columns", {})
    if drift_by_columns:
        print("   Per-Column Drift Breakdown:")
        for col_name, col_info in drift_by_columns.items():
            drift_status = "DRIFTED" if col_info.get("drift_detected") else "STABLE"
            p_val = col_info.get("p_value", 0.0)
            stattest = col_info.get("stattest_name", "test")
            print(f"     - {col_name:18s}: {drift_status:7s} | p-value={p_val:.4f} ({stattest})")

    return report_path, drift_summary


def generate_performance_report(dataset_name: str) -> Path:
    """Generate Evidently Model Performance Report comparing reference (train) vs current (test).

    Args:
        dataset_name: 'irish' or 'domain'

    Returns:
        Path: Path to saved HTML report
    """
    cfg = get_config(dataset_name)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{dataset_name}_performance.html"

    task_type = cfg["task_type"]
    _, X_train, X_test, y_train, y_test = load_dataset(dataset_name)

    # Train a standard model to evaluate performance
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        preset = ClassificationPreset()
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        preset = RegressionPreset()

    model.fit(X_train, y_train)

    # Build reference and current dataframes with target and prediction columns
    reference_eval = X_train.copy()
    reference_eval["target"] = y_train
    reference_eval["prediction"] = model.predict(X_train)

    current_eval = X_test.copy()
    current_eval["target"] = y_test
    current_eval["prediction"] = model.predict(X_test)

    print(f"\n[Evidently] Computing Model Performance for '{dataset_name}':")
    perf_report = Report(metrics=[preset])
    perf_report.run(reference_data=reference_eval, current_data=current_eval)
    perf_report.save_html(str(report_path))

    print(f"   Saved Performance HTML: {report_path}")
    return report_path


def run_and_log_reports(dataset_name: str = "irish") -> None:
    """Generate both reports and log them as artifacts to an MLflow run."""
    cfg = get_config(dataset_name)
    experiment_name = cfg["experiment_name"]

    print("=" * 65)
    print(f"EVIDENTLY REPORT GENERATION & MLFLOW LOGGING: {dataset_name.upper()}")
    print("=" * 65)

    # 1. Generate Evidently HTML reports
    drift_html, drift_metrics = generate_drift_report(dataset_name)
    perf_html = generate_performance_report(dataset_name)

    # 2. Log both HTML files to an MLflow run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{dataset_name}_evidently_monitoring") as run:
        # Log drift summary metrics
        mlflow.log_metric("drift_share", drift_metrics["share_of_drifted_columns"])
        mlflow.log_metric("drifted_columns_count", drift_metrics["number_of_drifted_columns"])
        mlflow.log_param("dataset_drift_detected", str(drift_metrics["dataset_drift_detected"]))

        # Log HTML artifacts to MLflow
        mlflow.log_artifact(str(drift_html), artifact_path="evidently_reports")
        mlflow.log_artifact(str(perf_html), artifact_path="evidently_reports")

        print(f"\n[MLflow] Successfully logged HTML artifacts to MLflow Run: {run.info.run_id}")
        print(f"   - Artifact: evidently_reports/{drift_html.name}")
        print(f"   - Artifact: evidently_reports/{perf_html.name}")
    print("=" * 65)


if __name__ == "__main__":
    run_and_log_reports("irish")

