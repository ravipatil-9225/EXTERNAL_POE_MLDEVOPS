"""Training and MLflow experiment tracking module for ML DevOps POE project.
Trains models, logs parameters, metrics (accuracy, F1, RMSE), and models
with signatures and input examples under the configured MLflow experiment.
"""

from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
)

try:
    from config import get_config
    from load_data import load_dataset
except ImportError:
    from src.config import get_config
    from src.load_data import load_dataset


def train_model(
    dataset_name: str,
    model_type: str = "rf",
    hyperparams: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> str:
    """Train a model on the specified dataset and log the run to MLflow.

    Args:
        dataset_name: Name of dataset in config ('irish' or 'domain').
        model_type: 'rf' for Random Forest, 'lr' for LogisticRegression / Ridge.
        hyperparams: Dictionary of hyperparameters.
        run_name: Optional display name for the MLflow run.

    Returns:
        str: MLflow Run ID.
    """
    if hyperparams is None:
        hyperparams = {}

    cfg = get_config(dataset_name)
    experiment_name = cfg["experiment_name"]
    task_type = cfg["task_type"]
    test_size = cfg.get("test_size", 0.2)

    # Set the experiment in MLflow
    mlflow.set_experiment(experiment_name)

    # Load cleaned data and train/test splits
    df, X_train, X_test, y_train, y_test = load_dataset(dataset_name)

    # Initialize model based on task type and chosen algorithm
    if task_type == "classification":
        if model_type == "rf":
            n_estimators = hyperparams.get("n_estimators", 100)
            max_depth = hyperparams.get("max_depth", None)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=cfg.get("random_state", 42),
            )
            model_name = "RandomForestClassifier"
        elif model_type == "lr":
            C = hyperparams.get("C", 1.0)
            max_iter = hyperparams.get("max_iter", 200)
            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=cfg.get("random_state", 42),
            )
            model_name = "LogisticRegression"
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' for classification.")
    else:  # regression
        if model_type == "rf":
            n_estimators = hyperparams.get("n_estimators", 100)
            max_depth = hyperparams.get("max_depth", None)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=cfg.get("random_state", 42),
            )
            model_name = "RandomForestRegressor"
        elif model_type == "lr":
            alpha = hyperparams.get("alpha", 1.0)
            model = Ridge(alpha=alpha, random_state=cfg.get("random_state", 42))
            model_name = "RidgeRegression"
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' for regression.")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n---> Starting MLflow Run: {run_name or run_id}")

        # 1. Fit the model
        model.fit(X_train, y_train)

        # 2. Make predictions
        y_pred = model.predict(X_test)

        # 3. Log Parameters
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", test_size)
        for param_key, param_val in hyperparams.items():
            mlflow.log_param(param_key, param_val)

        # 4. Compute and Log Metrics
        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            print(f"     Metrics: Accuracy = {acc:.4f} | F1 (weighted) = {f1:.4f}")
        else:
            # Regression metrics
            try:
                rmse = root_mean_squared_error(y_test, y_pred)
            except AttributeError:
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            print(f"     Metrics: RMSE = {rmse:.4f}")

        # 5. Log Model with Signature & Input Example
        input_example = X_train.iloc[:3]
        signature = infer_signature(X_train, model.predict(X_train))

        # Pass skops_trusted_types or serialization_format to avoid skops tree trust checks
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        print(f"     Logged model artifact to run: {run_id}")
        return run_id


def run_experiment_suite(dataset_name: str = "irish") -> None:
    """Run 3 distinct model configurations so the MLflow UI has meaningful runs to compare."""
    print("=" * 65)
    print(f"EXECUTING 3 MLFLOW RUNS FOR EXPERIMENT: {dataset_name.upper()}")
    print("=" * 65)

    # Run 1: Random Forest with 20 trees and shallow depth
    train_model(
        dataset_name=dataset_name,
        model_type="rf",
        hyperparams={"n_estimators": 20, "max_depth": 2},
        run_name="rf_20trees_depth2",
    )

    # Run 2: Random Forest with 100 trees and depth 5
    train_model(
        dataset_name=dataset_name,
        model_type="rf",
        hyperparams={"n_estimators": 100, "max_depth": 5},
        run_name="rf_100trees_depth5",
    )

    # Run 3: Logistic Regression with C=0.5
    train_model(
        dataset_name=dataset_name,
        model_type="lr",
        hyperparams={"C": 0.5, "max_iter": 300},
        run_name="logistic_regression_C0.5",
    )

    print("\n" + "=" * 65)
    print("All 3 MLflow runs logged successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_experiment_suite("irish")
