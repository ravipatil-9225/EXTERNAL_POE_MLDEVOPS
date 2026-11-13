"""Central configuration module for ML DevOps POE project.
All dataset settings (paths, target column, task type, split type, drop columns, experiment name)
are defined here in one place so no other file hardcodes dataset details.
Supports the Irish dataset and ready configurations for every domain.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset configurations for all domains
DATASETS = {
    # 1. Primary benchmark dataset (Iris flower)
    "irish": {
        "name": "irish",
        "file_path": DATA_DIR / "irish" / "Iris.csv",
        "target_col": "Species",
        "task_type": "classification",
        "split_type": "random",
        "test_size": 0.2,
        "random_state": 42,
        "drop_cols": ["Id"],
        "experiment_name": "irish_classification",
        "reference_ratio": 0.7,
    },
    # 2. Healthcare: Heart Disease (UCI Cleveland)
    "healthcare": {
        "name": "healthcare",
        "file_path": DATA_DIR / "domain" / "heart_disease.csv",
        "target_col": "target",
        "task_type": "classification",
        "split_type": "random",
        "test_size": 0.2,
        "random_state": 42,
        "drop_cols": [],
        "experiment_name": "domain_healthcare",
        "reference_ratio": 0.7,
    },
    # 3. Agriculture: Crop Recommendation
    "agriculture": {
        "name": "agriculture",
        "file_path": DATA_DIR / "domain" / "crop_recommendation.csv",
        "target_col": "label",
        "task_type": "classification",
        "split_type": "random",
        "test_size": 0.2,
        "random_state": 42,
        "drop_cols": [],
        "experiment_name": "domain_agriculture",
        "reference_ratio": 0.7,
    },
    # 4. Insurance: Medical Insurance Charges (Regression)
    "insurance": {
        "name": "insurance",
        "file_path": DATA_DIR / "domain" / "insurance.csv",
        "target_col": "charges",
        "task_type": "regression",
        "split_type": "random",
        "test_size": 0.2,
        "random_state": 42,
        "drop_cols": [],
        "experiment_name": "domain_insurance",
        "reference_ratio": 0.7,
    },
    # 5. Banking: Bank Marketing (Deposit Subscription)
    "banking": {
        "name": "banking",
        "file_path": DATA_DIR / "domain" / "bank_marketing.csv",
        "target_col": "y",
        "task_type": "classification",
        "split_type": "random",
        "test_size": 0.2,
        "random_state": 42,
        "drop_cols": [],
        "experiment_name": "domain_banking",
        "reference_ratio": 0.7,
    },
    # 6. FinTech: Credit Card Fraud Detection
    "fintech": {
        "name": "fintech",
        "file_path": DATA_DIR / "domain" / "credit_card_fraud.csv",
        "target_col": "Class",
        "task_type": "classification",
        "split_type": "random",
        "test_size": 0.2,
        "random_state": 42,
        "drop_cols": ["Time"],
        "experiment_name": "domain_fintech",
        "reference_ratio": 0.7,
    },
    # 7. Time Series: Airline Passenger Forecast (Regression with chronological split)
    "time_series": {
        "name": "time_series",
        "file_path": DATA_DIR / "domain" / "airline_passengers.csv",
        "target_col": "Passengers",
        "task_type": "regression",
        # For time-series data, split by time rather than randomly to avoid lookahead bias
        # (shuffling future observations into the training set would lead to data leakage).
        "split_type": "time",
        "test_size": 0.2,
        "random_state": 42,
        "drop_cols": [],
        "experiment_name": "domain_time_series",
        "reference_ratio": 0.7,
    },
}

# Alias "domain" points to healthcare by default
DATASETS["domain"] = DATASETS["healthcare"]


def get_config(dataset_name: str) -> dict:
    """Retrieve dataset configuration by name.

    Args:
        dataset_name: Name of the dataset ("irish", "healthcare", "agriculture", etc.)

    Returns:
        dict: Configuration dictionary for the chosen dataset
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available options: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name]
