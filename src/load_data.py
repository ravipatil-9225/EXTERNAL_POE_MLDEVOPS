"""Data loading and preprocessing module for ML DevOps POE project.
Loads dataset based on central config, handles missing values,
encodes categorical features, creates lag features for time series,
and splits into train and test sets using random or time split.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from config import get_config
except ImportError:
    from src.config import get_config


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, and split dataset by name using settings from config.py.

    Args:
        name: Identifier of the dataset ('irish', 'healthcare', 'agriculture', etc.).

    Returns:
        df: Full cleaned pandas DataFrame.
        X_train: Training features DataFrame.
        X_test: Testing features DataFrame.
        y_train: Training target Series.
        y_test: Testing target Series.
    """
    cfg = get_config(name)
    file_path = Path(cfg["file_path"])

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path}. Please place the file in the data folder."
        )

    # 1. Read raw CSV
    df = pd.read_csv(file_path)

    # 2. Time-series feature engineering if applicable (create lag features so X is not empty)
    if cfg.get("split_type") == "time" and "Month" in df.columns:
        # Convert date and generate lag1 and lag2 autoregressive features
        df = df.sort_values(by="Month").reset_index(drop=True)
        target_name = cfg["target_col"]
        df["lag_1"] = df[target_name].shift(1)
        df["lag_2"] = df[target_name].shift(2)
        df = df.dropna().reset_index(drop=True)
        # Drop raw string timestamp column
        if "Month" not in cfg.get("drop_cols", []):
            cfg.setdefault("drop_cols", []).append("Month")

    # 3. Drop unwanted columns (e.g., ID columns) if configured
    drop_cols = [c for c in cfg.get("drop_cols", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 4. Handle missing values: median for numeric, mode for categorical
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    target_col = cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    # 5. Separate features X and target y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 6. Encode categorical features in X if any (e.g. for insurance / banking)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes

    test_size = cfg.get("test_size", 0.2)
    random_state = cfg.get("random_state", 42)
    split_type = cfg.get("split_type", "random")

    # 7. Train / Test split
    if split_type == "time":
        # Time-series split: preserve chronological order (no random shuffle)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        # Standard random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

    return df, X_train, X_test, y_train, y_test


def inspect_dataset(name: str) -> None:
    """Print shape, column types, and missing values for viva demonstration."""
    cfg = get_config(name)
    print("=" * 60)
    print(f"DATASET INSPECTION: {name.upper()}")
    print("=" * 60)
    print(f"Source file : {cfg['file_path']}")
    print(f"Target col  : {cfg['target_col']}")
    print(f"Task type   : {cfg['task_type']}")
    print(f"Split type  : {cfg['split_type']}")

    df, X_train, X_test, y_train, y_test = load_dataset(name)

    print(f"\n1. Full DataFrame Shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})")
    print(f"   Train split shape   : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"   Test split shape    : X_test={X_test.shape}, y_test={y_test.shape}")

    print("\n2. Feature Column Data Types:")
    for col, dtype in X_train.dtypes.items():
        print(f"   - {col:20s}: {dtype}")

    print("\n3. Missing Value Counts per Column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        print(f"   - {col:20s}: {count} missing")

    print(f"\n4. Target Column ('{cfg['target_col']}') Summary:")
    if cfg["task_type"] == "classification":
        print(y_train.value_counts().head(5).to_string())
    else:
        print(f"   Mean = {y_train.mean():.2f}, Std = {y_train.std():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset("irish")
