"""Data validation test suite for ML DevOps POE project.
Tests data integrity, column availability, absence of nulls in target,
and proper separation of train and test index sets.
"""

import pandas as pd


def test_target_has_no_nulls(loaded_data, dataset_config):
    """Test 1: Verify that target column has zero null/missing values."""
    df, _, _, y_train, y_test = loaded_data
    target_col = dataset_config["target_col"]

    assert target_col in df.columns, f"Target column '{target_col}' missing from DataFrame."
    assert df[target_col].isnull().sum() == 0, f"Target column contains {df[target_col].isnull().sum()} null values."
    assert y_train.isnull().sum() == 0, "y_train contains unexpected nulls."
    assert y_test.isnull().sum() == 0, "y_test contains unexpected nulls."


def test_expected_columns_exist(loaded_data, dataset_config):
    """Test 2: Verify that expected features and target exist in the dataset."""
    df, X_train, X_test, _, _ = loaded_data
    target_col = dataset_config["target_col"]
    drop_cols = dataset_config.get("drop_cols", [])

    # The target must exist in full dataset
    assert target_col in df.columns, f"Target column '{target_col}' not found in DataFrame."

    # Dropped columns should NOT be present in features X_train or X_test
    for col in drop_cols:
        assert col not in X_train.columns, f"Dropped column '{col}' still present in X_train."
        assert col not in X_test.columns, f"Dropped column '{col}' still present in X_test."

    # Feature sets must have identical column structure and be non-empty
    assert len(X_train.columns) > 0, "Feature set X_train is empty."
    assert list(X_train.columns) == list(X_test.columns), "X_train and X_test column mismatch."


def test_no_train_test_index_leakage(loaded_data):
    """Test 3: Verify train and test sets have strictly disjoint, non-overlapping indices."""
    _, X_train, X_test, y_train, y_test = loaded_data

    # Check X splits
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)

    assert len(overlap) == 0, f"Data leakage detected! Overlapping indices between train and test: {overlap}"

    # Also verify target indices align exactly with feature indices
    assert list(X_train.index) == list(y_train.index), "X_train and y_train indices are misaligned."
    assert list(X_test.index) == list(y_test.index), "X_test and y_test indices are misaligned."

