"""Shared pytest fixtures for the ML DevOps POE project.
Provides clean, reusable fixtures for data loading and baseline models
so tests in test_data.py and test_model.py stay concise and DRY.
"""

import sys
from pathlib import Path
import pytest
from sklearn.ensemble import RandomForestClassifier

# Ensure 'src' is in python search path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import get_config
from load_data import load_dataset


@pytest.fixture(scope="session")
def dataset_name():
    """Default dataset under test."""
    return "irish"


@pytest.fixture(scope="session")
def dataset_config(dataset_name):
    """Retrieve central configuration dictionary."""
    return get_config(dataset_name)


@pytest.fixture(scope="session")
def loaded_data(dataset_name):
    """Load dataset tuple: (df, X_train, X_test, y_train, y_test)."""
    return load_dataset(dataset_name)


@pytest.fixture(scope="session")
def trained_model(loaded_data, dataset_config):
    """Train a simple baseline model on training split for model testing."""
    _, X_train, _, y_train, _ = loaded_data
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

