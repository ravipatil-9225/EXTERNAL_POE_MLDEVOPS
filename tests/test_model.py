"""Model quality and inference validation test suite for ML DevOps POE project.
Validates prediction shape/dimensions and confirms accuracy beats baseline
(e.g., better than the majority class naive guess).
"""

from sklearn.metrics import accuracy_score


def test_prediction_length_matches_input(trained_model, loaded_data):
    """Test 1: Model predictions must have the exact same length as input."""
    _, _, X_test, _, _ = loaded_data

    preds = trained_model.predict(X_test)

    assert len(preds) == len(X_test), (
        f"Prediction dimension mismatch: expected {len(X_test)} predictions, got {len(preds)}."
    )


def test_model_beats_majority_class_baseline(trained_model, loaded_data):
    """Test 2: Model accuracy must significantly outperform majority-class guessing baseline."""
    _, _, X_test, _, y_test = loaded_data

    # 1. Compute naive majority-class baseline accuracy
    majority_class = y_test.mode()[0]
    naive_predictions = [majority_class] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, naive_predictions)

    # 2. Compute trained model accuracy
    preds = trained_model.predict(X_test)
    model_accuracy = accuracy_score(y_test, preds)

    print(f"\n[Validation] Majority baseline accuracy: {baseline_accuracy:.4f}")
    print(f"[Validation] Trained model accuracy   : {model_accuracy:.4f}")

    # Model should beat baseline (e.g. baseline is 33% on balanced Iris, model is ~95-100%)
    assert model_accuracy > baseline_accuracy, (
        f"Model failed quality gate: model accuracy ({model_accuracy:.4f}) did not beat "
        f"majority-class baseline ({baseline_accuracy:.4f})."
    )

