import os
import numpy as np
import joblib

def test_model_and_preprocessor_exist():
    """Ensure that the training script actually saved the artifacts."""
    assert os.path.exists("models/best_model.joblib"), "Model artifact is missing."
    assert os.path.exists("models/preprocessor.joblib"), "Preprocessor artifact is missing."

def test_model_predictions():
    """Load the model and test data to ensure predictions are binary."""
    model = joblib.load("models/best_model.joblib")
    X_test = np.load("data/processed/X_test.npy")
    
    # Predict on a small batch of 5 records
    predictions = model.predict(X_test[:5])
    
    assert len(predictions) == 5, "Model did not output exactly 5 predictions."
    # Check that predictions are only 0 or 1
    assert set(predictions).issubset({0, 1}), f"Model predicted invalid classes: {set(predictions)}"
