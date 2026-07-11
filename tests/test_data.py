import os
import numpy as np

def test_raw_data_exists():
    """Check if the raw dataset was successfully downloaded."""
    assert os.path.exists("data/raw/processed.cleveland.data"), "Raw data file is missing."

def test_processed_data_shapes():
    """Verify that train and test arrays have matching row counts with their labels."""
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train row counts do not match."
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test row counts do not match."
    assert X_train.shape[1] == X_test.shape[1], "Feature columns mismatch between train and test."
