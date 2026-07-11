import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def load_data_and_features():
    """Loads the preprocessed data and the feature names from the saved preprocessor."""
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    preprocessor = joblib.load('models/preprocessor.joblib')
    return X_train, X_test, y_train, y_test, preprocessor.get_feature_names_out()

def save_visualizations(model, X_test, y_test, model_name, feature_names):
    """Generates and saves the Confusion Matrix, ROC Curve, and Feature Importance plots."""
    os.makedirs('plots', exist_ok=True)
    safe_name = model_name.replace(' ', '_').lower()

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(f"plots/{safe_name}_confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC Curve')
    plt.savefig(f"plots/{safe_name}_roc_curve.png", bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance
    plt.figure(figsize=(10, 8))
    if hasattr(model, 'coef_'):
        importance = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return
        
    indices = np.argsort(np.abs(importance))
    plt.barh(range(len(indices)), importance[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(f'{model_name} - Feature Importance')
    plt.tight_layout()
    plt.savefig(f"plots/{safe_name}_feature_importance.png", bbox_inches='tight')
    plt.close()

def train_and_tune_model(model, param_grid, X_train, y_train, model_name):
    """Performs hyperparameter tuning using GridSearchCV."""
    print(f"\nRunning GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names):
    """Evaluates the model, generates visualizations, and logs everything to MLflow."""
    print(f"\n--- Evaluating {model_name} ---")
    
    with mlflow.start_run(run_name=model_name):
        # Log Tuned Parameters
        mlflow.log_params(model.get_params())
        
        # Predict on Test Set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)
        test_f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metrics({
            "accuracy": test_acc,
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": test_f1,
            "roc_auc": test_auc
        })
        
        print(f"Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test ROC-AUC: {test_auc:.4f}")
        
        # Generate Visualizations and Log Artifacts
        save_visualizations(model, X_test, y_test, model_name, feature_names)
        mlflow.log_artifacts("plots", artifact_path="evaluation_plots")
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        return test_acc, test_auc

def main():
    # ---------------------------------------------------------
    # FIX: Force MLflow to use the current working directory's absolute path.
    # This prevents Mac/Linux path conflicts in GitHub Actions.
    # ---------------------------------------------------------
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    current_dir = os.path.abspath(os.getcwd())
    tracking_uri = "file://" + os.path.join(current_dir, "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI forcefully set to: {tracking_uri}")
    
    # Initialize the MLflow experiment
    mlflow.set_experiment("Heart_Disease_Prediction_MLOps")
    
    # Load Data
    X_train, X_test, y_train, y_test, feature_names = load_data_and_features()
    
    # Define Parameter Grids for Tuning
    log_reg_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    
    # Initialize Base Models
    log_reg_base = LogisticRegression(random_state=42, max_iter=1000)
    rf_base = RandomForestClassifier(random_state=42)
    
    # Tune Models
    best_log_reg = train_and_tune_model(log_reg_base, log_reg_params, X_train, y_train, "Logistic Regression")
    best_rf = train_and_tune_model(rf_base, rf_params, X_train, y_train, "Random Forest")
    
    # Evaluate and Track Models via MLflow
    lr_acc, lr_auc = evaluate_model(best_log_reg, X_train, y_train, X_test, y_test, "Logistic Regression", feature_names)
    rf_acc, rf_auc = evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Random Forest", feature_names)
    
    # Save the Best Performing Model for the API
    final_model = best_log_reg if lr_auc > rf_auc else best_rf
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/best_model.joblib')
    print("\nTraining complete. Best tuned model securely saved to 'models/best_model.joblib'.")

if __name__ == "__main__":
    main()
