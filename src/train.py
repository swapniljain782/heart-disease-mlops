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

# ... (Keep load_data_and_features, save_visualizations, train_and_tune_model as they are) ...

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names):
    """Evaluates the model, generates visualizations, and logs everything to MLflow."""
    print(f"\n--- Evaluating {model_name} ---")
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        })
        
        save_visualizations(model, X_test, y_test, model_name, feature_names)
        mlflow.log_artifacts("plots", artifact_path="evaluation_plots")
        
        # UPDATED: Log model with registration name for the Model Registry
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name="HeartDiseaseModel"
        )
        
        return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_proba)

def main():
    # ---------------------------------------------------------
    # UPDATED: Use Environment Variable first, fallback to local
    # ---------------------------------------------------------
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Fallback to local storage if running outside Jenkins/Azure environment
        current_dir = os.path.abspath(os.getcwd())
        mlflow.set_tracking_uri("file://" + os.path.join(current_dir, "mlruns"))
    
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    mlflow.set_experiment("Heart_Disease_Prediction_MLOps")
    
    X_train, X_test, y_train, y_test, feature_names = load_data_and_features()
    
    # ... (Keep param grids and model initialization as they are) ...
    log_reg_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    
    log_reg_base = LogisticRegression(random_state=42, max_iter=1000)
    rf_base = RandomForestClassifier(random_state=42)
    
    best_log_reg = train_and_tune_model(log_reg_base, log_reg_params, X_train, y_train, "Logistic Regression")
    best_rf = train_and_tune_model(rf_base, rf_params, X_train, y_train, "Random Forest")
    
    lr_acc, lr_auc = evaluate_model(best_log_reg, X_train, y_train, X_test, y_test, "Logistic Regression", feature_names)
    rf_acc, rf_auc = evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Random Forest", feature_names)
    
    final_model = best_log_reg if lr_auc > rf_auc else best_rf
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/best_model.joblib')
    print("\nTraining complete. Model saved locally and registered in MLflow.")

if __name__ == "__main__":
    main()
