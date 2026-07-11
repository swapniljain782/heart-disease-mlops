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

# 1. HELPER FUNCTIONS DEFINED FIRST
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(f"plots/{safe_name}_confusion_matrix.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC Curve')
    plt.savefig(f"plots/{safe_name}_roc_curve.png", bbox_inches='tight')
    plt.close()
    
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
    print(f"\nRunning GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names):
    print(f"\n--- Evaluating {model_name} ---")
    
    # We use 'run' to get the run_id explicitly
    with mlflow.start_run(run_name=model_name) as run:
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
        
        # 1. Log the model artifact only (no registration here)
        model_info = mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model"
        )
        
        # 2. Explicitly register the model using the captured model_info.model_uri
        # This is the standard, reliable pattern in MLflow 3.x
        try:
            result = mlflow.register_model(
                model_uri=model_info.model_uri,
                name="HeartDiseaseModel"
            )
            print(f"✅ Model registered successfully: {result.name} version {result.version}")
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
        
        return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_proba)
    
# 2. MAIN EXECUTION
def main():
    # Force MLflow configurations
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Priority: Env Var > Default Remote > Local fallback
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://20.17.177.233:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    mlflow.set_experiment("Heart_Disease_Prediction_MLOps")
    
    # Now load_data_and_features is defined and accessible
    X_train, X_test, y_train, y_test, feature_names = load_data_and_features()
    
    log_reg_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    
    log_reg_base = LogisticRegression(random_state=42, max_iter=1000)
    rf_base = RandomForestClassifier(random_state=42)
    
    best_log_reg = train_and_tune_model(log_reg_base, log_reg_params, X_train, y_train, "Logistic Regression")
    best_rf = train_and_tune_model(rf_base, rf_params, X_train, y_train, "Random Forest")
    
    evaluate_model(best_log_reg, X_train, y_train, X_test, y_test, "Logistic Regression", feature_names)
    evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Random Forest", feature_names)
    
    print("\nTraining complete. Model registered in MLflow.")

if __name__ == "__main__":
    main()
