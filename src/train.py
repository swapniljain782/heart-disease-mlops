import os
import glob
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

WORKSPACE = os.getcwd()

# 1. HELPER FUNCTIONS
def load_data_and_features():
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    preprocessor = joblib.load('models/preprocessor.joblib')
    return X_train, X_test, y_train, y_test, preprocessor.get_feature_names_out()

def save_visualizations(model, X_test, y_test, model_name, feature_names):
    # Use relative path (current working directory) to stay in the Jenkins workspace
    plot_dir = os.path.join(WORKSPACE, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').lower()

    # Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(f"{plot_dir}/{safe_name}_confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # Save ROC Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} - ROC Curve')
    plt.savefig(f"{plot_dir}/{safe_name}_roc_curve.png", bbox_inches='tight')
    plt.close()
    
    # Save Feature Importance
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
    plt.savefig(f"{plot_dir}/{safe_name}_feature_importance.png", bbox_inches='tight')
    plt.close()

def train_and_tune_model(model, param_grid, X_train, y_train, model_name):
    print(f"\nRunning GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names):
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
        
        # LOG ARTIFACTS SINGULARLY: Avoids recursive mkdir permission issues
        for plot in glob.glob("plots/*.png"):
            mlflow.log_artifact(plot, artifact_path="evaluation_plots")
        
        # LOG MODEL ONLY: Removed registered_model_name to prevent double-registration
        model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        
        # EXPLICIT REGISTRATION: Highly reliable for MLflow 3.x
        try:
            result = mlflow.register_model(model_uri=model_info.model_uri, name="HeartDiseaseModel")
            print(f"✅ Model registered: {result.name} version {result.version}")
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
        
        return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_proba)
  
    
# 2. MAIN EXECUTION
def main():
    # 1. Configuration for Remote Artifact Proxying
    # We DO NOT set MLFLOW_ARTIFACT_URI to a local path. 
    # We let MLflow use the Tracking URI to find the server's artifact endpoint.
    os.environ["MLFLOW_TRACKING_URI"] = "http://20.17.177.233:5000"
    os.environ["MLFLOW_HTTP_PROXY_ARTIFACTS"] = "true"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # 2. Set Tracking URI
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    
    # 3. Create ONLY the local directory for plots (your temporary scratchpad)
    # Do NOT set this as the Artifact URI, or it will try to write MLflow internals there
    local_plots = os.path.join(os.getcwd(), "plots")
    os.makedirs(local_plots, exist_ok=True)
    
    # Using V2 to ensure the remote artifact path is cleanly mapped
    mlflow.set_experiment("Heart_Disease_Prediction_V2")
    
    # 4. Load Data
    X_train, X_test, y_train, y_test, feature_names = load_data_and_features()
    
    # 5. Model Training
    log_reg_params = {'C': [0.1, 1], 'solver': ['liblinear']}
    rf_params = {'n_estimators': [50], 'max_depth': [10]}
    
    best_log_reg = train_and_tune_model(LogisticRegression(), log_reg_params, X_train, y_train, "Logistic Regression")
    best_rf = train_and_tune_model(RandomForestClassifier(), rf_params, X_train, y_train, "Random Forest")
    
    # 6. Evaluation
    evaluate_model(best_log_reg, X_train, y_train, X_test, y_test, "Logistic Regression", feature_names)
    evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Random Forest", feature_names)

    print("\n✅ Training and registration complete.")

if __name__ == "__main__":
    main()