import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def load_data_and_features(data_dir="data/processed", model_dir="models"):
    """Loads preprocessed data and extracts feature names."""
    print("Loading processed data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    feature_names = preprocessor.get_feature_names_out()
    
    return X_train, X_test, y_train, y_test, feature_names

def save_visualizations(model, X_test, y_test, model_name, feature_names):
    """Generates and saves standard ML evaluation plots locally."""
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
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.savefig(f"plots/{safe_name}_roc_curve.png", bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance / Coefficients
    plt.figure(figsize=(10, 8))
    if hasattr(model, 'coef_'):
        importance = model.coef_[0]
        plt.title(f'{model_name} - Feature Coefficients')
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        plt.title(f'{model_name} - Feature Importance')
    else:
        plt.close()
        return
        
    indices = np.argsort(np.abs(importance))
    plt.barh(range(len(indices)), importance[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance / Weight')
    plt.tight_layout()
    plt.savefig(f"plots/{safe_name}_feature_importance.png", bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_names):
    """Evaluates the model and logs everything to MLflow."""
    print(f"\n--- Training and Tracking {model_name} ---")
    
    # Start an MLflow run for this specific model
    with mlflow.start_run(run_name=model_name):
        
        # 1. Log Parameters
        mlflow.log_params(model.get_params())
        
        # 2. Cross-Validation
        scoring = ['accuracy', 'precision', 'recall', 'roc_auc']
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
        
        cv_accuracy = np.mean(cv_results['test_accuracy'])
        cv_precision = np.mean(cv_results['test_precision'])
        cv_recall = np.mean(cv_results['test_recall'])
        cv_roc_auc = np.mean(cv_results['test_roc_auc'])
        
        # Log CV metrics to MLflow
        mlflow.log_metrics({
            "cv_accuracy": cv_accuracy,
            "cv_precision": cv_precision,
            "cv_recall": cv_recall,
            "cv_roc_auc": cv_roc_auc
        })
        
        # 3. Train on Full Training Set and Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_proba)
        
        # Log Test metrics to MLflow
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_roc_auc": test_roc_auc
        })
        
        print(f"Test Accuracy: {test_accuracy:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
        
        # 4. Generate Visualizations and Log as Artifacts
        save_visualizations(model, X_test, y_test, model_name, feature_names)
        
        # This tells MLflow to upload everything in the /plots folder to this run
        mlflow.log_artifacts("plots", artifact_path="evaluation_plots")
        
        # 5. Log the actual model to MLflow (Satisfies Task 4 requirement)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        return test_accuracy, test_roc_auc

def main():
    # Set the MLflow Experiment name
    mlflow.set_experiment("Heart_Disease_Prediction_MLOps")
    
    X_train, X_test, y_train, y_test, feature_names = load_data_and_features()
    
    # Initialize models
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    
    # Evaluate and Track
    lr_acc, lr_auc = evaluate_model(log_reg, X_train, y_train, X_test, y_test, "Logistic Regression", feature_names)
    rf_acc, rf_auc = evaluate_model(rf_clf, X_train, y_train, X_test, y_test, "Random Forest", feature_names)
    
    # Save the absolute best model locally as well for easy access in the API later
    best_model = log_reg if lr_auc > rf_auc else rf_clf
    joblib.dump(best_model, 'models/best_model.joblib')
    print(f"\nTraining complete. Best model saved to models/best_model.joblib")

if __name__ == "__main__":
    main()
