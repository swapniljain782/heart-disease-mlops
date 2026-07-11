import pandas as pd
import numpy as np
import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_data(input_path, output_dir):
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(input_path, names=columns, na_values="?")
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_processed)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_processed)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.to_numpy())
    
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/processed.cleveland.data")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()
    prepare_data(args.input, args.output)
