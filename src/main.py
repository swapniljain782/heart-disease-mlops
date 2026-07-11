import mlflow
import mlflow.pyfunc
import os
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# 1. Setup MLflow Tracking URI from environment
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://20.17.177.233:5000")
mlflow.set_tracking_uri(tracking_uri)

# 2. Dynamically load the latest registered model
# 'HeartDiseaseModel' must match the registered_model_name in train.py
model_uri = "models:/HeartDiseaseModel/latest"

print(f"🔄 Loading model from MLflow: {model_uri}...")
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([data])
    
    # Generate prediction
    prediction = model.predict(input_df)
    
    return {"prediction": int(prediction[0])}

@app.get("/")
def health_check():
    return {"status": "API is running", "model_uri": model_uri}
