import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
# 1. Import the Prometheus Counter
from prometheus_client import Counter 

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

model = None
preprocessor = None

# 2. Define your Custom Metric
# We add a label called 'prediction_class' so we can split the graph by High vs Low risk
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of predictions made by the model",
    ["prediction_class"] 
)

class PatientData(BaseModel):
    # ... (Keep your existing Pydantic model exactly the same) ...
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        model = joblib.load("models/best_model.joblib")
        preprocessor = joblib.load("models/preprocessor.joblib")
        print("Model and preprocessor loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.post("/predict")
def predict_heart_disease(data: PatientData):
    try:
        input_data = pd.DataFrame([data.dict()])
        processed_data = preprocessor.transform(input_data)
        
        prediction = model.predict(processed_data)[0]
        confidence = model.predict_proba(processed_data)[0].tolist()
        
        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk"
        
        # 3. Increment the counter with the specific result label
        PREDICTION_COUNTER.labels(prediction_class=result).inc()
        
        return {
            "prediction_class": int(prediction),
            "prediction_label": result,
            "confidence_scores": {
                "class_0_probability": round(confidence[0], 4),
                "class_1_probability": round(confidence[1], 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

Instrumentator().instrument(app).expose(app)
