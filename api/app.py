import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter 

app = FastAPI(title="Heart Disease Prediction API")
model = None
preprocessor = None

PREDICTION_COUNTER = Counter("model_predictions_total", "Total predictions", ["prediction_class"])

class PatientData(BaseModel):
    age: float; sex: float; cp: float; trestbps: float; chol: float; fbs: float; restecg: float; thalach: float; exang: float; oldpeak: float; slope: float; ca: float; thal: float

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    model = joblib.load("models/best_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

@app.post("/predict")
def predict_heart_disease(data: PatientData):
    try:
        input_data = pd.DataFrame([data.dict()])
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)[0]
        confidence = model.predict_proba(processed_data)[0].tolist()
        result = "High Risk" if prediction == 1 else "Low Risk"
        PREDICTION_COUNTER.labels(prediction_class=result).inc()
        
        return {"prediction_class": int(prediction), "prediction_label": result, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

Instrumentator().instrument(app).expose(app)
