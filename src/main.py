import mlflow
import mlflow.sklearn  # Changed to sklearn to enable predict_proba()
import os
import pandas as pd
import joblib
import logging
from fastapi import FastAPI

# 1. Setup API Request Logging (Fulfills Task 8)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# 2. Setup MLflow Tracking URI
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://20.17.177.233:5000")
mlflow.set_tracking_uri(tracking_uri)

# 3. Load the Preprocessor Pipeline (Fulfills Task 4)
preprocessor = None
try:
    # Ensure this path matches where your Dockerfile places the models folder
    preprocessor = joblib.load("models/preprocessor.joblib")
    logger.info("✅ Preprocessor loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading preprocessor: {e}")

# 4. Dynamically load the latest registered model
model_uri = "models:/HeartDiseaseModel/latest"
logger.info(f"🔄 Loading model from MLflow: {model_uri}...")
try:
    # Use mlflow.sklearn instead of pyfunc to easily access probabilities
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

@app.post("/predict")
def predict(data: dict):
    # Log the incoming API request (Fulfills Task 8)
    logger.info(f"Incoming prediction request: {data}")
    
    if model is None or preprocessor is None:
        return {"error": "Model or preprocessor not loaded correctly."}
    
    try:
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([data])
        
        # Apply the exact same preprocessing used during training
        processed_data = preprocessor.transform(input_df)
        
        # Generate prediction and confidence score (Fulfills Task 6)
        prediction = int(model.predict(processed_data)[0])
        probabilities = model.predict_proba(processed_data)[0]
        confidence = float(probabilities[prediction])
        
        response = {
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }
        
        logger.info(f"Successful response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "API is running", "model_uri": model_uri}