# End-to-End MLOps Pipeline: Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white)
![Jenkins](https://img.shields.io/badge/Jenkins-D24939?style=flat&logo=jenkins&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat&logo=prometheus&logoColor=white)

## 1. Project Overview
This project is an end-to-end Machine Learning Operations (MLOps) pipeline that predicts heart disease risk based on a patient's clinical data. It provides real-time model inference via a REST API, ensuring high availability, scalable deployment, and continuous monitoring in a production-like Kubernetes environment.

## 2. Architecture & Technology Stack
*   **Model Training & Tracking:** Scikit-Learn, MLflow (logs metrics and saves `best_model.joblib` and `preprocessor.joblib`).
*   **API Framework:** FastAPI, Uvicorn, Pydantic (strict JSON data validation).
*   **Containerization:** Docker (utilizing a lightweight `python:3.12-slim` base image).
*   **Orchestration:** Kubernetes / Minikube (Deployments, LoadBalancer Services).
*   **CI/CD:** Jenkins (Automated pipeline for building, testing, and deploying).
*   **Monitoring:** Prometheus (`prometheus_fastapi_instrumentator`) and Grafana.

## 3. CI/CD Pipeline (Jenkins)
The delivery pipeline is fully automated using a `Jenkinsfile` with the following stages:
1.  **Checkout:** Pulls the latest code from the repository.
2.  **Build Image:** Builds the Docker image for the FastAPI application.
3.  **Push to Registry:** Pushes the built image to a container registry.
4.  **Deploy to Kubernetes:** Applies the `deployment.yaml` and `service.yaml` manifests to the Kubernetes cluster using `kubectl`.

*(Note: Include a screenshot of your successful Jenkins pipeline dashboard here.)*

## 4. API Endpoints & Usage
The API is exposed securely and provides self-documenting endpoints.

*   **Base URL:** `http://<your-vm-ip>:8080`
*   **Interactive Docs (Swagger UI):** `/docs`
*   **Health Check:** `GET /health` - Used by Kubernetes for readiness and liveness probes.
*   **Prediction Endpoint:** `POST /predict` 
    *   Accepts a JSON payload of 13 clinical features.
    *   Transforms data using the loaded `preprocessor.joblib`.
    *   Returns a binary classification (`High Risk` or `Low Risk`) and the model's prediction confidence array.

### Sample POST Request Payload:
```json
{
  "age": 52.0,
  "sex": 1.0,
  "cp": 3.0,
  "trestbps": 120.0,
  "chol": 233.0,
  "fbs": 0.0,
  "restecg": 1.0,
  "thalach": 163.0,
  "exang": 0.0,
  "oldpeak": 0.6,
  "slope": 2.0,
  "ca": 0.0,
  "thal": 2.0
}
```

## 5. Kubernetes Deployment Strategy
The application runs inside an isolated Kubernetes cluster to guarantee uptime and scalability.
*   **Deployment:** Manages the replica sets for the containerized FastAPI application, automatically restarting if a crash occurs.
*   **Service:** A `LoadBalancer` service maps external traffic from port `8080` directly to the container's internal Uvicorn server running on port `8000`.

*(Note: Include terminal output screenshots of `kubectl get pods` and `kubectl get svc` showing the `Running` state and the mapped ports.)*

## 6. Model Monitoring & Metrics
Observability is built directly into the API to track model performance and usage over time.
*   **Prometheus Integration:** FastAPI is instrumented using `prometheus_fastapi_instrumentator`.
*   **Custom Metrics:** A custom counter metric (`model_predictions_total`) tracks the exact number of predictions made, categorized dynamically by `prediction_class` (High Risk vs. Low Risk).
*   **Metrics Endpoint:** `GET /metrics`

*(Note: Include a screenshot of your `/metrics` endpoint showing the `model_predictions_total` data.)*
