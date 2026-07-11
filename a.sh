for i in {1..200}; do
  if [ $((i % 5)) -eq 0 ]; then
      # Intentionally trigger an HTTP 422/400 Validation Error
      curl -s -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"bad_data": "yes"}' > /dev/null
      echo "Triggered Error..."
  else
      # Valid prediction request
      curl -s -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"age":63.0,"sex":1.0,"cp":1.0,"trestbps":145.0,"chol":233.0,"fbs":1.0,"restecg":2.0,"thalach":150.0,"exang":0.0,"oldpeak":2.3,"slope":3.0,"ca":0.0,"thal":6.0}' > /dev/null
      echo "Valid Request..."
  fi
  sleep 0.2
done
