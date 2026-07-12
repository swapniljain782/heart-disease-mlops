# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app


# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the trained models into the container
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
