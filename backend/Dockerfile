FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY best_bone_age_model.pth /app/models/
COPY xgb_model.joblib /app/models/
COPY ridge_model.joblib /app/models/
COPY scaler.joblib /app/models/
COPY ensemble_weights.joblib /app/models/

# Copy backend code
COPY backend/ /app/

# Override model paths for containerized environment
ENV MODEL_DIR=/app/models
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
