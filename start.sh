#!/bin/bash
# ==========================================
# BoneAge AI — HF Spaces Startup Script
# Downloads model weights from Google Drive
# then starts the FastAPI server.
# ==========================================
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
PORT="${PORT:-7860}"

echo "================================================"
echo "  BoneAge AI — Starting Up"
echo "================================================"

# Download models if not already present
if [ ! -f "$MODEL_DIR/best_bone_age_model.pth" ]; then
    echo "🔽 Downloading model weights from Google Drive..."
    mkdir -p "$MODEL_DIR"
    python -c "
import gdown
import os

folder_url = 'https://drive.google.com/drive/folders/1K2w1VhoD-sniUc_wP7-AYfx9VDzWxn4Q'
output = os.environ.get('MODEL_DIR', '/app/models')
gdown.download_folder(folder_url, output=output, quiet=False, use_cookies=False)
print('✅ All model weights downloaded successfully!')
"
else
    echo "✅ Model weights already present, skipping download"
fi

# Verify models exist
echo ""
echo "📦 Checking model artifacts..."
for f in best_bone_age_model.pth xgb_model.joblib ridge_model.joblib scaler.joblib ensemble_weights.joblib; do
    if [ -f "$MODEL_DIR/$f" ]; then
        echo "   ✅ $f"
    else
        echo "   ❌ MISSING: $f"
        exit 1
    fi
done

echo ""
echo "🚀 Starting server on port $PORT..."
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
