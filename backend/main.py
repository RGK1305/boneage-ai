"""
FastAPI Application — Bone Age Assessment
==========================================
Main entry point for the backend API server.
Includes CORS, upload validation, model loading at startup, and prediction endpoint.
"""


import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import Optional

from model_manager import ModelManager
from pipeline_orchestrator import PipelineOrchestrator
from explainability import GradCAMGenerator

# ==========================================
# CONFIGURATION
# ==========================================
MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# LIFESPAN — Load models at startup
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts into memory at server startup."""
    logger.info("🚀 Server starting — loading models...")
    try:
        model_manager = ModelManager()
        model_manager.load_models()
        logger.info("✅ All models loaded. Server is ready.")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    yield
    logger.info("🛑 Server shutting down.")


# ==========================================
# APP INITIALIZATION
# ==========================================
app = FastAPI(
    title="Bone Age Assessment API",
    description="Automated bone age prediction using ResNet50 + XGBoost + Ridge ensemble with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    mm = ModelManager()
    return {
        "status": "healthy" if mm.is_loaded else "degraded",
        "models_loaded": mm.is_loaded,
        "device": str(mm.device)
    }


@app.post("/api/v1/predict")
async def predict(
    image: UploadFile = File(..., description="Pediatric hand radiograph (JPEG, PNG)"),
    patient_sex: str = Form(..., description="Biological sex: 'M' or 'F'"),
    chronological_age_months: Optional[float] = Form(None, description="Optional chronological age in months")
):
    """
    Predict bone age from a hand radiograph.

    Returns predicted bone age, developmental stage, Grad-CAM heatmap overlay,
    and optional delta from chronological age.
    """
    # 1. Validate sex input
    if patient_sex.upper() not in ("M", "F"):
        raise HTTPException(status_code=400, detail="patient_sex must be 'M' or 'F'")
    patient_sex = patient_sex.upper()

    # 2. Validate MIME type
    if image.content_type and image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {image.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_MIME_TYPES)}"
        )

    # 3. Read and validate file size
    image_bytes = await image.read()
    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(image_bytes) / 1024 / 1024:.1f}MB). "
                   f"Maximum allowed: {MAX_UPLOAD_SIZE_MB}MB"
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    # 4. Verify models are loaded
    mm = ModelManager()
    if not mm.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Server is starting up.")

    try:
        # 5. Run prediction pipeline
        orchestrator = PipelineOrchestrator()
        result = orchestrator.predict(image_bytes, patient_sex)

        # 6. Generate Grad-CAM overlay
        gradcam = GradCAMGenerator()
        gradcam_base64 = gradcam.generate(image_bytes, patient_sex)

        # 7. Build response
        response = {
            "status": "success",
            "predicted_bone_age_months": result["predicted_bone_age_months"],
            "developmental_stage": result["developmental_stage"],
            "gradcam_overlay_base64": gradcam_base64,
            "processing_time_ms": result["processing_time_ms"]
        }

        # 8. Optional delta computation
        if chronological_age_months is not None:
            delta = result["predicted_bone_age_months"] - chronological_age_months
            response["chronological_age_months"] = chronological_age_months
            response["delta_months"] = round(delta, 1)

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ==========================================
# STATIC FILE SERVING (production / HF Spaces)
# ==========================================
STATIC_DIR = os.environ.get("STATIC_DIR", "")
if STATIC_DIR and os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    logger.info(f"📁 Serving frontend from {STATIC_DIR}")
