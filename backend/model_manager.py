"""
Model Manager — Singleton Pattern
===================================
Loads PyTorch, XGBoost, Ridge models and ensemble weights at server startup.
Uses CUDA if available, falls back to CPU with robust device mapping.
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import joblib
import logging

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION CONSTANTS
# Must match training exactly
# ==========================================
IMG_SIZE = 384
MAX_AGE_SCALE = 240.0
SEX_FEATURE_SCALE = 5.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model artifacts relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTORCH_MODEL_PATH = os.path.join(PROJECT_ROOT, "best_bone_age_model.pth")
XGB_MODEL_PATH = os.path.join(PROJECT_ROOT, "xgb_model.joblib")
RIDGE_MODEL_PATH = os.path.join(PROJECT_ROOT, "ridge_model.joblib")
SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.joblib")
ENSEMBLE_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "ensemble_weights.joblib")


class ResNetBoneModel(nn.Module):
    """ResNet50-based bone age model — architecture must match training exactly."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Linear(num_features + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self.stage_head = nn.Sequential(
            nn.Linear(num_features + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, img, sex):
        features = self.backbone(img)
        combined = torch.cat([features, sex.unsqueeze(1)], dim=1)
        return self.age_head(combined), self.stage_head(combined)


class ModelManager:
    """
    Singleton model manager.
    Loads all model artifacts once at server startup.
    Thread-safe for inference (all models used in eval/read-only mode).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pytorch_model = None
        self.xgb_model = None
        self.ridge_model = None
        self.scaler = None
        self.ensemble_weights = None

    def load_models(self):
        """Load all model artifacts. Call once at server startup."""
        logger.info(f"Loading models on device: {self.device}")

        # 1. PyTorch model with robust device mapping
        if not os.path.exists(PYTORCH_MODEL_PATH):
            raise FileNotFoundError(f"PyTorch model not found: {PYTORCH_MODEL_PATH}")

        self.pytorch_model = ResNetBoneModel().to(self.device)
        state_dict = torch.load(
            PYTORCH_MODEL_PATH,
            map_location=self.device,
            weights_only=True
        )
        self.pytorch_model.load_state_dict(state_dict)
        self.pytorch_model.eval()
        logger.info("✅ PyTorch model loaded")

        # 2. XGBoost model
        if not os.path.exists(XGB_MODEL_PATH):
            raise FileNotFoundError(f"XGBoost model not found: {XGB_MODEL_PATH}")
        self.xgb_model = joblib.load(XGB_MODEL_PATH)
        logger.info("✅ XGBoost model loaded")

        # 3. Ridge model
        if not os.path.exists(RIDGE_MODEL_PATH):
            raise FileNotFoundError(f"Ridge model not found: {RIDGE_MODEL_PATH}")
        self.ridge_model = joblib.load(RIDGE_MODEL_PATH)
        logger.info("✅ Ridge model loaded")

        # 4. StandardScaler (exact normalization params from training)
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)
        logger.info("✅ StandardScaler loaded")

        # 5. Ensemble weights
        if not os.path.exists(ENSEMBLE_WEIGHTS_PATH):
            raise FileNotFoundError(f"Ensemble weights not found: {ENSEMBLE_WEIGHTS_PATH}")
        self.ensemble_weights = joblib.load(ENSEMBLE_WEIGHTS_PATH)
        logger.info(f"✅ Ensemble weights loaded: DL={self.ensemble_weights[0]:.4f}, "
                     f"XGB={self.ensemble_weights[1]:.4f}, Ridge={self.ensemble_weights[2]:.4f}")

        logger.info("🎉 All models loaded successfully!")

    @property
    def is_loaded(self):
        return all([
            self.pytorch_model is not None,
            self.xgb_model is not None,
            self.ridge_model is not None,
            self.scaler is not None,
            self.ensemble_weights is not None
        ])
