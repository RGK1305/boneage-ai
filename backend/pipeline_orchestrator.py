"""
Pipeline Orchestrator
=====================
Manages the full inference flow:
  1. Preprocess image (exact transform parity with training dataloader)
  2. Extract deep features via ResNet50 backbone
  3. Run ensemble prediction (DL + XGBoost + Ridge)
"""

import io
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model_manager import (
    ModelManager, IMG_SIZE, MAX_AGE_SCALE,
    SEX_FEATURE_SCALE, IMAGENET_MEAN, IMAGENET_STD
)


class PipelineOrchestrator:
    """Orchestrates preprocessing, feature extraction, and ensemble prediction."""

    def __init__(self):
        self.model_manager = ModelManager()

        # CRITICAL: These transforms must exactly match the validation
        # transforms used during training. Any mismatch will silently
        # degrade prediction accuracy.
        # Training val_tfm was:
        #   transforms.Resize((384, 384))
        #   transforms.ToTensor()
        #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.inference_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def preprocess_image(self, image_bytes: bytes) -> Image.Image:
        """Load and validate raw image bytes into a PIL Image."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image

    def encode_sex(self, patient_sex: str) -> float:
        """Encode sex string to numeric value matching training convention."""
        if patient_sex.upper() == "M":
            return 1.0
        return 0.0

    def predict(self, image_bytes: bytes, patient_sex: str) -> dict:
        """
        Full prediction pipeline.

        Returns dict with:
          - predicted_bone_age_months: float
          - developmental_stage: str
          - processing_time_ms: int
        """
        start_time = time.time()
        mm = self.model_manager
        device = mm.device

        # 1. Preprocess
        pil_image = self.preprocess_image(image_bytes)
        img_tensor = self.inference_transform(pil_image).unsqueeze(0).to(device)
        sex_value = self.encode_sex(patient_sex)
        sex_tensor = torch.tensor([sex_value], dtype=torch.float32).to(device)

        # 2. Deep Learning prediction (direct from full model)
        mm.pytorch_model.eval()
        with torch.no_grad():
            p_age, _ = mm.pytorch_model(img_tensor, sex_tensor)
            dl_pred = (p_age * MAX_AGE_SCALE).cpu().squeeze().item()

        # 3. Feature extraction for ensemble models
        with torch.no_grad():
            features = mm.pytorch_model.backbone(img_tensor)
            features = features.view(features.size(0), -1).cpu().numpy()

        # Concatenate sex feature (scaled by SEX_FEATURE_SCALE, matching training)
        sex_feature = np.array([[sex_value * SEX_FEATURE_SCALE]])
        features_with_sex = np.hstack([features, sex_feature])

        # 4. XGBoost prediction
        xgb_pred = mm.xgb_model.predict(features_with_sex)[0]

        # 5. Ridge prediction (using saved StandardScaler for exact normalization)
        features_scaled = mm.scaler.transform(features_with_sex)
        ridge_pred = mm.ridge_model.predict(features_scaled)[0]

        # 6. Ensemble weighted prediction
        w = mm.ensemble_weights
        final_pred = (w[0] * dl_pred) + (w[1] * xgb_pred) + (w[2] * ridge_pred)

        # 7. Derive developmental stage
        stage = self._get_stage(final_pred)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "predicted_bone_age_months": round(float(final_pred), 1),
            "developmental_stage": stage,
            "processing_time_ms": processing_time_ms
        }

    @staticmethod
    def _get_stage(months: float) -> str:
        """Derive developmental stage from predicted bone age."""
        if months < 144:
            return "Child"
        elif months < 216:
            return "Adolescent"
        else:
            return "Adult"
