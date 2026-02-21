"""
Grad-CAM Explainability Module
================================
Generates Grad-CAM heatmap overlays for the ResNet50 bone age model.
Hooks into model.backbone.layer4 (final convolutional layer).
"""

import io
import base64
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

from model_manager import (
    ModelManager, IMG_SIZE, MAX_AGE_SCALE,
    IMAGENET_MEAN, IMAGENET_STD
)


class GradCAMGenerator:
    """Generates Grad-CAM heatmaps for model explainability."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.gradients = None
        self.activations = None
        self._hooks = []

        self.inference_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def _register_hooks(self):
        """Register forward/backward hooks on the last conv layer."""
        self._remove_hooks()
        model = self.model_manager.pytorch_model

        # Hook into layer4 — the final convolutional block of ResNet50
        target_layer = model.backbone.layer4

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(self, image_bytes: bytes, patient_sex: str) -> str:
        """
        Generate Grad-CAM overlay and return as base64-encoded PNG.

        Args:
            image_bytes: Raw image file bytes
            patient_sex: "M" or "F"

        Returns:
            Base64-encoded PNG string with data URI prefix
        """
        mm = self.model_manager
        device = mm.device
        model = mm.pytorch_model

        # Load original image for overlay
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_w, original_h = pil_image.size

        # Prepare input tensor
        img_tensor = self.inference_transform(pil_image).unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        sex_value = 1.0 if patient_sex.upper() == "M" else 0.0
        sex_tensor = torch.tensor([sex_value], dtype=torch.float32).to(device)

        # Register hooks
        self._register_hooks()

        try:
            # Forward pass (need gradients, so temporarily enable training mode behavior)
            model.eval()
            # Enable gradient computation for this pass
            p_age, _ = model(img_tensor, sex_tensor)

            # Backward pass — compute gradients w.r.t. the age prediction
            model.zero_grad()
            target = p_age.squeeze()
            target.backward()

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

            # Grad-CAM computation
            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

            # Weighted sum of activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [B, 1, H, W]
            cam = torch.relu(cam)  # ReLU to keep only positive contributions

            # Normalize to [0, 1]
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            # Resize heatmap to original image dimensions
            heatmap = cv2.resize(cam, (original_w, original_h))
            heatmap = np.uint8(255 * heatmap)

            # Apply JET colormap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Convert original image to OpenCV format (BGR)
            original_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Overlay heatmap on original image
            overlay = cv2.addWeighted(original_cv, 0.6, heatmap_colored, 0.4, 0)

            # Convert back to RGB for encoding
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # Encode as PNG base64
            pil_overlay = Image.fromarray(overlay_rgb)
            buffer = io.BytesIO()
            pil_overlay.save(buffer, format='PNG')
            buffer.seek(0)
            b64_string = base64.b64encode(buffer.read()).decode('utf-8')

            return f"data:image/png;base64,{b64_string}"

        finally:
            self._remove_hooks()
            # Reset gradients
            self.gradients = None
            self.activations = None
