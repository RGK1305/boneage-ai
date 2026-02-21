<div align="center">

# 🦴 BoneAge AI

**Automated Bone Age Assessment using Deep Learning Ensemble with Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

*A production-ready web application for pediatric bone age assessment from hand radiographs, powered by a ResNet50 + XGBoost + Ridge ensemble with Grad-CAM explainability.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Docker Deployment](#-docker-deployment)
- [Model Details](#-model-details)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔬 Overview

BoneAge AI automates skeletal maturity assessment from pediatric hand X-rays. It combines:

- **ResNet50** deep learning backbone for feature extraction
- **XGBoost + Ridge** ensemble for robust regression
- **Grad-CAM** heatmaps highlighting anatomically relevant regions
- **Gender-aware** predictions using biological sex as an input feature

The system processes a hand radiograph and returns a predicted bone age (in months/years), developmental stage classification, and an explainable heatmap overlay.

---

## 🏗 Architecture

```
┌─────────────────┐         ┌───────────────────────────────────────────┐
│   React Frontend│  POST   │             FastAPI Backend               │
│   (Vite :5173)  │────────▶│               (:8000)                    │
│                 │         │  ┌─────────────────────────────────────┐  │
│  ┌────────────┐ │         │  │         ModelManager (Singleton)    │  │
│  │ Upload Zone│ │         │  │  ┌──────────┐ ┌───────┐ ┌───────┐  │  │
│  │ Sex Input  │ │         │  │  │ ResNet50 │ │XGBoost│ │ Ridge │  │  │
│  │ Age Input  │ │         │  │  └────┬─────┘ └───┬───┘ └───┬───┘  │  │
│  └────────────┘ │         │  │       │           │         │      │  │
│                 │◀────────│  │  Feature Ext.  ───┴─────────┘      │  │
│  ┌────────────┐ │  JSON   │  │       │      Ensemble Weighted     │  │
│  │ Results    │ │         │  │       ▼           Prediction       │  │
│  │ Dashboard  │ │         │  │   Grad-CAM ──▶ Heatmap Overlay     │  │
│  │ + Grad-CAM │ │         │  └─────────────────────────────────────┘  │
│  └────────────┘ │         └───────────────────────────────────────────┘
└─────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19 · Vite 7 · Vanilla CSS |
| **Backend** | FastAPI · Uvicorn · Python-Multipart |
| **ML/DL** | PyTorch · torchvision · XGBoost · scikit-learn |
| **Explainability** | Grad-CAM (layer4 hooks) · OpenCV |
| **Containerization** | Docker · Docker Compose · Nginx |

---

## 📁 Project Structure

```
prmlpro/
├── 📄 README.md
├── 📄 .gitignore
├── 📄 docker-compose.yml          # Container orchestration
├── 📄 save_models.py              # One-time model export script
├── 📄 PRML_CODE.ipynb             # Training notebook
│
├── 📂 backend/
│   ├── main.py                    # FastAPI app + endpoints
│   ├── model_manager.py           # Singleton model loader
│   ├── pipeline_orchestrator.py   # Inference pipeline
│   ├── explainability.py          # Grad-CAM generator
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile
│
└── 📂 frontend/
    ├── index.html
    ├── Dockerfile
    └── src/
        ├── main.jsx
        ├── index.css              # Clinical design system
        ├── App.jsx                # State management
        └── components/
            ├── UploadWorkspace.jsx
            ├── ProcessingIndicator.jsx
            └── ResultsDashboard.jsx
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **RSNA Bone Age dataset** (for model export step)
- *(Optional)* Docker & Docker Compose

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/boneage-ai.git
cd boneage-ai
```

### 2. Export Model Artifacts (One-time)

The PyTorch model (`best_bone_age_model.pth`) must be downloaded separately due to its size (~100MB). Place it in the project root, then:

```bash
# Create/activate Python environment
python -m venv bone_env
bone_env\Scripts\activate        # Windows
# source bone_env/bin/activate   # Linux/Mac

pip install torch torchvision xgboost scikit-learn pandas joblib

# Export XGBoost, Ridge, and scaler
python save_models.py
```

> **Note:** This requires the RSNA dataset. Update `BASE_DIR` in `save_models.py` if your dataset is at a different path.

### 3. Start Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Verify: open `http://localhost:8000/api/v1/health`

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 📡 API Reference

### Health Check

```
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda"
}
```

### Predict Bone Age

```
POST /api/v1/predict
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | ✅ | Hand radiograph (JPEG, PNG, BMP, TIFF) |
| `patient_sex` | String | ✅ | `"M"` or `"F"` |
| `chronological_age_months` | Float | ❌ | Patient's actual age for delta |

**Response:**
```json
{
  "status": "success",
  "predicted_bone_age_months": 142.5,
  "developmental_stage": "Adolescent",
  "gradcam_overlay_base64": "data:image/png;base64,...",
  "processing_time_ms": 850,
  "chronological_age_months": 150,
  "delta_months": -7.5
}
```

---

## 🐳 Docker Deployment

```bash
# Build and start both services
docker-compose up --build

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

The Nginx reverse proxy in the frontend container forwards `/api/` requests to the backend automatically.

---

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| **Backbone** | ResNet50 (ImageNet pretrained) |
| **Input** | 384×384 RGB image + biological sex |
| **Feature Dim** | 2048 (backbone) + 1 (sex × 5.0) = 2049 |
| **Ensemble** | DL (~3.6%) + XGBoost (~87.3%) + Ridge (~9.1%) |
| **Normalization** | ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| **Output Scale** | `MAX_AGE_SCALE = 240.0` months |
| **Explainability** | Grad-CAM on `backbone.layer4` |
| **Dataset** | RSNA Pediatric Bone Age Challenge |

### Developmental Stages

| Stage | Age Range |
|-------|-----------|
| Child | 0–143 months |
| Adolescent | 144–215 months |
| Adult | 216+ months |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for pediatric radiology**

</div>
