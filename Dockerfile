# ==========================================
# BoneAge AI — Hugging Face Spaces Dockerfile
# Combined container: React frontend + FastAPI backend
# ==========================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user

WORKDIR /app

# ── Python dependencies ──
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

# ── Build frontend ──
COPY frontend/package*.json /tmp/frontend/
WORKDIR /tmp/frontend
RUN npm ci

COPY frontend/ /tmp/frontend/
RUN npm run build \
    && mkdir -p /app/static \
    && cp -r /tmp/frontend/dist/* /app/static/ \
    && rm -rf /tmp/frontend

WORKDIR /app

# ── Copy backend code ──
COPY backend/ /app/

# ── Copy startup script ──
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── Prepare directories ──
RUN mkdir -p /app/models \
    && chown -R user:user /app

# ── Environment ──
ENV MODEL_DIR=/app/models
ENV STATIC_DIR=/app/static
ENV PYTHONUNBUFFERED=1

USER user

EXPOSE 7860
ENV PORT=7860

CMD ["/app/start.sh"]
