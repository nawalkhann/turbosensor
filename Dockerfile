FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
# CPU-only PyTorch keeps the image small; the CNN blend member needs it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[serve,deep]"

# Bake a trained FD001 model into the image so the container serves immediately.
RUN turbosensor download --data-dir data/raw \
    && turbosensor train --config configs/fd001.yaml \
    && rm -rf data/raw

EXPOSE 8000
CMD ["turbosensor", "serve", "--model", "artifacts/FD001/lgbm.joblib", "--host", "0.0.0.0", "--port", "8000"]
