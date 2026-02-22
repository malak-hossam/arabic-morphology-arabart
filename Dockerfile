FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY main.py /app/main.py
COPY README.md /app/README.md

EXPOSE 8000

# Mount trained model at /app/trained_arabart_morph_model or override with ARABART_MODEL_DIR.
ENV ARABART_MODEL_DIR=/app/trained_arabart_morph_model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
