# API Documentation

Base app:
- FastAPI app object: `main:app` (wrapper around `src.api.app`)

## Endpoints

### `GET /health`
Checks service status.

Response:
```json
{
  "status": "ok",
  "model_dir": "trained_arabart_morph_model"
}
```

### `POST /analyze`
Analyze one Arabic word or sentence.

Request body:
```json
{
  "text": "الكتاب جميل"
}
```

Response:
```json
{
  "result": [
    {
      "الكلمة": "الكتاب",
      "الصنف الصرفي": "اسم",
      "الحالة": "معرب",
      "الجذر": "كتب"
    },
    {
      "الكلمة": "جميل",
      "الصنف الصرفي": "صفة",
      "الحالة": "معرب",
      "الجذر": "جمل"
    }
  ]
}
```

## Run Locally
```bash
uvicorn main:app --reload
```

By default, the API reads model path from:
- `ARABART_MODEL_DIR` environment variable, or
- `trained_arabart_morph_model` in the project root.

