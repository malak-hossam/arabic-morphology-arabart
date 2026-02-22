# Model Card: AraBART Morphology Generator

## Model Details
- Model family: Transformer encoder-decoder (AraBART)
- Base checkpoint: `moussaKam/AraBART`
- Fine-tuning objective: generate Arabic morphology description text from a single token
- Implementation stack: Hugging Face `transformers`, `datasets`, PyTorch

## Intended Use
- Arabic morphology assistance for:
  - educational demos
  - linguistic analysis prototypes
  - downstream systems needing approximate root/category/state extraction

## Out-of-Scope / Not Intended
- Legal, religious, or high-stakes linguistic rulings without expert review
- Robust dialectal morphology for all Arabic varieties
- Real-time large-scale batch inference without dedicated optimization

## Inputs and Outputs
- Input: one Arabic token (or tokenized words from sentence input)
- Output per token:
  - `الكلمة`
  - `الصنف الصرفي`
  - `الحالة`
  - `الجذر`

## Training Data
- Derived from MASAQ annotations.
- Pipeline reconstructs word-level forms and generated targets from segmented morph annotations.

## Evaluation
- Training loop tracks `eval_loss`.
- Lightweight evaluation pipeline reports exact-match on generated formatted targets.
- For production-grade reporting, add token-level F1 and per-field extraction accuracy.

## Limitations
- Domain bias toward Quranic style and annotation conventions.
- Generated text format may drift from expected template.
- OOV and orthographic variants can degrade root extraction.
- Heuristic fallback root extraction is less reliable than full analyzer-backed roots.

## Ethical Considerations
- Model outputs are probabilistic and may contain linguistic errors.
- Users should validate outputs before educational publication or automated decisions.

