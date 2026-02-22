# Experiments



## Reproducible Run Commands

```bash
python -m src.pipelines.prepare_pipeline
python -m src.pipelines.split_pipeline
python -m src.pipelines.train_pipeline
python -m src.pipelines.eval_pipeline --max-samples 100
```

## Notes
- Keep run metadata in this file and store model artifacts outside git.
- If hyperparameters differ from `src/config/default.yaml`, log them explicitly in the `notes` column.

