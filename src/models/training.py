from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.models.arabart_model import load_model_and_tokenizer, save_model_and_tokenizer
from src.models.metrics import exact_match_accuracy
from src.models.tokenizer import tokenize_pair
from src.utils.logging import get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


@dataclass
class TrainingConfig:
    train_csv: str
    val_csv: str
    pretrained_model_name: str
    output_dir: str
    trained_model_dir: str
    max_input_length: int = 64
    max_target_length: int = 128
    learning_rate: float = 3e-5
    warmup_steps: int = 300
    train_batch_size: int = 16
    eval_batch_size: int = 2
    weight_decay: float = 0.01
    num_train_epochs: int = 6
    save_total_limit: int = 2
    early_stopping_patience: int = 2
    logging_steps: int = 20
    seed: int = 42


def _build_hf_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame):
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise ImportError(
            "datasets library is required for training. Install requirements.txt."
        ) from exc

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    return DatasetDict({"train": train_ds, "validation": val_ds})


def _compute_generation_metrics(tokenizer):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [
            [(token if token != -100 else tokenizer.pad_token_id) for token in sequence]
            for sequence in labels
        ]
        label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        return {"exact_match": exact_match_accuracy(pred_text, label_text)}

    return compute_metrics


def train_model(config: TrainingConfig) -> Dict[str, float]:
    try:
        import torch
        from transformers import (
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers/torch are required for training. Install requirements.txt."
        ) from exc

    set_seed(config.seed)

    train_df = pd.read_csv(config.train_csv)
    val_df = pd.read_csv(config.val_csv)
    dataset = _build_hf_dataset(train_df, val_df)

    model, tokenizer = load_model_and_tokenizer(config.pretrained_model_name)

    def preprocess_fn(example):
        return tokenize_pair(
            tokenizer=tokenizer,
            source_text=example["input"],
            target_text=example["target"],
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
        )

    tokenized_dataset = dataset.map(preprocess_fn, batched=False)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=_compute_generation_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    save_model_and_tokenizer(trainer.model, tokenizer, config.trained_model_dir)
    LOGGER.info("Training complete. Model saved to %s", config.trained_model_dir)
    return eval_metrics

