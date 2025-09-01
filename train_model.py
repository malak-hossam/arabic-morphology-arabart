import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
import torch

# === تحميل البيانات
train_df = pd.read_csv("D:/ai/project/morphems/data_ready/train.csv")
val_df = pd.read_csv("D:/ai/project/morphems/data_ready/val.csv")

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds
})

# === إعداد النموذج
model_name = "moussaKam/AraBART"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_input_length = 64
max_target_length = 128

# === تحضير البيانات
def preprocess(example):
    model_input = tokenizer(example["input"], max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        label = tokenizer(example["target"], max_length=max_target_length, truncation=True, padding="max_length")
    model_input["labels"] = label["input_ids"]
    return model_input

tokenized_dataset = dataset.map(preprocess, batched=True)

# === إعدادات التدريب
training_args = Seq2SeqTrainingArguments(
    output_dir="D:/ai/project/morphems/arabart_morph_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    warmup_steps=300,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    num_train_epochs=6,
    predict_with_generate=True,
    fp16=True,  # استخدم True إذا كان لديك GPU يدعم FP16
    logging_dir='./logs',
    logging_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# === التدريب
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# === حفظ النموذج
trainer.save_model("trained_arabart_morph_model")
tokenizer.save_pretrained("trained_arabart_morph_model")

print("✅ التدريب اكتمل وتم حفظ النموذج.")
