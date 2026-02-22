from typing import Dict


def tokenize_pair(
    tokenizer: object,
    source_text: str,
    target_text: str,
    max_input_length: int,
    max_target_length: int,
) -> Dict[str, list]:
    model_input = tokenizer(
        source_text,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=target_text,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )
    model_input["labels"] = labels["input_ids"]
    return model_input

