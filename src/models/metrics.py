from typing import Iterable


def exact_match_accuracy(predictions: Iterable[str], references: Iterable[str]) -> float:
    preds = list(predictions)
    refs = list(references)
    if not refs:
        return 0.0
    hits = sum(int(p.strip() == r.strip()) for p, r in zip(preds, refs))
    return hits / len(refs)

