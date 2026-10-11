"""Extraction: answer a question with a span from a passage.

SQuAD v1.1 (Rajpurkar et al., 2016), validation split. This is the shape of the
last step of retrieval-augmented answering, and of pulling a field out of a
document: the answer is in the text and the model has to find it.

Scored with SQuAD's own token F1 against the closest reference answer, so a
right answer wrapped in a sentence earns partial credit instead of none.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from .base import Task


def prompt(item: dict) -> str:
    return (
        "Answer the question using the passage. Reply with the exact words from the "
        "passage that answer it, and nothing else.\n\n"
        f"Passage: {item['context']}\n\nQuestion: {item['question']}\nAnswer:"
    )


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """SQuAD's answer normalisation: case, punctuation, articles, spacing."""
    return " ".join(_ARTICLES.sub(" ", text.lower().translate(_PUNCTUATION)).split())


def f1(prediction: str, reference: str) -> float:
    predicted, expected = normalise(prediction).split(), normalise(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    shared = sum((Counter(predicted) & Counter(expected)).values())
    if shared == 0:
        return 0.0
    precision, recall = shared / len(predicted), shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def score(item: dict, output: str) -> float:
    return max(f1(output, answer) for answer in item["answers"])


TASK = Task(name="extraction", metric="token_f1", max_tokens=48, prompt=prompt, score=score)
