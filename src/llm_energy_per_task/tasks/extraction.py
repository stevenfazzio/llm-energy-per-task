"""Extraction: answer a question with a span from a passage.

SQuAD v1.1 (Rajpurkar et al., 2016), validation split. This is the shape of the
last step of retrieval-augmented answering, and of pulling a field out of a
document: the answer is in the text and the model has to find it.

The score is whether a reference answer appears in the reply. SQuAD's own
metric, token F1, was built for models that could only return a span; a chat
model that wraps the right answer in a sentence loses most of its credit there,
which measures how it talks and not whether it found the answer. Its extra
words are already paid for in energy. F1 is still recorded, as `token_f1`.

Appearing in the reply is not the same as being the answer: a long reply can
mention the reference in passing, and a right answer in other words is missed.
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


def contains(prediction: str, reference: str) -> bool:
    """Whether the reference's words appear in the prediction, in order and unbroken.

    Word by word after SQuAD's normalisation, so "8000" is not found in "18000".
    """
    predicted, expected = normalise(prediction).split(), normalise(reference).split()
    if not expected:
        return not predicted
    return any(
        predicted[start : start + len(expected)] == expected
        for start in range(len(predicted) - len(expected) + 1)
    )


def score(item: dict, output: str) -> float:
    return float(any(contains(output, answer) for answer in item["answers"]))


def token_f1(item: dict, output: str) -> float:
    return max(f1(output, answer) for answer in item["answers"])


TASK = Task(
    name="extraction",
    metric="contains_answer",
    max_tokens=48,
    prompt=prompt,
    score=score,
    also={"token_f1": token_f1},
)
