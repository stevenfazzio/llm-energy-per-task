"""Intent classification: route a customer message to one of 77 banking intents.

Banking77 (Casanueva et al., 2020), test split. The whole label set goes in the
prompt and the model answers with a label name -- how a small model gets used as
a router when there is no labelled data to fine-tune a classifier on.
"""

from __future__ import annotations

import json
import re
from functools import cache

from .. import TASK_DATA
from .base import Task

LABELS_FILE = TASK_DATA / "intent_labels.json"


@cache
def labels() -> tuple[str, ...]:
    return tuple(json.loads(LABELS_FILE.read_text()))


def prompt(item: dict) -> str:
    return (
        "Classify the customer message into exactly one of the intents below. "
        "Reply with the intent name only.\n\n"
        "Intents:\n" + "\n".join(labels()) + f"\n\nMessage: {item['text']}\nIntent:"
    )


def normalise(text: str) -> str:
    """Letters and digits only, so case, quoting, spacing and punctuation are forgiven.

    Applied to the label as well as the answer: two of the dataset's own labels
    are irregular ("Refund_not_showing_up", "reverted_card_payment?").
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def score(item: dict, output: str) -> float:
    line = next((ln for ln in output.splitlines() if ln.strip()), "")
    answer = normalise(line).removeprefix("intent_")
    return float(answer == normalise(item["label"]))


TASK = Task(name="intent", metric="accuracy", max_tokens=24, prompt=prompt, score=score)
