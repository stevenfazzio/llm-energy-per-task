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


@cache
def names() -> frozenset[str]:
    return frozenset(normalise(label) for label in labels())


@cache
def _written() -> re.Pattern:
    """Any intent written as its label is, and not as part of a longer name."""
    longest_first = sorted(names(), key=len, reverse=True)
    return re.compile(
        r"(?<![a-z0-9_])(?:" + "|".join(map(re.escape, longest_first)) + r")(?![a-z0-9_])"
    )


def first_line(output: str) -> str:
    line = next((ln for ln in output.splitlines() if ln.strip()), "")
    return normalise(line).removeprefix("intent_")


def named(output: str) -> set[str]:
    """The intents a reply names.

    A name counts wherever it is written the way the label is, underscores and
    all, whatever is wrapped around it: 'The intent is "card_arrival".' The
    same words written as ordinary words do not count, because the labels read
    like phrases a reply can use without classifying anything: "the exchange
    rate is 0.75" names nothing. The one exception is a first line that is
    nothing but the name, where spacing and punctuation are forgiven.
    """
    found = set(_written().findall(output.lower()))
    alone = first_line(output)
    if alone in names():
        found.add(alone)
    return found


def score(item: dict, output: str) -> float:
    """Right when the reply names the item's intent and no other."""
    return float(named(output) == {normalise(item["label"])})


def first_line_match(item: dict, output: str) -> float:
    """The stricter score first used: the first line is the label and nothing else.

    It marks a model down for wrapping the right label in a sentence, which is
    a manner of speaking and not a wrong answer, so it is kept as a column and
    not held against a bar.
    """
    return float(first_line(output) == normalise(item["label"]))


TASK = Task(
    name="intent",
    metric="names_intent",
    max_tokens=24,
    prompt=prompt,
    score=score,
    also={"first_line": first_line_match},
)
