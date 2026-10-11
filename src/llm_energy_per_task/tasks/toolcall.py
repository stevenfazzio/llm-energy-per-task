"""Tool calling: turn a request into one call to a described function.

Items are the "simple" category of the Berkeley Function-Calling Leaderboard
(Patil et al.): one function, one call. The function is described in the prompt
and the model replies with JSON, so every model is asked the same way whatever
its own tool-calling format.

The scorer is this repo's own, modelled on the leaderboard's argument check but
not identical to it, so a score here is not a leaderboard score. A call is right
when the name matches, every argument holds one of the accepted values, nothing
unexpected is passed, and anything left out was optional.
"""

from __future__ import annotations

import json
import re

from .base import Task

_FORM = '{"name": "<function name>", "arguments": {"<argument>": <value>}}'


def prompt(item: dict) -> str:
    return (
        "You can call this function:\n\n"
        f"{json.dumps(item['function'], indent=2)}\n\n"
        "Write the call that fulfils the request below. Reply with only a JSON object "
        f"of the form {_FORM}\n\n"
        f"Request: {item['request']}"
    )


def parse_call(output: str) -> tuple[str, dict] | None:
    """The first JSON value in the answer, read as a call. None if it is not one."""
    decoder = json.JSONDecoder()
    for start, character in enumerate(output):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(output, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not isinstance(value, dict):
            return None
        name = value.get("name")
        arguments = value.get("arguments", value.get("parameters"))
        if isinstance(arguments, str):  # some models double-encode, as the OpenAI API does
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(name, str) and isinstance(arguments, dict):
            return name, arguments
        return None
    return None


_INSIGNIFICANT = re.compile(r"[ ,./\-_*^'\"]")


def _canonical(value):
    """Forgive what the leaderboard forgives: string spacing and case, and int for float."""
    if isinstance(value, bool):
        return ("bool", value)  # keep True from equalling 1
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return _INSIGNIFICANT.sub("", value).lower()
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {key: _canonical(v) for key, v in value.items()}
    return value


def _matches(value, option) -> bool:
    # A dict-valued argument lists accepted values per key, like the call itself,
    # and so does each dict in a list of them.
    if isinstance(option, dict) and isinstance(value, dict):
        return arguments_match(value, option)
    if isinstance(option, list) and any(isinstance(o, dict) for o in option):
        return (
            isinstance(value, list)
            and len(value) == len(option)
            and all(_matches(v, o) for v, o in zip(value, option, strict=True))
        )
    return _canonical(value) == _canonical(option)


def _accepted(value, acceptable: list) -> bool:
    return any(_matches(value, option) for option in acceptable)


def arguments_match(arguments: dict, expected: dict[str, list]) -> bool:
    if set(arguments) - set(expected):
        return False
    for name, acceptable in expected.items():
        if name not in arguments:
            if "" not in acceptable:  # "" marks an argument that may be left out
                return False
        elif not _accepted(arguments[name], acceptable):
            return False
    return True


def score(item: dict, output: str) -> float:
    call = parse_call(output)
    if call is None:
        return 0.0
    name, arguments = call
    return float(name == item["function"]["name"] and arguments_match(arguments, item["expected"]))


TASK = Task(name="toolcall", metric="accuracy", max_tokens=192, prompt=prompt, score=score)
