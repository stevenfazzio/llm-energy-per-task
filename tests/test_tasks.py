from llm_energy_per_task.tasks import extraction, intent, strip_thinking, toolcall


def test_strip_thinking_removes_closed_and_unclosed_blocks():
    assert strip_thinking("<think>hmm</think>\ncard_arrival") == "card_arrival"
    assert strip_thinking("<think>never finished") == ""
    assert strip_thinking("no reasoning here") == "no reasoning here"


def test_intent_forgives_formatting_but_not_the_wrong_label():
    item = {"label": "card_arrival"}
    for answer in [
        "card_arrival",
        " Card Arrival.",
        "`card_arrival`",
        "Intent: card-arrival\nbecause",
    ]:
        assert intent.score(item, answer) == 1.0, answer
    for answer in ["card_delivery_estimate", "", "The intent is card_arrival"]:
        assert intent.score(item, answer) == 0.0, answer


def test_intent_matches_the_datasets_irregular_labels():
    assert intent.score({"label": "Refund_not_showing_up"}, "refund_not_showing_up") == 1.0
    assert intent.score({"label": "reverted_card_payment?"}, "reverted_card_payment") == 1.0
    assert intent.score({"label": "reverted_card_payment?"}, "reverted_card_payment?") == 1.0


def test_extraction_f1_follows_squad():
    item = {"answers": ["Denver Broncos", "the Broncos"]}
    assert extraction.score(item, "The Denver Broncos.") == 1.0
    assert extraction.score(item, "Carolina Panthers") == 0.0
    # Two right tokens among four, against a two-token reference: P 0.5, R 1.0.
    assert abs(extraction.score(item, "Denver Broncos won it") - 2 / 3) < 1e-9
    assert extraction.score(item, "") == 0.0


FUNCTION = {"name": "calculate_triangle_area", "parameters": {}}
ITEM = {"function": FUNCTION, "expected": {"base": [10], "height": [5], "unit": ["units", ""]}}


def call(arguments, name="calculate_triangle_area"):
    import json

    return json.dumps({"name": name, "arguments": arguments})


def test_toolcall_accepts_a_right_call_however_it_is_wrapped():
    assert toolcall.score(ITEM, call({"base": 10, "height": 5})) == 1.0
    assert toolcall.score(ITEM, call({"base": 10.0, "height": 5, "unit": "Units"})) == 1.0
    fenced = "```json\n" + call({"base": 10, "height": 5}) + "\n```"
    assert toolcall.score(ITEM, "Here you go:\n" + fenced) == 1.0
    encoded = (
        '{"name": "calculate_triangle_area", "arguments": "{\\"base\\": 10, \\"height\\": 5}"}'
    )
    assert toolcall.score(ITEM, encoded) == 1.0


def test_toolcall_rejects_wrong_calls():
    assert toolcall.score(ITEM, call({"base": 10})) == 0.0  # height is required
    assert toolcall.score(ITEM, call({"base": 10, "height": 6})) == 0.0
    assert (
        toolcall.score(ITEM, call({"base": "10", "height": 5})) == 0.0
    )  # a string is not a number
    assert toolcall.score(ITEM, call({"base": 10, "height": 5, "colour": "red"})) == 0.0
    assert toolcall.score(ITEM, call({"base": 10, "height": 5}, name="area")) == 0.0
    assert toolcall.score(ITEM, "I would call calculate_triangle_area(10, 5)") == 0.0


def test_toolcall_keeps_true_apart_from_one():
    item = {"function": {"name": "f"}, "expected": {"flag": [True]}}
    assert toolcall.score(item, '{"name": "f", "arguments": {"flag": true}}') == 1.0
    assert toolcall.score(item, '{"name": "f", "arguments": {"flag": 1}}') == 0.0


def test_toolcall_checks_dict_arguments_key_by_key():
    item = {
        "function": {"name": "f"},
        "expected": {"where": [{"city": ["Paris"], "zip": ["", 75001]}]},
    }
    assert toolcall.score(item, '{"name": "f", "arguments": {"where": {"city": "paris"}}}') == 1.0
    assert toolcall.score(item, '{"name": "f", "arguments": {"where": {"city": "Lyon"}}}') == 0.0


def test_toolcall_checks_a_list_of_dicts_element_by_element():
    conditions = [{"field": ["age"], "value": ["25"]}, {"field": ["job"], "value": ["engineer"]}]
    item = {"function": {"name": "f"}, "expected": {"conditions": [conditions]}}

    def answer(*pairs):
        import json

        rows = [{"field": field, "value": value} for field, value in pairs]
        return json.dumps({"name": "f", "arguments": {"conditions": rows}})

    assert toolcall.score(item, answer(("age", "25"), ("job", "Engineer"))) == 1.0
    assert toolcall.score(item, answer(("age", "25"))) == 0.0
    assert toolcall.score(item, answer(("age", "30"), ("job", "engineer"))) == 0.0
