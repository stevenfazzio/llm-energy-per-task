import pytest

from llm_energy_per_task.frontier import chart_bars, interval, line


def row(model, released, score, joules, spread=0.0):
    return {
        "model": model,
        "released": released,
        "score": score,
        "score_low": score - spread,
        "score_high": score + spread,
        "joules_per_item": joules,
    }


ROWS = [
    row("big-2023", "2023-06-01", 0.85, 10.0),
    row("small-2023", "2023-09-01", 0.60, 1.0),  # frugal, but below the bar
    row("mid-2024", "2024-04-01", 0.82, 3.0),
    row("mid-2024-worse", "2024-06-01", 0.90, 4.0),  # clears the bar, but not a new low
    row("small-2025", "2025-03-01", 0.80, 0.5),
]


def steps(rows, bar):
    return [r["model"] for kind, r in line(rows, bar) if kind != "near_miss"]


def test_steps_down_only_on_a_new_low_at_or_above_the_bar():
    assert steps(ROWS, 0.8) == ["big-2023", "mid-2024", "small-2025"]


def test_a_lower_bar_admits_the_earlier_frugal_model():
    assert steps(ROWS, 0.6) == ["big-2023", "small-2023", "small-2025"]


def test_a_bar_nothing_clears_gives_no_line():
    assert line(ROWS, 0.95) == []


def test_same_day_releases_yield_one_step():
    rows = [row("a", "2024-01-01", 0.9, 2.0), row("b", "2024-01-01", 0.9, 1.0)]
    assert steps(rows, 0.8) == ["b"]


def test_a_new_low_within_three_percent_of_the_last_step_is_not_a_step():
    rows = [
        row("first", "2023-01-01", 0.9, 10.0),
        row("a-hair-lower", "2024-01-01", 0.9, 9.8),
        # Measured against the last step, not against the low that did not count.
        row("another-hair", "2024-06-01", 0.9, 9.6),
        row("clearly-lower", "2025-01-01", 0.9, 9.0),
    ]
    assert steps(rows, 0.8) == ["first", "another-hair", "clearly-lower"]


def test_a_step_whose_interval_reaches_below_the_bar_is_marked():
    rows = [row("sure", "2023-01-01", 0.90, 10.0, 0.02), row("just", "2024-01-01", 0.81, 5.0, 0.02)]
    assert [(kind, r["model"]) for kind, r in line(rows, 0.8)] == [
        ("step", "sure"),
        ("step_straddling", "just"),
    ]


def test_a_near_miss_is_recorded_and_does_not_move_the_line():
    rows = [
        row("first", "2023-01-01", 0.90, 10.0),
        row("just-under", "2024-01-01", 0.79, 2.0, 0.02),
        row("well-under", "2024-02-01", 0.70, 1.0, 0.02),  # its interval does not reach
        row("under-and-costly", "2024-03-01", 0.79, 11.0, 0.02),  # would not have been a step
        # A step, because the near miss before it left the line where it was.
        row("later", "2025-01-01", 0.85, 5.0),
    ]
    assert [(kind, r["model"]) for kind, r in line(rows, 0.8)] == [
        ("step", "first"),
        ("near_miss", "just-under"),
        ("step", "later"),
    ]


def test_the_chart_bars_are_the_highest_2023_clears_and_the_highest_three_clear():
    rows = [
        row("old", "2023-06-01", 0.62, 10.0),
        row("a", "2024-01-01", 0.91, 5.0),
        row("b", "2025-01-01", 0.87, 4.0),
        row("c", "2026-01-01", 0.86, 3.0),
    ]
    assert chart_bars(rows) == {0.60: "2023", 0.85: "three_models"}
    # One bar can be both, and a task with nothing from 2023 has only the other.
    same = [row(name, "2023-06-01", 0.62, 10.0) for name in "abc"]
    assert chart_bars(same) == {0.60: "2023+three_models"}
    assert chart_bars([r for r in rows if r["model"] != "old"]) == {0.85: "three_models"}


def test_the_interval_narrows_with_more_items_and_stays_inside_zero_and_one():
    low, high = interval(0.5, 2000)
    assert (low, high) == pytest.approx((0.478, 0.522), abs=0.001)
    wide_low, wide_high = interval(0.5, 400)
    assert wide_low < low and wide_high > high
    assert interval(1.0, 400)[1] == pytest.approx(1.0)
    assert interval(0.0, 400)[0] == pytest.approx(0.0)
    assert 0.98 < interval(1.0, 400)[0] < 1.0
