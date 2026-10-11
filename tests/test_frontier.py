from llm_energy_per_task.frontier import frontier


def row(model, released, score, joules):
    return {"model": model, "released": released, "score": score, "joules_per_item": joules}


ROWS = [
    row("big-2023", "2023-06-01", 0.85, 10.0),
    row("small-2023", "2023-09-01", 0.60, 1.0),  # frugal, but below the bar
    row("mid-2024", "2024-04-01", 0.82, 3.0),
    row("mid-2024-worse", "2024-06-01", 0.90, 4.0),  # clears the bar, but not a new low
    row("small-2025", "2025-03-01", 0.80, 0.5),
]


def test_steps_down_only_on_a_new_low_at_or_above_the_bar():
    assert [r["model"] for r in frontier(ROWS, 0.8)] == ["big-2023", "mid-2024", "small-2025"]


def test_a_lower_bar_admits_the_earlier_frugal_model():
    assert [r["model"] for r in frontier(ROWS, 0.6)] == ["big-2023", "small-2023", "small-2025"]


def test_a_bar_nothing_clears_gives_no_line():
    assert frontier(ROWS, 0.95) == []


def test_same_day_releases_yield_one_step():
    rows = [row("a", "2024-01-01", 0.9, 2.0), row("b", "2024-01-01", 0.9, 1.0)]
    assert [r["model"] for r in frontier(rows, 0.8)] == ["b"]
