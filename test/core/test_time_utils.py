from datetime import date, datetime, timedelta
from typing import List

import pytest

from core.time_utils import Bounds

expected_bounds: List[Bounds] = [
    Bounds.for_days(date(2025, 1, 1), date(2025, 1, 4)),
    Bounds.for_days(date(2025, 1, 4), date(2025, 1, 7)),
    Bounds.for_days(date(2025, 1, 7), date(2025, 1, 10)),
    Bounds.for_days(date(2025, 1, 10), date(2025, 1, 13)),
    Bounds.for_days(date(2025, 1, 13), date(2025, 1, 16)),
    Bounds.for_days(date(2025, 1, 16), date(2025, 1, 19)),
    Bounds.for_days(date(2025, 1, 19), date(2025, 1, 22)),
    Bounds.for_days(date(2025, 1, 22), date(2025, 1, 25)),
    Bounds.for_days(date(2025, 1, 25), date(2025, 1, 28)),
    Bounds.for_days(date(2025, 1, 28), date(2025, 1, 31)),
    Bounds.for_days(date(2025, 1, 31), date(2025, 2, 1)),
]


def test_bounds_generate_overlapping_bounds():
    bounds: Bounds = Bounds.for_days(date(2025, 1, 1), date(2025, 2, 1))
    sub_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=timedelta(days=3), interval=timedelta(days=3))

    assert sub_bounds == expected_bounds


def test_for_days_rejects_equal_start_and_end():
    """for_days used to silently produce an inverted Bounds when start == end, yielding zero rows."""
    with pytest.raises(ValueError):
        Bounds.for_days(date(2025, 1, 1), date(2025, 1, 1))


def test_for_days_rejects_inverted_range():
    with pytest.raises(ValueError):
        Bounds.for_days(date(2025, 1, 2), date(2025, 1, 1))


def test_for_days_end_exclusive_semantics_preserved():
    """`for_days` encodes a literal half-open midnight boundary."""
    bounds: Bounds = Bounds.for_days(date(2025, 1, 1), date(2025, 1, 2))
    assert bounds.start_inclusive == datetime(2025, 1, 1, 0, 0, 0)
    assert bounds.end_exclusive == datetime(2025, 1, 2, 0, 0, 0)
    assert bounds.day0 == date(2025, 1, 1)
    assert bounds.day1 == date(2025, 1, 1)


def test_direct_bounds_reject_empty_or_inverted_ranges():
    with pytest.raises(ValueError):
        Bounds(datetime(2025, 1, 1), datetime(2025, 1, 1))
    with pytest.raises(ValueError):
        Bounds(datetime(2025, 1, 2), datetime(2025, 1, 1))


def test_midnight_exclusive_bounds_do_not_include_boundary_day():
    """Genuinely exclusive midnight bounds must exclude the boundary day from contain_days."""
    bounds: Bounds = Bounds(
        start_inclusive=datetime(2024, 11, 1, 0, 0, 0),
        end_exclusive=datetime(2024, 12, 1, 0, 0, 0),
    )
    assert bounds.contain_days(date(2024, 11, 1))
    assert bounds.contain_days(date(2024, 11, 30))
    assert not bounds.contain_days(date(2024, 12, 1))


def test_midnight_exclusive_bounds_date_range_stops_before_boundary():
    bounds: Bounds = Bounds(
        start_inclusive=datetime(2024, 11, 28, 0, 0, 0),
        end_exclusive=datetime(2024, 12, 1, 0, 0, 0),
    )
    assert list(bounds.date_range()) == [date(2024, 11, 28), date(2024, 11, 29), date(2024, 11, 30)]


def test_midnight_exclusive_bounds_year_month_strings_stops_before_boundary():
    bounds: Bounds = Bounds(
        start_inclusive=datetime(2024, 11, 1, 0, 0, 0),
        end_exclusive=datetime(2024, 12, 1, 0, 0, 0),
    )
    assert bounds.generate_year_month_strings() == ["202411"]


def test_for_days_year_month_strings_matches_existing_docstring_examples():
    """The generate_year_month_strings examples in the docstring must keep working."""
    assert Bounds.for_days(date(2025, 1, 1), date(2025, 3, 1)).generate_year_month_strings() == ["202501", "202502"]
    assert Bounds.for_days(date(2025, 1, 1), date(2025, 3, 2)).generate_year_month_strings() == [
        "202501",
        "202502",
        "202503",
    ]


def test_bounds_eq_returns_notimplemented_for_non_bounds():
    bounds: Bounds = Bounds.for_days(date(2025, 1, 1), date(2025, 1, 2))
    # `bounds == other` should not crash on unrelated types and must not accidentally treat them
    # as equal. NotImplemented lets Python fall back to identity comparison, which is False here.
    assert bounds != 5
    assert bounds != "not a bounds"
    assert bounds.__eq__(5) is NotImplemented
