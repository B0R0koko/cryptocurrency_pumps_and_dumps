from datetime import date, timedelta
from typing import List

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
]


def test_bounds_generate_overlapping_bounds():
    bounds: Bounds = Bounds.for_days(
        date(2025, 1, 1), date(2025, 2, 1)
    )
    sub_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=timedelta(days=3), interval=timedelta(days=3))

    for sub_bound, expected in zip(sub_bounds, expected_bounds):
        assert sub_bound == expected, f"Generated {str(sub_bound)} and expected {str(expected)} bounds do not match"
