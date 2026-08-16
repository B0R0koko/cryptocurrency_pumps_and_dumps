from dataclasses import dataclass
from datetime import date, timedelta, time, datetime
from enum import Enum
from typing import List

import pandas as pd


def start_of_the_day(day: date) -> datetime:
    """Converts date to datetime with 0:00 time"""
    return datetime.combine(date=day, time=time(hour=0, minute=0, second=0))


def end_of_the_day(day: date) -> datetime:
    """Converts date to datetime with 23:59:59:9999 time"""
    return start_of_the_day(day=day) + timedelta(days=1) - timedelta(microseconds=1)


class NamedTimeDelta(Enum):
    ONE_MINUTE = (timedelta(minutes=1), "1MIN")
    TWO_MINUTES = (timedelta(minutes=2), "2MIN")
    THREE_MINUTES = (timedelta(minutes=3), "3MIN")
    FOUR_MINUTES = (timedelta(minutes=4), "4MIN")
    FIVE_MINUTES = (timedelta(minutes=5), "5MIN")
    FIFTEEN_MINUTES = (timedelta(minutes=15), "15MIN")
    ONE_HOUR = (timedelta(hours=1), "1H")
    TWO_HOURS = (timedelta(hours=2), "2H")
    FOUR_HOURS = (timedelta(hours=4), "4H")
    TWELVE_HOURS = (timedelta(hours=12), "12H")
    ONE_DAY = (timedelta(days=1), "1D")
    TWO_DAYS = (timedelta(days=2), "2D")
    ONE_WEEK = (timedelta(weeks=1), "7D")
    TWO_WEEKS = (timedelta(weeks=2), "14D")

    def get_td(self) -> timedelta:
        return self.value[0]

    def get_slug(self) -> str:
        return self.value[1]


@dataclass
class Bounds:
    start_inclusive: datetime
    end_exclusive: datetime

    def __post_init__(self) -> None:
        if self.end_exclusive <= self.start_inclusive:
            raise ValueError(
                "Bounds requires end_exclusive > start_inclusive; "
                f"got start_inclusive={self.start_inclusive}, end_exclusive={self.end_exclusive}"
            )

    @classmethod
    def for_days(cls, start_inclusive: date, end_exclusive: date) -> "Bounds":
        """
        For instance, if we pass start_inclusive = date(2024, 11, 1) and end_exclusive = date(2024, 12, 1),
        Final Bounds will have the half-open datetimes
        ``[2024-11-01 00:00:00, 2024-12-01 00:00:00)``.

        Raises ValueError when end_exclusive <= start_inclusive; an empty or inverted range is almost
        always a caller bug and previously produced a silently inverted Bounds that yielded zero rows.
        """
        return cls(
            start_inclusive=start_of_the_day(day=start_inclusive),
            end_exclusive=start_of_the_day(day=end_exclusive),
        )

    @classmethod
    def for_day(cls, day: date) -> "Bounds":
        return cls(
            start_inclusive=start_of_the_day(day=day),
            end_exclusive=start_of_the_day(day=day + timedelta(days=1)),
        )

    @property
    def day0(self) -> date:
        return self.start_inclusive.date()

    @property
    def day1(self) -> date:
        """
        Last calendar day covered by this Bounds.

        We subtract a single microsecond so exact-midnight-exclusive bounds do not accidentally
        include the boundary day (e.g. Bounds(Nov 1 00:00, Dec 1 00:00) has day1 = Nov 30), while
        `for_days`-shaped bounds (end_exclusive = 23:59:59.999999) still resolve to the same day.
        """
        return (self.end_exclusive - timedelta(microseconds=1)).date()

    def __str__(self) -> str:
        return (
            f"Bounds: {self.start_inclusive.strftime("%Y-%m-%d %H:%M:%S")} - "
            f"{self.end_exclusive.strftime("%Y-%m-%d %H:%M:%S")}"
        )

    def generate_overlapping_bounds(self, step: timedelta, interval: timedelta) -> List["Bounds"]:
        """Returns a list of bounds created from a parent Bounds interval with a certain interval size and step.

        Bounds are half-open and adjacent windows therefore share no rows. The
        final sub-bound may be shorter than `interval` when the parent range is
        not an exact multiple of `step`.
        """
        intervals: List["Bounds"] = []

        lb = self.start_inclusive

        while lb < self.end_exclusive:
            rb: datetime = min(lb + interval, self.end_exclusive)
            intervals.append(
                Bounds(
                    start_inclusive=lb,
                    end_exclusive=rb,
                )
            )
            lb += step

        return intervals

    def contain_days(self, day: date) -> bool:
        return self.day0 <= day <= self.day1

    def date_range(self):
        for dt in pd.date_range(self.day0, self.day1, freq="1D", inclusive="both"):
            yield dt.date()

    def generate_year_month_strings(self) -> List[str]:
        """
        For Bounds.for_days(date(2025, 1, 1), date(2025, 3, 1)) returns -> ["202501", "202502"]
        For Bounds.for_days(date(2025, 1, 1), date(2025, 3, 2)) returns -> ["202501", "202502", "202503"]
        """
        y, m = self.day0.year, self.day0.month
        last_year, last_month = self.day1.year, self.day1.month

        months: List[str] = []
        # step month by month
        while (y < last_year) or (y == last_year and m <= last_month):
            months.append(f"{y:04d}{m:02d}")
            # increment month
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1

        return months

    def __eq__(self, other) -> bool:
        if not isinstance(other, Bounds):
            return NotImplemented
        return self.start_inclusive == other.start_inclusive and self.end_exclusive == other.end_exclusive
