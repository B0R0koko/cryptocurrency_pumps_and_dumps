from datetime import datetime

from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.pump_event import PumpEvent


def _make(cp_str: str, time_str: str) -> PumpEvent:
    return PumpEvent(
        currency_pair=CurrencyPair.from_string(cp_str),
        time=datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S"),
        exchange=Exchange.BINANCE_SPOT,
    )


def test_pump_event_hash_round_trip() -> None:
    pump: PumpEvent = _make("ACM-BTC", "2021-06-05 18:00:13")
    inferred: PumpEvent = PumpEvent.from_pump_hash(str(pump))
    assert inferred == pump


def test_lt_orders_by_time_then_currency_pair() -> None:
    """Equal-time events must not silently compare as equal under ordering.

    Previously __lt__ only compared time, so `a < b` and `b < a` were both False for
    equal-time events even though they were unequal by dataclass __eq__, which breaks
    functools.total_ordering-based sorts and containers.
    """
    same_time: str = "2021-06-05 18:00:13"
    acm: PumpEvent = _make("ACM-BTC", same_time)
    zzz: PumpEvent = _make("ZZZ-BTC", same_time)

    assert acm != zzz
    assert acm < zzz
    assert not zzz < acm
    assert zzz > acm

    # Sorting must be deterministic and total for equal-time events.
    sorted_events = sorted([zzz, acm])
    assert sorted_events == [acm, zzz]


def test_lt_still_orders_by_time_when_times_differ() -> None:
    earlier: PumpEvent = _make("ZZZ-BTC", "2021-06-05 18:00:13")
    later: PumpEvent = _make("ACM-BTC", "2021-06-05 18:00:14")
    assert earlier < later
    assert not later < earlier


def test_lt_returns_notimplemented_for_non_pump_event() -> None:
    pump: PumpEvent = _make("ACM-BTC", "2021-06-05 18:00:13")
    assert pump.__lt__(5) is NotImplemented
