"""Checks the raw and standardized CPI surprise calculations."""

import math
import statistics
from datetime import UTC, date, datetime, time

import pandas as pd  # type: ignore[import-untyped]
import pytest

from macro_surprise.analysis.surprises import add_cpi_surprises


# Identifies missing scores when a surprise cannot yet be standardized.
def _is_missing(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


# Builds example CPI records with the fields needed by the surprise calculation.
def _releases(
    entries: list[tuple[date, str, float, float]],
) -> pd.DataFrame:
    rows = []
    for release_date, series, actual, consensus in entries:
        rows.append(
            {
                "release_date": release_date,
                "release_time_et": time(8, 30),
                "release_timestamp_utc": datetime(
                    release_date.year,
                    release_date.month,
                    release_date.day,
                    13,
                    30,
                    tzinfo=UTC,
                ),
                "series": series,
                "actual": actual,
                "consensus": consensus,
                "prior_original": math.nan,
                "prior_revised": math.nan,
                "unit": "percent",
                "source": "fixture",
                "verified": True,
            }
        )
    return pd.DataFrame(rows)


# Creates a monthly CPI history with chosen surprises so the expected scaling is known.
def _monthly(
    series: str,
    surprises: list[float],
    *,
    start: int = 0,
) -> pd.DataFrame:
    entries: list[tuple[date, str, float, float]] = []
    for offset, surprise in enumerate(surprises):
        index = start + offset
        year = 2010 + index // 12
        month = index % 12 + 1
        entries.append((date(year, month, 15), series, surprise, 0.0))
    return _releases(entries)


# Selects one example release by date and CPI series for a precise result check.
def _at(frame: pd.DataFrame, release_date: date, series: str) -> pd.Series:
    match = frame[(frame["release_date"] == release_date) & (frame["series"] == series)]
    assert len(match) == 1
    return match.iloc[0]


# Checks that surprises use actuals and forecasts without changing the input.
def test_raw_surprise_is_actual_minus_consensus() -> None:
    source = _releases(
        [
            (date(2024, 1, 11), "headline_mom", 0.3, 0.2),
            (date(2024, 2, 13), "headline_mom", 0.1, 0.4),
        ]
    )
    source.loc[0, "prior_revised"] = 9.9
    snapshot = source.copy()

    result = add_cpi_surprises(source)

    pd.testing.assert_frame_equal(source, snapshot)
    pd.testing.assert_frame_equal(result[list(snapshot.columns)], snapshot)
    assert result["raw_surprise"].tolist() == pytest.approx([0.1, -0.3])


# Checks that interleaved headline and core releases retain separate histories.
def test_series_histories_stay_separate() -> None:
    headline = _monthly("headline_mom", [1.0, 3.0, 5.0])
    core = _monthly("core_mom", [10.0, 10.0, 10.0])
    pieces = []
    for index in range(3):
        pieces.append(core.iloc[[index]])
        pieces.append(headline.iloc[[index]])
    result = add_cpi_surprises(pd.concat(pieces, ignore_index=True))

    headline_row = _at(result, date(2010, 3, 15), "headline_mom")
    core_row = _at(result, date(2010, 3, 15), "core_mom")

    assert headline_row["historical_surprise_count"] == 2
    assert core_row["historical_surprise_count"] == 2
    assert headline_row["historical_surprise_std"] == pytest.approx(
        statistics.stdev([1.0, 3.0])
    )
    assert core_row["historical_surprise_std"] == 0


# Checks that the current surprise is excluded from its historical scaling.
def test_current_event_is_excluded_from_its_historical_standard_deviation() -> None:
    surprises = [float(value) for value in range(1, 13)] + [100.0]
    source = _monthly("headline_mom", surprises).iloc[::-1].reset_index(drop=True)

    result = add_cpi_surprises(source)

    earliest = _at(result, date(2010, 1, 15), "headline_mom")
    latest = _at(result, date(2011, 1, 15), "headline_mom")
    history = [float(value) for value in range(1, 13)]
    history_only = statistics.stdev(history)
    history_plus_current = statistics.stdev([*history, 100.0])

    assert list(result["release_date"]) == list(source["release_date"])
    assert earliest["historical_surprise_count"] == 0
    assert latest["historical_surprise_count"] == 12
    assert latest["historical_surprise_std"] == pytest.approx(history_only)
    assert latest["historical_surprise_std"] != pytest.approx(history_plus_current)
    assert latest["standardized_surprise"] == pytest.approx(100.0 / history_only)


# Checks that adding a future release cannot change an earlier standardized score.
def test_future_event_does_not_change_an_earlier_standardized_surprise() -> None:
    history = _monthly("headline_mom", [float(value) for value in range(1, 14)])
    before = _at(
        add_cpi_surprises(history),
        date(2011, 1, 15),
        "headline_mom",
    )
    extended = pd.concat(
        [_monthly("headline_mom", [1000.0], start=13), history],
        ignore_index=True,
    )

    after = _at(
        add_cpi_surprises(extended),
        date(2011, 1, 15),
        "headline_mom",
    )
    future = _at(add_cpi_surprises(extended), date(2011, 2, 15), "headline_mom")

    assert after["historical_surprise_count"] == before["historical_surprise_count"]
    assert after["historical_surprise_std"] == pytest.approx(
        before["historical_surprise_std"]
    )
    assert after["standardized_surprise"] == pytest.approx(
        before["standardized_surprise"]
    )
    assert future["historical_surprise_std"] != pytest.approx(
        before["historical_surprise_std"]
    )


# Checks that standardization starts only after twelve earlier releases are available.
def test_standardized_surprise_is_missing_until_twelve_earlier_observations() -> None:
    twelve = add_cpi_surprises(
        _monthly("headline_mom", [float(value) for value in range(1, 13)])
    )
    twelfth = _at(twelve, date(2010, 12, 15), "headline_mom")

    assert twelfth["historical_surprise_count"] == 11
    assert all(_is_missing(value) for value in twelve["standardized_surprise"])

    thirteen = add_cpi_surprises(
        _monthly("headline_mom", [float(value) for value in range(1, 14)])
    )
    still_early = _at(thirteen, date(2010, 12, 15), "headline_mom")
    first_ready = _at(thirteen, date(2011, 1, 15), "headline_mom")

    assert still_early["historical_surprise_count"] == 11
    assert _is_missing(still_early["standardized_surprise"])
    assert first_ready["historical_surprise_count"] == 12
    assert not _is_missing(first_ready["standardized_surprise"])


# Checks that a constant history produces a missing score rather than infinity.
def test_zero_historical_standard_deviation_has_missing_standardized_surprise() -> None:
    result = add_cpi_surprises(_monthly("core_yoy", [0.1] * 12 + [0.5]))
    row = _at(result, date(2011, 1, 15), "core_yoy")

    assert row["historical_surprise_count"] == 12
    assert row["raw_surprise"] == pytest.approx(0.5)
    assert row["historical_surprise_std"] == 0
    assert _is_missing(row["standardized_surprise"])
    assert not math.isinf(row["standardized_surprise"])


# Checks that older releases contribute to scaling when only dates are available.
def test_release_date_orders_history_when_old_timestamps_are_missing() -> None:
    source = _monthly("headline_mom", [float(value) for value in range(1, 14)])
    source.loc[source.index[:12], "release_timestamp_utc"] = pd.NaT

    result = add_cpi_surprises(source)
    latest = _at(result, date(2011, 1, 15), "headline_mom")

    assert latest["historical_surprise_count"] == 12
    assert latest["standardized_surprise"] == pytest.approx(
        13.0 / statistics.stdev([float(value) for value in range(1, 13)])
    )
