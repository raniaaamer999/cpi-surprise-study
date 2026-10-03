"""Checks the market reaction calculations and trading session matching."""

from datetime import date

import pandas as pd
import pytest

from macro_surprise.analysis.reactions import ReactionError, calculate_daily_reactions


# Creates one example CPI release with a known standardized surprise.
def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "release_date": date(2024, 1, 11),
                "series": "core_mom",
                "standardized_surprise": 1.25,
            }
        ]
    )


# Creates market values with gaps for a weekend and holiday to test session counting.
def _market() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
            "2024-01-16",
            "2024-01-17",
            "2024-01-18",
            "2024-01-19",
        ]
    ).date
    rows = []
    for instrument, asset_class, values in (
        ("us_2y_yield", "rates", [4.00, 4.05, 4.07, 4.10, 4.08, 4.11, 4.12]),
        ("sp500", "equity", [100.0, 101.0, 102.0, 104.0, 103.0, 104.0, 105.0]),
    ):
        for observation_date, value in zip(dates, values, strict=True):
            rows.append(
                {
                    "observation_date": observation_date,
                    "instrument": instrument,
                    "asset_class": asset_class,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


# Checks that yield reactions use basis points and share the same starting observation.
def test_yield_reactions_are_cumulative_basis_point_changes() -> None:
    result = calculate_daily_reactions(_events(), _market())
    row = result[result["instrument"] == "us_2y_yield"].iloc[0]

    assert row["reaction_eod"] == pytest.approx(5.0)
    assert row["reaction_1d"] == pytest.approx(7.0)
    assert row["reaction_5d"] == pytest.approx(12.0)
    assert row["reaction_unit"] == "basis_points"
    assert row["plus_5_session_date"] == date(2024, 1, 19)


# Checks that price returns share the same starting value before the release.
def test_price_reactions_are_cumulative_percentage_returns() -> None:
    result = calculate_daily_reactions(_events(), _market())
    row = result[result["instrument"] == "sp500"].iloc[0]

    assert row["reaction_eod"] == pytest.approx(1.0)
    assert row["reaction_1d"] == pytest.approx(2.0)
    assert row["reaction_5d"] == pytest.approx(5.0)
    assert row["reaction_unit"] == "percent_return"


# Checks that a missing release day stays missing even when a later market value exists.
def test_missing_release_day_is_not_replaced_with_next_session() -> None:
    market = _market()
    market = market[
        ~(
            market["instrument"].eq("sp500")
            & market["observation_date"].eq(date(2024, 1, 11))
        )
    ]

    result = calculate_daily_reactions(_events(), market)
    row = result[result["instrument"] == "sp500"].iloc[0]

    assert pd.isna(row["event_session_date"])
    assert pd.isna(row["reaction_eod"])
    assert row["reaction_1d"] == pytest.approx(2.0)


# Checks that duplicate market observations cannot enter the reaction calculation.
def test_duplicate_instrument_date_is_rejected() -> None:
    market = pd.concat([_market(), _market().iloc[[0]]], ignore_index=True)

    with pytest.raises(ReactionError, match="duplicate"):
        calculate_daily_reactions(_events(), market)
