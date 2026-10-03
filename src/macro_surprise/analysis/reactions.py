"""Matches CPI releases with market data and calculates each reaction window."""

from __future__ import annotations

from datetime import date

import pandas as pd  # type: ignore[import-untyped]

EVENT_COLUMNS = ("release_date", "series", "standardized_surprise")
MARKET_COLUMNS = ("observation_date", "instrument", "asset_class", "value")
RATE_INSTRUMENTS = frozenset({"us_2y_yield", "us_10y_yield"})

REACTION_COLUMNS = (
    "release_date",
    "series",
    "instrument",
    "asset_class",
    "standardized_surprise",
    "baseline_date",
    "event_session_date",
    "plus_1_session_date",
    "plus_5_session_date",
    "baseline_value",
    "event_session_value",
    "plus_1_session_value",
    "plus_5_session_value",
    "reaction_eod",
    "reaction_1d",
    "reaction_5d",
    "reaction_unit",
)


class ReactionError(ValueError):
    pass


# Matches each CPI release with every instrument and calculates three reaction windows.
# Session counts use available market observations rather than calendar days.
def calculate_daily_reactions(
    events: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(events, EVENT_COLUMNS, "events")
    _require_columns(market, MARKET_COLUMNS, "market")
    event_frame = events.copy()
    market_frame = market.copy()
    event_frame["release_date"] = pd.to_datetime(
        event_frame["release_date"], errors="coerce"
    ).dt.date
    market_frame["observation_date"] = pd.to_datetime(
        market_frame["observation_date"], errors="coerce"
    ).dt.date
    market_frame["value"] = pd.to_numeric(market_frame["value"], errors="coerce")
    _validate_events(event_frame)
    _validate_market(market_frame)

    instruments = market_frame[["instrument", "asset_class"]].drop_duplicates()
    # Removes missing prices so session counts use actual trading observations.
    levels_by_instrument = {
        str(name): group[group["value"].notna()].sort_values("observation_date")
        for name, group in market_frame.groupby("instrument", sort=False)
    }
    rows: list[dict[str, object]] = []
    for event in event_frame.to_dict(orient="records"):
        release_date = event["release_date"]
        if not isinstance(release_date, date):
            raise ReactionError("event release_date is invalid")
        for instrument_row in instruments.to_dict(orient="records"):
            instrument = str(instrument_row["instrument"])
            rows.append(
                _one_reaction(
                    event,
                    levels_by_instrument[instrument],
                    instrument_row,
                    release_date,
                )
            )
    return pd.DataFrame.from_records(rows, columns=list(REACTION_COLUMNS))


# Finds the starting value and the three later values for one event and instrument.
# All windows start at the last valid observation before the release.
def _one_reaction(
    event: dict[str, object],
    levels: pd.DataFrame,
    instrument_row: dict[str, object],
    release_date: date,
) -> dict[str, object]:
    before = levels[levels["observation_date"] < release_date]
    on_date = levels[levels["observation_date"] == release_date]
    after = levels[levels["observation_date"] > release_date]
    baseline = before.iloc[-1] if not before.empty else None
    event_session = on_date.iloc[0] if not on_date.empty else None
    # Count available sessions after the release, skipping holidays and missing values.
    plus_1 = after.iloc[0] if len(after) >= 1 else None
    plus_5 = after.iloc[4] if len(after) >= 5 else None
    instrument = str(instrument_row["instrument"])
    # Reports yield changes in basis points and other changes in percent.
    unit = "basis_points" if instrument in RATE_INSTRUMENTS else "percent_return"
    return {
        "release_date": release_date,
        "series": event["series"],
        "instrument": instrument,
        "asset_class": instrument_row["asset_class"],
        "standardized_surprise": event["standardized_surprise"],
        "baseline_date": _field(baseline, "observation_date"),
        "event_session_date": _field(event_session, "observation_date"),
        "plus_1_session_date": _field(plus_1, "observation_date"),
        "plus_5_session_date": _field(plus_5, "observation_date"),
        "baseline_value": _field(baseline, "value"),
        "event_session_value": _field(event_session, "value"),
        "plus_1_session_value": _field(plus_1, "value"),
        "plus_5_session_value": _field(plus_5, "value"),
        "reaction_eod": _reaction(baseline, event_session, instrument),
        "reaction_1d": _reaction(baseline, plus_1, instrument),
        "reaction_5d": _reaction(baseline, plus_5, instrument),
        "reaction_unit": unit,
    }


# Converts a pair of market values into a yield change or a percentage return.
# Missing observations stay missing rather than appearing to show no movement.
def _reaction(
    baseline: pd.Series | None,
    endpoint: pd.Series | None,
    instrument: str,
) -> float | None:
    # Leaves the reaction missing when either endpoint is unavailable.
    if baseline is None or endpoint is None:
        return None
    start = float(baseline["value"])
    end = float(endpoint["value"])
    if instrument in RATE_INSTRUMENTS:
        # A change of one percentage point in a yield equals one hundred basis points.
        return (end - start) * 100.0
    if start <= 0:
        raise ReactionError(f"{instrument} has non-positive baseline value")
    return (end / start - 1.0) * 100.0


# Retrieves a date or value from a market observation, leaving unmatched values empty.
def _field(row: pd.Series | None, column: str) -> object | None:
    return None if row is None else row[column]


# Checks that the event or market table contains the fields needed for matching.
def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], label: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ReactionError(f"{label} missing columns: {', '.join(missing)}")


# Rejects incomplete or duplicate CPI events before they can enter the reaction dataset.
def _validate_events(events: pd.DataFrame) -> None:
    if events[list(EVENT_COLUMNS)].isna().any().any():
        raise ReactionError("events contain missing required values")
    if events.duplicated(["release_date", "series"]).any():
        raise ReactionError("events contain duplicate release_date and series rows")


# Checks that market dates and instrument labels are complete and consistent.
# Duplicate observations are rejected because session matching would be ambiguous.
def _validate_market(market: pd.DataFrame) -> None:
    key_columns = ["observation_date", "instrument", "asset_class"]
    if market[key_columns].isna().any().any():
        raise ReactionError(
            "market contains missing dates, instruments, or asset classes"
        )
    if market.duplicated(["observation_date", "instrument"]).any():
        raise ReactionError("market contains duplicate instrument-date rows")
    if (market.groupby("instrument")["asset_class"].nunique() > 1).any():
        raise ReactionError("an instrument maps to multiple asset classes")
