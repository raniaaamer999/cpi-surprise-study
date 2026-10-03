"""Calculates CPI surprises without using information from later releases."""

import math
import statistics

import pandas as pd  # type: ignore[import-untyped]

MIN_EARLIER_SURPRISES = 12

SURPRISE_COLUMNS = (
    "raw_surprise",
    "historical_surprise_count",
    "historical_surprise_std",
    "standardized_surprise",
)

_INPUT_COLUMNS = ("series", "actual", "consensus")


# Calculates raw and standardized CPI surprises while preserving the input row order.
# Each CPI series uses its own earlier releases to set the scale.
def add_cpi_surprises(releases: pd.DataFrame) -> pd.DataFrame:
    frame = releases.copy()
    _require_columns(frame)
    if frame.empty:
        return _with_empty_surprise_columns(frame)

    order_column = _order_column(frame)
    ordered = frame.sort_values(order_column, kind="mergesort")
    _require_unique_order(ordered, order_column)
    calculated = _surprises_by_series(ordered)
    for column in SURPRISE_COLUMNS:
        frame[column] = calculated[column]
    return frame


# Checks that the CPI series, published values and survey forecasts are available.
def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _INPUT_COLUMNS if column not in frame.columns]
    if missing:
        required = ", ".join(_INPUT_COLUMNS)
        found = ", ".join(missing)
        raise ValueError(
            f"CPI surprise calculation requires columns {required}. Missing: {found}"
        )


# Orders releases by date, using the timestamp if no date column exists.
def _order_column(frame: pd.DataFrame) -> str:
    if "release_date" in frame.columns:
        return "release_date"
    if "release_timestamp_utc" in frame.columns:
        return "release_timestamp_utc"
    raise ValueError(
        "CPI surprise calculation requires release_date or release_timestamp_utc"
    )


# Rejects missing dates and duplicates that would make release ordering ambiguous.
def _require_unique_order(ordered: pd.DataFrame, order_column: str) -> None:
    if ordered[order_column].isna().any():
        raise ValueError(f"{order_column} must not be missing")
    duplicated = ordered.duplicated(["series", order_column], keep=False)
    if bool(duplicated.any()):
        raise ValueError(
            f"{order_column} must be unique within each CPI series "
            "so earlier surprises are unambiguous"
        )


# Returns the expected surprise columns even when there are no releases to process.
def _with_empty_surprise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame["raw_surprise"] = pd.Series(index=frame.index, dtype="float64")
    frame["historical_surprise_count"] = pd.Series(index=frame.index, dtype="int64")
    frame["historical_surprise_std"] = pd.Series(index=frame.index, dtype="float64")
    frame["standardized_surprise"] = pd.Series(index=frame.index, dtype="float64")
    return frame


# Keeps CPI measures separate so headline and core histories cannot affect each other.
def _surprises_by_series(ordered: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        _series_surprises(group) for _, group in ordered.groupby("series", sort=False)
    ]
    return pd.concat(pieces)


# Builds the surprise history for one CPI measure in release order.
# The current release is excluded from the history used to scale its surprise.
def _series_surprises(group: pd.DataFrame) -> pd.DataFrame:
    raw_values = [
        _raw_surprise(actual, consensus)
        for actual, consensus in zip(
            group["actual"].tolist(),
            group["consensus"].tolist(),
            strict=True,
        )
    ]
    counts: list[int] = []
    stds: list[float] = []
    standardized: list[float] = []
    for end, surprise in enumerate(raw_values):
        # Uses only earlier releases so future information does not affect the result.
        history = raw_values[:end]
        std = _sample_std(history)
        counts.append(len(history))
        stds.append(std)
        standardized.append(_standardized_surprise(surprise, len(history), std))
    return pd.DataFrame(
        {
            "raw_surprise": raw_values,
            "historical_surprise_count": counts,
            "historical_surprise_std": stds,
            "standardized_surprise": standardized,
        },
        index=group.index,
    )


# Subtracts the forecast from the published CPI value and rejects invalid numbers.
def _raw_surprise(actual: object, consensus: object) -> float:
    try:
        surprise = float(actual) - float(consensus)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("actual and consensus must be finite numbers") from exc
    if not math.isfinite(surprise):
        raise ValueError("actual and consensus must be finite numbers")
    return surprise


# Measures the spread of earlier surprises using the sample standard deviation.
def _sample_std(history: list[float]) -> float:
    # Needs at least two earlier surprises to calculate the sample spread.
    if len(history) < 2:
        return float("nan")
    return statistics.stdev(history)


# Requires twelve earlier observations with a positive spread before scaling.
# Leaves the score missing when the history cannot support standardization.
def _standardized_surprise(surprise: float, count: int, std: float) -> float:
    if count >= MIN_EARLIER_SURPRISES and math.isfinite(std) and std > 0:
        return surprise / std
    return float("nan")
