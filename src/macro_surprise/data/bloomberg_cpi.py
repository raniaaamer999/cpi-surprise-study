"""Cleans the private Bloomberg workbook into one row for each CPI measure."""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKBOOK = _REPO_ROOT / "data" / "private" / "cpi_bloomberg_private.xlsx"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "private" / "cpi_events_clean.csv"

SERIES_TICKERS = {
    "headline_yoy": "CPI YOY Index",
    "headline_mom": "CPI CHNG Index",
    "core_mom": "CPUPXCHG Index",
    "core_yoy": "CPI XYOY Index",
}

OUTPUT_COLUMNS = (
    "release_date",
    "release_time_et",
    "release_timestamp_utc",
    "series",
    "ticker",
    "actual",
    "consensus",
    "raw_surprise",
    "prior_original",
    "latest_value",
    "calendar_actual",
    "calendar_consensus",
    "actual_matches_calendar",
    "consensus_matches_calendar",
    "source",
)

_NEW_YORK = ZoneInfo("America/New_York")


class BloombergCpiError(ValueError):
    pass


def load_bloomberg_cpi(path: str | Path | None = None) -> pd.DataFrame:
    # Keeps the older CPI history because it is needed to scale later surprises.
    workbook = DEFAULT_WORKBOOK if path is None else Path(path)
    if not workbook.is_file():
        raise BloombergCpiError(f"Bloomberg CPI workbook not found: {workbook}")
    try:
        excel = pd.ExcelFile(workbook)
    except (OSError, ValueError, ImportError) as exc:
        raise BloombergCpiError(
            f"Could not open Bloomberg workbook: {workbook}"
        ) from exc

    _require_sheets(excel.sheet_names)
    calendar = _load_calendar(workbook, excel.sheet_names)
    frames = [
        _load_series(workbook, series, ticker, calendar)
        for series, ticker in SERIES_TICKERS.items()
    ]
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["release_date", "series"]).reset_index(drop=True)
    if result.duplicated(["release_date", "series"]).any():
        raise BloombergCpiError("Duplicate release_date and series rows after cleaning")
    return result.loc[:, list(OUTPUT_COLUMNS)]


def _require_sheets(sheet_names: list[str]) -> None:
    missing = set(SERIES_TICKERS) - set(sheet_names)
    calendar_sheets = [name for name in sheet_names if name.startswith("release_time_")]
    if missing:
        raise BloombergCpiError(f"Missing CPI sheets: {', '.join(sorted(missing))}")
    if not calendar_sheets:
        raise BloombergCpiError("No release_time_YYYY calendar sheets found")


def _load_calendar(workbook: Path, sheet_names: list[str]) -> pd.DataFrame:
    # Combines the yearly calendar sheets and keeps only the four CPI series.
    sheets = sorted(name for name in sheet_names if name.startswith("release_time_"))
    pieces: list[pd.DataFrame] = []
    required = {
        "calendar().release_date",
        "calendar().release_time",
        "calendar().survey_median",
        "calendar().actual",
        "calendar().ticker",
    }
    for sheet in sheets:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        missing = required - set(frame.columns)
        if missing:
            raise BloombergCpiError(
                f"Sheet {sheet} is missing columns: {', '.join(sorted(missing))}"
            )
        pieces.append(frame)
    calendar = pd.concat(pieces, ignore_index=True).rename(
        columns=lambda name: str(name).removeprefix("calendar().")
    )
    calendar = calendar[calendar["ticker"].isin(SERIES_TICKERS.values())].copy()
    calendar["release_date"] = pd.to_datetime(
        calendar["release_date"], errors="coerce"
    ).dt.date
    calendar["release_time"] = calendar["release_time"].map(_parse_clock)
    calendar["actual"] = pd.to_numeric(calendar["actual"], errors="coerce")
    calendar["survey_median"] = pd.to_numeric(
        calendar["survey_median"], errors="coerce"
    )
    calendar = calendar.dropna(subset=["release_date", "release_time"])
    if calendar.duplicated(["release_date", "ticker"]).any():
        raise BloombergCpiError(
            "Calendar contains duplicate release_date and ticker rows"
        )
    return calendar


def _load_series(
    workbook: Path,
    series: str,
    ticker: str,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    # Reads the alternating Bloomberg rows and keeps rows with an actual and forecast.
    raw = pd.read_excel(workbook, sheet_name=series, header=None)
    if raw.shape[1] < 5:
        raise BloombergCpiError(f"Sheet {series} must contain at least five columns")
    raw = raw.iloc[:, :5].copy()
    raw.columns = ["excel_date", "eco_release_date", "actual", "consensus", "latest"]
    raw["release_date"] = pd.to_datetime(
        pd.to_numeric(raw["excel_date"], errors="coerce"),
        unit="D",
        origin="1899-12-30",
        errors="coerce",
    ).dt.date
    raw["actual"] = pd.to_numeric(raw["actual"], errors="coerce")
    raw["consensus"] = pd.to_numeric(raw["consensus"], errors="coerce")
    # Takes the latest value from the metadata row that follows each release row.
    raw["latest_value"] = pd.to_numeric(raw["latest"].shift(-1), errors="coerce")
    rows = raw.dropna(subset=["release_date", "actual", "consensus"]).copy()
    if rows.empty:
        raise BloombergCpiError(f"Sheet {series} has no usable actual/consensus rows")
    rows["series"] = series
    rows["ticker"] = ticker
    rows["raw_surprise"] = rows["actual"] - rows["consensus"]
    # Uses the previous original release as the prior value.
    rows["prior_original"] = rows["actual"].shift(1)

    selected_calendar = calendar.loc[
        calendar["ticker"].eq(ticker),
        ["release_date", "release_time", "actual", "survey_median"],
    ].rename(
        columns={
            "release_time": "release_time_et",
            "actual": "calendar_actual",
            "survey_median": "calendar_consensus",
        }
    )
    rows = rows.merge(
        selected_calendar,
        on="release_date",
        how="left",
        validate="one_to_one",
    )
    rows["release_timestamp_utc"] = rows.apply(_timestamp_or_missing, axis=1)
    rows["actual_matches_calendar"] = _close_or_missing(
        rows["actual"], rows["calendar_actual"]
    )
    rows["consensus_matches_calendar"] = _close_or_missing(
        rows["consensus"], rows["calendar_consensus"]
    )
    rows["source"] = "Bloomberg Terminal (private licensed input)"
    return rows


def _parse_clock(value: object) -> time | None:
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    if isinstance(value, datetime):
        return value.time().replace(tzinfo=None)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    for pattern in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, pattern).time()
        except ValueError:
            continue
    return None


def _timestamp_or_missing(row: pd.Series) -> datetime | pd.NaT:
    # Converts the New York release time to UTC when an exact time is available.
    release_time = row["release_time_et"]
    if not isinstance(release_time, time):
        return pd.NaT
    local = datetime.combine(row["release_date"], release_time, tzinfo=_NEW_YORK)
    return local.astimezone(UTC)


def _close_or_missing(left: pd.Series, right: pd.Series) -> pd.Series:
    available = left.notna() & right.notna()
    result = pd.Series(pd.NA, index=left.index, dtype="boolean")
    result.loc[available] = (left.loc[available] - right.loc[available]).abs() < 1e-9
    return result


def write_clean_bloomberg_cpi(
    workbook: str | Path | None = None,
    output: str | Path | None = None,
) -> pd.DataFrame:
    # Saves the cleaned event rows in the private data folder.
    destination = DEFAULT_OUTPUT if output is None else Path(output)
    frame = load_bloomberg_cpi(workbook)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = write_clean_bloomberg_cpi(args.workbook, args.output)
    timed = frame["release_timestamp_utc"].notna().sum()
    print(
        f"Wrote {len(frame)} CPI measure rows "
        f"({timed} with exact times) to {args.output}"
    )


if __name__ == "__main__":
    main()
