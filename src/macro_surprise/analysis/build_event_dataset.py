"""Builds the dataset that joins CPI surprises with daily market reactions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from macro_surprise.analysis.reactions import calculate_daily_reactions
from macro_surprise.analysis.surprises import add_cpi_surprises
from macro_surprise.data.bloomberg_cpi import DEFAULT_OUTPUT, load_bloomberg_cpi
from macro_surprise.data.market_client import DEFAULT_MARKET_DATA_PATH

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVENT_DATASET = _REPO_ROOT / "data" / "private" / "cpi_event_reactions.csv"


# Joins usable CPI surprises with the market reactions for each instrument.
# Older releases help scale surprises even when their exact timing is unavailable.
def build_event_dataset(
    cpi_path: str | Path = DEFAULT_OUTPUT,
    market_path: str | Path = DEFAULT_MARKET_DATA_PATH,
) -> pd.DataFrame:
    cpi_file = Path(cpi_path)
    cpi = pd.read_csv(cpi_file) if cpi_file.is_file() else load_bloomberg_cpi()
    cpi["release_date"] = pd.to_datetime(cpi["release_date"], errors="raise").dt.date
    surprised = add_cpi_surprises(cpi)
    events = surprised[
        surprised["release_timestamp_utc"].notna()
        & surprised["standardized_surprise"].notna()
    ].copy()
    # Matches each usable CPI release with the five market instruments.
    market = pd.read_csv(market_path)
    reactions = calculate_daily_reactions(events, market)
    event_columns = [
        "release_date",
        "series",
        "release_time_et",
        "release_timestamp_utc",
        "actual",
        "consensus",
        "raw_surprise",
        "historical_surprise_count",
        "historical_surprise_std",
        "standardized_surprise",
        "actual_matches_calendar",
        "consensus_matches_calendar",
    ]
    return reactions.drop(columns=["standardized_surprise"]).merge(
        events[event_columns], on=["release_date", "series"], validate="many_to_one"
    )


# Builds and saves the event dataset in the private data folder by default.
def write_event_dataset(
    cpi_path: str | Path = DEFAULT_OUTPUT,
    market_path: str | Path = DEFAULT_MARKET_DATA_PATH,
    output: str | Path = DEFAULT_EVENT_DATASET,
) -> pd.DataFrame:
    frame = build_event_dataset(cpi_path, market_path)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


# Reads the input paths, builds the dataset and reports its size.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpi", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVENT_DATASET)
    args = parser.parse_args()
    frame = write_event_dataset(args.cpi, args.market, args.output)
    print(
        f"Wrote {len(frame)} event and asset rows across "
        f"{frame['release_date'].nunique()} CPI dates to {args.output}"
    )


if __name__ == "__main__":
    main()
