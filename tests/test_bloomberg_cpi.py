"""Checks that the Bloomberg CPI workbook is cleaned correctly."""

from datetime import UTC, date, datetime, time
from pathlib import Path

import pandas as pd
import pytest

from macro_surprise.data.bloomberg_cpi import (
    SERIES_TICKERS,
    BloombergCpiError,
    load_bloomberg_cpi,
)


def _write_workbook(path: Path, *, duplicate_calendar: bool = False) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for series in SERIES_TICKERS:
            pd.DataFrame(
                [
                    [43112, None, 2.1, 2.0, None],
                    [43145, 20180313, None, None, 2.1],
                    [43145, None, 2.2, 2.1, None],
                    [43172, 20180411, None, None, 2.2],
                ]
            ).to_excel(writer, sheet_name=series, header=False, index=False)
        ticker_rows = [
            {
                "ID": "US Country",
                "calendar().release_date": "2018-01-12",
                "calendar().release_time": "08:30",
                "calendar().event_name": series,
                "calendar().survey_median": 2.0,
                "calendar().actual": 2.1,
                "calendar().prior": 1.9,
                "calendar().revision": None,
                "calendar().ticker": ticker,
            }
            for series, ticker in SERIES_TICKERS.items()
        ]
        if duplicate_calendar:
            ticker_rows.append(ticker_rows[0].copy())
        pd.DataFrame(ticker_rows).to_excel(
            writer, sheet_name="release_time_2018", index=False
        )


def test_loads_interleaved_bdh_rows_and_attaches_release_time(tmp_path: Path) -> None:
    path = tmp_path / "cpi.xlsx"
    _write_workbook(path)

    frame = load_bloomberg_cpi(path)

    assert len(frame) == 8
    row = frame[
        (frame["series"] == "headline_yoy")
        & (frame["release_date"] == date(2018, 1, 12))
    ].iloc[0]
    assert row["release_time_et"] == time(8, 30)
    assert row["release_timestamp_utc"] == datetime(2018, 1, 12, 13, 30, tzinfo=UTC)
    assert row["actual"] == 2.1
    assert row["consensus"] == 2.0
    assert row["raw_surprise"] == pytest.approx(0.1)
    assert row["latest_value"] == 2.1
    assert bool(row["actual_matches_calendar"])
    assert bool(row["consensus_matches_calendar"])


def test_rejects_duplicate_calendar_ticker_date(tmp_path: Path) -> None:
    path = tmp_path / "cpi.xlsx"
    _write_workbook(path, duplicate_calendar=True)

    with pytest.raises(BloombergCpiError, match="duplicate"):
        load_bloomberg_cpi(path)


def test_rejects_missing_required_sheet(tmp_path: Path) -> None:
    path = tmp_path / "cpi.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([[1]]).to_excel(writer, sheet_name="headline_yoy", index=False)
        pd.DataFrame([[1]]).to_excel(
            writer, sheet_name="release_time_2018", index=False
        )

    with pytest.raises(BloombergCpiError, match="Missing CPI sheets"):
        load_bloomberg_cpi(path)
