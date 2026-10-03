"""Checks that daily market data is downloaded and validated correctly."""

import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd  # type: ignore[import-untyped]
import pytest

from macro_surprise.data.market_client import (
    OUTPUT_COLUMNS,
    MarketDataError,
    fetch_daily_market_data,
    main,
)

START = date(2024, 1, 2)
END = date(2024, 1, 4)
RETRIEVED_AT = datetime(2026, 9, 25, 16, 0, tzinfo=UTC)


# Identifies missing prices in the downloaded example table.
def _is_missing(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


# Creates a small FRED style CSV response for testing without network access.
def _fred(series_id: str, rows: list[tuple[str, str]]) -> str:
    lines = [f"observation_date,{series_id}"]
    lines.extend(f"{day},{value}" for day, value in rows)
    return "\n".join(lines) + "\n"


# Creates a timestamp in New York time for testing trading date conversion.
def _ny_unix(day: str, hour: int = 9, minute: int = 30) -> int:
    parsed = date.fromisoformat(day)
    local = datetime(
        parsed.year,
        parsed.month,
        parsed.day,
        hour,
        minute,
        tzinfo=ZoneInfo("America/New_York"),
    )
    return int(local.timestamp())


# Builds a sample Yahoo response with controllable prices and metadata.
def _yahoo_chart(
    bars: Sequence[tuple[int, Any]],
    *,
    adjusted: Sequence[Any] | None = None,
    symbol: str = "GLD",
    currency: str = "USD",
    timezone_name: str | None = "America/New_York",
) -> str:
    indicators: dict[str, object] = {
        "quote": [{"close": [price for _, price in bars]}],
    }
    if adjusted is not None:
        indicators["adjclose"] = [{"adjclose": adjusted}]
    meta: dict[str, object] = {"currency": currency, "symbol": symbol}
    if timezone_name is not None:
        meta["exchangeTimezoneName"] = timezone_name
    payload: dict[str, object] = {
        "chart": {
            "result": [
                {
                    "meta": meta,
                    "timestamp": [stamp for stamp, _ in bars],
                    "indicators": indicators,
                }
            ],
            "error": None,
        }
    }
    return json.dumps(payload)


# Uses distinct regular and adjusted prices to reveal which field was chosen.
def _sample_gld() -> str:
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    closes = [180.25, 181.10, 181.40]
    adjusted = [190.25, 191.10, 191.40]
    bars = [(_ny_unix(day), close) for day, close in zip(days, closes, strict=True)]
    return _yahoo_chart(bars, adjusted=adjusted)


# Provides example responses for all five instruments without network access.
def _sample_bodies() -> dict[str, str]:
    days = [("2024-01-02", "1"), ("2024-01-03", "1"), ("2024-01-04", "1")]
    return {
        "DGS2": _fred(
            "DGS2",
            [("2024-01-02", "4.25"), ("2024-01-03", "4.30"), ("2024-01-04", "4.40")],
        ),
        "DGS10": _fred(
            "DGS10",
            [("2024-01-02", "4.00"), ("2024-01-03", "4.05"), ("2024-01-04", "4.10")],
        ),
        "DEXUSEU": _fred("DEXUSEU", [("2024-01-02", "1.095"), *days[1:]]),
        "SP500": _fred(
            "SP500",
            [("2024-01-02", "4700.5"), ("2024-01-03", "4710"), ("2024-01-04", "4720")],
        ),
        "GLD": _sample_gld(),
    }


# Identifies which instrument a simulated download request is asking for.
def _body_key(request: httpx.Request) -> str:
    series_id = request.url.params.get("id")
    if isinstance(series_id, str) and series_id:
        return series_id
    if request.url.path.rstrip("/").endswith("/chart/GLD"):
        return "GLD"
    return ""


# Returns the example data with the response type expected from its provider.
def _mocked_response(request: httpx.Request, bodies: dict[str, str]) -> httpx.Response:
    key = _body_key(request)
    body = bodies.get(key)
    if body is None:
        return httpx.Response(404, text=str(request.url))
    if body.lstrip().startswith("<"):
        content_type = "text/html"
    elif key == "GLD":
        content_type = "application/json"
    else:
        content_type = "text/csv"
    return httpx.Response(200, text=body, headers={"content-type": content_type})


# Runs the downloader against simulated responses and records the requested URLs.
def _fetch(
    bodies: dict[str, str],
    start: date = START,
    end: date = END,
) -> pd.DataFrame:
    urls: list[str] = []

    # Returns a simulated provider response so this test does not need internet access.
    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return _mocked_response(request, bodies)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        frame = fetch_daily_market_data(
            start,
            end,
            client=client,
            retrieved_at=RETRIEVED_AT,
        )
    frame.attrs["urls"] = urls
    return frame


# Selects one instrument and date from the downloaded table for checking.
def _at(frame: pd.DataFrame, day: date, instrument: str) -> pd.Series:
    match = frame[
        (frame["observation_date"] == day) & (frame["instrument"] == instrument)
    ]
    assert len(match) == 1
    return match.iloc[0]


# Checks labels, units, source details and the preference for adjusted GLD prices.
def test_labels_proxy_and_preserves_yield_units() -> None:
    frame = _fetch(_sample_bodies())

    assert list(frame.columns) == list(OUTPUT_COLUMNS)
    assert set(frame["instrument"]) == {
        "us_2y_yield",
        "us_10y_yield",
        "eurusd",
        "sp500",
        "gold_gld_proxy",
    }
    yield_row = _at(frame, START, "us_2y_yield")
    assert yield_row["value"] == 4.25
    assert yield_row["price_field"] == "DGS2"
    assert yield_row["unit"] == "percentage_points"
    assert yield_row["asset_class"] == "rates"
    assert bool(yield_row["is_proxy"]) is False
    assert yield_row["source_name"] == "FRED"
    ten_year = _at(frame, START, "us_10y_yield")
    assert ten_year["value"] == 4.00
    assert ten_year["unit"] == "percentage_points"
    euro = _at(frame, START, "eurusd")
    assert euro["value"] == pytest.approx(1.095)
    assert euro["unit"] == "usd_per_eur"
    assert euro["asset_class"] == "fx"
    sp500 = _at(frame, START, "sp500")
    assert sp500["value"] == pytest.approx(4700.5)
    assert sp500["unit"] == "index_level"
    assert sp500["asset_class"] == "equity"
    gold = _at(frame, START, "gold_gld_proxy")
    assert gold["value"] == pytest.approx(190.25)
    assert gold["price_field"] == "adjclose"
    assert gold["unit"] == "usd_per_share"
    assert gold["asset_class"] == "commodity"
    assert bool(gold["is_proxy"]) is True
    assert gold["source_name"] == "Yahoo Finance"
    assert gold["source_url"] == "https://query1.finance.yahoo.com/v8/finance/chart/GLD"
    assert set(frame.loc[frame["is_proxy"].astype(bool), "instrument"]) == {
        "gold_gld_proxy"
    }
    urls = frame.attrs["urls"]
    assert isinstance(urls, list)
    assert any("id=DGS2" in url and "api_key" not in url.lower() for url in urls)
    assert any("cosd=2024-01-02" in url and "coed=2024-01-04" in url for url in urls)
    assert any(
        "/chart/GLD" in url
        and "interval=1d" in url
        and "includeAdjustedClose=true" in url
        for url in urls
    )
    assert all("stooq" not in url for url in urls)
    assert all(value == RETRIEVED_AT for value in frame["retrieved_at_utc"])


# Checks that a gap in FRED data is not replaced with the previous observation.
def test_missing_observations_are_not_forward_filled() -> None:
    bodies = _sample_bodies()
    bodies["DGS2"] = _fred(
        "DGS2",
        [("2024-01-02", "4.25"), ("2024-01-03", "."), ("2024-01-04", "4.40")],
    )

    frame = _fetch(bodies)
    gap = _at(frame, date(2024, 1, 3), "us_2y_yield")
    following = _at(frame, date(2024, 1, 4), "us_2y_yield")

    assert _is_missing(gap["value"])
    assert following["value"] == 4.40
    assert len(frame[frame["instrument"] == "us_2y_yield"]) == 3


# Checks that text in a numeric market field causes a clear error.
def test_rejects_malformed_values() -> None:
    bodies = _sample_bodies()
    bodies["DEXUSEU"] = _fred(
        "DEXUSEU",
        [("2024-01-02", "abc"), ("2024-01-03", "1.09"), ("2024-01-04", "1.09")],
    )

    with pytest.raises(MarketDataError, match="must be numeric"):
        _fetch(bodies)


# Checks that unexpected CSV columns are rejected rather than misread as prices.
def test_rejects_duplicate_dates() -> None:
    bodies = _sample_bodies()
    bodies["DGS10"] = _fred(
        "DGS10",
        [("2024-01-02", "4.00"), ("2024-01-02", "4.10"), ("2024-01-03", "4.05")],
    )

    with pytest.raises(MarketDataError, match="Duplicate"):
        _fetch(bodies)


# Checks that FRED observations outside the requested dates are rejected.
def test_rejects_unexpected_schema() -> None:
    bodies = _sample_bodies()
    bodies["SP500"] = "date,close\n2024-01-02,4700\n2024-01-03,4710\n"

    with pytest.raises(MarketDataError, match="SP500"):
        _fetch(bodies)


# Checks that FRED observations outside the requested dates are rejected.
def test_rejects_dates_outside_the_requested_range() -> None:
    bodies = _sample_bodies()
    bodies["DGS2"] = _fred(
        "DGS2",
        [("2023-12-29", "4.10"), ("2024-01-02", "4.25"), ("2024-01-03", "4.30")],
    )

    with pytest.raises(MarketDataError, match="outside"):
        _fetch(bodies, start=START, end=date(2024, 1, 3))


# Checks that a failed download stops the run instead of leaving an incomplete table.
def test_rejects_incomplete_source() -> None:
    bodies = _sample_bodies()
    del bodies["GLD"]

    with pytest.raises(MarketDataError, match="HTTP 404"):
        _fetch(bodies)


# Checks that the download command saves all five instruments with the expected units.
def test_command_writes_daily_market_csv(tmp_path: Path) -> None:
    output = tmp_path / "daily_market_data.csv"
    bodies = _sample_bodies()

    # Returns a simulated provider response so this test does not need internet access.
    def handler(request: httpx.Request) -> httpx.Response:
        return _mocked_response(request, bodies)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        main(
            ["--start", "2024-01-02", "--end", "2024-01-04", "--output", str(output)],
            client=client,
        )

    saved = pd.read_csv(output)
    assert set(saved["instrument"]) == {
        "us_2y_yield",
        "us_10y_yield",
        "eurusd",
        "sp500",
        "gold_gld_proxy",
    }
    assert set(saved.loc[saved["instrument"] == "us_2y_yield", "unit"]) == {
        "percentage_points"
    }
    assert "reaction" not in output.read_text(encoding="utf-8")


# Checks that regular closes are used when adjusted prices are absent.
def test_gld_uses_close_when_adjusted_close_is_absent() -> None:
    bodies = _sample_bodies()
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    closes: list[object] = [180.25, None, 181.40]
    bars = [(_ny_unix(day), close) for day, close in zip(days, closes, strict=True)]
    bodies["GLD"] = _yahoo_chart(bars)

    frame = _fetch(bodies)
    first = _at(frame, START, "gold_gld_proxy")
    gap = _at(frame, date(2024, 1, 3), "gold_gld_proxy")
    last = _at(frame, date(2024, 1, 4), "gold_gld_proxy")

    assert first["value"] == pytest.approx(180.25)
    assert first["price_field"] == "close"
    assert _is_missing(gap["value"])
    assert gap["price_field"] == ""
    assert last["value"] == pytest.approx(181.40)
    assert last["price_field"] == "close"
    assert last["value"] != pytest.approx(180.25)


# Checks that one missing adjusted price falls back to that day's regular close.
def test_gld_falls_back_to_close_for_a_missing_adjusted_bar() -> None:
    bodies = _sample_bodies()
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    closes = [180.25, 181.10, 181.40]
    adjusted: list[object] = [190.25, None, 191.40]
    bars = [(_ny_unix(day), close) for day, close in zip(days, closes, strict=True)]
    bodies["GLD"] = _yahoo_chart(bars, adjusted=adjusted)

    frame = _fetch(bodies)
    first = _at(frame, START, "gold_gld_proxy")
    gap = _at(frame, date(2024, 1, 3), "gold_gld_proxy")
    last = _at(frame, date(2024, 1, 4), "gold_gld_proxy")

    assert first["value"] == pytest.approx(190.25)
    assert first["price_field"] == "adjclose"
    assert gap["value"] == pytest.approx(181.10)
    assert gap["price_field"] == "close"
    assert last["value"] == pytest.approx(191.40)
    assert last["price_field"] == "adjclose"


# Checks that UTC timestamps map to the correct New York trading dates.
def test_gld_timestamps_use_new_york_trading_dates() -> None:
    start = date(2024, 1, 1)
    end = date(2024, 1, 2)
    bodies = _sample_bodies()
    for series_id in ("DGS2", "DGS10", "DEXUSEU", "SP500"):
        bodies[series_id] = _fred(
            series_id,
            [("2024-01-01", "1.5"), ("2024-01-02", "1.6")],
        )
    previous_evening_utc = int(datetime(2024, 1, 2, 4, 30, tzinfo=UTC).timestamp())
    bodies["GLD"] = _yahoo_chart(
        [
            (previous_evening_utc, 100.0),
            (_ny_unix("2024-01-02"), 200.0),
        ],
        adjusted=[110.0, 210.0],
    )

    frame = _fetch(bodies, start=start, end=end)
    evening = _at(frame, date(2024, 1, 1), "gold_gld_proxy")
    session = _at(frame, date(2024, 1, 2), "gold_gld_proxy")

    assert evening["value"] == pytest.approx(110.0)
    assert session["value"] == pytest.approx(210.0)
    assert evening["price_field"] == "adjclose"


# Checks rejection of browser pages, broken responses and invalid instrument data.
def test_rejects_yahoo_html_and_malformed_chart() -> None:
    html = _sample_bodies()
    html["GLD"] = (
        "<!doctype html><html>"
        "This site requires JavaScript to verify your browser."
        "</html>"
    )
    with pytest.raises(MarketDataError, match="HTML"):
        _fetch(html)

    malformed = _sample_bodies()
    malformed["GLD"] = "{not json"
    with pytest.raises(MarketDataError, match="not valid JSON"):
        _fetch(malformed)

    failed = _sample_bodies()
    failed["GLD"] = json.dumps(
        {"chart": {"result": None, "error": {"description": "No data found"}}}
    )
    with pytest.raises(MarketDataError, match="request failed"):
        _fetch(failed)

    wrong_symbol = _sample_bodies()
    wrong_symbol["GLD"] = _yahoo_chart(
        [(_ny_unix("2024-01-02"), 10.0)],
        symbol="IAU",
    )
    with pytest.raises(MarketDataError, match="symbol must be GLD"):
        _fetch(wrong_symbol)

    wrong_currency = _sample_bodies()
    wrong_currency["GLD"] = _yahoo_chart(
        [(_ny_unix("2024-01-02"), 10.0)],
        currency="EUR",
    )
    with pytest.raises(MarketDataError, match="currency must be USD"):
        _fetch(wrong_currency)

    negative = _sample_bodies()
    negative["GLD"] = _yahoo_chart(
        [(_ny_unix("2024-01-02"), 10.0)],
        adjusted=[-1.0],
    )
    with pytest.raises(MarketDataError, match="finite positive"):
        _fetch(negative)


# Checks that GLD observations outside the requested dates are rejected.
def test_rejects_duplicate_gld_trading_dates() -> None:
    bodies = _sample_bodies()
    bodies["GLD"] = _yahoo_chart(
        [
            (_ny_unix("2024-01-02", 9, 30), 190.0),
            (_ny_unix("2024-01-02", 16, 0), 191.0),
            (_ny_unix("2024-01-03"), 192.0),
        ],
        adjusted=[190.0, 191.0, 192.0],
    )

    with pytest.raises(MarketDataError, match="Duplicate"):
        _fetch(bodies)


# Checks that GLD observations outside the requested trading dates are rejected.
def test_rejects_gld_dates_outside_the_requested_range() -> None:
    bodies = _sample_bodies()
    for series_id in ("DGS2", "DGS10", "DEXUSEU", "SP500"):
        bodies[series_id] = _fred(
            series_id,
            [("2024-01-02", "1.5"), ("2024-01-03", "1.6")],
        )
    bodies["GLD"] = _yahoo_chart(
        [
            (_ny_unix("2023-12-29"), 180.0),
            (_ny_unix("2024-01-02"), 190.0),
        ],
        adjusted=[180.0, 190.0],
    )

    with pytest.raises(
        MarketDataError, match="gold_gld_proxy date 2023-12-29 is outside"
    ):
        _fetch(bodies, start=START, end=date(2024, 1, 3))


# Checks that the download command can show its help without a Python runtime warning.
def test_module_entrypoint_does_not_warn() -> None:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "macro_surprise.data.market_client",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
