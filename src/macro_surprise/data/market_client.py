"""Downloads and checks the daily market data used in the CPI study."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from io import StringIO
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import httpx
import pandas as pd  # type: ignore[import-untyped]


class AssetClass(StrEnum):
    RATES = "rates"
    FX = "fx"
    EQUITY = "equity"
    COMMODITY = "commodity"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GLD"
_NEW_YORK = ZoneInfo("America/New_York")
_USER_AGENT = "cpi-surprise-event-study/0.1 (research; public market data)"
_MISSING_TOKENS = frozenset({"", "."})

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MARKET_DATA_PATH = _REPO_ROOT / "data" / "external" / "daily_market_data.csv"
_ALLOWED_HOSTS = frozenset(
    {
        "fred.stlouisfed.org",
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
    }
)

OUTPUT_COLUMNS = (
    "observation_date",
    "instrument",
    "asset_class",
    "value",
    "price_field",
    "unit",
    "source_name",
    "source_url",
    "is_proxy",
    "retrieved_at_utc",
)


class MarketDataError(ValueError):
    pass


@dataclass(frozen=True)
class _Instrument:
    instrument: str
    asset_class: AssetClass
    unit: str
    source_name: str
    provider_id: str
    is_proxy: bool
    require_positive: bool
    kind: str


MARKET_INSTRUMENTS = (
    _Instrument(
        "us_2y_yield",
        AssetClass.RATES,
        "percentage_points",
        "FRED",
        "DGS2",
        False,
        False,
        "fred",
    ),
    _Instrument(
        "us_10y_yield",
        AssetClass.RATES,
        "percentage_points",
        "FRED",
        "DGS10",
        False,
        False,
        "fred",
    ),
    _Instrument(
        "eurusd",
        AssetClass.FX,
        "usd_per_eur",
        "FRED",
        "DEXUSEU",
        False,
        True,
        "fred",
    ),
    _Instrument(
        "sp500",
        AssetClass.EQUITY,
        "index_level",
        "FRED",
        "SP500",
        False,
        True,
        "fred",
    ),
    _Instrument(
        "gold_gld_proxy",
        AssetClass.COMMODITY,
        "usd_per_share",
        "Yahoo Finance",
        "GLD",
        True,
        True,
        "yahoo",
    ),
)
_INSTRUMENTS_BY_NAME = {
    instrument.instrument: instrument for instrument in MARKET_INSTRUMENTS
}


def fetch_daily_market_data(
    start: date,
    end: date,
    *,
    client: httpx.Client | None = None,
    retrieved_at: datetime | None = None,
) -> pd.DataFrame:
    # Downloads all five instruments and keeps missing source values empty.
    start_date = _require_date(start, "start")
    end_date = _require_date(end, "end")
    _validate_range(start_date, end_date)
    retrieved = _as_utc(retrieved_at)
    owns_client = client is None
    http = client if client is not None else httpx.Client(timeout=30.0)
    try:
        rows: list[dict[str, object]] = []
        for spec in MARKET_INSTRUMENTS:
            rows.extend(
                _download_instrument(http, spec, start_date, end_date, retrieved)
            )
    finally:
        if owns_client:
            http.close()
    return _as_frame(rows)


def save_daily_market_data(
    frame: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    # Checks the table again before saving it to the external data folder.
    output = DEFAULT_MARKET_DATA_PATH if path is None else Path(path)
    _require_market_table(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(OUTPUT_COLUMNS)].to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None, *, client: httpx.Client | None = None) -> None:
    # Handles the command line inputs and saves the finished market file.
    args = _parse_args(argv)
    output = Path(args.output)
    try:
        frame = fetch_daily_market_data(
            args.start,
            args.end,
            client=client,
        )
        saved = save_daily_market_data(frame, output)
    except MarketDataError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Wrote {len(frame)} daily market rows to {saved}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download free daily market levels for the CPI event study. "
            "Yields stay in percentage points. GLD is a gold proxy."
        )
    )
    parser.add_argument(
        "--start",
        type=_cli_date,
        required=True,
        help="First observation date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        type=_cli_date,
        required=True,
        help="Last observation date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MARKET_DATA_PATH,
        help="CSV path. Default: data/external/daily_market_data.csv.",
    )
    return parser.parse_args(argv)


def _cli_date(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a YYYY-MM-DD date") from exc


def _require_date(value: object, label: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise MarketDataError(f"{label} must be a calendar date")
    return value


def _validate_range(start: date, end: date) -> None:
    if start > end:
        raise MarketDataError("start must not be after end")
    if start.year < 1900:
        raise MarketDataError("start must be 1900 or later")
    latest = datetime.now(UTC).date() + timedelta(days=1)
    if end > latest:
        raise MarketDataError(f"end must be {latest.isoformat()} or earlier")


def _as_utc(retrieved_at: datetime | None) -> datetime:
    stamp = datetime.now(UTC) if retrieved_at is None else retrieved_at
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise MarketDataError("retrieved_at must be timezone-aware")
    return stamp.astimezone(UTC)


def _download_instrument(
    client: httpx.Client,
    spec: _Instrument,
    start: date,
    end: date,
    retrieved_at: datetime,
) -> list[dict[str, object]]:
    # Sends each instrument to the parser for its own data source.
    if spec.kind == "fred":
        text = _get_text(client, _fred_url(spec.provider_id, start, end))
        rows = _parse_fred(text, spec, start, end, retrieved_at)
    elif spec.kind == "yahoo":
        text = _get_text(
            client,
            _yahoo_url(start, end),
            accept="application/json",
        )
        rows = _parse_yahoo(text, spec, start, end, retrieved_at)
    else:
        raise MarketDataError(f"Unknown market source {spec.kind!r}")
    if not rows:
        raise MarketDataError(
            f"{spec.source_name} {spec.provider_id} returned no daily observations"
        )
    return rows


def _fred_url(series_id: str, start: date, end: date) -> str:
    return (
        f"{FRED_CSV_URL}?id={series_id}&cosd={start.isoformat()}&coed={end.isoformat()}"
    )


def _yahoo_url(start: date, end: date) -> str:
    period_start = _new_york_midnight(start)
    period_end = _new_york_midnight(end + timedelta(days=1))
    return (
        f"{YAHOO_CHART_URL}?period1={period_start}&period2={period_end}"
        "&interval=1d&includeAdjustedClose=true"
    )


def _new_york_midnight(day: date) -> int:
    local = datetime.combine(day, time.min, tzinfo=_NEW_YORK)
    return int(local.timestamp())


def _get_text(
    client: httpx.Client,
    url: str,
    *,
    accept: str = "text/csv",
) -> str:
    # Accepts responses only from the data providers used by this project.
    try:
        response = client.get(
            url,
            headers={"User-Agent": _USER_AGENT, "Accept": accept},
            timeout=30.0,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise MarketDataError(
            f"Market request could not be completed for {url}"
        ) from exc
    host = response.url.host or ""
    if host not in _ALLOWED_HOSTS:
        raise MarketDataError(f"Refusing market data from {host or 'an unknown host'}")
    if response.status_code != 200:
        raise MarketDataError(
            f"Market request failed with HTTP {response.status_code} for {url}"
        )
    if accept == "application/json":
        _reject_html(response)
    return response.text


def _reject_html(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "")
    body = response.text.lstrip()
    if "html" in content_type.lower() or body.startswith("<"):
        raise MarketDataError("Yahoo Finance returned HTML instead of GLD chart JSON")


def _parse_fred(
    text: str,
    spec: _Instrument,
    start: date,
    end: date,
    retrieved_at: datetime,
) -> list[dict[str, object]]:
    # Keeps FRED missing values empty instead of filling them with older values.
    header, records = _csv_table(text, spec.provider_id)
    expected = ["observation_date", spec.provider_id]
    if header != expected:
        found = ", ".join(header)
        raise MarketDataError(
            f"FRED {spec.provider_id} columns must be {', '.join(expected)}. "
            f"Found: {found}"
        )
    rows: list[dict[str, object]] = []
    seen: set[date] = set()
    for record in records:
        observed = _observation_date(
            record["observation_date"],
            spec.instrument,
            start,
            end,
        )
        _reject_duplicate(spec.instrument, observed, seen)
        value = _number(
            record[spec.provider_id],
            label=f"{spec.instrument} {observed.isoformat()}",
            require_positive=spec.require_positive,
        )
        rows.append(
            _row(
                spec,
                observed,
                value,
                _fred_source_url(spec),
                retrieved_at,
                spec.provider_id,
            )
        )
    return rows


def _parse_yahoo(
    text: str,
    spec: _Instrument,
    start: date,
    end: date,
    retrieved_at: datetime,
) -> list[dict[str, object]]:
    chart = _yahoo_chart(text)
    timestamps = _yahoo_timestamps(chart)
    closes = _yahoo_closes(chart, len(timestamps))
    adjusted = _yahoo_adjusted(chart, len(timestamps))
    rows: list[dict[str, object]] = []
    seen: set[date] = set()
    for index, timestamp in enumerate(timestamps):
        observed = _trading_date(timestamp, spec.instrument, start, end)
        _reject_duplicate(spec.instrument, observed, seen)
        value, price_field = _yahoo_price(
            closes[index],
            None if adjusted is None else adjusted[index],
            f"{spec.instrument} {observed.isoformat()}",
        )
        rows.append(
            _row(spec, observed, value, YAHOO_CHART_URL, retrieved_at, price_field)
        )
    return rows


def _yahoo_chart(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MarketDataError("Yahoo Finance GLD response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise MarketDataError("Yahoo Finance GLD response was not a JSON object")
    chart = cast(dict[str, object], payload).get("chart")
    if not isinstance(chart, dict):
        raise MarketDataError("Yahoo Finance GLD response did not include chart")
    chart_object = cast(dict[str, object], chart)
    if chart_object.get("error") is not None:
        raise MarketDataError(
            f"Yahoo Finance GLD request failed: {chart_object.get('error')}"
        )
    result = chart_object.get("result")
    if (
        not isinstance(result, list)
        or len(result) != 1
        or not isinstance(result[0], dict)
    ):
        raise MarketDataError(
            "Yahoo Finance GLD response did not include one chart result"
        )
    item = cast(dict[str, object], result[0])
    _require_gld_meta(item.get("meta"))
    return item


def _require_gld_meta(meta: object) -> None:
    if not isinstance(meta, dict):
        raise MarketDataError("Yahoo Finance GLD response did not include meta")
    details = cast(dict[str, object], meta)
    if details.get("symbol") != "GLD":
        raise MarketDataError(
            f"Yahoo Finance symbol must be GLD, got {details.get('symbol')!r}"
        )
    if details.get("currency") != "USD":
        raise MarketDataError(
            f"Yahoo Finance GLD currency must be USD, got {details.get('currency')!r}"
        )
    timezone_name = details.get("exchangeTimezoneName")
    if timezone_name is not None and timezone_name != "America/New_York":
        raise MarketDataError(
            "Yahoo Finance GLD timezone must be America/New_York, "
            f"got {timezone_name!r}"
        )


def _yahoo_timestamps(chart: dict[str, object]) -> list[object]:
    timestamps = chart.get("timestamp")
    if not isinstance(timestamps, list):
        raise MarketDataError("Yahoo Finance GLD response did not include timestamps")
    return list(timestamps)


def _yahoo_closes(chart: dict[str, object], count: int) -> list[object]:
    indicators = chart.get("indicators")
    if not isinstance(indicators, dict):
        raise MarketDataError("Yahoo Finance GLD response did not include indicators")
    quote = cast(dict[str, object], indicators).get("quote")
    if not isinstance(quote, list) or len(quote) != 1 or not isinstance(quote[0], dict):
        raise MarketDataError("Yahoo Finance GLD response did not include quotes")
    closes = cast(dict[str, object], quote[0]).get("close")
    if not isinstance(closes, list) or len(closes) != count:
        raise MarketDataError(
            "Yahoo Finance GLD close prices do not match the timestamps"
        )
    return list(closes)


def _yahoo_adjusted(chart: dict[str, object], count: int) -> list[object] | None:
    indicators = chart.get("indicators")
    if not isinstance(indicators, dict):
        raise MarketDataError("Yahoo Finance GLD response did not include indicators")
    adjusted = cast(dict[str, object], indicators).get("adjclose")
    if adjusted is None:
        return None
    if (
        not isinstance(adjusted, list)
        or len(adjusted) != 1
        or not isinstance(adjusted[0], dict)
    ):
        raise MarketDataError("Yahoo Finance GLD adjusted close was malformed")
    values = cast(dict[str, object], adjusted[0]).get("adjclose")
    if not isinstance(values, list) or len(values) != count:
        raise MarketDataError(
            "Yahoo Finance GLD adjusted closes do not match the timestamps"
        )
    return list(values)


def _trading_date(
    timestamp: object,
    instrument: str,
    start: date,
    end: date,
) -> date:
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise MarketDataError(
            f"{instrument} timestamp must be unix seconds, got {timestamp!r}"
        )
    observed = datetime.fromtimestamp(timestamp, _NEW_YORK).date()
    if observed < start or observed > end:
        raise MarketDataError(
            f"{instrument} date {observed.isoformat()} is outside "
            f"{start.isoformat()} to {end.isoformat()}"
        )
    return observed


def _yahoo_price(
    close: object,
    adjusted: object,
    label: str,
) -> tuple[float | None, str]:
    close_price = _json_price(close, label)
    if adjusted is None and close is None:
        return None, ""
    if adjusted is None:
        return close_price, "" if close_price is None else "close"
    adjusted_price = _json_price(adjusted, label)
    if adjusted_price is not None:
        return adjusted_price, "adjclose"
    if close_price is not None:
        return close_price, "close"
    return None, ""


def _json_price(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MarketDataError(f"{label} must be a finite positive price, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise MarketDataError(f"{label} must be a finite positive price, got {value!r}")
    return number


def _csv_table(text: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.reader(StringIO(text.lstrip("\ufeff")))
    try:
        raw_header = next(reader)
    except StopIteration as exc:
        raise MarketDataError(f"{label} market file was empty") from exc
    header = [cell.strip() for cell in raw_header]
    if not header or any(column == "" for column in header):
        raise MarketDataError(f"{label} market file is missing column names")
    if len(header) != len(set(header)):
        raise MarketDataError(f"{label} market file repeats a column name")
    records: list[dict[str, str]] = []
    for line_number, raw_row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in raw_row):
            continue
        row = [cell.strip() for cell in raw_row]
        if len(row) != len(header):
            raise MarketDataError(
                f"{label} line {line_number} has {len(row)} columns, "
                f"expected {len(header)}"
            )
        records.append(dict(zip(header, row, strict=True)))
    return header, records


def _observation_date(
    text: str,
    instrument: str,
    start: date,
    end: date,
) -> date:
    try:
        observed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise MarketDataError(
            f"{instrument} date must be YYYY-MM-DD, got {text!r}"
        ) from exc
    if observed < start or observed > end:
        raise MarketDataError(
            f"{instrument} date {observed.isoformat()} is outside "
            f"{start.isoformat()} to {end.isoformat()}"
        )
    return observed


def _reject_duplicate(instrument: str, observed: date, seen: set[date]) -> None:
    if observed in seen:
        raise MarketDataError(
            f"Duplicate {instrument} observation on {observed.isoformat()}"
        )
    seen.add(observed)


def _number(text: str, *, label: str, require_positive: bool) -> float | None:
    if text in _MISSING_TOKENS:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise MarketDataError(f"{label} must be numeric, got {text!r}") from exc
    if not math.isfinite(number):
        raise MarketDataError(f"{label} must be a finite number, got {text!r}")
    if require_positive and number <= 0:
        raise MarketDataError(f"{label} must be positive, got {text!r}")
    return number


def _row(
    spec: _Instrument,
    observed: date,
    value: float | None,
    source_url: str,
    retrieved_at: datetime,
    price_field: str,
) -> dict[str, object]:
    return {
        "observation_date": observed,
        "instrument": spec.instrument,
        "asset_class": spec.asset_class.value,
        "value": float("nan") if value is None else value,
        "price_field": price_field,
        "_order": _instrument_order(spec.instrument),
        "unit": spec.unit,
        "source_name": spec.source_name,
        "source_url": source_url,
        "is_proxy": spec.is_proxy,
        "retrieved_at_utc": retrieved_at,
    }


def _fred_source_url(spec: _Instrument) -> str:
    return f"{FRED_CSV_URL}?id={spec.provider_id}"


def _instrument_order(name: str) -> int:
    for position, spec in enumerate(MARKET_INSTRUMENTS):
        if spec.instrument == name:
            return position
    raise MarketDataError(f"Unknown instrument {name}")


def _as_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    found = {str(row["instrument"]) for row in rows}
    expected = set(_INSTRUMENTS_BY_NAME)
    if found != expected:
        missing = sorted(expected - found)
        raise MarketDataError(
            "Market download is missing instruments: " + ", ".join(missing)
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["observation_date", "_order"],
        kind="mergesort",
    ).drop(columns="_order")
    frame = frame.reset_index(drop=True)
    _reject_duplicate_frame_rows(frame)
    return frame.loc[:, list(OUTPUT_COLUMNS)]


def _reject_duplicate_frame_rows(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(["instrument", "observation_date"], keep=False)
    if bool(duplicated.any()):
        raise MarketDataError("Duplicate instrument and observation_date rows")


def _require_market_table(frame: pd.DataFrame) -> None:
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise MarketDataError("Market table is missing columns: " + ", ".join(missing))
    if frame.empty:
        raise MarketDataError("Refusing to save an empty market table")
    found = set(frame["instrument"].astype(str))
    expected = set(_INSTRUMENTS_BY_NAME)
    if found != expected:
        raise MarketDataError(
            "Market table instruments must be "
            + ", ".join(spec.instrument for spec in MARKET_INSTRUMENTS)
        )
    proxy_names = set(
        frame.loc[frame["is_proxy"].astype(bool), "instrument"].astype(str)
    )
    if proxy_names != {"gold_gld_proxy"}:
        raise MarketDataError(
            "Only gold_gld_proxy may be labeled as a proxy. "
            f"Found: {sorted(proxy_names) or 'none'}"
        )
    for name in ("us_2y_yield", "us_10y_yield"):
        units = set(frame.loc[frame["instrument"] == name, "unit"].astype(str))
        if units != {"percentage_points"}:
            raise MarketDataError(
                f"{name} must stay in percentage_points. Found: {sorted(units)}"
            )


if __name__ == "__main__":
    main()
