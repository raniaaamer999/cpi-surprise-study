# US CPI Surprise and Cross Asset Reaction Study

This project studies how financial markets respond when United States inflation differs from what economists expected. It focuses on Treasury yields, the EUR/USD exchange rate, the S&P 500 and gold. The analysis compares the CPI value reported at release with the survey median recorded beforehand.

## Research question

When a US CPI release is higher or lower than economists expected, how do these five markets respond on the release day, after one trading session and after five trading sessions?

## Why I chose this question

I first studied inflation and interest rates while preparing for my O Level Economics exam. Textbooks gave me a clear idea of what should happen. Later, when I started following financial news at university, markets did not always move in the direction I expected. For a moment, I wondered whether I had misunderstood what I had learned.

I then began to focus on expectations. A high inflation number may not surprise the market if investors already expected it. I built this study to examine the difference between the reported CPI value and the forecast available before its release, and to measure how that surprise relates to market movements.

## Study design and data

The sample covers 104 CPI announcement dates from January 2018 through September 2026. It contains 414 CPI measure observations, matched with five financial instruments to form 2,070 event and asset observations. The study uses four CPI measures and three reaction windows.

| CPI measure | What it compares |
|---|---|
| Headline monthly | CPI including food and energy, compared with the previous month |
| Headline yearly | CPI including food and energy, compared with the same month one year earlier |
| Core monthly | CPI excluding food and energy, compared with the previous month |
| Core yearly | CPI excluding food and energy, compared with the same month one year earlier |

| Instrument | What it represents |
|---|---|
| 2 year US Treasury yield | Shorter term interest rate expectations |
| 10 year US Treasury yield | Longer term interest rate, inflation and growth expectations |
| EUR/USD | The value of the euro relative to the US dollar |
| S&P 500 | A broad measure of large US company share prices |
| GLD | A gold exchange traded fund used as a tradable gold proxy |

The reaction windows are the release day, the next trading session and the fifth trading session.

Bloomberg provided the originally reported CPI values, survey medians available before release, and release dates and times. FRED provided daily observations for Treasury yields, EUR/USD and the S&P 500. Yahoo Finance provided adjusted closing prices for GLD. The licensed Bloomberg workbook is not included in this repository. Any use of Bloomberg data must follow the terms that apply to the user's access.

## How the analysis works

First, the CPI surprise is calculated as:

```text
raw surprise = original CPI value - survey median recorded before release
```

A positive surprise means that CPI was higher than expected. A negative surprise means it was lower than expected.

The raw surprise is then divided by the historical standard deviation of earlier surprises from the same CPI measure. At least 12 earlier observations are required. Only information available before each release is used, so a later CPI release cannot affect an earlier surprise score. This puts different CPI measures on a more comparable scale.

For each event, market movements are measured from the last valid market observation before release. Treasury yield changes are measured in basis points. EUR/USD, the S&P 500 and GLD are measured as percentage returns. The code keeps a missing release day value missing rather than filling it with an earlier price.

The main analysis contains 60 regression specifications: four CPI measures multiplied by five instruments multiplied by three reaction windows. Each regression estimates the average market movement associated with a one standard deviation CPI surprise. Ordinary least squares is used to estimate this relationship. HC3 standard errors provide the main uncertainty estimates. HAC standard errors are used as an additional check. The Benjamini-Hochberg procedure adjusts for testing many relationships at once.

## Data checks and additional tests

The project includes checks designed to assess data quality and test how the analysis behaves under different choices:

* Bloomberg's historical CPI tables are compared with its economic calendar records for agreement on actual values and survey medians.
* The main analysis can be repeated using only observations where those Bloomberg records agree.
* HC3 and HAC uncertainty estimates are compared.
* Additional tests examine whether market movements continue or reverse after the release day.
* The analysis compares estimates from before 2020 with estimates from 2020 onward.
* Tests check for duplicate events, duplicate market records and missing observations.

This README describes the checks but does not report their empirical outcomes.

## Conclusion and limitations

This project estimates relationships between CPI surprises and market movements around scheduled releases. It does not establish that CPI alone caused each movement. Daily prices can also reflect other news and market conditions. GLD is an exchange traded fund, not the spot gold price. The four CPI measures are related and are released together, so they are not independent tests of inflation. The analysis also does not directly measure investors' Federal Reserve expectations or test a trading strategy.

This README focuses on the research question, data structure and analysis method. It does not state the market findings or list regression estimates. The analysis is designed to test how CPI surprises relate to market movements, not to prove what caused each move.

## Running the project

Python 3.12 or newer is required. To install the project and run its automated tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,dev]'
pytest
```

The full pipeline can be run with:

```bash
python run_pipeline.py
```

The full run requires the Bloomberg workbook at `data/private/cpi_bloomberg_private.xlsx` and daily market data at `data/external/daily_market_data.csv`. The pipeline creates analysis files and charts locally. The Bloomberg workbook is not supplied here, and users must have authorized access before working with Bloomberg data. The automated tests use small example inputs and do not require the Bloomberg workbook or network access.

## Repository guide

| Location | Purpose |
|---|---|
| `run_pipeline.py` | Runs the CPI data cleaning, event matching, analysis, chart creation and tests in order |
| `src/macro_surprise/data/` | Downloads public market data and reads the private Bloomberg workbook |
| `src/macro_surprise/analysis/` | Calculates CPI surprises, market reactions and regression statistics |
| `scripts/` | Builds the research charts |
| `tests/` | Checks calculations, data handling and error cases |
| `data/external/` | Stores public daily market data |
| `data/private/` | Local location for the licensed Bloomberg workbook; excluded from Git |
