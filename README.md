# US CPI Surprise and Cross Asset Reaction Study

This project studies how financial markets respond when United States inflation differs from what economists expected. It focuses on Treasury yields, the EUR/USD exchange rate, the S&P 500 and gold.

## Research question

When a US CPI release is higher or lower than economists expected, how do these five markets respond on the release day, after one trading session and after five trading sessions?

## Why I chose this question

I first studied inflation and interest rates while preparing for my O Level Economics exam. Textbooks gave me a clear idea of what should happen. Later, when I started following financial news at university, markets did not always move in the direction I expected. For a moment, I wondered whether I had misunderstood what I had learned.

I then began to focus on expectations. A high inflation number may not surprise the market if investors already expected it. I built this study to examine the difference between the reported CPI value and the forecast available before its release and to measure how that surprise relates to market movements.

## Study design and data

The sample covers 104 CPI announcement dates from January 2018 through September 2026. Four measures across 104 dates create 416 possible date and measure combinations. Of these, 414 had usable CPI observations. Each usable observation is matched with five financial instruments, giving 2,070 event and asset rows. The study includes three reaction windows.

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

Bloomberg provided the originally reported CPI values, survey medians available before release, and release dates and times. FRED provided daily observations for Treasury yields, EUR/USD and the S&P 500. Yahoo Finance provided adjusted closing prices for GLD.

## Methodology

### Step 1: Calculate the CPI surprise

For each CPI measure, the surprise is the original reported value minus the survey median recorded before release.

```text
raw surprise = original CPI value - pre-release survey median
```

A positive surprise means CPI was higher than expected. A negative surprise means it was lower than expected. A zero surprise means the reported value matched the survey median.

### Step 2: Put surprises on a comparable scale

Headline and core CPI measures have different typical ranges. To compare them, each raw surprise is divided by the historical standard deviation of earlier surprises from the same measure.

```text
standardized surprise = raw surprise / standard deviation of earlier surprises
```

At least 12 earlier observations are required. Only information available before the release is used, so a later event cannot affect an earlier surprise score.

### Step 3: Measure the market response

There are 414 usable CPI measure observations. Matching each one with five instruments gives:

```text
414 CPI measure observations x 5 instruments = 2,070 event and asset rows
```

Each row represents one CPI measure on one release date matched with one market instrument. The row contains the market reaction for three windows: the release day, the next valid trading session and the fifth valid trading session. Each window starts from the last valid market observation before the release.

Treasury yield changes are measured in basis points. EUR/USD, the S&P 500 and GLD are measured as percentage returns. A missing release day price is kept missing rather than filled using an earlier price.

### Step 4: Group the rows and estimate the relationships

The analysis asks a separate question for each CPI measure, instrument and reaction window. First, four CPI measures and five instruments form 20 CPI and instrument groups:

```text
4 CPI measures x 5 instruments = 20 groups
```

Each group has three reaction windows. One regression is run for each window:

```text
20 groups x 3 reaction windows = 60 ordinary least squares regressions
```

Each regression uses about 102 to 104 release observations. Ordinary least squares, or OLS, finds the straight line that makes the squared gaps between the observed market reactions and the line's estimates as small as possible. The slope of that line is called beta. It estimates the average market movement linked with a one standard deviation increase in the CPI surprise.

### Step 5: Check uncertainty and repeated testing

HC3 standard errors are used for the main confidence intervals and p-values. HAC standard errors provide an additional check that allows nearby observations to be related.

The analysis tests 60 primary relationships. Benjamini-Hochberg false discovery rate correction adjusts the p-values together, reducing the chance of treating results as reliable simply because many relationships were tested. A corrected q-value below 0.05 is the chosen threshold for statistical significance.

## Data quality and additional checks

The project includes several checks. Bloomberg's historical CPI tables are compared with its economic calendar records for agreement on actual values and survey medians. The analysis can also be repeated using only observations where the two records agree. HC3 and HAC estimates are compared as an uncertainty check.

Separate tests examine whether there is additional market movement after the release day and whether estimates differ before 2020 and from 2020 onward. Other checks look for duplicate releases, duplicate market records and missing prices.

## Data access and publication restrictions

The Bloomberg workbook is licensed data and is not included in this repository. Bloomberg advised that Bloomberg data and results based on it should not be posted on public sites without an appropriate redistribution agreement. Therefore, this README describes the study design and analysis method.

## Scope and limitations

This project is designed to measure how CPI surprises relate to market movements around scheduled releases. It does not prove that CPI alone caused a market move. Daily prices can also reflect other news and market conditions. GLD is an exchange traded fund, not the spot gold price. The four CPI measures are related and are released together, so they are not independent tests. The analysis does not directly measure investors' Federal Reserve expectations or test a trading strategy.

## Running the project

Python 3.12 or newer is required. To install the project and run its automated tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,dev]'
pytest
```

The automated tests use small example inputs and do not require Bloomberg data or network access. `run_pipeline.py` coordinates the full analysis, which requires licensed CPI input. Use it only in an environment and workflow permitted by Bloomberg and the relevant university guidance. The licensed input is not supplied here. Generated analysis files, charts and reports are not included in this repository.

## Repository guide

| Location | Purpose |
|---|---|
| `run_pipeline.py` | Coordinates the full analysis in an authorized environment |
| `src/macro_surprise/data/` | Downloads public market data and reads licensed CPI input |
| `src/macro_surprise/analysis/` | Calculates CPI surprises, market reactions and regression statistics |
| `scripts/` | Builds the research charts |
| `tests/` | Checks calculations, data handling and error cases |
| `data/external/` | Stores public daily market data |
