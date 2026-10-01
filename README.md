# US CPI Surprise and Cross Asset Reaction Study

This project studies what happened in financial markets when United States
inflation was different from what economists expected. It focuses on a simple
question: when a CPI release surprised the market, how did Treasury yields,
the United States dollar, the S&P 500, and gold respond?

## Why I built it

I first studied inflation and interest rates while preparing for my O Level
Economics exam. Textbooks gave me a clear idea of what should happen, but when
I later followed financial news at university, markets did not always move in
the direction I expected. The missing part was expectations. A high inflation
number can still be unsurprising if the market already expected it. I built
this study to separate the published CPI number from the surprise inside that
number and then measure the market response.

## What the study covers

The final sample contains 104 CPI announcement dates from January 2018 through
September 2026. It includes four CPI measures:

* headline CPI compared with the previous month
* headline CPI compared with the same month one year earlier
* core CPI compared with the previous month
* core CPI compared with the same month one year earlier

Core CPI excludes food and energy. The analysis follows five market
instruments:

* the two year United States Treasury yield
* the ten year United States Treasury yield
* the EUR/USD exchange rate
* the S&P 500 index
* GLD, which is used as a tradable proxy for gold

Each CPI measure is matched with each market instrument. This produces 2,070
CPI measure and market instrument observations. Reactions are measured on the
release day, after one trading session, and after five trading sessions.

## Data sources

Bloomberg provides the original CPI values, the economist survey medians, and
the release dates and times. These licensed records are private and are never
committed to this repository.

The Federal Reserve Bank of St. Louis provides the two Treasury yields,
EUR/USD, and the S&P 500 through FRED. Yahoo Finance provides adjusted closing
prices for GLD. The public market file is included so that the daily reaction
calculations can be inspected.

The private Bloomberg workbook is cleaned into 798 historical CPI measure
rows. The exact timing data identify 414 usable CPI measure releases from 2018
through 2026. Matching those 414 releases with five market instruments gives
the 2,070 rows in the final event dataset. There are 104 unique announcement
dates because several CPI measures are released together on the same day.

## How the analysis works

For every CPI measure, the surprise is calculated as the original published
value minus the survey median available before the release.

```text
raw surprise = original published CPI value minus survey median
```

Headline and core measures have different normal ranges, so each raw surprise
is divided by the historical variation of earlier surprises from the same
series. Only earlier releases are used. This prevents later information from
changing the score of an older event.

```text
standardized surprise = raw surprise divided by the earlier historical spread
```

For Treasury yields, the market response is measured in basis points. For
EUR/USD, the S&P 500, and GLD, it is measured as a percentage return. Each
regression then estimates the average market movement associated with a one
standard deviation CPI surprise.

There are 60 main regressions because the study combines four CPI measures,
five market instruments, and three reaction windows. HC3 confidence intervals
are used for the main uncertainty estimate because the amount of variation can
differ across events. HAC results provide a second check that allows nearby
observations to be related. The Benjamini Hochberg procedure controls the false
discovery rate across the 60 tests. This reduces the chance that a relationship
looks significant only because many comparisons were made.

Three additional checks are included:

* The Bloomberg agreement check repeats the analysis only when the historical
  and calendar tables report the same CPI actual and survey median.
* The persistence check asks whether the release day response continues into
  the following sessions.
* The period check compares estimates before 2020 with estimates from 2020
  onward.

## Main findings

The strongest evidence appears on the release day. A positive headline
monthly CPI surprise was associated with the following average movements for
a one standard deviation surprise:

* the two year Treasury yield increased by 3.09 basis points
* the ten year Treasury yield increased by 2.37 basis points
* EUR/USD fell by 0.18 percent, which means the United States dollar strengthened
* GLD fell by 0.23 percent

The S&P 500 estimate was negative, but it did not remain statistically
significant after the correction for repeated testing.

Across the full study, 23 of the 60 main relationships remained significant
after the false discovery rate correction. Sixteen of the twenty release day
relationships remained significant. Of the twenty relationships measured after
one session, seven remained significant. None of the twenty relationships
measured after five sessions remained significant.

This pattern supports a careful conclusion. CPI surprises were clearly related
to the first market response, especially in Treasury yields and EUR/USD. The
evidence did not support a dependable continuation over five sessions. The
analysis also found no reliable difference between the estimated relationships
before 2020 and those from 2020 onward after the same correction.

## Data protection

The Bloomberg workbook, cleaned CPI records, event dataset, and detailed result
tables are stored under `data/private/`. That folder is excluded by
`.gitignore`. The public repository contains only code, public market data,
aggregate charts, and aggregate findings. Bloomberg is cited as the source of
the licensed CPI information.

## Quality controls

* The analysis uses original CPI releases rather than later revised values.
* Survey medians are the forecasts recorded before each release.
* Standardization uses only earlier events from the same CPI series.
* Missing release day market observations remain missing.
* Duplicate events and duplicate market records stop the pipeline with an error.
* Bloomberg history and calendar values are compared as a separate data check.
* Insignificant results are retained rather than removed.
* Thirty four automated tests cover data cleaning, surprise calculations,
  trading session matching, regressions, and error handling.

## Run the project

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,dev]'
python run_pipeline.py
```

The master runner cleans the private CPI input, builds the event dataset, runs
the statistical analysis, creates the six charts, and runs the tests in the
correct order. It does not download public data unless that option is selected.

To preview the steps without changing files:

```bash
python run_pipeline.py --dry-run
```

To download the public market data again before rebuilding the study:

```bash
python run_pipeline.py --refresh-public-data
```

The individual commands are also available:

```bash
download-market-data --start 2018-01-01 --end 2026-09-25
clean-bloomberg-cpi --workbook data/private/cpi_bloomberg_private.xlsx
build-cpi-event-dataset
analyze-cpi-reactions
python scripts/build_charts.py
pytest
```

The Bloomberg cleaning command requires a properly licensed private workbook.
The automated tests use small created examples, so they do not require
Bloomberg access or internet access.

## Repository structure

```text
src/macro_surprise/data/       downloads public data and cleans Bloomberg CPI data
src/macro_surprise/analysis/   calculates surprises, reactions, and statistical results
tests/                         checks the main calculations and error handling
scripts/                       creates the six aggregate charts
data/external/                 stores the public daily market data
output/charts/                 stores the aggregate research figures
run_pipeline.py                runs the complete study in the correct order
```

## Limits

Daily market movements can include other news released during the same trading
session. GLD is a tradable gold proxy rather than the spot gold price. The four
CPI measures are related to one another, so their results should not be read as
four fully independent experiments. The regressions measure associations around
scheduled CPI releases. They do not prove that CPI was the only cause of every
market movement, and they do not establish a trading strategy.
