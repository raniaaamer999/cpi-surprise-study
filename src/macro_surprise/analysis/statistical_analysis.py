"""Tests how CPI surprises relate to market reactions across each time window."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import statsmodels.api as sm  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]

from macro_surprise.analysis.build_event_dataset import DEFAULT_EVENT_DATASET

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = _REPO_ROOT / "data" / "private" / "results"
WINDOWS = ("reaction_eod", "reaction_1d", "reaction_5d")


class StatisticalAnalysisError(ValueError):
    pass


def estimate_surprise_reactions(events: pd.DataFrame) -> pd.DataFrame:
    # Runs one regression for every CPI measure, market instrument, and window.
    frame = _prepare(events)
    rows: list[dict[str, object]] = []
    keys = ["series", "instrument", "asset_class", "reaction_unit"]
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        for window in WINDOWS:
            sample = group[["standardized_surprise", window]].dropna()
            fit = _simple_fit(sample["standardized_surprise"], sample[window])
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "window": window,
                    **fit,
                }
            )
    result = pd.DataFrame(rows)
    # Corrects the p values because the analysis tests many relationships.
    result["q_value_bh"] = _benjamini_hochberg(result["p_value_hc3"])
    result["significant_5pct_raw"] = result["p_value_hc3"] < 0.05
    result["significant_5pct_fdr"] = result["q_value_bh"] < 0.05
    return result


def estimate_regime_changes(events: pd.DataFrame) -> pd.DataFrame:
    # Compares the relationships before 2020 with those from 2020 onward.
    frame = _prepare(events)
    frame["post_2020"] = (frame["release_date"] >= pd.Timestamp("2020-01-01")).astype(
        "float64"
    )
    rows: list[dict[str, object]] = []
    keys = ["series", "instrument", "asset_class", "reaction_unit"]
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        for window in WINDOWS:
            columns = ["standardized_surprise", "post_2020", window]
            sample = group[columns].dropna()
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "window": window,
                    **_regime_fit(sample, window),
                }
            )
    result = pd.DataFrame(rows)
    result["interaction_q_value_bh"] = _benjamini_hochberg(
        result["interaction_p_value_hc3"]
    )
    result["regime_change_significant_5pct_fdr"] = (
        result["interaction_q_value_bh"] < 0.05
    )
    return result


def estimate_persistence(events: pd.DataFrame) -> pd.DataFrame:
    # Checks whether the first market reaction continues after the release day.
    frame = _prepare(events)
    frame["drift_after_eod_to_1d"] = frame["reaction_1d"] - frame["reaction_eod"]
    frame["drift_after_eod_to_5d"] = frame["reaction_5d"] - frame["reaction_eod"]
    rows: list[dict[str, object]] = []
    keys = ["series", "instrument", "asset_class", "reaction_unit"]
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        for window in ("drift_after_eod_to_1d", "drift_after_eod_to_5d"):
            sample = group[["standardized_surprise", window]].dropna()
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "drift_window": window,
                    **_simple_fit(sample["standardized_surprise"], sample[window]),
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = _benjamini_hochberg(result["p_value_hc3"])
    result["significant_5pct_fdr"] = result["q_value_bh"] < 0.05
    return result


def estimate_calendar_matched_sensitivity(events: pd.DataFrame) -> pd.DataFrame:
    # Repeats the main tests only where both Bloomberg sources agree.
    required = {"actual_matches_calendar", "consensus_matches_calendar"}
    missing = required - set(events.columns)
    if missing:
        raise StatisticalAnalysisError(
            "Calendar-matched sensitivity requires Bloomberg comparison flags"
        )
    actual_match = _as_boolean(events["actual_matches_calendar"])
    consensus_match = _as_boolean(events["consensus_matches_calendar"])
    matched = events[actual_match & consensus_match].copy()
    result = estimate_surprise_reactions(matched)
    result.insert(0, "sensitivity_sample", "calendar_values_match")
    return result


def write_statistical_results(
    event_path: str | Path = DEFAULT_EVENT_DATASET,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, pd.DataFrame]:
    events = pd.read_csv(event_path)
    outputs = {
        "surprise_reactions": estimate_surprise_reactions(events),
        "calendar_matched_sensitivity": estimate_calendar_matched_sensitivity(events),
        "regime_changes": estimate_regime_changes(events),
        "persistence": estimate_persistence(events),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    return outputs


def _prepare(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "release_date",
        "series",
        "instrument",
        "asset_class",
        "reaction_unit",
        "standardized_surprise",
        *WINDOWS,
    }
    missing = required - set(events.columns)
    if missing:
        raise StatisticalAnalysisError(
            f"Event dataset missing columns: {', '.join(sorted(missing))}"
        )
    frame = events.copy()
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    numeric = ["standardized_surprise", *WINDOWS]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["release_date"].isna().any():
        raise StatisticalAnalysisError("Event dataset contains invalid release dates")
    if frame.duplicated(["release_date", "series", "instrument"]).any():
        raise StatisticalAnalysisError(
            "Event dataset contains duplicate event and asset rows"
        )
    return frame


def _as_boolean(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.casefold()
    if not normalized.isin(["true", "false"]).all():
        raise StatisticalAnalysisError("Bloomberg comparison flags must be boolean")
    return normalized.eq("true")


def _simple_fit(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    # Uses HC3 for the main uncertainty estimate and HAC as a second check.
    if len(x) < 20 or x.nunique() < 2:
        raise StatisticalAnalysisError(
            "A regression requires 20 rows and varying surprise"
        )
    design = sm.add_constant(x.astype("float64"), has_constant="add")
    response = y.astype("float64")
    ordinary = sm.OLS(response, design).fit()
    hc3 = ordinary.get_robustcov_results(cov_type="HC3")
    hac = ordinary.get_robustcov_results(cov_type="HAC", maxlags=3)
    return {
        "n": int(ordinary.nobs),
        "intercept": float(ordinary.params.iloc[0]),
        "beta": float(ordinary.params.iloc[1]),
        "std_error_hc3": float(hc3.bse[1]),
        "t_stat_hc3": float(hc3.tvalues[1]),
        "p_value_hc3": float(hc3.pvalues[1]),
        "ci_95_low_hc3": float(hc3.conf_int(alpha=0.05)[1, 0]),
        "ci_95_high_hc3": float(hc3.conf_int(alpha=0.05)[1, 1]),
        "p_value_hac3": float(hac.pvalues[1]),
        "r_squared": float(ordinary.rsquared),
        "correlation": float(x.corr(y)),
        "mean_reaction": float(y.mean()),
    }


def _regime_fit(sample: pd.DataFrame, window: str) -> dict[str, float | int]:
    # Uses an interaction term to measure the change in the slope after 2020.
    if len(sample) < 40 or sample["post_2020"].nunique() != 2:
        raise StatisticalAnalysisError(
            "Regime regression needs both periods and 40 rows"
        )
    x = sample["standardized_surprise"].astype("float64")
    post = sample["post_2020"].astype("float64")
    design = pd.DataFrame(
        {"const": 1.0, "surprise": x, "post_2020": post, "interaction": x * post}
    )
    ordinary = sm.OLS(sample[window].astype("float64"), design).fit()
    robust = ordinary.get_robustcov_results(cov_type="HC3")
    covariance = np.asarray(robust.cov_params())
    beta_pre = float(ordinary.params["surprise"])
    interaction = float(ordinary.params["interaction"])
    beta_post = beta_pre + interaction
    post_variance = covariance[1, 1] + covariance[3, 3] + 2 * covariance[1, 3]
    post_se = math.sqrt(max(0.0, float(post_variance)))
    post_z = beta_post / post_se if post_se > 0 else math.nan
    return {
        "n": int(ordinary.nobs),
        "n_pre_2020": int((post == 0).sum()),
        "n_post_2020": int((post == 1).sum()),
        "beta_pre_2020": beta_pre,
        "beta_post_2020": beta_post,
        "beta_post_2020_std_error_hc3": post_se,
        "beta_post_2020_p_value_hc3": float(2 * norm.sf(abs(post_z))),
        "interaction": interaction,
        "interaction_std_error_hc3": float(robust.bse[3]),
        "interaction_p_value_hc3": float(robust.pvalues[3]),
        "r_squared": float(ordinary.rsquared),
    }


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    # Limits the share of false positives across the full group of tests.
    values = p_values.astype("float64").to_numpy()
    count = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(count, dtype="float64")
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=p_values.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    outputs = write_statistical_results(args.events, args.output_dir)
    primary = outputs["surprise_reactions"]
    print(
        f"Wrote {len(primary)} primary regressions and three supporting analyses "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
