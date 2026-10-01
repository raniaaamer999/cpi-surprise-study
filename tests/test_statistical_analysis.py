"""Checks the regression, persistence, regime, and sensitivity results."""

from datetime import date

import pandas as pd
import pytest

from macro_surprise.analysis.statistical_analysis import (
    StatisticalAnalysisError,
    estimate_calendar_matched_sensitivity,
    estimate_persistence,
    estimate_regime_changes,
    estimate_surprise_reactions,
)


def _events() -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    rows = []
    for index, stamp in enumerate(dates):
        surprise = float((index % 9) - 4) / 2.0
        post = stamp.date() >= date(2020, 1, 1)
        eod_beta = 2.0 if not post else 4.0
        eod = eod_beta * surprise + float(index % 3) / 20.0
        rows.append(
            {
                "release_date": stamp.date(),
                "series": "core_mom",
                "instrument": "us_2y_yield",
                "asset_class": "rates",
                "reaction_unit": "basis_points",
                "standardized_surprise": surprise,
                "reaction_eod": eod,
                "reaction_1d": eod + 0.5 * surprise,
                "reaction_5d": eod - 1.0 * surprise,
                "actual_matches_calendar": index % 5 != 0,
                "consensus_matches_calendar": index % 7 != 0,
            }
        )
    return pd.DataFrame(rows)


def test_primary_regressions_recover_positive_relationship() -> None:
    result = estimate_surprise_reactions(_events())
    eod = result[result["window"] == "reaction_eod"].iloc[0]

    assert len(result) == 3
    assert eod["n"] == 72
    assert eod["beta"] > 3.0
    assert eod["p_value_hc3"] < 0.001
    assert eod["q_value_bh"] < 0.001
    assert bool(eod["significant_5pct_fdr"])


def test_regime_interaction_detects_stronger_post_2020_slope() -> None:
    result = estimate_regime_changes(_events())
    eod = result[result["window"] == "reaction_eod"].iloc[0]

    assert eod["n_pre_2020"] == 24
    assert eod["n_post_2020"] == 48
    assert eod["beta_pre_2020"] == pytest.approx(2.0, abs=0.05)
    assert eod["beta_post_2020"] == pytest.approx(4.0, abs=0.05)
    assert eod["interaction_p_value_hc3"] < 0.001


def test_persistence_measures_post_event_drift() -> None:
    result = estimate_persistence(_events())
    five_day = result[result["drift_window"] == "drift_after_eod_to_5d"].iloc[0]

    assert five_day["beta"] == pytest.approx(-1.0)
    assert five_day["p_value_hc3"] < 0.001


def test_duplicate_event_asset_rows_are_rejected() -> None:
    events = pd.concat([_events(), _events().iloc[[0]]], ignore_index=True)

    with pytest.raises(StatisticalAnalysisError, match="duplicate"):
        estimate_surprise_reactions(events)


def test_calendar_sensitivity_uses_only_matching_rows() -> None:
    result = estimate_calendar_matched_sensitivity(_events())

    assert result["sensitivity_sample"].eq("calendar_values_match").all()
    assert result["n"].max() < 72
