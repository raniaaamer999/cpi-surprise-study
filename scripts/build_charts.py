"""Builds the six charts that summarize the CPI study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/private/results"
CHARTS = ROOT / "output/charts"

CYAN = "#2bd5c4"
INK = "#172332"
LIGHT = "#e8eef3"

SERIES_LABELS = {
    "headline_mom": "Headline MoM",
    "headline_yoy": "Headline YoY",
    "core_mom": "Core MoM",
    "core_yoy": "Core YoY",
}
ASSET_LABELS = {
    "us_2y_yield": "2Y yield",
    "us_10y_yield": "10Y yield",
    "eurusd": "EUR/USD",
    "sp500": "S&P 500",
    "gold_gld_proxy": "Gold (GLD)",
}
WINDOW_LABELS = {
    "reaction_eod": "Release day",
    "reaction_1d": "+1 session",
    "reaction_5d": "+5 sessions",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": "#b8c4ce",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: Figure, name: str) -> Path:
    path = CHARTS / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def cpi_forest(cpi: pd.DataFrame, series: str, number: int) -> Path:
    # Shows the release day response for one of the four CPI measures.
    frame = cpi[(cpi.series == series) & (cpi.window == "reaction_eod")].copy()
    panels = (
        (["us_2y_yield", "us_10y_yield"], "Rates (basis points)", " bp"),
        (["eurusd", "sp500", "gold_gld_proxy"], "Other markets (%)", "%"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), width_ratios=[0.9, 1.25])
    for ax, (order, title, suffix) in zip(axes, panels, strict=True):
        panel = frame.set_index("instrument").loc[order].reset_index()
        y = np.arange(len(panel))[::-1]
        ax.axvline(0, color="#7c8c9a", lw=1)
        for index, row in panel.iterrows():
            color = CYAN if row.significant_5pct_fdr else "#9aa8b4"
            ax.errorbar(
                row.beta,
                y[index],
                xerr=[
                    [row.beta - row.ci_95_low_hc3],
                    [row.ci_95_high_hc3 - row.beta],
                ],
                fmt="o",
                color=color,
                capsize=3,
                lw=1.5,
            )
            ax.text(
                0.98,
                y[index],
                f"{row.beta:+.2f}{suffix}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
                color=INK,
            )
        ax.set_yticks(y, [ASSET_LABELS[item] for item in order])
        ax.set_xlabel("Effect per 1 SD surprise")
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.grid(axis="x", color=LIGHT, lw=0.8)
    fig.suptitle(
        f"{SERIES_LABELS[series]} CPI: release day response across markets",
        x=0.08,
        ha="left",
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=CYAN,
                lw=0,
                label="Significant after FDR",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#9aa8b4",
                lw=0,
                label="Not significant after FDR",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    fig.subplots_adjust(bottom=0.23, top=0.82, wspace=0.55)
    return save(fig, f"{number:02d}_cpi_release_day_{series}_forest.png")


def significance_heatmap(cpi: pd.DataFrame) -> Path:
    # Shows which results remain significant after the correction for many tests.
    frame = cpi.copy()
    frame["row"] = (
        frame.series.map(SERIES_LABELS) + ": " + frame.window.map(WINDOW_LABELS)
    )
    pivot = frame.pivot(index="row", columns="instrument", values="q_value_bh")
    row_order = [
        f"{SERIES_LABELS[s]}: {WINDOW_LABELS[w]}"
        for s in SERIES_LABELS
        for w in WINDOW_LABELS
    ]
    col_order = list(ASSET_LABELS)
    pivot = pivot.loc[row_order, col_order]
    score = -np.log10(pivot.clip(lower=1e-4))
    cmap = LinearSegmentedColormap.from_list("sig", ["#e9eef2", "#8bded6", "#137c79"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    image = ax.imshow(score, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(
        range(len(col_order)),
        [ASSET_LABELS[x] for x in col_order],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(range(len(row_order)), row_order, fontsize=7)
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            q = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                "*" if q < 0.05 else "",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("Negative log10 of the FDR q value")
    ax.set_title(
        "Where CPI relationships survive multiple testing correction",
        loc="left",
        fontweight="bold",
    )
    return save(fig, "05_cpi_significance_heatmap.png")


def persistence_chart(persistence: pd.DataFrame) -> Path:
    # Summarizes whether the release day reaction continues in later sessions.
    order = ["drift_after_eod_to_1d", "drift_after_eod_to_5d"]
    summary = persistence.groupby("drift_window").agg(
        smallest_q=("q_value_bh", "min"),
        significant=("significant_5pct_fdr", "sum"),
        tests=("significant_5pct_fdr", "size"),
    )
    summary = summary.loc[order]
    labels = ["After release day to +1", "After release day to +5"]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    bars = ax.bar(labels, summary.smallest_q, color=[CYAN, "#7c91a5"], width=0.55)
    ax.axhline(0.05, color="#b34f5b", ls="--", lw=1, label="FDR threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Smallest FDR q value")
    ax.set_title(
        "Initial CPI reactions do not persist reliably", loc="left", fontweight="bold"
    )
    for bar, sig, tests in zip(
        bars, summary.significant, summary.tests, strict=True
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(sig)} of {int(tests)} significant",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.grid(axis="y", color=LIGHT)
    return save(fig, "06_cpi_persistence.png")


def main() -> None:
    style()
    CHARTS.mkdir(parents=True, exist_ok=True)
    cpi = pd.read_csv(RESULTS / "surprise_reactions.csv")
    persistence = pd.read_csv(RESULTS / "persistence.csv")
    charts = (
        cpi_forest(cpi, "headline_mom", 1),
        cpi_forest(cpi, "headline_yoy", 2),
        cpi_forest(cpi, "core_mom", 3),
        cpi_forest(cpi, "core_yoy", 4),
        significance_heatmap(cpi),
        persistence_chart(persistence),
    )
    print(f"Wrote {len(charts)} charts to {CHARTS}")


if __name__ == "__main__":
    main()
