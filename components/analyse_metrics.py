#!/usr/bin/env python3
"""
analyze_judge_scores.py

Parses a CSV file where spreadsheet column I (the 9th column) contains a
text blob like:

    Judge LLM Coverage Score: 0.7
    Judge LLM Coverage Score Explanation: ...
    Judge LLM Uniqueness Score: 0.8
    Judge LLM Uniqueness Score Explanation: ...
    Judge LLM Semantic Validity Score:0.9
    Judge LLM Semantic Validity Score Explanation: ...
    Judge LLM Consistency Score: 1.0
    Judge LLM Consistency Score Explanation: ...

For each row, it extracts the four numeric scores (Coverage, Uniqueness,
Semantic Validity, Consistency) and computes descriptive statistics
(mean, median, quantiles, std, min/max, IQR, skew, kurtosis) for each
category across all rows.

Usage:
    python analyze_judge_scores.py input.csv
    python analyze_judge_scores.py input.csv --plot

The plot is always generated and saved. By default it is saved only
(not shown); pass --plot to also display it on screen.

Outputs (always written to <script_dir>/../stats/<input_filename_without_extension>/):
    - extracted_scores.csv               one row per input row, one column per category
    - summary_stats.csv                  one row per category, one column per statistic
    - boxplot.png                        boxplot of the four categories (means only, no outliers)
    - distribution_<category>.png        one histogram per category (Coverage, Uniqueness,
                                          Semantic Validity, Consistency)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The Judge LLM text blob always lives in spreadsheet column I (9th column,
# 0-indexed position 8) — regardless of what that column is actually named
# in the CSV header (e.g. "LLM judgments evaluation").
COLUMN_INDEX = 8

# Outputs always go to <script_dir>/../stats, regardless of current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTDIR = SCRIPT_DIR.parent / "stats"

# Regex patterns to pull each score out of the text blob.
# \s*:\s* tolerates "Score: 0.7", "Score:0.7", "Score : 0.7", etc.
CATEGORIES = {
    "Coverage": r"Judge\s+LLM\s+Coverage\s+Score\s*:\s*([-+]?\d*\.?\d+)",
    "Uniqueness": r"Judge\s+LLM\s+Uniqueness\s+Score\s*:\s*([-+]?\d*\.?\d+)",
    "Semantic Validity": r"Judge\s+LLM\s+Semantic\s+Validity\s+Score\s*:\s*([-+]?\d*\.?\d+)",
    "Consistency": r"Judge\s+LLM\s+Consistency\s+Score\s*:\s*([-+]?\d*\.?\d+)",
}


def extract_scores(text, patterns=CATEGORIES):
    """Extract each category's numeric score from one cell of text.

    Returns a dict {category_name: float or np.nan}.
    """
    result = {}
    if not isinstance(text, str):
        for cat in patterns:
            result[cat] = np.nan
        return result
    for cat, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        result[cat] = float(match.group(1)) if match else np.nan
    return result


def compute_stats(series: pd.Series) -> dict:
    """Compute descriptive statistics for one category's score column."""
    clean = series.dropna()
    n_total = len(series)
    n_valid = len(clean)
    n_missing = n_total - n_valid

    if n_valid == 0:
        return {"count": 0, "missing": n_missing}

    stats = {
        "count": n_valid,
        "missing": n_missing,
        "mean": clean.mean(),
        "std": clean.std(ddof=1) if n_valid > 1 else 0.0,
        "min": clean.min(),
        "q1 (25%)": clean.quantile(0.25),
        "median (50%)": clean.median(),
        "q3 (75%)": clean.quantile(0.75),
        "max": clean.max(),
        "iqr": clean.quantile(0.75) - clean.quantile(0.25),
        "range": clean.max() - clean.min(),
        "skew": clean.skew() if n_valid > 2 else np.nan,
        "kurtosis": clean.kurt() if n_valid > 3 else np.nan,
    }
    return stats


def maybe_plot(scores_df: pd.DataFrame, outdir: Path, show: bool = False) -> None:
    """Save a boxplot figure and one score-distribution histogram per category.

    Files are always saved:
        - boxplot.png                       boxplot only, all categories
        - distribution_<category>.png       one histogram per category

    If show is True, each figure is additionally displayed on screen (plt.show()).
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot generation (pip install matplotlib to enable).")
        return

    categories = list(CATEGORIES.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Boxplot (its own figure, no outlier markers, mean shown as orange line) ---
    data = [scores_df[cat].dropna().values for cat in categories]
    fig_box, ax_box = plt.subplots(figsize=(7, 5))
    bp = ax_box.boxplot(
        data,
        tick_labels=categories,
        showmeans=True,
        meanline=True,
        showfliers=False,
        medianprops={"linewidth": 0},  # hide the median line
        meanprops={"color": "orange", "linewidth": 2, "linestyle": "-"},
    )
    ax_box.legend(bp["means"][:1], ["Mean"], loc="lower right")
    ax_box.set_title("Judge LLM Scores by Category")
    ax_box.set_ylabel("Score")
    ax_box.tick_params(axis="x", rotation=20)
    fig_box.tight_layout()

    box_path = outdir / "boxplot.png"
    fig_box.savefig(box_path, dpi=150)
    print(f"Boxplot saved to:                  {box_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_box)

    # --- One histogram per category, black-outlined bars ---
    for i, cat in enumerate(categories):
        fig_hist, ax_hist = plt.subplots(figsize=(7, 5))
        ax_hist.hist(
            scores_df[cat].dropna(),
            bins=10,
            color=colors[i % len(colors)],
            edgecolor="black",
        )
        ax_hist.set_title(f"{cat} Score Distribution")
        ax_hist.set_xlabel("Score")
        ax_hist.set_ylabel("Frequency")
        fig_hist.tight_layout()

        safe_name = cat.lower().replace(" ", "_")
        hist_path = outdir / f"distribution_{safe_name}.png"
        fig_hist.savefig(hist_path, dpi=150)
        print(f"{cat} distribution saved to: {hist_path}")

        if show:
            plt.show()
        else:
            plt.close(fig_hist)


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics on Judge LLM scores embedded in a CSV column."
    )
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument(
        "--encoding", default="utf-8", help="CSV file encoding (default: utf-8)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also display the plot on screen (it is always saved to ../stats/boxplot.png)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        sys.exit(f"Error: file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding=args.encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    if len(df.columns) <= COLUMN_INDEX:
        sys.exit(
            f"Error: CSV only has {len(df.columns)} column(s); "
            f"expected at least {COLUMN_INDEX + 1} (spreadsheet column I)."
        )
    column_name = df.columns[COLUMN_INDEX]

    # Extract the four scores from every row of the target column
    extracted = df[column_name].apply(extract_scores)
    scores_df = pd.DataFrame(list(extracted))

    outdir = DEFAULT_OUTDIR / csv_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    # Per-row extracted scores (useful for spot-checking / further analysis)
    scores_out_path = outdir / "extracted_scores.csv"
    scores_df.to_csv(scores_out_path, index=False)

    # Summary stats per category
    all_stats = {cat: compute_stats(scores_df[cat]) for cat in CATEGORIES}
    stats_df = pd.DataFrame(all_stats).T  # categories as rows, stats as columns
    stats_out_path = outdir / "summary_stats.csv"
    stats_df.to_csv(stats_out_path)

    # Console output
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== Judge LLM Score Summary Statistics ===\n")
    print(stats_df.to_string())

    for cat in CATEGORIES:
        n_missing = scores_df[cat].isna().sum()
        if n_missing:
            print(f"\nNote: {n_missing} row(s) had no parseable '{cat}' score.")

    print(f"\nPer-row extracted scores saved to: {scores_out_path}")
    print(f"Summary statistics saved to:       {stats_out_path}")

    maybe_plot(scores_df, outdir, show=args.plot)


if __name__ == "__main__":
    main()