#!/usr/bin/env python3
"""
analyze_judge_scores.py

Parses a CSV file where one column contains a text blob like:

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
    python analyze_judge_scores.py input.csv --column I --outdir stats_out
    python analyze_judge_scores.py input.csv --plot

Outputs (written to --outdir, default "judge_score_stats"):
    - extracted_scores.csv   one row per input row, one column per category
    - summary_stats.csv      one row per category, one column per statistic
    - boxplot.png            (only if --plot is passed and matplotlib is available)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def maybe_plot(scores_df: pd.DataFrame, outdir: Path) -> None:
    """Save a boxplot + histogram figure of the four categories, if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping --plot (pip install matplotlib to enable).")
        return

    categories = list(CATEGORIES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot across categories
    data = [scores_df[cat].dropna().values for cat in categories]
    axes[0].boxplot(data, tick_labels=categories, showmeans=True)
    axes[0].set_title("Judge LLM Scores by Category")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=20)

    # Overlaid histograms
    for cat in categories:
        axes[1].hist(scores_df[cat].dropna(), bins=10, alpha=0.5, label=cat)
    axes[1].set_title("Score Distributions")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    plot_path = outdir / "boxplot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to:                     {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics on Judge LLM scores embedded in a CSV column."
    )
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument(
        "--column",
        default="I",
        help="Name of the column containing the judge text blob (default: 'I')",
    )
    parser.add_argument(
        "--outdir",
        default="judge_score_stats",
        help="Directory to write outputs to (default: judge_score_stats)",
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="CSV file encoding (default: utf-8)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also generate a boxplot + histogram PNG (requires matplotlib)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        sys.exit(f"Error: file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding=args.encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    if args.column not in df.columns:
        sys.exit(
            f"Error: column '{args.column}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Extract the four scores from every row of the target column
    extracted = df[args.column].apply(extract_scores)
    scores_df = pd.DataFrame(list(extracted))

    outdir = Path(args.outdir)
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

    if args.plot:
        maybe_plot(scores_df, outdir)


if __name__ == "__main__":
    main()