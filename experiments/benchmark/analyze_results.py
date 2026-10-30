# -*- coding: utf-8 -*-
"""Statistical analysis of the benchmark results: mean +/- std summary tables,
Holm-corrected Wilcoxon comparisons, and bootstrap confidence intervals.

Reads the formal comparison results and writes
  - summary_statistics.csv: per-model mean and std of each test metric
  - model_comparisons.csv: paired `full` vs each other model with Holm-adjusted p values
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS = Path(__file__).resolve().parents[2] / "results" / "benchmark"


def load_formal_results():
    """Load the formal comparison results (6 models x 10 seeds)."""
    models = ["full", "no_info", "hgt", "simplehgn", "distmult", "mlp"]
    seeds = list(range(101, 111))
    records = []
    for model in models:
        for seed in seeds:
            path = RESULTS / f"{model}_seed{seed}.json"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    rec = json.load(f)
                    rec["model"] = model
                    records.append(rec)
    return pd.DataFrame(records)


def extract_metrics(df):
    """Extract the test metrics of interest from the result records."""
    rows = []
    for _, rec in df.iterrows():
        row = {
            "model": rec["model"],
            "seed": rec["seed"],
            "test_macro_NDCG@50": rec.get("test", {}).get("macro_NDCG@50"),
            "test_pair_MRR": rec.get("test", {}).get("pair_MRR"),
            "test_macro_AUROC": rec.get("test_binary", {}).get("macro_AUROC"),
            "test_macro_AUPRC": rec.get("test_binary", {}).get("macro_AUPRC"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def holm_correction(p_values, alpha=0.05):
    """Holm-Bonferroni step-down correction.

    Returns the adjusted p values in the original order.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = []
    for i, p in enumerate(sorted_p):
        adj_p = p * (n - i)
        adjusted.append(min(adj_p, 1.0))
    # restore the original order
    result = [0.0] * n
    for i, idx in enumerate(sorted_idx):
        result[idx] = adjusted[i]
    return np.array(result)


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Percentile bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def compare_models(df_metrics):
    """Paired comparison of `full` against each other model (Wilcoxon
    signed-rank test, one-sided greater)."""
    full = df_metrics[df_metrics["model"] == "full"]
    others = df_metrics[df_metrics["model"] != "full"]
    comparisons = []
    for model in others["model"].unique():
        other = df_metrics[df_metrics["model"] == model]
        # align by seed
        merged = full.merge(other, on="seed", suffixes=("_full", f"_{model}"))
        for metric in ["test_macro_NDCG@50", "test_pair_MRR", "test_macro_AUROC"]:
            full_vals = merged[f"{metric}_full"].dropna()
            other_vals = merged[f"{metric}_{model}"].dropna()
            if len(full_vals) > 0 and len(other_vals) > 0:
                stat, p = stats.wilcoxon(full_vals, other_vals, alternative="greater")
                comparisons.append({
                    "model": model,
                    "metric": metric,
                    "full_mean": full_vals.mean(),
                    "full_std": np.std(full_vals),
                    "other_mean": other_vals.mean(),
                    "other_std": np.std(other_vals),
                    "p_value": p,
                })
    return pd.DataFrame(comparisons)


def std(values):
    """Population standard deviation (ddof=0)."""
    return float(np.std(np.asarray(values)))


def generate_summary_table(df_metrics):
    """Aggregate summary table (mean +/- std per model)."""
    summary = df_metrics.groupby("model").agg({
        "test_macro_NDCG@50": ["mean", std],
        "test_pair_MRR": ["mean", std],
        "test_macro_AUROC": ["mean", std],
        "test_macro_AUPRC": ["mean", std],
    })
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    # formatted mean +/- std column for quick reading
    for col in summary.columns:
        if "mean" in col:
            mean_col = col
            std_col = col.replace("mean", "std")
            summary[f"{col}_formatted"] = summary.apply(
                lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}", axis=1
            )
    return summary


def main():
    """Run the full statistical analysis."""
    print("Loading formal experiment results...", flush=True)
    df = load_formal_results()
    if len(df) == 0:
        print("No results found. Run the formal phase first.", flush=True)
        return

    print(f"Loaded {len(df)} results", flush=True)
    df_metrics = extract_metrics(df)

    print("\nGenerating summary table...", flush=True)
    summary = generate_summary_table(df_metrics)
    print(summary, flush=True)

    print("\nComparing full model vs others (Wilcoxon test)...", flush=True)
    comparisons = compare_models(df_metrics)
    if len(comparisons) > 0:
        # Holm correction within each metric family
        for metric in comparisons["metric"].unique():
            mask = comparisons["metric"] == metric
            p_vals = comparisons.loc[mask, "p_value"].tolist()
            adj_p = holm_correction(p_vals)
            comparisons.loc[mask, "p_holm"] = adj_p
        print(comparisons, flush=True)

    # save
    summary_path = RESULTS / "summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}", flush=True)

    comp_path = RESULTS / "model_comparisons.csv"
    comparisons.to_csv(comp_path, index=False)
    print(f"Comparisons saved to {comp_path}", flush=True)

    return summary, comparisons


if __name__ == "__main__":
    main()
