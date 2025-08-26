# ex2a_side_by_side_plots_and_regressions.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -------- Config --------
INPUT_CSV = "ex2a_results.csv"  # change if needed
FIG_PATH  = "ex2a_collective_vs_distributive.png"
REG_CSV   = "regression_2a.csv"
# ------------------------

def pick_col(df, prefer, fallback=None):
    if prefer in df.columns:
        return prefer
    if fallback and fallback in df.columns:
        print(f"[warn] Using fallback '{fallback}' for '{prefer}'")
        return fallback
    raise KeyError(f"Missing column '{prefer}'" + (f" (also tried '{fallback}')" if fallback else ""))

def scatter_with_refline(ax, x, y, xlabel, ylabel, title):
    sub = pd.DataFrame({"x": x, "y": y}).dropna().astype(float)
    if sub.empty:
        ax.set_title(f"{title} (no data)")
        return
    r = sub["x"].corr(sub["y"])

    ax.scatter(sub["x"], sub["y"], alpha=0.6, edgecolor="none")
    lo, hi = float(min(sub["x"].min(), sub["y"].min())), float(max(sub["x"].max(), sub["y"].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    # Fit line
    try:
        a, b = np.polyfit(sub["x"], sub["y"], 1)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, a*xs + b, linewidth=1)
    except Exception:
        pass

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n n={len(sub)} | r={r:.3f}")

def run_ols(y, X, label):
    sub = pd.concat([y, X], axis=1).dropna()
    if sub.empty:
        raise ValueError(f"No data for regression: {label}")
    yv = sub.iloc[:, 0].astype(float)
    Xv = sm.add_constant(sub.iloc[:, 1].astype(float), has_constant="add")
    model = sm.OLS(yv, Xv).fit()
    out = pd.DataFrame({
        "model": label,
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "n": len(sub),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj
    })
    return model, out

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}. Update INPUT_CSV.")

    df = pd.read_csv(INPUT_CSV)

    # Columns
    bart_collective_col      = pick_col(df, "bart_collective")
    bart_distributive_col    = pick_col(df, "bart_distributive")
    roberta_collective_col   = pick_col(df, "roberta_collective", fallback="roverta_collective")
    roberta_distributive_col = pick_col(df, "roberta_distributive", fallback="roverta_distributive")

    # ---- Combined Figure with 2 subplots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    scatter_with_refline(
        axes[0],
        df[bart_collective_col], df[bart_distributive_col],
        xlabel="BART (collective)",
        ylabel="BART (distributive)",
        title="BART: collective vs distributive"
    )

    scatter_with_refline(
        axes[1],
        df[roberta_collective_col], df[roberta_distributive_col],
        xlabel="RoBERTa (collective)",
        ylabel="RoBERTa (distributive)",
        title="RoBERTa: collective vs distributive"
    )

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()
    print(f"[ok] Saved side-by-side plots to {FIG_PATH}")

    # ---- Regressions ----
    _, reg_bart = run_ols(df[bart_collective_col], df[bart_distributive_col], "bart")
    _, reg_rob  = run_ols(df[roberta_collective_col], df[roberta_distributive_col], "roberta")

    reg_all = pd.concat([reg_bart, reg_rob], ignore_index=True)
    reg_all.to_csv(REG_CSV, index=False)
    print(f"[ok] Regression results saved to {REG_CSV}")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your CSV
csv_path = Path("ex2a_results.csv")

# Load CSV
df = pd.read_csv(csv_path)

# Make column names easier to match
cols = {c.lower(): c for c in df.columns}

# Identify the right columns
item_col = cols.get("item")
rob_col = next((v for k, v in cols.items() if "roberta_collective" in k), None)

if item_col is None or rob_col is None:
    raise ValueError(f"Could not find 'Item' or 'roberta_collective' column. Found: {list(df.columns)}")

# Rank by roberta_collective (ascending)
df_ranked = df.sort_values(by=rob_col, ascending=True).reset_index(drop=True)
df_ranked["Rank"] = range(1, len(df_ranked) + 1)

# Keep only the relevant columns
df_ranked = df_ranked[["Rank", item_col, rob_col]]

# Save ranked results to a new CSV
df_ranked.to_csv("ex2a_roberta_collective_ranked.csv", index=False)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df_ranked["Rank"], df_ranked[rob_col], marker="o")
plt.xticks(ticks=df_ranked["Rank"], labels=df_ranked[item_col].astype(str), rotation=90)
plt.xlabel("Item Number (from 'Item' column, ranked by RoBERTa collective)")
plt.ylabel(rob_col)
plt.title("RoBERTa Collective — Sentences Ranked by Score")
plt.tight_layout()
plt.savefig("ex2a_roberta_collective_ranked.png", dpi=200)
plt.show()