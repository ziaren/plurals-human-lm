# ex1a_collective_plots.py
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "ex1_results.csv"

def main():
    df = pd.read_csv(INPUT)

    # Basic checks
    needed = {"Item", "collective_large", "collective_small"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT}: {missing}")

    # -----------------------------
    # Plot 1: Compare large vs small
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df["Item"], df["collective_large"], marker="o", linewidth=1, label="Collective (Large)")
    plt.plot(df["Item"], df["collective_small"], marker="s", linewidth=1, label="Collective (Small)")
    plt.xlabel("Item")
    plt.ylabel("Cosine similarity")
    plt.title("Collective similarity: Large vs Small (by Item)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_collective_large_vs_small.png", dpi=200)
    plt.show()

    # -------------------------------------------------------
    # Plot 2: Rank collective_large from low to high on x-axis
    # x-axis marks the Item number for the corresponding sentence
    # -------------------------------------------------------
    ranked = df[["Item", "collective_large"]].sort_values("collective_large").reset_index(drop=True)
    x_rank = range(1, len(ranked) + 1)  # 1..N
    y_vals = ranked["collective_large"].values
    item_labels = ranked["Item"].astype(str).tolist()  # Item numbers as labels in ranked order

    plt.figure(figsize=(12, 5))
    plt.plot(x_rank, y_vals, marker=".", linewidth=1)
    plt.xlabel("Rank (low → high) — x-axis labeled by corresponding Item number")
    plt.ylabel("Cosine similarity (collective_large)")
    plt.title("Collective (Large) ranked from low to high")

    # Set tick labels to Item numbers corresponding to each rank
    plt.xticks(ticks=list(x_rank), labels=item_labels, rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_collective_large_ranked_by_item_labels.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()

# ex1a regression comparisons
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

INPUT = "ex1_results.csv"
OUTPUT = "regression_1a.csv"

def paired_summary(a, b):
    a, b = np.asarray(a), np.asarray(b)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    diff = b - a
    n = diff.size
    return n, float(a.mean()), float(b.mean()), float(diff.mean()), float(diff.std(ddof=1)), diff

def paired_tests(a, b, label_a, label_b):
    n, mean_a, mean_b, mean_d, sd_d, diff = paired_summary(a, b)

    # Paired t-test
    t_res = stats.ttest_rel(b, a, nan_policy="omit")
    t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)

    # Wilcoxon
    try:
        w_stat, w_pval = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
    except ValueError:
        w_stat, w_pval = np.nan, np.nan

    # Effect size
    dz = mean_d / sd_d if sd_d and not np.isnan(sd_d) and sd_d != 0 else np.nan

    # 95% CI
    if n > 1 and sd_d:
        t_crit = stats.t.ppf(0.975, df=n-1)
        se = sd_d / np.sqrt(n)
        ci_low = mean_d - t_crit * se
        ci_high = mean_d + t_crit * se
    else:
        ci_low = ci_high = np.nan

    return {
        "Comparison": f"{label_b} - {label_a}",
        "n": n,
        "Mean_A": mean_a,
        "Mean_B": mean_b,
        "Mean_Diff": mean_d,
        "CI_Low": ci_low,
        "CI_High": ci_high,
        "t_stat": t_stat,
        "t_pval": t_pval,
        "Wilcoxon_stat": w_stat,
        "Wilcoxon_pval": w_pval,
        "Cohens_dz": dz
    }

def ols_with_item_fixed_effects(df, col_a, col_b, item_col="Item"):
    long = pd.concat([
        pd.DataFrame({item_col: df[item_col], "value": df[col_a], "condition": 0}),
        pd.DataFrame({item_col: df[item_col], "value": df[col_b], "condition": 1}),
    ], ignore_index=True).dropna(subset=["value"])

    model = smf.ols(formula=f"value ~ condition + C({item_col})", data=long).fit()
    coef = model.params.get("condition", np.nan)
    pval = model.pvalues.get("condition", np.nan)
    se = model.bse.get("condition", np.nan)
    return coef, se, pval

def main():
    df = pd.read_csv(INPUT)

    results = []

    # 1) collective_large vs distributive_large
    res1 = paired_tests(df["collective_large"], df["distributive_large"],
                        "collective_large", "distributive_large")
    coef, se, pval = ols_with_item_fixed_effects(df, "collective_large", "distributive_large")
    res1.update({"OLS_coef": coef, "OLS_se": se, "OLS_pval": pval})
    results.append(res1)

    # 2) collective_small vs distributive_small
    res2 = paired_tests(df["collective_small"], df["distributive_small"],
                        "collective_small", "distributive_small")
    coef, se, pval = ols_with_item_fixed_effects(df, "collective_small", "distributive_small")
    res2.update({"OLS_coef": coef, "OLS_se": se, "OLS_pval": pval})
    results.append(res2)

    # 3) collective_large vs collective_small
    res3 = paired_tests(df["collective_large"], df["collective_small"],
                        "collective_large", "collective_small")
    coef, se, pval = ols_with_item_fixed_effects(df, "collective_large", "collective_small")
    res3.update({"OLS_coef": coef, "OLS_se": se, "OLS_pval": pval})
    results.append(res3)

    # Save results to CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT, index=False)
    print(f"✅ Regression results saved to {OUTPUT}")
    print(out_df)

if __name__ == "__main__":
    main()