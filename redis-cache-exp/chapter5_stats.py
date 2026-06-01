import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf


warnings.filterwarnings("ignore")

RAW_FILE = "redis_cache_results.csv"
ANALYSIS_DIR = "analysis_output"
OUTPUT_DIR = "chapter5_output"


def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def theory_uniform(N, C):
    C = int(min(max(C, 1), N))
    return C / N


def theory_zipf(N, C, alpha):
    C = int(min(max(C, 1), N))
    ranks = np.arange(1, N + 1)
    weights = ranks ** (-alpha)
    return float(weights[:C].sum() / weights.sum())


def load_data():
    """
    优先读取 analysis_output/results_with_actual_theory.csv。
    如果不存在，则读取 redis_cache_results.csv 并自动补充 actual theory。
    """
    actual_file = os.path.join(ANALYSIS_DIR, "results_with_actual_theory.csv")

    if os.path.exists(actual_file):
        df = pd.read_csv(actual_file)
    elif os.path.exists(RAW_FILE):
        df = pd.read_csv(RAW_FILE)
    else:
        raise FileNotFoundError(
            "找不到 results_with_actual_theory.csv 或 redis_cache_results.csv"
        )

    # 兼容旧版本列名
    if "theory_hit_rate_actual" not in df.columns:
        actual_theories = []
        for _, row in df.iterrows():
            N = int(row["N"])
            C_actual = int(row["dbsize_after"])
            dist = row["distribution"]

            if dist == "uniform":
                actual_theory = theory_uniform(N, C_actual)
            elif dist == "zipf":
                actual_theory = theory_zipf(N, C_actual, float(row["alpha"]))
            else:
                actual_theory = np.nan

            actual_theories.append(actual_theory)

        df["theory_hit_rate_actual"] = actual_theories

    if "error_actual" not in df.columns:
        df["error_actual"] = df["redis_hit_rate"] - df["theory_hit_rate_actual"]

    if "abs_error_actual" not in df.columns:
        df["abs_error_actual"] = df["error_actual"].abs()

    df["is_lfu"] = (df["policy"] == "allkeys-lfu").astype(int)
    df["actual_capacity_ratio"] = df["dbsize_after"] / df["N"]

    return df


def confidence_interval_mean(values, confidence=0.95):
    """
    返回均值、标准差、标准误、t 置信区间。
    """
    values = pd.Series(values).dropna()
    n = len(values)
    mean = values.mean()
    std = values.std(ddof=1)

    if n <= 1 or std == 0 or np.isnan(std):
        return mean, std, np.nan, np.nan, np.nan

    se = std / math.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    low = mean - t_crit * se
    high = mean + t_crit * se

    return mean, std, se, low, high


def make_descriptive_stats(df):
    """
    生成第五章第一部分：描述性统计与重复实验稳定性。
    """
    group_cols = ["distribution", "alpha", "capacity_ratio", "policy"]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        distribution, alpha, capacity_ratio, policy = keys

        mean, std, se, ci_low, ci_high = confidence_interval_mean(
            g["redis_hit_rate"]
        )

        error_mean, error_std, error_se, error_ci_low, error_ci_high = (
            confidence_interval_mean(g["abs_error_actual"])
        )

        rows.append({
            "distribution": distribution,
            "alpha": alpha,
            "capacity_ratio": capacity_ratio,
            "policy": policy,
            "n": len(g),
            "redis_hit_rate_mean": mean,
            "redis_hit_rate_std": std,
            "redis_hit_rate_se": se,
            "redis_hit_rate_ci95_low": ci_low,
            "redis_hit_rate_ci95_high": ci_high,
            "theory_design_mean": g["theory_hit_rate"].mean(),
            "theory_actual_mean": g["theory_hit_rate_actual"].mean(),
            "abs_error_actual_mean": error_mean,
            "abs_error_actual_std": error_std,
            "abs_error_actual_ci95_low": error_ci_low,
            "abs_error_actual_ci95_high": error_ci_high,
            "C_design": g["C"].iloc[0],
            "C_actual_mean": g["dbsize_after"].mean(),
            "C_actual_std": g["dbsize_after"].std(ddof=1),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["distribution", "alpha", "capacity_ratio", "policy"],
        na_position="first"
    )

    out.to_csv(
        os.path.join(OUTPUT_DIR, "descriptive_stats.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return out


def make_strategy_paired_test(descriptive):
    """
    以实验组均值为单位做 LRU vs LFU 配对 t 检验。
    每一对具有相同 alpha 和 capacity_ratio，只改变 policy。
    """
    zipf = descriptive[descriptive["distribution"] == "zipf"].copy()

    pivot = zipf.pivot_table(
        index=["alpha", "capacity_ratio"],
        columns="policy",
        values="redis_hit_rate_mean",
        aggfunc="mean"
    )

    pivot = pivot.dropna(subset=["allkeys-lru", "allkeys-lfu"])
    pivot["lfu_minus_lru"] = pivot["allkeys-lfu"] - pivot["allkeys-lru"]

    diff = pivot["lfu_minus_lru"]

    mean_diff, std_diff, se_diff, ci_low, ci_high = confidence_interval_mean(diff)

    t_two, p_two = stats.ttest_1samp(
        diff,
        0,
        alternative="two-sided"
    )

    t_greater, p_greater = stats.ttest_1samp(
        diff,
        0,
        alternative="greater"
    )

    # 小样本补充非参数检验
    try:
        wilcoxon_two = stats.wilcoxon(diff, alternative="two-sided").pvalue
        wilcoxon_greater = stats.wilcoxon(diff, alternative="greater").pvalue
    except Exception:
        wilcoxon_two = np.nan
        wilcoxon_greater = np.nan

    cohen_dz = mean_diff / std_diff if std_diff and not np.isnan(std_diff) else np.nan

    paired_summary = pd.DataFrame([{
        "n_pairs": len(diff),
        "mean_lfu_minus_lru": mean_diff,
        "std_lfu_minus_lru": std_diff,
        "se_lfu_minus_lru": se_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_two_sided_t": t_two,
        "paired_t_two_sided_p": p_two,
        "paired_t_greater_t": t_greater,
        "paired_t_greater_p": p_greater,
        "wilcoxon_two_sided_p": wilcoxon_two,
        "wilcoxon_greater_p": wilcoxon_greater,
        "cohen_dz": cohen_dz,
    }])

    pivot.to_csv(
        os.path.join(OUTPUT_DIR, "strategy_paired_groups.csv"),
        encoding="utf-8-sig"
    )

    paired_summary.to_csv(
        os.path.join(OUTPUT_DIR, "strategy_paired_test_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return pivot, paired_summary


def make_strategy_welch_tests(df):
    """
    对每个固定 alpha 和 capacity_ratio，分别比较 LRU 与 LFU 的重复实验结果。
    使用 Welch t 检验，避免假定两组方差相等。
    """
    z = df[df["distribution"] == "zipf"].copy()

    rows = []
    for (alpha, cap), g in z.groupby(["alpha", "capacity_ratio"]):
        policies = set(g["policy"])
        if "allkeys-lru" not in policies or "allkeys-lfu" not in policies:
            continue

        lru = g[g["policy"] == "allkeys-lru"]["redis_hit_rate"]
        lfu = g[g["policy"] == "allkeys-lfu"]["redis_hit_rate"]

        t_two = stats.ttest_ind(
            lfu,
            lru,
            equal_var=False,
            alternative="two-sided"
        )

        t_greater = stats.ttest_ind(
            lfu,
            lru,
            equal_var=False,
            alternative="greater"
        )

        diff = lfu.mean() - lru.mean()

        # Cohen's d，使用 pooled std，仅作效应量参考
        n1, n2 = len(lfu), len(lru)
        s1, s2 = lfu.std(ddof=1), lru.std(ddof=1)
        pooled_std = math.sqrt(
            ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
        ) if n1 + n2 > 2 else np.nan

        cohen_d = diff / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan

        rows.append({
            "alpha": alpha,
            "capacity_ratio": cap,
            "n_lru": len(lru),
            "n_lfu": len(lfu),
            "lru_mean": lru.mean(),
            "lru_std": lru.std(ddof=1),
            "lfu_mean": lfu.mean(),
            "lfu_std": lfu.std(ddof=1),
            "lfu_minus_lru": diff,
            "welch_t_two_sided": t_two.statistic,
            "welch_p_two_sided": t_two.pvalue,
            "welch_t_greater": t_greater.statistic,
            "welch_p_greater": t_greater.pvalue,
            "cohen_d": cohen_d,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["alpha", "capacity_ratio"])

    out.to_csv(
        os.path.join(OUTPUT_DIR, "strategy_welch_test_by_condition.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return out


def regression_result_to_table(model, model_name):
    """
    将 statsmodels 回归结果整理成 CSV 表。
    """
    table = pd.DataFrame({
        "model": model_name,
        "variable": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci95_low": model.conf_int()[0].values,
        "ci95_high": model.conf_int()[1].values,
    })

    table["r_squared"] = model.rsquared
    table["adj_r_squared"] = model.rsquared_adj
    table["nobs"] = model.nobs

    return table


def make_regressions(df):
    """
    回归一：解释 Redis 实测命中率。
    回归二：解释 Redis 实测值与实际容量理论值之间的绝对偏差。
    回归三、四：使用实际容量比例替代设计容量比例，作为稳健性分析。
    """
    z = df[df["distribution"] == "zipf"].copy()

    # 主模型 1：命中率
    model_hit = smf.ols(
        "redis_hit_rate ~ alpha + capacity_ratio + is_lfu",
        data=z
    ).fit(cov_type="HC3")

    # 主模型 2：理论偏差
    model_error = smf.ols(
        "abs_error_actual ~ alpha + capacity_ratio + is_lfu",
        data=z
    ).fit(cov_type="HC3")

    # 稳健性模型 1：用实际容量比例解释命中率
    model_hit_actual_cap = smf.ols(
        "redis_hit_rate ~ alpha + actual_capacity_ratio + is_lfu",
        data=z
    ).fit(cov_type="HC3")

    # 稳健性模型 2：用实际容量比例解释理论偏差
    model_error_actual_cap = smf.ols(
        "abs_error_actual ~ alpha + actual_capacity_ratio + is_lfu",
        data=z
    ).fit(cov_type="HC3")

    # 探索性模型：加入交互项，观察 LFU 优势是否随 alpha / 容量变化
    model_hit_interaction = smf.ols(
        "redis_hit_rate ~ alpha + capacity_ratio + is_lfu + alpha:is_lfu + capacity_ratio:is_lfu",
        data=z
    ).fit(cov_type="HC3")

    models = [
        ("hit_rate_design_capacity", model_hit),
        ("abs_error_design_capacity", model_error),
        ("hit_rate_actual_capacity", model_hit_actual_cap),
        ("abs_error_actual_capacity", model_error_actual_cap),
        ("hit_rate_interaction", model_hit_interaction),
    ]

    all_tables = []
    with open(os.path.join(OUTPUT_DIR, "regression_full_summary.txt"), "w", encoding="utf-8") as f:
        for name, model in models:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(name + "\n")
            f.write("=" * 80 + "\n")
            f.write(model.summary().as_text())
            f.write("\n\n")

            all_tables.append(regression_result_to_table(model, name))

    reg_table = pd.concat(all_tables, ignore_index=True)
    reg_table.to_csv(
        os.path.join(OUTPUT_DIR, "regression_coefficients.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return models, reg_table


def plot_strategy_difference(pivot):
    """
    画 LFU-LRU 策略差异图，供第五章使用。
    """
    plot_df = pivot.reset_index()

    labels = [
        f"α={row['alpha']:.1f}, C={row['capacity_ratio']*100:.0f}%"
        for _, row in plot_df.iterrows()
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5))
    plt.bar(x, plot_df["lfu_minus_lru"])
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("LFU - LRU hit rate")
    plt.title("Paired Difference between LFU and LRU")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "fig_strategy_paired_difference.png"),
        dpi=300
    )
    plt.close()


def plot_regression_fitted(df, models):
    """
    画回归拟合值与观测值，用于检查模型拟合效果。
    """
    z = df[df["distribution"] == "zipf"].copy()

    model_dict = dict(models)

    hit_model = model_dict["hit_rate_design_capacity"]
    err_model = model_dict["abs_error_design_capacity"]

    z["hit_fitted"] = hit_model.predict(z)
    z["error_fitted"] = err_model.predict(z)

    plt.figure(figsize=(6, 6))
    plt.scatter(z["redis_hit_rate"], z["hit_fitted"])
    min_v = min(z["redis_hit_rate"].min(), z["hit_fitted"].min())
    max_v = max(z["redis_hit_rate"].max(), z["hit_fitted"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Observed Redis hit rate")
    plt.ylabel("Fitted Redis hit rate")
    plt.title("Observed vs Fitted Hit Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_reg_hit_observed_fitted.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(z["abs_error_actual"], z["error_fitted"])
    min_v = min(z["abs_error_actual"].min(), z["error_fitted"].min())
    max_v = max(z["abs_error_actual"].max(), z["error_fitted"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Observed absolute error")
    plt.ylabel("Fitted absolute error")
    plt.title("Observed vs Fitted Theoretical Error")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_reg_error_observed_fitted.png"), dpi=300)
    plt.close()


def write_chapter5_report(
    descriptive,
    paired_groups,
    paired_summary,
    welch_by_condition,
    reg_table
):
    """
    输出一个便于你复制给我的汇总文本。
    """
    report_path = os.path.join(OUTPUT_DIR, "chapter5_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("第五章统计分析结果汇总\n")
        f.write("=" * 80 + "\n\n")

        f.write("一、描述性统计摘要\n")
        f.write("-" * 80 + "\n")
        f.write(descriptive[
            [
                "distribution",
                "alpha",
                "capacity_ratio",
                "policy",
                "n",
                "redis_hit_rate_mean",
                "redis_hit_rate_std",
                "theory_actual_mean",
                "abs_error_actual_mean",
                "C_actual_mean",
            ]
        ].to_string(index=False))
        f.write("\n\n")

        f.write("二、LRU 与 LFU 配对比较\n")
        f.write("-" * 80 + "\n")
        f.write(paired_groups.to_string())
        f.write("\n\n")
        f.write(paired_summary.to_string(index=False))
        f.write("\n\n")

        f.write("三、每个条件下的 Welch t 检验\n")
        f.write("-" * 80 + "\n")
        f.write(welch_by_condition.to_string(index=False))
        f.write("\n\n")

        f.write("四、回归系数表\n")
        f.write("-" * 80 + "\n")
        f.write(reg_table.to_string(index=False))
        f.write("\n\n")

        f.write("请把本文件和以下 CSV 一起发送给 ChatGPT：\n")
        f.write("1. descriptive_stats.csv\n")
        f.write("2. strategy_paired_test_summary.csv\n")
        f.write("3. strategy_paired_groups.csv\n")
        f.write("4. strategy_welch_test_by_condition.csv\n")
        f.write("5. regression_coefficients.csv\n")
        f.write("6. regression_full_summary.txt\n")

    return report_path


def main():
    ensure_dir()

    df = load_data()

    # 保存增强版原始数据
    df.to_csv(
        os.path.join(OUTPUT_DIR, "chapter5_input_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    descriptive = make_descriptive_stats(df)
    paired_groups, paired_summary = make_strategy_paired_test(descriptive)
    welch_by_condition = make_strategy_welch_tests(df)
    models, reg_table = make_regressions(df)

    plot_strategy_difference(paired_groups)
    plot_regression_fitted(df, models)

    report_path = write_chapter5_report(
        descriptive,
        paired_groups,
        paired_summary,
        welch_by_condition,
        reg_table
    )

    print("Done.")



if __name__ == "__main__":
    main()