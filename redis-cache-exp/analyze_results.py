import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "redis_cache_results.csv"
OUTPUT_DIR = "analysis_output"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def theory_uniform(N, C):
    return C / N


def theory_zipf(N, C, alpha):
    """
    按实际缓存对象数 C 重新计算 Zipf 理论命中率。
    """
    C = int(min(max(C, 1), N))
    ranks = np.arange(1, N + 1)
    weights = ranks ** (-alpha)
    return float(weights[:C].sum() / weights.sum())


def add_actual_capacity_theory(df):
    """
    原始 CSV 里的 theory_hit_rate 是按设计容量 C 计算的。
    这里额外增加一列 theory_hit_rate_actual，
    按 Redis 实际 dbsize_after 重新估计理论命中率。
    """
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

    df = df.copy()
    df["theory_hit_rate_actual"] = actual_theories
    df["error_actual"] = df["redis_hit_rate"] - df["theory_hit_rate_actual"]
    df["abs_error_actual"] = df["error_actual"].abs()
    return df


def make_summary_tables(df):
    """
    生成论文中可以直接使用的汇总表。
    """
    group_cols = ["distribution", "alpha", "capacity_ratio", "policy"]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            N=("N", "first"),
            C_design=("C", "first"),
            C_actual_mean=("dbsize_after", "mean"),
            C_actual_std=("dbsize_after", "std"),
            theory_design_mean=("theory_hit_rate", "mean"),
            theory_actual_mean=("theory_hit_rate_actual", "mean"),
            redis_hit_rate_mean=("redis_hit_rate", "mean"),
            redis_hit_rate_std=("redis_hit_rate", "std"),
            error_design_mean=("error", "mean"),
            abs_error_design_mean=("abs_error", "mean"),
            error_actual_mean=("error_actual", "mean"),
            abs_error_actual_mean=("abs_error_actual", "mean"),
            repeat_count=("redis_hit_rate", "count"),
        )
        .reset_index()
    )

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "summary_all.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 表1：访问分布影响
    distribution_effect = summary[
        (summary["capacity_ratio"] == 0.10)
        & (summary["policy"] == "allkeys-lru")
    ].copy()

    # 排序：uniform 放前面，zipf 按 alpha 排
    distribution_effect["sort_key"] = distribution_effect.apply(
        lambda r: -1 if r["distribution"] == "uniform" else r["alpha"],
        axis=1,
    )
    distribution_effect = distribution_effect.sort_values("sort_key")
    distribution_effect = distribution_effect.drop(columns=["sort_key"])

    distribution_effect.to_csv(
        os.path.join(OUTPUT_DIR, "distribution_effect.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 表2：缓存容量影响，固定 Zipf alpha=1.0，LRU
    capacity_effect = summary[
        (summary["distribution"] == "zipf")
        & (summary["alpha"] == 1.0)
        & (summary["policy"] == "allkeys-lru")
    ].copy()

    capacity_effect = capacity_effect.sort_values("capacity_ratio")

    capacity_effect.to_csv(
        os.path.join(OUTPUT_DIR, "capacity_effect.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 表3：LRU/LFU 策略对比
    policy_compare = summary[
        (summary["distribution"] == "zipf")
        & (
            (
                (summary["capacity_ratio"] == 0.10)
                & (summary["alpha"].isin([0.8, 1.0, 1.2]))
            )
            | (
                (summary["alpha"] == 1.0)
                & (summary["capacity_ratio"].isin([0.05, 0.10, 0.20]))
            )
        )
    ].copy()

    policy_compare = policy_compare.sort_values(
        ["alpha", "capacity_ratio", "policy"]
    )

    policy_compare.to_csv(
        os.path.join(OUTPUT_DIR, "policy_compare.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return summary, distribution_effect, capacity_effect, policy_compare


def plot_distribution_effect(distribution_effect):
    """
    图1：访问分布对命中率的影响。
    固定容量比例 10%，策略 allkeys-lru。
    """
    labels = []
    for _, row in distribution_effect.iterrows():
        if row["distribution"] == "uniform":
            labels.append("Uniform")
        else:
            labels.append(f"Zipf α={row['alpha']:.1f}")

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x,
        distribution_effect["redis_hit_rate_mean"],
        yerr=distribution_effect["redis_hit_rate_std"],
        marker="o",
        capsize=4,
        label="Redis measured",
    )
    plt.plot(
        x,
        distribution_effect["theory_design_mean"],
        marker="s",
        linestyle="--",
        label="Theory by design C",
    )
    plt.plot(
        x,
        distribution_effect["theory_actual_mean"],
        marker="^",
        linestyle=":",
        label="Theory by actual dbsize",
    )

    plt.xticks(x, labels, rotation=30)
    plt.xlabel("Request distribution")
    plt.ylabel("Hit rate")
    plt.title("Effect of Request Distribution on Cache Hit Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_distribution_effect.png"), dpi=300)
    plt.close()


def plot_capacity_effect(capacity_effect):
    """
    图2：缓存容量比例对命中率的影响。
    固定 Zipf alpha=1.0，策略 allkeys-lru。
    """
    x = capacity_effect["capacity_ratio"] * 100

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x,
        capacity_effect["redis_hit_rate_mean"],
        yerr=capacity_effect["redis_hit_rate_std"],
        marker="o",
        capsize=4,
        label="Redis measured",
    )
    plt.plot(
        x,
        capacity_effect["theory_design_mean"],
        marker="s",
        linestyle="--",
        label="Theory by design C",
    )
    plt.plot(
        x,
        capacity_effect["theory_actual_mean"],
        marker="^",
        linestyle=":",
        label="Theory by actual dbsize",
    )

    plt.xlabel("Cache capacity ratio (%)")
    plt.ylabel("Hit rate")
    plt.title("Effect of Cache Capacity on Hit Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_capacity_effect.png"), dpi=300)
    plt.close()


def plot_policy_alpha(summary):
    """
    图3：不同 Zipf alpha 下 LRU 与 LFU 对比。
    固定容量比例 10%。
    """
    data = summary[
        (summary["distribution"] == "zipf")
        & (summary["capacity_ratio"] == 0.10)
        & (summary["alpha"].isin([0.8, 1.0, 1.2]))
        & (summary["policy"].isin(["allkeys-lru", "allkeys-lfu"]))
    ].copy()

    pivot = data.pivot_table(
        index="alpha",
        columns="policy",
        values="redis_hit_rate_mean",
        aggfunc="mean",
    ).sort_index()

    plt.figure(figsize=(8, 5))

    if "allkeys-lru" in pivot.columns:
        plt.plot(
            pivot.index,
            pivot["allkeys-lru"],
            marker="o",
            label="LRU",
        )

    if "allkeys-lfu" in pivot.columns:
        plt.plot(
            pivot.index,
            pivot["allkeys-lfu"],
            marker="s",
            label="LFU",
        )

    plt.xlabel("Zipf alpha")
    plt.ylabel("Redis hit rate")
    plt.title("LRU vs LFU under Different Zipf Alpha")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_policy_alpha.png"), dpi=300)
    plt.close()


def plot_policy_capacity(summary):
    """
    图4：不同缓存容量下 LRU 与 LFU 对比。
    固定 Zipf alpha=1.0。
    """
    data = summary[
        (summary["distribution"] == "zipf")
        & (summary["alpha"] == 1.0)
        & (summary["capacity_ratio"].isin([0.05, 0.10, 0.20]))
        & (summary["policy"].isin(["allkeys-lru", "allkeys-lfu"]))
    ].copy()

    pivot = data.pivot_table(
        index="capacity_ratio",
        columns="policy",
        values="redis_hit_rate_mean",
        aggfunc="mean",
    ).sort_index()

    x = pivot.index * 100

    plt.figure(figsize=(8, 5))

    if "allkeys-lru" in pivot.columns:
        plt.plot(
            x,
            pivot["allkeys-lru"],
            marker="o",
            label="LRU",
        )

    if "allkeys-lfu" in pivot.columns:
        plt.plot(
            x,
            pivot["allkeys-lfu"],
            marker="s",
            label="LFU",
        )

    plt.xlabel("Cache capacity ratio (%)")
    plt.ylabel("Redis hit rate")
    plt.title("LRU vs LFU under Different Cache Capacity")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "fig_policy_capacity.png"), dpi=300)
    plt.close()


def main():
    ensure_output_dir()

    df = pd.read_csv(INPUT_FILE)

    print("Loaded data:")
    print(df.head())
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()

    df = add_actual_capacity_theory(df)

    # 保存带 actual theory 的原始数据
    df.to_csv(
        os.path.join(OUTPUT_DIR, "results_with_actual_theory.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary, distribution_effect, capacity_effect, policy_compare = make_summary_tables(df)

    plot_distribution_effect(distribution_effect)
    plot_capacity_effect(capacity_effect)
    plot_policy_alpha(summary)
    plot_policy_capacity(summary)

    print("Generated files in:", OUTPUT_DIR)
    print()
    print("Main summary:")
    print(summary[
        [
            "distribution",
            "alpha",
            "capacity_ratio",
            "policy",
            "C_design",
            "C_actual_mean",
            "theory_design_mean",
            "theory_actual_mean",
            "redis_hit_rate_mean",
            "redis_hit_rate_std",
            "abs_error_design_mean",
            "abs_error_actual_mean",
            "repeat_count",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()