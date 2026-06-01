# ProbabilityTheory

这个仓库用于概率论课程与统计小论文相关的实验整理。目前主要内容是 `redis-cache-exp`：用 Redis 的 LRU/LFU 淘汰策略模拟缓存系统，在不同访问分布和缓存容量下比较实测命中率、理论命中率和策略差异，并生成论文中可用的表格、图像与统计分析结果。

## 仓库结构

```text
.
├── README.md
└── redis-cache-exp
    ├── experiment.py                 # 运行 Redis 缓存实验，生成原始实验数据
    ├── analyze_results.py            # 汇总原始数据，生成论文基础表格和图像
    ├── chapter5_stats.py             # 生成第五章统计检验、回归分析和报告
    ├── redis_cache_results.csv       # 原始实验结果
    ├── analysis_output/              # 基础分析输出
    └── chapter5_output/              # 第五章统计分析输出
```

## 环境要求

需要本机可以运行 Python 和 Redis。

Python 依赖：

```bash
pip install redis numpy pandas matplotlib scipy statsmodels
```

Redis 需要在本机 `localhost:6379` 可连接。脚本会修改 Redis 的 `maxmemory` 和 `maxmemory-policy`，并会执行 `flushall` 清空当前 Redis 数据库，所以不要在存有重要数据的 Redis 实例上运行。

可以用下面的命令检查 Redis 是否可用：

```bash
redis-cli ping
```

返回 `PONG` 即表示连接正常。

## 运行方法

所有脚本都使用相对路径读写文件，建议先进入实验目录：

```bash
cd redis-cache-exp
```

### 1. 运行原始缓存实验

```bash
python experiment.py
```

该脚本会：

- 估算 Redis 中单个 key-value 的平均内存占用；
- 在不同访问分布下运行缓存实验，包括 Uniform 分布和 Zipf 分布；
- 在不同缓存容量比例下运行实验；
- 比较 Redis `allkeys-lru` 和 `allkeys-lfu` 策略；
- 每组实验重复 3 次；
- 输出原始结果到 `redis_cache_results.csv`。

主要实验设置在 `experiment.py` 中：

- 对象总数：`N = 10000`
- 预热请求数：`warmup_T = 50000`
- 正式统计请求数：`measure_T = 100000`
- 容量比例：如 `0.05`、`0.10`、`0.20`、`0.30`、`0.40`
- Zipf 参数：如 `0.6`、`0.8`、`1.0`、`1.2`

如果只想复现实验后的分析，仓库中已经保留了 `redis_cache_results.csv`，可以直接从下一步开始。

### 2. 生成基础分析结果

```bash
python analyze_results.py
```

该脚本读取 `redis_cache_results.csv`，并输出到 `analysis_output/`：

- `results_with_actual_theory.csv`：在原始数据基础上，加入按 Redis 实际缓存对象数重新计算的理论命中率；
- `summary_all.csv`：所有实验组的汇总统计；
- `distribution_effect.csv`：访问分布对命中率影响的汇总表；
- `capacity_effect.csv`：缓存容量比例对命中率影响的汇总表；
- `policy_compare.csv`：LRU 与 LFU 策略对比表；
- `fig_distribution_effect.png`：访问分布影响图；
- `fig_capacity_effect.png`：容量比例影响图；
- `fig_policy_alpha.png`：不同 Zipf 参数下 LRU/LFU 对比图；
- `fig_policy_capacity.png`：不同容量下 LRU/LFU 对比图。

### 3. 生成第五章统计分析

```bash
python chapter5_stats.py
```

该脚本优先读取 `analysis_output/results_with_actual_theory.csv`；如果不存在，则读取 `redis_cache_results.csv` 并自动补充实际容量理论值。输出位于 `chapter5_output/`：

- `chapter5_input_data.csv`：第五章分析使用的增强版输入数据；
- `descriptive_stats.csv`：描述性统计、标准误和 95% 置信区间；
- `strategy_paired_groups.csv`：按相同实验条件配对后的 LRU/LFU 命中率；
- `strategy_paired_test_summary.csv`：LRU 与 LFU 的配对 t 检验和 Wilcoxon 检验摘要；
- `strategy_welch_test_by_condition.csv`：每个实验条件下的 Welch t 检验；
- `regression_coefficients.csv`：回归模型系数表；
- `regression_full_summary.txt`：statsmodels 输出的完整回归结果；
- `chapter5_report.txt`：便于写论文时复制和整理的统计分析摘要；
- `fig_strategy_paired_difference.png`：LFU-LRU 配对差异图；
- `fig_reg_hit_observed_fitted.png`：命中率回归的观测值与拟合值图；
- `fig_reg_error_observed_fitted.png`：理论误差回归的观测值与拟合值图。

## 推荐流程

完整复现实验和分析：

```bash
cd redis-cache-exp
python experiment.py
python analyze_results.py
python chapter5_stats.py
```

只重新生成图表和统计结果：

```bash
cd redis-cache-exp
python analyze_results.py
python chapter5_stats.py
```

## 实验含义

这个实验把 Redis 当作一个真实缓存系统，用 cache-aside 方式模拟请求过程：

1. 每次请求先 `GET obj:id`；
2. 如果未命中，则 `SET obj:id` 写入缓存；
3. Redis 根据配置的 `maxmemory-policy` 自动淘汰 key；
4. 通过 Redis 的 `keyspace_hits` 和 `keyspace_misses` 统计实测命中率。

理论命中率主要用于和实测结果对照：

- Uniform 分布下，理想命中率近似为 `C / N`；
- Zipf 分布下，理想命中率近似为最热门前 `C` 个对象的概率质量之和；
- 因为 Redis 的 `maxmemory` 以字节为单位，实际缓存对象数可能不同于设计容量，所以分析脚本额外计算了基于 `dbsize_after` 的实际容量理论命中率。

这些结果可以支持论文中关于访问分布、缓存容量、LRU/LFU 策略差异，以及理论模型与真实系统偏差的讨论。
