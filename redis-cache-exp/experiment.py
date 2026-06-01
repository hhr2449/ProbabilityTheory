import random
import math
import time
import csv
from typing import List, Dict

import numpy as np
import redis


HOST = "localhost"
PORT = 6379

# 固定 value 大小，尽量让每个对象大小相同
VALUE_SIZE = 128
VALUE = "x" * VALUE_SIZE


def connect_redis():
    return redis.Redis(host=HOST, port=PORT, decode_responses=True)


def reset_redis(r: redis.Redis):
    r.flushall()
    r.config_resetstat()


def set_redis_policy(r: redis.Redis, policy: str, maxmemory_bytes: int):
    # policy 可选：allkeys-lru / allkeys-lfu
    r.config_set("maxmemory-policy", policy)
    r.config_set("maxmemory", maxmemory_bytes)


def get_used_memory(r: redis.Redis) -> int:
    return int(r.info("memory")["used_memory"])


def calibrate_avg_key_size(r: redis.Redis, sample_keys: int = 5000) -> float:
    """
    估算单个 key-value 平均占用多少 Redis 内存。
    注意：这是近似值，不是严格值。
    """
    r.flushall()
    r.config_set("maxmemory", 0)  # 校准阶段先不限制内存

    before = get_used_memory(r)

    pipe = r.pipeline()
    for i in range(sample_keys):
        pipe.set(f"calib:{i}", VALUE)
    pipe.execute()

    after = get_used_memory(r)
    avg = (after - before) / sample_keys

    r.flushall()
    return avg


def generate_uniform_requests(N: int, T: int, seed: int) -> List[int]:
    random.seed(seed)
    return [random.randint(1, N) for _ in range(T)]


def generate_zipf_requests(N: int, T: int, alpha: float, seed: int) -> List[int]:
    """
    生成 1..N 上的截断 Zipf 分布请求。
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, N + 1)
    weights = ranks ** (-alpha)
    probs = weights / weights.sum()
    return rng.choice(ranks, size=T, p=probs).tolist()


def theoretical_hit_rate_uniform(N: int, C: int) -> float:
    return C / N


def theoretical_hit_rate_zipf(N: int, C: int, alpha: float) -> float:
    ranks = np.arange(1, N + 1)
    weights = ranks ** (-alpha)
    return float(weights[:C].sum() / weights.sum())


def run_cache_requests(r: redis.Redis, requests: List[int]):
    """
    cache-aside 模式：
    GET key
    miss 后 SET key
    """
    pipe = r.pipeline()
    # 为了简单可靠，这里不用 pipeline 批量 GET/SET，因为命中判断依赖每一步结果
    for obj_id in requests:
        key = f"obj:{obj_id}"
        val = r.get(key)
        if val is None:
            r.set(key, VALUE)


def get_hit_rate_from_info(r: redis.Redis) -> Dict[str, float]:
    stats = r.info("stats")
    hits = int(stats.get("keyspace_hits", 0))
    misses = int(stats.get("keyspace_misses", 0))
    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0

    return {
        "hits": hits,
        "misses": misses,
        "total": total,
        "hit_rate": hit_rate,
    }


def run_one_experiment(
    N: int,
    capacity_ratio: float,
    distribution: str,
    alpha: float,
    policy: str,
    warmup_T: int,
    measure_T: int,
    seed: int,
    avg_key_size: float,
):
    r = connect_redis()
    reset_redis(r)

    C = int(N * capacity_ratio)

    # Redis maxmemory 是字节，不是对象数。
    # 这里用校准得到的平均 key 大小近似换算。
    base_memory = get_used_memory(r)
    maxmemory_bytes = int(base_memory + C * avg_key_size)

    set_redis_policy(r, policy, maxmemory_bytes)

    if distribution == "uniform":
        warmup_requests = generate_uniform_requests(N, warmup_T, seed)
        measure_requests = generate_uniform_requests(N, measure_T, seed + 100000)
        theory = theoretical_hit_rate_uniform(N, C)
    elif distribution == "zipf":
        warmup_requests = generate_zipf_requests(N, warmup_T, alpha, seed)
        measure_requests = generate_zipf_requests(N, measure_T, alpha, seed + 100000)
        theory = theoretical_hit_rate_zipf(N, C, alpha)
    else:
        raise ValueError("distribution must be uniform or zipf")

    # 预热阶段：不统计
    run_cache_requests(r, warmup_requests)

    # 清空统计量，但保留缓存内容
    r.config_resetstat()

    # 正式统计阶段
    run_cache_requests(r, measure_requests)

    result = get_hit_rate_from_info(r)
    dbsize = r.dbsize()

    return {
        "N": N,
        "C": C,
        "capacity_ratio": capacity_ratio,
        "distribution": distribution,
        "alpha": alpha if distribution == "zipf" else "",
        "policy": policy,
        "warmup_T": warmup_T,
        "measure_T": measure_T,
        "seed": seed,
        "avg_key_size": avg_key_size,
        "maxmemory_bytes": maxmemory_bytes,
        "dbsize_after": dbsize,
        "theory_hit_rate": theory,
        "redis_hit_rate": result["hit_rate"],
        "hits": result["hits"],
        "misses": result["misses"],
        "error": result["hit_rate"] - theory,
        "abs_error": abs(result["hit_rate"] - theory),
    }


def main():
    r = connect_redis()

    print("Calibrating average key size...")
    avg_key_size = calibrate_avg_key_size(r, sample_keys=5000)
    print(f"Average key size ≈ {avg_key_size:.2f} bytes")

    N = 10000
    warmup_T = 50000
    measure_T = 100000

    experiments = []

    # 实验 A：访问分布影响，固定容量 10%，LRU
    for dist, alpha in [
        ("uniform", 0.0),
        ("zipf", 0.6),
        ("zipf", 0.8),
        ("zipf", 1.0),
        ("zipf", 1.2),
    ]:
        experiments.append({
            "N": N,
            "capacity_ratio": 0.10,
            "distribution": dist,
            "alpha": alpha,
            "policy": "allkeys-lru",
            "warmup_T": warmup_T,
            "measure_T": measure_T,
        })

    # 实验 B：缓存容量影响，固定 Zipf alpha=1.0，LRU
    for ratio in [0.05, 0.10, 0.20, 0.30, 0.40]:
        experiments.append({
            "N": N,
            "capacity_ratio": ratio,
            "distribution": "zipf",
            "alpha": 1.0,
            "policy": "allkeys-lru",
            "warmup_T": warmup_T,
            "measure_T": measure_T,
        })

    # 实验 C：LRU / LFU 对比
    for alpha, ratio in [
        (0.8, 0.10),
        (1.0, 0.10),
        (1.2, 0.10),
        (1.0, 0.05),
        (1.0, 0.20),
    ]:
        for policy in ["allkeys-lru", "allkeys-lfu"]:
            experiments.append({
                "N": N,
                "capacity_ratio": ratio,
                "distribution": "zipf",
                "alpha": alpha,
                "policy": policy,
                "warmup_T": warmup_T,
                "measure_T": measure_T,
            })

    rows = []

    # 每组实验重复 3 次
    repeat = 3
    for idx, exp in enumerate(experiments):
        for rep in range(repeat):
            seed = 20240601 + idx * 100 + rep
            print(f"Running exp {idx + 1}/{len(experiments)}, repeat {rep + 1}/{repeat}: {exp}")

            row = run_one_experiment(
                N=exp["N"],
                capacity_ratio=exp["capacity_ratio"],
                distribution=exp["distribution"],
                alpha=exp["alpha"],
                policy=exp["policy"],
                warmup_T=exp["warmup_T"],
                measure_T=exp["measure_T"],
                seed=seed,
                avg_key_size=avg_key_size,
            )
            rows.append(row)

            print(
                f"  theory={row['theory_hit_rate']:.4f}, "
                f"redis={row['redis_hit_rate']:.4f}, "
                f"error={row['error']:.4f}, "
                f"dbsize={row['dbsize_after']}"
            )

    output_file = "redis_cache_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Results saved to {output_file}")


if __name__ == "__main__":
    main()