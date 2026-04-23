"""极简批次迭代器 + IterationRecord 定义。

与原 sampling.py 的对比：
- 原：按分层加权采样 section、EWMA 产出率、多种 sampling_mode、渐近曲线拟合
- 新：顺序切分 unique_keys 成定长 batch；IterationRecord 瘦身为 5 个字段

渐近曲线拟合函数 fit_asymptote 保留（visualization 需要），但不再参与停止逻辑。
"""
from typing import Iterator, TypedDict

import numpy as np


class IterationRecord(TypedDict):
    iteration: int
    keys_processed_this_iter: int
    cumulative_keys_processed: int
    cumulative_canonical_size: int
    new_canonical_keys_this_iter: int


def iter_batches(
    unique_keys: list[str],
    batch_size: int,
    start_index: int = 0,
) -> Iterator[tuple[int, list[str]]]:
    """按字典序从 start_index 开始切 batch。

    yield: (batch_start_index_in_unique_keys, batch_keys)
    """
    n = len(unique_keys)
    i = start_index
    while i < n:
        end = min(i + batch_size, n)
        yield i, unique_keys[i:end]
        i = end


def _asymptotic_model(n: np.ndarray, s_max: float, k: float) -> np.ndarray:
    """S(N) = S_max * (1 - exp(-k * N))"""
    return s_max * (1.0 - np.exp(-k * n))


def fit_asymptote(
    history: list[IterationRecord],
) -> tuple[float, float] | None:
    """拟合累积曲线到渐近模型，失败返回 None。

    x = cumulative_keys_processed, y = cumulative_canonical_size
    """
    from scipy.optimize import curve_fit

    xs = [float(r["cumulative_keys_processed"]) for r in history]
    ys = [float(r["cumulative_canonical_size"]) for r in history]
    if len(xs) < 10 or ys[-1] == 0:
        return None
    x = np.array(xs)
    y = np.array(ys)
    try:
        popt, _ = curve_fit(
            _asymptotic_model, x, y,
            p0=[y[-1] * 1.2, 1.0 / max(x[-1] * 0.3, 1.0)],
            bounds=([y[-1] * 0.8, 1e-8], [y[-1] * 5.0, 1.0]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return None
