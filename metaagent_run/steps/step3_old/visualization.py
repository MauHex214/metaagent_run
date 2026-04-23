"""分组饱和曲线可视化（精简版）。

与原 visualization.py 的差异：
- 横轴从 "iteration" 改为 "cumulative unique keys processed"
- 去掉触发点标注、阶段着色、确认窗口等饱和相关元素
- 保留渐近模型拟合（虚线 + S_max 水平参考线）用于论文图的渐近分析
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .runtime import LOGGER
from .sampling import IterationRecord, fit_asymptote, _asymptotic_model


def plot_saturation_curve(
    history: list[IterationRecord],
    output_path: str,
) -> None:
    """画 "处理的唯一 key 数 vs canonical 组数" 的累积曲线。

    - 实心点折线：真实数据
    - 灰色虚线：渐近模型拟合 S(N)=S_max*(1-exp(-k*N))
    - 水平参考线：拟合得到的 S_max
    """
    if not history:
        LOGGER.warning("[曲线] history 为空，跳过作图")
        return

    x = np.array([r["cumulative_keys_processed"] for r in history], dtype=float)
    y = np.array([r["cumulative_canonical_size"] for r in history], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(x, y, marker="o", markersize=4, linewidth=1.4,
            color="#1f77b4", label="canonical size (observed)")

    # 渐近拟合
    fit = fit_asymptote(history)
    if fit is not None:
        s_max, k = fit
        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_fit = _asymptotic_model(x_smooth, s_max, k)
        ax.plot(x_smooth, y_fit, linestyle="--", linewidth=1.3,
                color="#888888",
                label=f"asymptotic fit: S(N)=S_max(1−e^(−kN)), S_max≈{s_max:.1f}, k≈{k:.2e}")
        ax.axhline(y=s_max, linestyle=":", linewidth=1.0, color="#cc4444",
                   label=f"fitted S_max ≈ {s_max:.1f}")
        coverage = y[-1] / s_max if s_max > 0 else float("nan")
        LOGGER.info("[曲线拟合] S_max=%.2f, k=%.4e, 当前覆盖率=%.2f%%",
                    s_max, k, 100 * coverage)
    else:
        LOGGER.info("[曲线拟合] 数据不足或拟合失败，未画渐近线")

    ax.set_xlabel("Cumulative unique keys processed")
    ax.set_ylabel("Canonical schema size")
    ax.set_title("Step 3 canonical grouping — key accumulation curve")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("[曲线] 已保存到 %s", output_path)
