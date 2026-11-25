"""从实际采样轨迹独立计算误差、路程与得分。越小越好。"""
from statistics import mean
from typing import NamedTuple

import numpy as np

from .design import BUDGET, ED_TOL, MODELS

WEIGHTS = (3, 5, 2)


def measure_path(points, best_index):
    # 沿用 batch_test 的口径：含末次采样回到最终最佳设定点；回位不再采样。
    route = np.vstack((points, points[best_index]))
    steps = np.abs(np.diff(route, axis=0))
    by_axis = steps.sum(axis=0)
    # 每段三轴同步移动，取该段最大的轴位移，再把各段相加。
    return by_axis, float(by_axis.max()), float(steps.max(axis=1).sum())


def metrics(history, peak, span):
    points, ed = history["X"], history["y"]
    best_index = int(np.argmax(ed))
    best = float(ed[best_index])
    error = abs(best - peak) / abs(peak)
    axes, max_axis_total, sum_step_max = measure_path(points, best_index)
    n = len(ed)
    limits = (ED_TOL, BUDGET, BUDGET * span)
    normalized = [value / limit for value, limit in zip((error, n, sum_step_max), limits)]
    loss = sum(w * v for w, v in zip(WEIGHTS, normalized)) / sum(WEIGHTS)
    return dict(initial=history["n_init"], bo=n - history["n_init"], samples=n,
                stop=history["stop_reason"], success=int(history["stop_reason"] == "converged"),
                design_ed=peak, best_ed=best, best_sample=best_index + 1,
                best_x=points[best_index, 0], best_y=points[best_index, 1],
                best_z=points[best_index, 2], error=error, travel_x=axes[0],
                travel_y=axes[1], travel_z=axes[2], travel=max_axis_total,
                sum_step_max=sum_step_max, loss=loss)


class RecipeScore(NamedTuple):
    # 先满足训练场次全部收敛的约束，然后最小化各工况的等权平均 loss。
    has_failure: bool
    mean_loss: float
    worst_failure_rate: float
    mean_failure_rate: float


def rank_trials(trials):
    """全达标配方中选平均 loss 最小者；尚无全达标配方时以 loss 引导下一代。"""
    groups = [[r for r in trials if r["model"] == model] for model in MODELS]
    misses = [1 - mean(r["success"] for r in group) for group in groups]
    losses = [mean(r["loss"] for r in group) for group in groups]
    return RecipeScore(any(misses), mean(losses), max(misses), mean(misses))
