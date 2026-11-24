"""从实际采样轨迹独立计算误差、路程与得分。越小越好。"""
from statistics import mean

import numpy as np

from .design import BUDGET, ED_TOL, MODELS

WEIGHTS = (3, 5, 2)


def measure_path(points, best_index):
    # 沿用 batch_test 的口径：含末次采样回到最终最佳设定点；回位不再采样。
    route = np.vstack((points, points[best_index]))
    steps = np.abs(np.diff(route, axis=0))
    by_axis = steps.sum(axis=0)
    # 两种常见的路程各自单列。评分只使用三条轴累计行程中的最大值。
    return by_axis, float(by_axis.max()), float(steps.max(axis=1).sum())


def metrics(history, peak, span):
    points, ed = history["X"], history["y"]
    best_index = int(np.argmax(ed))
    best = float(ed[best_index])
    error = abs(best - peak) / abs(peak)
    axes, distance, simultaneous = measure_path(points, best_index)
    n = len(ed)
    limits = (ED_TOL, BUDGET, BUDGET * span)
    normalized = [value / limit for value, limit in zip((error, n, distance), limits)]
    loss = sum(w * v for w, v in zip(WEIGHTS, normalized)) / sum(WEIGHTS)
    return dict(initial=history["n_init"], bo=n - history["n_init"], samples=n,
                stop=history["stop_reason"], success=int(history["stop_reason"] == "converged"),
                design_ed=peak, best_ed=best, best_sample=best_index + 1,
                best_x=points[best_index, 0], best_y=points[best_index, 1],
                best_z=points[best_index, 2], error=error, travel_x=axes[0],
                travel_y=axes[1], travel_z=axes[2], travel=distance,
                sum_step_max=simultaneous, loss=loss)


def rank_trials(trials):
    """排序先保护最弱工况，再看全部失败数，最后比较四工况等权的综合分。"""
    groups = [[r for r in trials if r["model"] == model] for model in MODELS]
    misses = [1 - mean(r["success"] for r in group) for group in groups]
    losses = [mean(r["loss"] for r in group) for group in groups]
    return max(misses), mean(misses), mean(losses)
