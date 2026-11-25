"""从实际采样轨迹独立计算误差、路程与得分。越小越好。"""
from statistics import mean

import numpy as np

from .design import BUDGET, ED_TOL, MODELS

WEIGHTS = (3, 5, 2)
FAILURE_WEIGHT = 0.50


def measure_path(points, best_index):
    # 含末次采样回到最终最佳设定点；回位不再采样。
    route = np.vstack((points, points[best_index]))
    steps = np.abs(np.diff(route, axis=0))
    by_axis = steps.sum(axis=0)
    # 每段三轴同步移动，取该段最大的轴位移，再把各段相加。
    return by_axis, float(by_axis.max()), float(steps.max(axis=1).sum())


def metrics(history, peak, span, patience_round):
    points, ed = history["X"], history["y"]
    best_index = int(np.argmax(ed))
    best = float(ed[best_index])
    error = abs(best - peak) / abs(peak)
    axes, max_axis_total, sum_step_max = measure_path(points, best_index)
    n = len(ed)
    limits = (ED_TOL, BUDGET, BUDGET * span)
    normalized = [value / limit for value, limit in zip((error, n, sum_step_max), limits)]
    base_loss = sum(w * v for w, v in zip(WEIGHTS, normalized)) / sum(WEIGHTS)
    # BO 在预算内曾满足 30 轮判据即算通过这一项；ED 准确度另有连续误差项。
    failed = int(patience_round is None)
    loss = (1 - FAILURE_WEIGHT) * base_loss + FAILURE_WEIGHT * failed
    return dict(initial=history["n_init"], bo=n - history["n_init"], samples=n,
                stop=history["stop_reason"], success=int(history["stop_reason"] == "converged"),
                design_ed=peak, best_ed=best, best_sample=best_index + 1,
                best_x=points[best_index, 0], best_y=points[best_index, 1],
                best_z=points[best_index, 2], error=error, travel_x=axes[0],
                travel_y=axes[1], travel_z=axes[2], travel=max_axis_total,
                sum_step_max=sum_step_max, patience_round=patience_round, patience_failed=failed,
                base_loss=base_loss, loss=loss)


def recipe_means(trials):
    """先对各模型的重复试验取平均，再给三个模型相同权重。"""
    groups = [[row for row in trials if row["model"] == model] for model in MODELS]
    return {key: mean(mean(row[key] for row in group) for group in groups)
            for key in ("base_loss", "patience_failed", "loss")}
