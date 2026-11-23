"""模型 3（LT 精细）带抖动 ED 的遗传算法调参。运行 python GA.py。"""
import os
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from statistics import mean
from typing import NamedTuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from fitter import EDFitter, EDPFModel
from LHS import get_initial_points
from measurement_data import MODEL_DATA
from search import BOUNDS, fit_gp, get_next_sample


class Recipe(NamedTuple):
    gp_restarts: int
    sigma3_rel: float
    xi_init: float
    xi_final: float
    ls_lo: float
    ls_hi: float
    init_points: int
    global_prob: float


# 每行对应 Recipe 中的一个基因。
CHOICES = (
    (2, 5, 10),                    # gp_restarts
    (0.0, 0.03, 0.10),             # sigma3_rel
    (0.0001, 0.0003, 0.001, 0.01), # xi_init，单位为 ED
    (0.0, 0.00001, 0.00003),       # xi_final
    (20, 40, 80),                  # ls_lo
    (200, 600, 2000),              # ls_hi
    (10, 20, 30, 40),              # init_points
    (0.15, 0.40, 0.70),            # global_prob
)
BASELINE = Recipe(10, 0.0, 0.01, 0.0, 40, 2000, 40, 0.40)
DESIGN_ED = float(MODEL_DATA[3][:, 3].max())
BUDGET = 80                         # 初始点与 BO 点合计
TRAIN_SEEDS = range(42, 47)
TEST_SEEDS = range(500, 530)
POPULATION = 8
GENERATIONS = 4


def run_once(recipe, seed):
    """每个坐标从 fitter 取一发带抖动 ED；仅在 BO 阶段累计无改进轮数。"""
    np.random.seed(seed)
    sensor = EDPFModel(EDFitter(3, "rbf"), seed=seed)
    x = get_initial_points(recipe.init_points, *BOUNDS)
    y = np.array([sensor.measure(point)[0] for point in x])
    best = float(y.max())
    stale = 0
    rounds = BUDGET - len(y)
    sigma_a = recipe.sigma3_rel * np.ptp(y) / 3
    sigma_used = sigma_a
    bounds = (recipe.ls_lo, recipe.ls_hi)
    span = np.mean(BOUNDS[:, 1] - BOUNDS[:, 0])
    length = np.clip(span / 2, *bounds)

    low_ed = 0.90 * DESIGN_ED
    high_ed = 1.10 * DESIGN_ED
    for i in range(rounds):
        variance = (sigma_a / (y.std() + 1e-12)) ** 2
        gp = fit_gp(x, y, max(variance, 1e-6), length, bounds,
                    n_restarts=recipe.gp_restarts, use_white=recipe.sigma3_rel > 0)
        t = i / (rounds - 1)
        xi = recipe.xi_init * (1 - t)
        if recipe.xi_final > 0:
            ratio = recipe.xi_final / recipe.xi_init
            xi = recipe.xi_init * ratio ** t
        best_norm = (best - 3 * sigma_used - gp.y_mean) / gp.y_std
        point, _ = get_next_sample(x, y, BOUNDS, gp, best_norm, xi / gp.y_std,
                                   global_prob=recipe.global_prob, min_step_i=2 * (1 - t)**2,
                                   i=i, max_iter=rounds)
        if recipe.sigma3_rel > 0:
            learned = np.sqrt(gp.kernel.k2.noise_level) * gp.y_std
            sigma_used = max(learned, 0.5 * sigma_a)

        ed, _ = sensor.measure(point)
        x = np.vstack((x, point))
        y = np.append(y, ed)
        stale += 1
        if ed > best:
            best = ed
            stale = 0
        converged = stale >= 10 and low_ed <= best <= high_ed
        if converged:
            break

    error = abs(best - DESIGN_ED) / DESIGN_ED
    position = tuple(x[np.argmax(y)].tolist())
    return dict(seed=seed, converged=converged, bo_rounds=i + 1, total=len(y),
                stale=stale, best_ed=float(best), error_pct=100 * float(error), xyz=position)


def train_score(recipe):
    rows = [run_once(recipe, seed) for seed in TRAIN_SEEDS]
    failures = sum(not row["converged"] for row in rows)
    return failures, mean(r["total"] for r in rows), mean(r["error_pct"] for r in rows)


def evolve(pool):
    rng = random.Random(2026)
    population = [BASELINE]
    while len(population) < POPULATION:
        genes = [rng.choice(values) for values in CHOICES]
        population.append(Recipe(*genes))
    scores = {}
    for generation in range(1, GENERATIONS + 1):
        fresh = [r for r in dict.fromkeys(population) if r not in scores]
        scores.update(zip(fresh, pool.map(train_score, fresh)))
        ranked = sorted(population, key=scores.get)
        print(f"第 {generation} 代，训练成绩 {scores[ranked[0]]}", flush=True)
        if generation == GENERATIONS:
            break
        population = [ranked[0]]
        weights = range(POPULATION, 0, -1)
        while len(population) < POPULATION:
            parent_a, parent_b = rng.choices(ranked, weights=weights, k=2)
            genes = []
            for a, b, options in zip(parent_a, parent_b, CHOICES):
                gene = rng.choice((a, b))
                if rng.random() < 0.18:
                    gene = rng.choice(options)
                genes.append(gene)
            child = Recipe(*genes)
            if child not in population:
                population.append(child)
    print(f"共评估 {len(scores)} 套 recipe")
    return min(scores, key=scores.get)


def main():
    with ProcessPoolExecutor(max_workers=4) as pool:
        winner = evolve(pool)
        print("最佳 recipe", winner)
        for label, recipe in [("GA", winner), ("默认配置", BASELINE)]:
            rows = list(pool.map(run_once, repeat(recipe), TEST_SEEDS))
            for row in rows:
                print(label, row)
            hits = sum(r["converged"] for r in rows)
            print(f"{label} 验证收敛 {hits}/{len(rows)}")
            for field in ("bo_rounds", "total", "best_ed", "error_pct"):
                print(field, mean(r[field] for r in rows))


if __name__ == "__main__":
    main()
