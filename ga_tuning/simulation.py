"""用 search.run_search 跑一场带抖动的 ED 寻优，整理试验和采样明细。"""
import numpy as np

import search
from .design import BUDGET, ED_TOL, PATIENCE
from .objective import metrics


def default_recipe():
    """读取 search.run_search 使用的 BO 默认参数。"""
    params = search.BO_PARAMETERS
    lo, hi = params["ls_bounds"]
    return dict(sigma3_rel=params["sigma3_rel"], n_restarts=params["n_restarts"],
                n_restarts_ei=params["n_restarts_ei"],
                xi_init=params["xi_init"], xi_final=params["xi_final"],
                global_prob=params["global_prob"], ls_lo=lo, ls_hi=hi,
                init_points=params["init_points"])


def simulate(task):
    recipe, name, case, seed = task
    peak = case.reference[3]
    hist = search.run_search(
        **case.options, target="ed", seed=seed,
        init_points=recipe["init_points"], max_iter=BUDGET - recipe["init_points"],
        n_restarts=recipe["n_restarts"], n_restarts_ei=recipe["n_restarts_ei"],
        xi_init=recipe["xi_init"], xi_final=recipe["xi_final"], sigma3_rel=recipe["sigma3_rel"],
        global_prob=recipe["global_prob"], ls_bounds=(recipe["ls_lo"], recipe["ls_hi"]),
        stop_no_improve=PATIENCE, reference_value=peak, relative_tolerance=ED_TOL,
        verbose=False, show=False,
    )
    span = float(np.ptp(search.BOUNDS, axis=1).max())
    result = metrics(hist, peak, span, hist["patience_round"])
    result.update(model=name, seed=seed)
    points = []
    best = -float("inf")
    for idx, (xyz, ed) in enumerate(zip(hist["X"], hist["ed"])):
        bo = idx - hist["n_init"]
        best = max(best, ed)
        points.append(dict(sample=idx + 1, phase="LHS" if bo < 0 else "BO",
                           bo_step=None if bo < 0 else bo + 1, x=xyz[0], y=xyz[1], z=xyz[2],
                           ed=ed, pf=hist["pf"][idx], best_ed=best,
                           error=abs(best - peak) / peak,
                           xi=None if bo < 0 else hist["xi_used"][bo],
                           sigma=None if bo < 0 else hist["sigma_abs_used"][bo]))
    return result, points
