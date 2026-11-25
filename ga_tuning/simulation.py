"""适配仓库已有的 fitted/RBF、抖动、GPR/EI 和联合停止入口。"""
from unittest.mock import patch

import numpy as np

import batch_test
import search
from .design import BUDGET, ED_TOL, PATIENCE, reference
from .objective import metrics


def default_recipe():
    """直接读取现行 BO 配置。两种重启次数的回退值与 batch_test 的实际调用一致。"""
    params = search.BO_PARAMETERS
    lo, hi = params["ls_bounds"]
    return dict(sigma3_rel=params["sigma3_rel"], n_restarts=10,
                n_restarts_ei=params.get("n_restarts_ei", 20),
                xi_init=params["xi_init"], xi_final=params["xi_final"],
                global_prob=params["global_prob"], ls_lo=lo, ls_hi=hi,
                init_points=params["init_points"])


def simulate(task):
    recipe, model, seed = task
    peak = reference(model)[3]
    initial = recipe["init_points"]
    case = dict(data_model=model, seed=seed, init_points=initial, target="ed")
    params = {k: recipe[k] for k in ("sigma3_rel", "xi_init", "xi_final",
                                       "global_prob", "n_restarts_ei")}
    params["ls_bounds"] = (recipe["ls_lo"], recipe["ls_hi"])

    # 现有批测内部写死了 n_restarts=10。临时转发给同一个原始 fit_gp，
    # 只覆盖这一个实参。每个 worker 是独立进程，退出 with 即恢复。
    original_fit = search.fit_gp

    def fit_recipe(*args, **kwargs):
        kwargs["n_restarts"] = recipe["n_restarts"]
        return original_fit(*args, **kwargs)

    with patch.dict(search.BO_PARAMETERS, params), patch.object(search, "fit_gp", fit_recipe):
        hist = batch_test.run_search_with_accuracy(
            case, {"reference_value": peak}, BUDGET - initial, PATIENCE, 100 * ED_TOL)
    span = float(np.ptp(search.BOUNDS, axis=1).max())
    # 用原 BO 的逐轮噪声和显著改善规则复核；不在此处叠加 ED 的 10% 门槛。
    patience_round = batch_test.get_convergence_iteration(hist, PATIENCE, peak, tolerance_pct=np.inf)
    result = metrics(hist, peak, span, patience_round)
    result.update(model=model, seed=seed)
    points = []
    best = -float("inf")
    for idx, (xyz, ed) in enumerate(zip(hist["X"], hist["ed"])):
        bo = idx - initial
        best = max(best, ed)
        points.append(dict(sample=idx + 1, phase="LHS" if bo < 0 else "BO",
                           bo_step=None if bo < 0 else bo + 1, x=xyz[0], y=xyz[1], z=xyz[2],
                           ed=ed, pf=hist["pf"][idx], best_ed=best,
                           error=abs(best - peak) / peak,
                           xi=None if bo < 0 else hist["xi_used"][bo],
                           sigma=None if bo < 0 else hist["sigma_abs_used"][bo]))
    return result, points
