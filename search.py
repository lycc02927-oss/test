"""新版三维 GPR + EI 寻优；运行 python search.py 即可。

模型与抖动复用 op 的 fitter.py；图表使用本版四宫格绘图函数。
每次只量测一发，ED、PF、ED/PF 与寻优目标来自同一份结果。
"""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from kernel import SumKernel, MaternKernel, WhiteKernel
from GPR import GaussianProcessRegressor
from EI import expected_improvement
from LHS import get_initial_points
from fitter import EDFitter, EDPFModel, FORMULA_CONFIGS, build_formula_ed_model
from measurement_data import MODEL_NAMES


# -------------------- 修改这里切换模型和运行参数 --------------------
TARGET = "ed"  # 可选："ed" 最大化 ED；"ed_pf_ratio" 最大化 ED/PF。
ED_SOURCE = "fitted"  # "fitted" 数据拟合；"formula" 数学公式。
DATA_MODEL = 2  # 1=P3试验，2=LT整机仿真，3/4/5=中心/偏置/多峰高斯。
FIT_METHOD = "rbf"  # rbf / manual_gpr / linear / polynomial / svm。
FORMULA_MODEL = "formula_gaussian"  # 平台区用 formula_plateau，截断峰用 formula_truncated_peak。
RANDOM_SEED = 42
BOUNDS = np.array([[-300.0, 300.0]] * 3)
BO_PARAMETERS = dict(
    max_iter=40,
    init_points=40,
    sigma3_rel=0.0,  # >0 时启用 WhiteKernel；相对于初始目标值范围的 3σ。
    xi_init=0.01,
    xi_final=0.0,
    global_prob=0.40,
    r_frac_min=0.05,
    stop_no_improve=30,
    n_restarts=10,
    n_restarts_ei=20,
    ls_bounds=(40.0, 2000.0),
)

_energy_model = None


def set_ed_model(source="fitted", *, formula_model="formula_gaussian",
                 data_model=DATA_MODEL, fit_method="rbf", fit_parameters=None,
                 seed=42, peaks=None, **sampling_parameters):
    """选择 ED 空间模型；量测与抖动共用原 fitter.EDPFModel。

    sampling_parameters 可传 position_half_width、ed_time_parameters、
    pf_time_parameters。每次调用重置量测随机数流。
    """
    global _energy_model
    if source == "formula":
        spatial_model = build_formula_ed_model(formula_model, peaks=peaks)
    elif source == "fitted":
        spatial_model = EDFitter(data_model, fit_method, fit_parameters)
    else:
        raise ValueError("source must be 'fitted' or 'formula'")
    _energy_model = EDPFModel(spatial_model, seed=seed, **sampling_parameters)
    return _energy_model


def energy_function(x, target=None):
    """量测一发并保存其 ED/PF，返回选定目标；打印时不再次量测。"""
    if _energy_model is None:
        set_ed_model(ED_SOURCE, formula_model=FORMULA_MODEL,
                     data_model=DATA_MODEL, fit_method=FIT_METHOD, seed=RANDOM_SEED)
    ed, pf = _energy_model.measure(x)
    measurement = {"ed": float(ed), "pf": float(pf), "ed_pf_ratio": float(ed / pf)}
    energy_function.last_measurement = measurement
    target = TARGET if target is None else target
    return {"ed": measurement["ed"], "ed_pf_ratio": measurement["ed_pf_ratio"]}[target]


def make_objective(target="ed"):
    """让寻优器读取同一次量测的 ED、PF 和 ED/PF。"""
    if target not in ("ed", "ed_pf_ratio"):
        raise ValueError("target must be 'ed' or 'ed_pf_ratio'")

    def objective(x):
        value = energy_function(x, target)
        objective.last_measurement = energy_function.last_measurement
        return value

    objective.target = target
    return objective


def fit_gp(X_obs, y_obs, alpha_eff, ls_init, ls_bounds, n_restarts=10,
           use_white=True, noise_bounds=(1e-6, 1e-2)):
    """拟合 GP；alpha_eff 是标准化空间噪声方差，作为 WhiteKernel 初值。"""
    if use_white:
        kernel = SumKernel(
            MaternKernel(length_scale=ls_init, length_scale_bounds=ls_bounds),
            WhiteKernel(noise_level=max(float(alpha_eff), 1e-10),
                        noise_level_bounds=noise_bounds))
    else:
        kernel = MaternKernel(length_scale=ls_init, length_scale_bounds=ls_bounds)
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-10, normalize_y=True,
        n_restarts_optimizer=n_restarts,
    ).fit(X_obs, y_obs)


def get_next_sample(X_obs, y_obs, bounds, gpr, best_norm, xi_norm,
                    r_frac_min=0.05, global_prob=0.0,
                    n_restarts_ei=20, min_step_i=0.0,
                    i=0, max_iter=70, span=None, verbose=False):
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = bounds.shape[0]
    if span is None:
        span = float(np.mean(hi - lo))
    best_x_cur = X_obs[int(np.argmax(y_obs))]
    y_range = float(np.ptp(y_obs)) + 1e-12
    t = i / max(1, max_iter - 1)
    r_frac = max(r_frac_min, 0.5 * (1.0 - t) ** 2)
    r_explore = r_frac * span

    # 后半程 L-BFGS-B 的边界收缩到 best_x_cur ± 5*r_explore。
    if t >= 0.5:
        lb = np.maximum(best_x_cur - 5.0 * r_explore, lo)
        ub = np.minimum(best_x_cur + 5.0 * r_explore, hi)
        search_bounds = np.column_stack([lb, ub])
    else:
        search_bounds = bounds

    best_c, best_ei = None, np.inf
    for _ in range(n_restarts_ei):
        if np.random.rand() < global_prob:
            x0 = lo + (hi - lo) * np.random.rand(dim)
        else:
            direction = np.random.randn(dim)
            nd = np.linalg.norm(direction)
            if nd > 1e-12:
                direction /= nd
            radius = r_explore * np.random.rand() ** (1.0 / dim)
            x0 = np.clip(best_x_cur + direction * radius, lo, hi)
        res = minimize(expected_improvement, x0,
                       args=(gpr, best_norm, xi_norm),
                       bounds=search_bounds, method='L-BFGS-B')
        if np.isfinite(res.fun) and res.fun < best_ei:
            best_ei, best_c = float(res.fun), np.asarray(res.x)
    next_x = (lo + (hi - lo) * np.random.rand(dim)
              if best_c is None else best_c)

    # EI 与阈值必须同在标准化空间。
    ei_flat = (-best_ei) < (1e-3 * y_range / gpr.y_std)
    if ei_flat:
        r_local = 0.5 * r_explore
        cand = np.clip(best_x_cur + r_local * np.random.randn(256, dim), lo, hi)
        if min_step_i > 1e-9 and len(X_obs) > 0:
            d_ok = np.min(cdist(cand, X_obs), axis=1) >= min_step_i
            if np.any(d_ok):
                cand = cand[d_ok]
        mu_c, _ = gpr.predict_normalized(cand, return_std=True)
        next_x = cand[int(np.argmax(np.ravel(mu_c)))]

    d_to_obs = float(np.min(cdist([next_x], X_obs)))
    if min_step_i > 1e-9 and d_to_obs < min_step_i:
        for _ in range(1000):
            x_c = lo + (hi - lo) * np.random.rand(dim)
            if np.min(cdist([x_c], X_obs)) >= min_step_i:
                next_x = x_c
                d_to_obs = float(np.min(cdist([next_x], X_obs)))
                break
    return next_x, d_to_obs


def bayesian_optimization(objective_func, bounds, max_iter=70, init_points=30,
                          sigma3_rel=0.0, xi_init=None, xi_final=None,
                          r_frac_min=0.05, global_prob=0.0, ls_bounds=None,
                          stop_no_improve=10, n_restarts_ei=20,
                          verbose=True, random_seed=42, *, n_restarts=10,
                          reference_value=None, relative_tolerance=0.10):
    """使用新版 GPR/EI 最大化目标，history 的 ED/PF 与 X、y 逐行对应。

    random_seed 控制 LHS、GP 重启及 EI 选点；传 None 可沿用外部随机状态。
    标量目标函数仍可使用；没有量测信息时 ED/PF 历史以 NaN 填充。
    传 reference_value 时，只在本轮仍无显著改善且相对偏差达标时提前结束。
    patience_round 记录首次满足停滞条件的 BO 轮数（从 1 开始），与偏差门槛独立。
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = float(np.mean(hi - lo))
    if ls_bounds is None:
        ls_bounds = (max(span / 100.0, 1e-3), max(span / 25.0, 1e-2))
    sigma3_rel = max(float(sigma3_rel), 0.0)
    use_white = sigma3_rel > 0
    noise_bounds = (1e-6, 1e-2) if use_white else (1e-10, 1e-2)

    # 沿用 op 的中心集中 LHS，保持初始采样方式。
    X_raw = np.asarray(get_initial_points(init_points, bounds[0], bounds[1], bounds[2]),
                       dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw[None, :]
    X_init = X_raw[np.all((X_raw >= lo) & (X_raw <= hi), axis=1)]

    measurements = {key: [] for key in ('ed', 'pf', 'ed_pf_ratio')}
    measurement_text = ""

    def evaluate(x):
        nonlocal measurement_text
        value = float(objective_func(x))
        measurement = getattr(objective_func, 'last_measurement', {})
        for key in measurements:
            measurements[key].append(float(measurement.get(key, np.nan)))
        measurement_text = (
            f"ED={measurement['ed']:.6f}, PF={measurement['pf']:.6f}, "
            f"ED/PF={measurement['ed_pf_ratio']:.6f}, "
            if measurement else "")
        return value

    y_init = []
    for index, x in enumerate(X_init):
        value = evaluate(x)
        y_init.append(value)
        if verbose:
            print(f"Init {index + 1:02d}/{len(X_init)}: {measurement_text}V={value:.6f}")
    y_init = np.asarray(y_init)
    y_range0 = float(np.ptp(y_init)) + 1e-12
    sigma_a = sigma3_rel / 3.0 * y_range0

    xi_def_init = 0.01 * y_range0
    xi_def_final = 0.001 * y_range0
    if sigma_a > 0:
        xi_def_init = max(xi_def_init, 0.5 * sigma_a)
        xi_def_final = max(xi_def_final, 0.1 * sigma_a)
    if xi_init is None and xi_final is None:
        xi_init, xi_final = xi_def_init, xi_def_final
    elif xi_init is None:
        xi_init = max(float(xi_final), xi_def_init)
    elif xi_final is None:
        xi_final = min(float(xi_init), xi_def_final)
    xi_init = float(max(xi_init, 0.0))
    xi_final = float(max(xi_final, 0.0))
    if xi_init <= 1e-12:
        xi_init = xi_final = 0.0
    elif xi_final > xi_init:
        warnings.warn(f"xi_final ({xi_final:.4g}) > xi_init ({xi_init:.4g})，"
                      f"已设为 xi_final = {xi_init:.4g}", RuntimeWarning)
        xi_final = xi_init

    history = {
        'X': X_init.copy(), 'y': y_init.copy(),
        'best_value': [float(np.max(y_init))],
        'best_x': [X_init[np.argmax(y_init)].copy()], 'time': [0.0],
        'distance_to_nearest': [], 'n_init': len(X_init),
        'noise_level': [], 'sigma_abs_used': [], 'xi_used': [],
        'target': getattr(objective_func, 'target', 'objective'),
        'patience_round': None, 'stop_reason': 'max_iter',
        **measurements,
    }
    start_time = time.time()
    if verbose:
        print(f"[BO] r_frac_min={r_frac_min:.3f} | global_prob={global_prob:.2f} | "
              f"use_white={use_white}")
        print(f"Initial best: {history['best_value'][0]:.4f} | n_init: {len(X_init)} | "
              f"xi: {xi_init:.4g} → {xi_final:.4g} | "
              f"3σ_rel: {sigma3_rel:.3g} (σ_a={sigma_a:.4g})")

    last_improve_best, last_improve_iter = -np.inf, 0
    X_obs, y_obs = X_init.copy(), y_init.copy()
    gpr = None
    for i in range(max_iter):
        y_range = float(np.ptp(y_obs)) + 1e-12
        y_std = float(y_obs.std()) + 1e-12
        current_best = float(np.max(y_obs))
        alpha_eff = max((sigma_a / y_std) ** 2, 1e-6) if use_white else 1e-10
        if gpr is not None and hasattr(gpr.kernel, 'k2'):
            sigma_learned = float(np.sqrt(gpr.kernel.k2.noise_level)) * gpr.y_std
            sigma_used = max(sigma_learned, 0.5 * sigma_a)
        else:
            sigma_used = sigma_a
        best_robust = current_best - 3.0 * sigma_used
        tol_abs = max(1e-6 * y_range, 3.0 * sigma_used)
        if best_robust > last_improve_best + tol_abs:
            last_improve_best, last_improve_iter = best_robust, i

        t = i / max(1, max_iter - 1)
        if xi_init <= 1e-12:
            xi_i = 0.0
        elif xi_final == 0.0:
            # 指数调度无法以 0 为终值；此时线性衰减至 0。
            xi_i = xi_init * (1.0 - t)
        else:
            xi_i = max(xi_init * ((xi_final / xi_init) ** t), xi_final)

        ls_init = float(np.clip(span / 2.0, ls_bounds[0], ls_bounds[1]))
        gpr = fit_gp(X_obs, y_obs, alpha_eff, ls_init, ls_bounds,
                     n_restarts=n_restarts, use_white=use_white, noise_bounds=noise_bounds)
        best_norm = (best_robust - gpr.y_mean) / gpr.y_std
        xi_norm = xi_i / gpr.y_std
        min_step_i = 2.0 * (1.0 - t) ** 2
        next_x, d_to_obs = get_next_sample(
            X_obs, y_obs, bounds, gpr, best_norm, xi_norm,
            r_frac_min=r_frac_min, global_prob=global_prob,
            n_restarts_ei=n_restarts_ei, min_step_i=min_step_i,
            i=i, max_iter=max_iter, span=span, verbose=verbose)
        history['distance_to_nearest'].append(d_to_obs)

        next_y = evaluate(next_x)
        X_obs = np.vstack([X_obs, next_x[None, :]])
        y_obs = np.append(y_obs, next_y)
        best_idx = int(np.argmax(y_obs))
        current_best = float(y_obs[best_idx])
        history['X'], history['y'] = X_obs.copy(), y_obs.copy()
        history['best_value'].append(current_best)
        history['best_x'].append(X_obs[best_idx].copy())
        history['time'].append(time.time() - start_time)
        history['noise_level'].append(float(gpr.kernel.k2.noise_level)
                                      if hasattr(gpr.kernel, 'k2') else 0.0)
        history['sigma_abs_used'].append(sigma_used)
        history['xi_used'].append(xi_i)
        if verbose:
            nl_str = (f"{gpr.kernel.k2.noise_level:.3e}"
                      if hasattr(gpr.kernel, 'k2') else "off")
            print(f"Iter {i + 1:02d}/{max_iter}: xi={xi_i:.4g}, {measurement_text}"
                  f"V={next_y:.6f}, Best={current_best:.6f}, noise={nl_str}")
        if (i - last_improve_iter) >= stop_no_improve:
            improved_now = current_best - 3.0 * sigma_used > last_improve_best + tol_abs
            if not improved_now and history['patience_round'] is None:
                history['patience_round'] = i + 1
            # GA 提供设计峰值。单独运行且未指定参考值时沿用原有停机方式。
            if reference_value is not None:
                allowed_error = abs(reference_value) * (relative_tolerance + 1e-11)
                if improved_now or abs(current_best - reference_value) > allowed_error:
                    continue
            history['stop_reason'] = 'converged'
            if verbose:
                print(f"\nStop: no improve for {i - last_improve_iter} iters")
            break

    for key in ('X', 'y', 'best_value', 'best_x', 'time', 'distance_to_nearest',
                'noise_level', 'sigma_abs_used', 'xi_used', 'ed', 'pf', 'ed_pf_ratio'):
        history[key] = np.asarray(history[key])
    return history


def plot_optimization_results(history, true_optimum=None,
                              dist_log=False, min_step=2.0, show=True):
    """最新版四宫格图；show=False 时返回 Figure，方便无界面保存。"""
    from matplotlib.font_manager import fontManager
    plt.rcParams['font.sans-serif'] = (
        ['SimHei'] if any(font.name == 'SimHei' for font in fontManager.ttflist)
        else ['DejaVu Sans'])
    plt.rcParams['axes.unicode_minus'] = False
    X = np.asarray(history['X'])
    y = np.asarray(history['y'])
    n_init = history.get('n_init', 0)
    dim = X.shape[1]
    r_norm = np.linalg.norm(X, axis=1)
    N = len(y)
    eval_idx = np.arange(N)
    best_aligned = np.maximum.accumulate(y)
    best_per_iter = np.asarray(history['best_value'])
    best_idx = int(np.argmax(y))
    best_value = float(y[best_idx])
    best_position = X[best_idx]
    target_label = {'ed': 'ED', 'ed_pf_ratio': 'ED/PF'}.get(
        history.get('target'), 'Objective')
    position_text = ', '.join(f'{axis}={value:.3f}'
                              for axis, value in zip(('X', 'Y', 'Z'), best_position))
    best_text = f'Max {target_label}={best_value:.6g}\n{position_text}'
    fig = plt.figure(figsize=(15, 10))

    # (1) BO 每轮改进量 + xi 调度。
    ax1 = fig.add_subplot(2, 2, 1)
    deltas = np.diff(best_per_iter)
    iter_x = 1 + np.arange(len(deltas))
    bar_colors = np.where(deltas > 1e-12, 'forestgreen', 'lightgray')
    ax1.bar(iter_x, deltas, width=0.8, color=bar_colors, alpha=0.75,
            label='improvement per iter')
    ax1.axhline(0.0, color='black', lw=0.5, alpha=0.5)
    ax1.set_xlabel('BO Iteration')
    ax1.set_ylabel('Improvement over best')
    ax1.set_title(f'BO Efficiency (initial best={best_per_iter[0]:.4g})')
    ax1.grid(True, alpha=0.4, axis='y')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    xi_arr = history.get('xi_used', None)
    if xi_arr is not None and len(xi_arr) > 0:
        ax1b = ax1.twinx()
        ax1b.plot(iter_x, np.asarray(xi_arr), 'o--', ms=3, lw=1.4,
                  color='darkorange', alpha=0.85, label='xi (explore strength)')
        ax1b.set_ylabel('xi', color='darkorange')
        ax1b.tick_params(axis='y', colors='darkorange')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc='best')
    else:
        ax1.legend(fontsize=8, loc='best')

    # (2) 探索距离，含初始扫描。
    ax2 = fig.add_subplot(2, 2, 2)
    dtn_init = np.full(n_init, np.nan)
    for k in range(1, n_init):
        dtn_init[k] = float(np.min(cdist([X[k]], X[:k])))
    dtn_iter = np.asarray(history['distance_to_nearest'])
    dtn_all = np.concatenate([dtn_init, dtn_iter])
    if dist_log:
        ax2.semilogy(eval_idx, np.maximum(dtn_all, 1e-12),
                     '-', lw=1.0, color='gray', alpha=0.5)
        if n_init > 0:
            ax2.semilogy(eval_idx[:n_init], np.maximum(dtn_init, 1e-12),
                         'o', ms=3.5, color='lightgray', label=f'Initial scan ({n_init})')
        if n_init < N:
            ax2.semilogy(eval_idx[n_init:], dtn_iter, 'o-', ms=3,
                         color='steelblue', label='BO iterations')
    else:
        ax2.plot(eval_idx, dtn_all, '-', lw=1.0, color='gray', alpha=0.5)
        if n_init > 0:
            ax2.plot(eval_idx[:n_init], dtn_init, 'o', ms=3.5,
                     color='lightgray', label=f'Initial scan ({n_init})')
        if n_init < N:
            ax2.plot(eval_idx[n_init:], dtn_iter, 'o-', ms=3,
                     color='steelblue', label='BO iterations')
    if min_step is not None:
        ax2.axhline(min_step, color='red', ls='--', alpha=0.6, label=f'min_step={min_step}')
    if 0 < n_init < N:
        ax2.axvline(n_init - 0.5, color='gray', ls=':', alpha=0.6)
    ax2.set_xlabel('Evaluation Index')
    ax2.set_ylabel('Dist to Nearest')
    ax2.set_title('Exploration Distance (initial scan included)'
                  + (' (log)' if dist_log else ' (linear)'))
    ax2.grid(True, alpha=0.4)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax2.legend(fontsize=8)

    # (3) 样本分布。
    if dim >= 3:
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        sc = ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(sc, ax=ax3, shrink=0.7, pad=0.1, label='objective')
        if n_init > 0:
            ax3.scatter(X[:n_init, 0], X[:n_init, 1], X[:n_init, 2],
                        c='red', s=50, label=f'Initial ({n_init})')
        if n_init < N:
            ax3.scatter(X[n_init:, 0], X[n_init:, 1], X[n_init:, 2],
                        c='steelblue', s=20, alpha=0.5, label=f'BO ({N - n_init})')
        ax3.scatter(X[-5:, 0], X[-5:, 1], X[-5:, 2], c='blue', s=50, label='Final 5')
        ax3.scatter(*best_position, c='gold', s=200, marker='*', edgecolors='black',
                    label=f'Max {target_label}={best_value:.6g}')
        ax3.text2D(0.98, 0.98, best_text, transform=ax3.transAxes, fontsize=8,
                   ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white',
                                          edgecolor='darkgoldenrod', alpha=0.95))
        if true_optimum is not None:
            ax3.scatter(*np.asarray(true_optimum), c='green', s=200,
                        marker='X', edgecolors='black', label='True Optimum')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Sample Distribution (3D)')
        ax3.legend(fontsize=7, loc='upper left')
    else:
        ax3 = fig.add_subplot(2, 2, 3)
        sc = ax3.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30)
        plt.colorbar(sc, ax=ax3, label='objective')
        ax3.set_title(f'Sample Distribution ({dim}D)')
        ax3.grid(True, alpha=0.3)

    # (4) 全部评估点 + 累计最优。
    ax4 = fig.add_subplot(2, 2, 4)
    if n_init > 0:
        ax4.plot(eval_idx[:n_init], y[:n_init], 'o', ms=3.5,
                 color='lightgray', alpha=0.95, label=f'Initial scan ({n_init})')
    if n_init < N:
        ax4.plot(eval_idx[n_init:], y[n_init:], 'o-', ms=3,
                 color='steelblue', alpha=0.65, label=f'BO evaluations ({N - n_init})')
    ax4.plot(eval_idx, best_aligned, '-', lw=1.5, color='black', label='Best so far')
    ax4.scatter(best_idx, best_value, c='gold', s=160, marker='*',
                edgecolors='black', zorder=5, label='Maximum')
    ax4.annotate(best_text, xy=(best_idx, best_value),
                 xytext=(0.03, 0.97), textcoords='axes fraction',
                 ha='left', va='top', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor='darkgoldenrod', alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color='darkgoldenrod'), zorder=6)
    if 0 < n_init < N:
        ax4.axvline(n_init - 0.5, color='gray', ls=':', label='end of init')
    ax4.set_xlabel('Evaluation Index')
    ax4.set_ylabel('Objective Value')
    ax4.set_title(f'Convergence (best={best_aligned[-1]:.4g})')
    ax4.grid(True, alpha=0.4)
    ax4.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax4.legend(fontsize=8, loc='lower right')
    r_str = ""
    if true_optimum is not None:
        r_str = (f" | r_best={r_norm[np.argmax(y)]:.2f}"
                 f" | r_min={r_norm.min():.2f}")
    fig.suptitle(f'BO[EI]: Max {target_label}={best_value:.6g} | '
                 f'evals={N} (init={n_init} + BO={N - n_init}){r_str}\n'
                 f'Best position: {position_text}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if show:
        plt.show()
    return fig


def run_search(*, source=None, target=None, formula_model=None, data_model=None,
               fit_method=None, seed=None, bounds=None, show=True,
               fit_parameters=None, sampling_parameters=None, peaks=None, **bo_parameters):
    """建立模型、寻优、打印最佳单发结果并按需绘图，返回 history。"""
    source = ED_SOURCE if source is None else source
    target = TARGET if target is None else target
    formula_model = FORMULA_MODEL if formula_model is None else formula_model
    data_model = DATA_MODEL if data_model is None else data_model
    fit_method = FIT_METHOD if fit_method is None else fit_method
    seed = RANDOM_SEED if seed is None else seed
    bounds = BOUNDS if bounds is None else bounds
    sensor = set_ed_model(source, formula_model=formula_model, data_model=data_model,
                          fit_method=fit_method, fit_parameters=fit_parameters,
                          seed=seed, peaks=peaks, **(sampling_parameters or {}))
    parameters = {**BO_PARAMETERS, **bo_parameters}
    history = bayesian_optimization(make_objective(target), bounds,
                                     random_seed=seed, **parameters)
    best_idx = int(np.argmax(history['y']))
    if parameters.get('verbose', True):
        print("\n=== Final (recorded measurement) ===")
        print(f"best_x = {history['X'][best_idx]}")
        print(f"|best_x| = {np.linalg.norm(history['X'][best_idx]):.4f}")
        print(f"ED={history['ed'][best_idx]:.6f}, PF={history['pf'][best_idx]:.6f}, "
              f"ED/PF={history['ed_pf_ratio'][best_idx]:.6f}")
        print(f"Best {target} = {history['y'][best_idx]:.6f}")
        print(f"iters = {len(history['best_value']) - 1}")
    if show:
        true_optimum = None
        if source == 'formula':
            true_optimum = sensor.spatial_model.single_peak_position(bounds)
        plot_optimization_results(history, true_optimum=true_optimum,
                                  dist_log=False, min_step=2.0)
    return history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('fitted', 'formula'), default=ED_SOURCE)
    parser.add_argument('--target', choices=('ed', 'ed_pf_ratio'), default=TARGET)
    parser.add_argument('--formula-model', choices=tuple(FORMULA_CONFIGS), default=FORMULA_MODEL)
    parser.add_argument('--data-model', type=int, choices=tuple(MODEL_NAMES), default=DATA_MODEL)
    parser.add_argument('--fit-method', choices=('rbf', 'manual_gpr', 'linear', 'polynomial', 'svm'),
                        default=FIT_METHOD)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--max-iter', type=int, default=BO_PARAMETERS['max_iter'])
    parser.add_argument('--init-points', type=int, default=BO_PARAMETERS['init_points'])
    parser.add_argument('--sigma3-rel', type=float, default=BO_PARAMETERS['sigma3_rel'])
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    run_search(source=args.source, target=args.target, formula_model=args.formula_model,
               data_model=args.data_model, fit_method=args.fit_method, seed=args.seed,
               max_iter=args.max_iter, init_points=args.init_points,
               sigma3_rel=args.sigma3_rel, show=not args.no_plot)
