"""上层批测：5 个 DATA_MODEL × 2 个目标 × 4 种初始点数 × 多个种子。

上层控制迭代，复用 search.py 的模型、fit_gp() 和 get_next_sample()，不修改底层文件。
默认每个模型 10 个种子，共 400 次。统一使用 fitted + rbf，最多 200 轮。
同一模型的两种目标和四种初始点数复用同一组种子，便于配对比较。
收敛必须同时满足：连续 stop_no_improve 轮无显著改进，且最佳值相对参考值偏差
不超过 10%。偏差超标时继续迭代，达到预算仍未同时满足则不计作收敛。

运行：python3 batch_test.py
续跑：python3 batch_test.py --resume
预览：python3 batch_test.py --plan-only
短流程验证：python3 batch_test.py --repeats 1 --max-iter 2 --output-dir smoke_results
已有 CSV 生成 Excel 和图：python3 batch_test.py --report-only --output-dir batch_results
从 Excel 单独重画：python3 batch_test.py --plot-excel batch_results/batch_results.xlsx

结果：plan.csv、results.csv、summary.csv、comparison.csv、run_config.json，
以及 reference_optima.csv、iteration_results.csv、iterations/ 和 histories/。
每轮记录最佳值/位置，以及 X、Y、Z 各轴的绝对距离偏差和相对位置偏差。
各轴相对位置偏差 = |当前最佳坐标 - 参考坐标| / 该轴搜索跨度 × 100%。
值相对偏差 = |当前最佳值 - 参考值| / |参考值| × 100%；参考值为 0 且非精确命中时留空，
同时保留值的绝对偏差。value_bias_pct 保留正负号，可识别观测值高于参考值。
iteration=0 表示初始扫描。
总路程按实际采样中心顺序计算：第一个初始点 → 全部初始点/BO采样点 → 最终最佳点。
X/Y/Z 分别累计相邻位置的绝对位移，包含末次采样回到最佳点的一段。
comparison.csv 新增 total_distance_x_mean/y_mean/z_mean 三列，每列按 init_10/20/30/40
列出有效运行的平均总路程；results.csv 保留每次路程，summary.csv 保留分组数值。
已有结果可用 --report-only 从 histories/*.npz 补算，无需重新寻优。
默认参考值是无抖动 RBF 模型的多起点数值参考最优，不保证是全局解析最优。
--reference-file 可提供已知理论值，CSV 列为 data_model,target,value,x,y,z，
参考表使用当前编号（1–5）。沿用旧表时，删掉旧 2，把旧 3–6 改为 2–5。
每个模型分别填写 ed 和 ed_pf_ratio 两行。CSV 可直接用 Excel 打开。
批测后自动生成 batch_results.xlsx、convergence_ed.png、convergence_ed_pf_ratio.png。
Excel 导出/读取需要 openpyxl：python3 -m pip install openpyxl。
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import warnings

# 在导入 NumPy/SciPy/Matplotlib 前设置；已有环境设置仍优先。
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import search
from fitter import PF_BASELINE
from measurement_data import MODEL_NAMES


DATA_MODELS = tuple(MODEL_NAMES)
TARGETS = ('ed', 'ed_pf_ratio')
INIT_POINTS = (10, 20, 30, 40)
FIT_METHOD = 'rbf'
REPEATS = 10
BASE_SEED = 42
MAX_ITER = 200
STOP_NO_IMPROVE = search.BO_PARAMETERS.get('stop_no_improve', 10)
REFERENCE_GRID = 11
REFERENCE_STARTS = 32
REFERENCE_SEED = 2026
VALUE_TOLERANCE_PCT = 10.0

PLAN_FIELDS = ['case_id', 'data_model', 'model_name', 'target', 'init_points', 'repeat', 'seed']
POSITION_FIELDS = [f'{axis}_{metric}' for axis in ('x', 'y', 'z')
                   for metric in ('abs_distance', 'relative_position_error_pct')]
TRAVEL_FIELDS = [f'total_distance_{axis}' for axis in ('x', 'y', 'z')]
TRAVEL_MEAN_FIELDS = [field + '_mean' for field in TRAVEL_FIELDS]
RESULT_FIELDS = PLAN_FIELDS + [
    'status', 'accuracy_passed', 'value_tolerance_pct',
    'convergence_iters', 'bo_iters', 'n_init_actual', 'total_evaluations',
    'best_found_iter', 'best_value', 'best_x', 'best_y', 'best_z',
    'best_ed', 'best_pf', 'best_ed_pf_ratio',
    'reference_method', 'reference_value', 'reference_x', 'reference_y', 'reference_z',
    'value_abs_error', 'value_bias_pct', 'value_relative_error_pct',
    *POSITION_FIELDS,
    *TRAVEL_FIELDS,
    'elapsed_seconds', 'history_file', 'iterations_file', 'error',
]
ITERATION_FIELDS = PLAN_FIELDS + [
    'iteration', 'total_evaluations', 'best_value', 'best_x', 'best_y', 'best_z',
    'reference_method', 'reference_value', 'reference_x', 'reference_y', 'reference_z',
    'value_abs_error', 'value_bias_pct', 'value_relative_error_pct', 'accuracy_passed',
    *POSITION_FIELDS,
]
REFERENCE_FIELDS = ['data_model', 'model_name', 'target',
                    'reference_method', 'reference_value', 'reference_x', 'reference_y', 'reference_z']
SUMMARY_FIELDS = [
    'data_model', 'model_name', 'target', 'init_points', 'planned_runs',
    'attempted_runs', 'valid_runs', 'converged_runs', 'accuracy_passed_runs', 'max_iter_runs', 'error_runs',
    'convergence_rate', 'convergence_iters_mean', 'convergence_iters_std',
    'convergence_iters_median', 'convergence_iters_min', 'convergence_iters_max',
    'bo_iters_mean', 'total_evaluations_mean', 'best_value_mean', 'elapsed_seconds_mean',
    'value_relative_error_pct_mean',
] + [field + '_mean' for field in POSITION_FIELDS] + TRAVEL_MEAN_FIELDS


def make_plan(models=DATA_MODELS, repeats=REPEATS, base_seed=BASE_SEED):
    """按模型编号分配种子，默认模型 2 使用 52..61，模型 5 使用 82..91。"""
    plan = []
    for model in models:
        for repeat in range(repeats):
            seed = base_seed + (model - 1) * repeats + repeat
            for target in TARGETS:
                for n_init in INIT_POINTS:
                    plan.append(dict(
                        case_id=f'm{model}_{target}_n{n_init}_s{seed}',
                        data_model=model, model_name=MODEL_NAMES[model], target=target,
                        init_points=n_init, repeat=repeat + 1, seed=seed))
    return plan


def make_references(models, grid=REFERENCE_GRID, n_starts=REFERENCE_STARTS,
                    seed=REFERENCE_SEED, reference_file=None):
    """建立静态无抖动参考；RBF 数值参考与用户提供的理论值明确区分。"""
    lo, hi = np.asarray(search.BOUNDS, dtype=float).T
    if reference_file is not None:
        references = {}
        with Path(reference_file).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                model, target = int(row['data_model']), row['target']
                if model not in models or target not in TARGETS:
                    continue
                key = (model, target)
                if key in references:
                    raise ValueError(f'理论值重复：{key}')
                value = float(row['value'])
                position = np.array([float(row[axis]) for axis in ('x', 'y', 'z')])
                if not np.isfinite(value) or not np.all(np.isfinite(position)):
                    raise ValueError(f'理论值存在非有限数值：{key}')
                if np.any(position < lo) or np.any(position > hi):
                    raise ValueError(f'理论最优位置不在当前搜索边界内：{key}')
                references[key] = dict(
                    data_model=model, model_name=MODEL_NAMES[model], target=target,
                    reference_method='provided_theoretical', reference_value=value,
                    **{f'reference_{axis}': float(v)
                       for axis, v in zip(('x', 'y', 'z'), position)})
        missing = [(m, t) for m in models for t in TARGETS if (m, t) not in references]
        if missing:
            raise ValueError(f'理论值 CSV 缺少模型/目标：{missing}')
        return list(references.values())

    # 单位立方体优化避免不同位置尺度影响数值精度；使用独立随机流。
    unit_grid = np.stack(np.meshgrid(*([np.linspace(0, 1, grid)] * 3), indexing='ij'), axis=-1)
    unit_grid = unit_grid.reshape(-1, 3)
    references = []
    for model in models:
        spatial = search.EDFitter(model, FIT_METHOD)
        values = spatial.model.predict(lo + unit_grid * (hi - lo))
        best_idx = int(np.argmax(values))
        best_u, best_value = unit_grid[best_idx].copy(), float(values[best_idx])
        scale = max(float(np.max(np.abs(values))), 1e-12)
        n_grid_starts = min(8, n_starts)
        starts = unit_grid[np.argsort(values)[-n_grid_starts:]]
        rng = np.random.default_rng(seed + model)
        starts = np.vstack([starts, rng.random((n_starts - n_grid_starts, 3))])
        for start in starts:
            result = minimize(lambda u: -spatial.predict(lo + u * (hi - lo)) / scale,
                              start, method='L-BFGS-B', bounds=[(0., 1.)] * 3,
                              options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9})
            position = np.clip(result.x, 0., 1.)
            value = spatial.predict(lo + position * (hi - lo))
            if np.isfinite(value) and value > best_value:
                best_u, best_value = position.copy(), float(value)
        best_position = lo + best_u * (hi - lo)
        for target in TARGETS:
            references.append(dict(
                data_model=model, model_name=MODEL_NAMES[model], target=target,
                reference_method='numerical_rbf_no_jitter',
                reference_value=best_value if target == 'ed' else best_value / PF_BASELINE,
                **{f'reference_{axis}': float(v)
                   for axis, v in zip(('x', 'y', 'z'), best_position)}))
        print(f'参考模型 {model}：无抖动 RBF 数值参考 ED={best_value:.8g}，'
              f'XYZ={best_position}', flush=True)
    return references


def relative_value_error(value, reference):
    if reference == 0:
        return 0.0 if value == 0 else np.inf
    return abs(value - reference) / abs(reference)


def accuracy_passed(value, reference, tolerance_pct=VALUE_TOLERANCE_PCT):
    return relative_value_error(value, reference) * 100.0 <= tolerance_pct + 1e-9


def iteration_deviations(case, history, reference, tolerance_pct=VALUE_TOLERANCE_PCT):
    """值偏差用 |参考值| 归一化；XYZ 分别用各轴搜索跨度归一化。"""
    widths = np.ptp(np.asarray(search.BOUNDS, dtype=float), axis=1)
    reference_position = np.array([reference[f'reference_{axis}'] for axis in ('x', 'y', 'z')])
    reference_value = float(reference['reference_value'])
    rows = []
    for iteration, (value, position) in enumerate(zip(history['best_value'], history['best_x'])):
        value = float(value)
        distances = np.abs(np.asarray(position) - reference_position)
        relative = 100.0 * relative_value_error(value, reference_value)
        bias = (100.0 * (value - reference_value) / abs(reference_value)
                if reference_value != 0 else 0.0 if value == 0 else '')
        rows.append(dict(
            **case, iteration=iteration, total_evaluations=int(history['n_init']) + iteration,
            best_value=value, **{f'best_{axis}': float(v)
                               for axis, v in zip(('x', 'y', 'z'), position)},
            **{name: reference[name] for name in REFERENCE_FIELDS[3:]},
            value_abs_error=abs(value - reference_value), value_bias_pct=bias,
            value_relative_error_pct=relative if np.isfinite(relative) else '',
            accuracy_passed=accuracy_passed(value, reference_value, tolerance_pct),
            **{f'{axis}_abs_distance': float(v)
               for axis, v in zip(('x', 'y', 'z'), distances)},
            **{f'{axis}_relative_position_error_pct': float(v)
               for axis, v in zip(('x', 'y', 'z'), 100.0 * distances / widths)},
        ))
    return rows


def get_convergence_iteration(history, patience, reference_value,
                              tolerance_pct=VALUE_TOLERANCE_PCT):
    """复核无改进和相对偏差两个条件，区分最后一轮收敛与单纯达到上限。

    使用每轮实际保存的 sigma_abs_used，不重新量测，也不另设阈值。
    返回 BO 轮数（不含初始扫描）；未触发时返回 None。
    """
    y = np.asarray(history['y'], dtype=float)
    best = np.asarray(history['best_value'], dtype=float)
    sigma = np.asarray(history['sigma_abs_used'], dtype=float)
    running_min = np.minimum.accumulate(y)
    running_max = np.maximum.accumulate(y)
    n_init = int(history['n_init'])
    last_best, last_iter = -np.inf, 0
    for i in range(len(best) - 1):
        end = n_init + i - 1  # 该轮新量测加入之前的最后一个样本。
        y_range = float(running_max[end] - running_min[end]) + 1e-12
        robust_best = float(best[i]) - 3.0 * float(sigma[i])
        tolerance = max(1e-6 * y_range, 3.0 * float(sigma[i]))
        if robust_best > last_best + tolerance:
            last_best, last_iter = robust_best, i
        improved_now = float(best[i + 1]) - 3.0 * float(sigma[i]) > last_best + tolerance
        if (not improved_now and i - last_iter >= patience
                and accuracy_passed(float(best[i + 1]), reference_value, tolerance_pct)):
            return i + 1
    return None


def run_search_with_accuracy(case, reference, max_iter, patience,
                             tolerance_pct=VALUE_TOLERANCE_PCT, verbose=False):
    """复用原有数值步骤，仅由上层实施联合停止，超出 10% 时不会提前结束。"""
    parameters = search.BO_PARAMETERS
    search.set_ed_model('fitted', data_model=case['data_model'], fit_method=FIT_METHOD,
                        seed=case['seed'])
    objective = search.make_objective(case['target'])
    np.random.seed(case['seed'])
    bounds = np.asarray(search.BOUNDS, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = float(np.mean(hi - lo))
    ls_bounds = parameters.get('ls_bounds')
    if ls_bounds is None:
        ls_bounds = (max(span / 100., 1e-3), max(span / 25., 1e-2))
    sigma3_rel = max(float(parameters.get('sigma3_rel', 0.)), 0.)
    use_white = sigma3_rel > 0
    noise_bounds = (1e-6, 1e-2) if use_white else (1e-10, 1e-2)
    X = np.asarray(search.get_initial_points(case['init_points'], *bounds), dtype=float)
    X = np.atleast_2d(X)
    X = X[np.all((X >= lo) & (X <= hi), axis=1)]
    measurements = {key: [] for key in ('ed', 'pf', 'ed_pf_ratio')}

    def evaluate(position):
        value = float(objective(position))
        for key in measurements:
            measurements[key].append(float(objective.last_measurement[key]))
        return value

    y = np.array([evaluate(position) for position in X])
    y_range0 = float(np.ptp(y)) + 1e-12
    sigma_a = sigma3_rel / 3.0 * y_range0
    xi_default_init = max(.01 * y_range0, .5 * sigma_a)
    xi_default_final = max(.001 * y_range0, .1 * sigma_a)
    xi_init, xi_final = parameters.get('xi_init'), parameters.get('xi_final')
    if xi_init is None and xi_final is None:
        xi_init, xi_final = xi_default_init, xi_default_final
    elif xi_init is None:
        xi_init = max(float(xi_final), xi_default_init)
    elif xi_final is None:
        xi_final = min(float(xi_init), xi_default_final)
    xi_init, xi_final = max(float(xi_init), 0.), max(float(xi_final), 0.)
    if xi_init <= 1e-12:
        xi_init = xi_final = 0.
    elif xi_final > xi_init:
        warnings.warn('xi_final > xi_init，已设为 xi_init', RuntimeWarning)
        xi_final = xi_init
    best_idx = int(np.argmax(y))
    history = dict(X=X.copy(), y=y.copy(), best_value=[float(y[best_idx])],
                   best_x=[X[best_idx].copy()], time=[0.], n_init=len(X),
                   target=case['target'], distance_to_nearest=[], noise_level=[],
                   sigma_abs_used=[], xi_used=[], **measurements)
    start_time = time.time()
    last_best, last_iter, gpr = -np.inf, 0, None
    stop_reason = 'max_iter'
    for i in range(max_iter):
        y_range, y_std = float(np.ptp(y)) + 1e-12, float(y.std()) + 1e-12
        alpha_eff = max((sigma_a / y_std) ** 2, 1e-6) if use_white else 1e-10
        sigma_used = sigma_a
        if gpr is not None and hasattr(gpr.kernel, 'k2'):
            learned = float(np.sqrt(gpr.kernel.k2.noise_level)) * gpr.y_std
            sigma_used = max(learned, .5 * sigma_a)
        robust_best = float(np.max(y)) - 3.0 * sigma_used
        tolerance = max(1e-6 * y_range, 3.0 * sigma_used)
        if robust_best > last_best + tolerance:
            last_best, last_iter = robust_best, i
        t = i / max(1, max_iter - 1)
        if xi_init <= 1e-12:
            xi = 0.
        elif xi_final == 0.:
            xi = xi_init * (1. - t)
        else:
            xi = max(xi_init * ((xi_final / xi_init) ** t), xi_final)
        gpr = search.fit_gp(X, y, alpha_eff, float(np.clip(span / 2., *ls_bounds)),
                            ls_bounds, n_restarts=10, use_white=use_white,
                            noise_bounds=noise_bounds)
        next_x, distance = search.get_next_sample(
            X, y, bounds, gpr, (robust_best - gpr.y_mean) / gpr.y_std, xi / gpr.y_std,
            r_frac_min=parameters.get('r_frac_min', .05),
            global_prob=parameters.get('global_prob', 0.),
            n_restarts_ei=parameters.get('n_restarts_ei', 20),
            min_step_i=2. * (1. - t) ** 2, i=i, max_iter=max_iter, span=span, verbose=verbose)
        next_y = evaluate(next_x)
        X, y = np.vstack([X, next_x[None, :]]), np.append(y, next_y)
        best_idx = int(np.argmax(y))
        best_value = float(y[best_idx])
        history['best_value'].append(best_value)
        history['best_x'].append(X[best_idx].copy())
        history['time'].append(time.time() - start_time)
        history['distance_to_nearest'].append(distance)
        history['noise_level'].append(float(gpr.kernel.k2.noise_level)
                                      if hasattr(gpr.kernel, 'k2') else 0.)
        history['sigma_abs_used'].append(sigma_used)
        history['xi_used'].append(xi)
        within = accuracy_passed(best_value, reference['reference_value'], tolerance_pct)
        if verbose:
            print(f"Iter {i + 1}/{max_iter}: ED={measurements['ed'][-1]:.6g}, "
                  f"PF={measurements['pf'][-1]:.6g}, ED/PF={measurements['ed_pf_ratio'][-1]:.6g}, "
                  f"Best={best_value:.6g}, 偏差={relative_value_error(best_value, reference['reference_value']):.2%}",
                  flush=True)
        improved_now = best_value - 3.0 * sigma_used > last_best + tolerance
        if not improved_now and i - last_iter >= patience and within:
            stop_reason = 'converged'
            break
    history['X'], history['y'] = X, y
    for key in ('best_value', 'best_x', 'time', 'distance_to_nearest', 'noise_level',
                'sigma_abs_used', 'xi_used', 'ed', 'pf', 'ed_pf_ratio'):
        history[key] = np.asarray(history[key])
    history['stop_reason'] = stop_reason
    history['value_tolerance_pct'] = tolerance_pct
    return history


def write_csv(path, fields, rows):
    """每次原子替换，避免中断时破坏已经完成的明细。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def calculate_total_distances(history):
    """设定中心逐轴累计路程，最后返回最终最佳点；不计首点之前的运动。"""
    positions = np.asarray(history['X'], dtype=float)
    final_best = np.asarray(history['best_x'][-1], dtype=float)
    distances = np.abs(np.diff(positions, axis=0)).sum(axis=0)
    distances += np.abs(positions[-1] - final_best)
    return dict(zip(TRAVEL_FIELDS, (float(value) for value in distances)))


def backfill_total_distances(directory, records):
    """从已保存的真实采样轨迹补算，缺少轨迹时保留空值。"""
    missing = []
    for row in records.values():
        if row['status'] == 'error' or all(row.get(field) not in ('', None) for field in TRAVEL_FIELDS):
            continue
        history_file = row.get('history_file')
        if not history_file or not (directory / history_file).exists():
            missing.append(row['case_id'])
            continue
        with np.load(directory / history_file, allow_pickle=False) as history:
            row.update(calculate_total_distances(history))
    if missing:
        print(f'{len(missing)} 条结果缺少原始 history，累计路程留空；请补齐 histories 文件后重建报告。',
              flush=True)
    return missing


def summarize(plan, records):
    """仅对已触发停止条件的运行统计收敛次数；上限与失败分别计数。"""
    groups = {}
    for case in plan:
        key = (case['data_model'], case['target'], case['init_points'])
        # 历史计划自带工况名称，重建报告时仍可读取已下线的模型。
        name = case.get('model_name') or MODEL_NAMES[case['data_model']]
        group = groups.setdefault(key, {'model_name': name, 'planned': 0, 'rows': []})
        group['planned'] += 1
        if case['case_id'] in records:
            group['rows'].append(records[case['case_id']])
    summary = []
    for (model, target, n_init), group in groups.items():
        rows = group['rows']
        valid = [row for row in rows if row['status'] != 'error']
        converged = [row for row in valid if row['status'] == 'converged']
        iterations = [int(row['convergence_iters']) for row in converged]
        summary.append(dict(
            data_model=model, model_name=group['model_name'], target=target,
            init_points=n_init, planned_runs=group['planned'], attempted_runs=len(rows),
            valid_runs=len(valid), converged_runs=len(converged),
            accuracy_passed_runs=sum(str(row.get('accuracy_passed', '')).lower() in ('true', '1')
                                     for row in valid),
            max_iter_runs=sum(row['status'] == 'max_iter' for row in valid),
            error_runs=len(rows) - len(valid),
            convergence_rate=len(converged) / len(valid) if valid else '',
            convergence_iters_mean=statistics.mean(iterations) if iterations else '',
            convergence_iters_std=(statistics.stdev(iterations) if len(iterations) > 1
                                   else 0.0 if iterations else ''),
            convergence_iters_median=statistics.median(iterations) if iterations else '',
            convergence_iters_min=min(iterations) if iterations else '',
            convergence_iters_max=max(iterations) if iterations else '',
            **{column + '_mean': statistics.mean(float(row[column]) for row in valid
                                                 if row.get(column) not in ('', None))
               if any(row.get(column) not in ('', None) for row in valid) else ''
               for column in ('bo_iters', 'total_evaluations', 'best_value', 'elapsed_seconds',
                              'value_relative_error_pct', *POSITION_FIELDS, *TRAVEL_FIELDS)},
        ))
    return summary


def write_reports(directory, plan, records):
    ordered = [records[case['case_id']] for case in plan if case['case_id'] in records]
    write_csv(directory / 'results.csv', RESULT_FIELDS, ordered)
    summary = summarize(plan, records)
    write_csv(directory / 'summary.csv', SUMMARY_FIELDS, summary)
    comparison = {}
    for row in summary:
        key = (row['data_model'], row['target'])
        entry = comparison.setdefault(key, dict(
            data_model=row['data_model'], model_name=row['model_name'], target=row['target']))
        mean = row['convergence_iters_mean']
        value = (f"{mean:.2f} +/- {row['convergence_iters_std']:.2f}"
                 if mean != '' else 'N/A')
        entry[f"init_{row['init_points']}"] = (
            f"{value} ({row['converged_runs']}/{row['valid_runs']})")
        for field in TRAVEL_MEAN_FIELDS:
            mean_distance = row[field]
            distance_text = f'{mean_distance:.2f}' if mean_distance != '' else 'N/A'
            entry.setdefault(field, []).append(f"init_{row['init_points']}={distance_text}")
    for entry in comparison.values():
        for field in TRAVEL_MEAN_FIELDS:
            entry[field] = '; '.join(entry[field])
    write_csv(directory / 'comparison.csv',
              ['data_model', 'model_name', 'target'] + [f'init_{n}' for n in INIT_POINTS]
              + TRAVEL_MEAN_FIELDS,
              comparison.values())
    return summary


def refresh_travel_reports(output_dir):
    """兼容既有批测目录：补齐单次路程并重建比较表，不重新调用寻优。"""
    directory = Path(output_dir).resolve()
    if not (directory / 'results.csv').exists():
        return
    with (directory / 'results.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = {row['case_id']: row for row in csv.DictReader(stream)}
    plan_file = directory / 'plan.csv'
    if plan_file.exists():
        with plan_file.open(encoding='utf-8-sig', newline='') as stream:
            plan = list(csv.DictReader(stream))
    else:
        plan = [{field: row[field] for field in PLAN_FIELDS} for row in records.values()]
    for case in plan:
        for field in ('data_model', 'init_points', 'repeat', 'seed'):
            case[field] = int(case[field])
    backfill_total_distances(directory, records)
    write_reports(directory, plan, records)


def merge_iterations(directory, plan, records):
    """合并每次运行的逐轮 CSV；续跑后从明细重建，避免重复追加。"""
    destination = directory / 'iteration_results.csv'
    temporary = destination.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=ITERATION_FIELDS)
        writer.writeheader()
        for case in plan:
            row = records.get(case['case_id'])
            if row is None or row['status'] == 'error':
                continue
            with (directory / row['iterations_file']).open(encoding='utf-8-sig', newline='') as source:
                writer.writerows(csv.DictReader(source))
    temporary.replace(destination)


def _excel_support():
    try:
        import openpyxl
        return openpyxl
    except ImportError as exc:
        raise RuntimeError('Excel 结果需要 openpyxl，请先运行：python3 -m pip install openpyxl') from exc


def export_excel_results(output_dir):
    """把批测 CSV 汇总为真实 xlsx，数值和百分比保留为可计算单元格。"""
    openpyxl = _excel_support()
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    directory = Path(output_dir).resolve()
    if not (directory / 'summary.csv').exists():
        raise FileNotFoundError(f"未找到批测汇总：{directory / 'summary.csv'}")
    workbook = openpyxl.Workbook(write_only=True)

    def add_sheet(name, fields):
        sheet = workbook.create_sheet(name)
        sheet.freeze_panes = 'A2'
        sheet.row_dimensions[1].height = 42
        for i, field in enumerate(fields, start=1):
            sheet.column_dimensions[get_column_letter(i)].width = min(38, max(14, len(field) + 2))
        header = []
        for field in fields:
            cell = WriteOnlyCell(sheet, value=field)
            cell.font = Font(name='Calibri', bold=True, color='FFFFFF', size=10)
            cell.fill = PatternFill('solid', fgColor='17365D')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
            header.append(cell)
        sheet.append(header)
        return sheet

    def excel_value(raw, field):
        if raw == '' or raw is None:
            return None, 'General'
        if str(raw) in ('True', 'False'):
            return str(raw) == 'True', 'General'
        try:
            number = float(raw)
            if not np.isfinite(number):
                return str(raw), 'General'
        except (ValueError, TypeError):
            return str(raw), 'General'
        if field == 'convergence_rate':
            return number, '0.0%'
        if field.endswith('_pct') or field.endswith('_pct_mean'):
            return number / 100., '0.00%'
        if field.endswith('_runs') or field in (
                'data_model', 'init_points', 'repeat', 'seed', 'iteration',
                'convergence_iters', 'bo_iters', 'n_init_actual', 'total_evaluations', 'best_found_iter'):
            return int(number), '#,##0'
        if field.startswith('convergence_iters_') or field.endswith('seconds_mean'):
            return number, '0.00'
        if any(word in field for word in ('best_', 'reference_', 'distance', 'error', 'deviation')):
            return number, '0.000000'
        return (int(number), '#,##0') if number.is_integer() else (number, '0.00')

    for name, filename in (
            ('Summary', 'summary.csv'), ('Comparison', 'comparison.csv'),
            ('Runs', 'results.csv'), ('Iterations', 'iteration_results.csv'),
            ('Reference', 'reference_optima.csv'), ('Plan', 'plan.csv')):
        source_path = directory / filename
        if not source_path.exists():
            continue
        with source_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            sheet = add_sheet(name, fields)
            count, part = 0, 1
            for row in reader:
                if count == 1_048_575:
                    sheet.auto_filter.ref = f'A1:{get_column_letter(len(fields))}{count + 1}'
                    part += 1
                    sheet, count = add_sheet(f'{name}_{part}', fields), 0
                cells = []
                for field in fields:
                    value, number_format = excel_value(row[field], field)
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.number_format = number_format
                    if name == 'Comparison':
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
                    cells.append(cell)
                if name == 'Comparison':
                    sheet.row_dimensions[count + 2].height = 60
                sheet.append(cells)
                count += 1
            sheet.auto_filter.ref = f'A1:{get_column_letter(len(fields))}{count + 1}'
    config_path = directory / 'run_config.json'
    if config_path.exists():
        sheet = add_sheet('Parameters', ['parameter', 'value'])
        sheet.column_dimensions['A'].width = 30
        sheet.column_dimensions['B'].width = 70
        for key, value in json.loads(config_path.read_text(encoding='utf-8')).items():
            sheet.append([key, json.dumps(value, ensure_ascii=False)
                          if isinstance(value, (dict, list)) else value])
    destination = directory / 'batch_results.xlsx'
    temporary = directory / 'batch_results.tmp.xlsx'
    workbook.save(temporary)
    temporary.replace(destination)
    return destination


def plot_convergence_results(excel_file, output_dir=None, show=False):
    """从 Excel 的 Summary 表读取统计，ED/ED-PF 各一幅图，模型颜色固定。"""
    workbook = _excel_support().load_workbook(excel_file, read_only=True, data_only=True)
    try:
        data = workbook['Summary'].iter_rows(values_only=True)
        fields = next(data)
        rows = [dict(zip(fields, row)) for row in data]
        settings = dict(workbook['Parameters'].iter_rows(min_row=2, max_col=2, values_only=True)) \
            if 'Parameters' in workbook.sheetnames else {}
    finally:
        workbook.close()
    directory = Path(output_dir) if output_dir is not None else Path(excel_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    colors = {1: '#0072B2', 2: '#E69F00', 3: '#009E73',
              4: '#D55E00', 5: '#CC79A7', 6: '#56B4E9'}
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v', 6: 'P'}
    threshold = settings.get('value_tolerance_pct')
    criterion = (f'value error <= {float(threshold):g}%'
                 if threshold is not None else 'criterion in saved legacy results')
    outputs = []
    for target in TARGETS:
        selected = [row for row in rows if row['target'] == target]
        figure, axis = plt.subplots(figsize=(10.5, 6.5))
        any_converged = False
        for model in sorted({int(row['data_model']) for row in selected}):
            lookup = {int(row['init_points']): row for row in selected if int(row['data_model']) == model}
            means, deviations = [], []
            converged = sum(int(row['converged_runs']) for row in lookup.values())
            valid = sum(int(row['valid_runs']) for row in lookup.values())
            for n in INIT_POINTS:
                row = lookup.get(n)
                exists = row is not None and int(row['converged_runs']) > 0
                means.append(float(row['convergence_iters_mean']) if exists else np.nan)
                deviations.append(float(row['convergence_iters_std'] or 0.) if exists else 0.)
            any_converged = any_converged or np.any(np.isfinite(means))
            label = f'DATA_MODEL={model} ({converged}/{valid})'
            axis.errorbar(INIT_POINTS, means, yerr=deviations, color=colors[model],
                          marker=markers[model], capsize=3, linewidth=1.6, label=label)
        axis.set(xlabel='init_points', ylabel='Mean convergence iterations (BO only)',
                 title=f"{'ED' if target == 'ed' else 'ED/PF'} optimization")
        axis.set_xticks(INIT_POINTS)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        axis.margins(y=.15)
        axis.set_ylim(bottom=0)
        if not any_converged:
            axis.set_ylim(0, 1)
            axis.text(.5, .5, 'No runs satisfy the convergence criteria.',
                      transform=axis.transAxes, ha='center', va='center')
        axis.grid(True, alpha=.25)
        if selected:
            axis.legend(fontsize=8, title='Converged / valid (all init_points)',
                        title_fontsize=9, loc='best')
        figure.text(.5, .02, f'Converged runs only; {criterion}. '
                    'Error bars: sample SD when n >= 2.\n'
                    'Missing points: no converged runs. Counts in legend combine all init_points.',
                    ha='center', fontsize=8)
        figure.tight_layout(rect=[0, .07, 1, 1])
        destination = directory / f'convergence_{target}.png'
        figure.savefig(destination, dpi=180)
        outputs.append(destination)
        if show:
            plt.show()
        plt.close(figure)
    return outputs


def generate_excel_and_plots(output_dir):
    refresh_travel_reports(output_dir)
    excel_file = export_excel_results(output_dir)
    plots = plot_convergence_results(excel_file)
    print(f'Excel：{excel_file}\n绘图：' + ', '.join(str(path) for path in plots), flush=True)
    return excel_file, plots


def make_config(models, repeats, base_seed, max_iter, patience,
                reference_grid, reference_starts, reference_seed, reference_file,
                value_tolerance_pct):
    root = Path(search.__file__).resolve().parent
    sources = [root / name for name in (
        'search.py', 'GPR.py', 'kernel.py', 'EI.py', 'LHS.py',
        'fitter.py', 'formula_models.py', 'measurement_data.py')]
    sources.extend(sorted((root / 'fit_models').glob('*.py')))
    config = dict(
        format_version=4, models=list(models), targets=list(TARGETS),
        init_points=list(INIT_POINTS), repeats=repeats, base_seed=base_seed,
        source='fitted', fit_method=FIT_METHOD, bounds=np.asarray(search.BOUNDS).tolist(),
        bo_parameters={**search.BO_PARAMETERS, 'max_iter': max_iter,
                       'stop_no_improve': patience},
        value_tolerance_pct=value_tolerance_pct,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        reference_grid=reference_grid, reference_starts=reference_starts,
        reference_seed=reference_seed,
        reference_file_sha256=(hashlib.sha256(Path(reference_file).read_bytes()).hexdigest()
                               if reference_file is not None else None),
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sources})
    return json.loads(json.dumps(config))  # 统一元组等可序列化类型，供续跑比较。


def run_batch(*, models=DATA_MODELS, repeats=REPEATS, base_seed=BASE_SEED,
              max_iter=MAX_ITER, patience=STOP_NO_IMPROVE,
              output_dir='batch_results', resume=False, verbose=False, plan_only=False,
              reference_grid=REFERENCE_GRID, reference_starts=REFERENCE_STARTS,
              reference_seed=REFERENCE_SEED, reference_file=None,
              value_tolerance_pct=VALUE_TOLERANCE_PCT):
    if repeats < 1 or max_iter < 1 or patience < 1:
        raise ValueError('repeats、max_iter、stop_no_improve 必须大于 0')
    if not 0 <= base_seed <= 2**32 - max(DATA_MODELS) * repeats:
        raise ValueError('种子范围必须适用于 np.random.seed')
    if reference_grid < 2 or reference_starts < 1:
        raise ValueError('reference_grid 至少为 2，reference_starts 至少为 1')
    if not np.isfinite(value_tolerance_pct) or value_tolerance_pct < 0:
        raise ValueError('value_tolerance_pct 必须是非负有限数值')
    plan = make_plan(models, repeats, base_seed)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = make_config(models, repeats, base_seed, max_iter, patience,
                         reference_grid, reference_starts, reference_seed, reference_file,
                         value_tolerance_pct)
    config_file = directory / 'run_config.json'
    results_file = directory / 'results.csv'
    if results_file.exists() and not resume:
        raise FileExistsError(f'{results_file} 已存在；用 --resume 续跑或换 --output-dir')
    if results_file.exists() and not config_file.exists():
        raise ValueError('已有结果缺少 run_config.json，无法核对续跑配置')
    if config_file.exists():
        previous = json.loads(config_file.read_text(encoding='utf-8'))
        if previous != config:
            raise ValueError('参数或算法源文件已改变，请使用新的 --output-dir，避免混合结果')
    else:
        config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(directory / 'plan.csv', PLAN_FIELDS, plan)
    print(f'测试计划：{len(models)} 模型 × 2 目标 × 4 初始点数 × {repeats} 种子'
          f' = {len(plan)} 次；RBF；上限 {max_iter} 轮；无显著改进 {patience} 轮，'
          f'且相对偏差 ≤ {value_tolerance_pct:g}% 才停止。', flush=True)
    print(f'输出目录：{directory}', flush=True)
    if plan_only:
        return []
    _excel_support()  # 批测前检查 Excel 依赖，避免运行结束后才发现缺少依赖。

    reference_path = directory / 'reference_optima.csv'
    if reference_path.exists():
        with reference_path.open(encoding='utf-8-sig', newline='') as stream:
            references = list(csv.DictReader(stream))
        for row in references:
            row['data_model'] = int(row['data_model'])
            for key in ('reference_value', 'reference_x', 'reference_y', 'reference_z'):
                row[key] = float(row[key])
    else:
        references = make_references(models, reference_grid, reference_starts,
                                      reference_seed, reference_file)
        write_csv(reference_path, REFERENCE_FIELDS, references)
    references = {(row['data_model'], row['target']): row for row in references}
    records = {}
    if resume and results_file.exists():
        with results_file.open(encoding='utf-8-sig', newline='') as stream:
            records = {row['case_id']: row for row in csv.DictReader(stream)}
        backfill_total_distances(directory, records)
    history_dir = directory / 'histories'
    history_dir.mkdir(exist_ok=True)
    iterations_dir = directory / 'iterations'
    iterations_dir.mkdir(exist_ok=True)
    for index, case in enumerate(plan, start=1):
        case_id = case['case_id']
        if case_id in records and records[case_id]['status'] != 'error':
            continue
        row = {field: '' for field in RESULT_FIELDS}
        row.update(case)
        start = time.perf_counter()
        try:
            reference = references[(case['data_model'], case['target'])]
            history = run_search_with_accuracy(case, reference, max_iter, patience,
                                                value_tolerance_pct, verbose)
            convergence = get_convergence_iteration(history, patience,
                                                     reference['reference_value'], value_tolerance_pct)
            best_idx = int(np.argmax(history['y']))
            n_init_actual = int(history['n_init'])
            row.update(
                status='converged' if convergence is not None else 'max_iter',
                accuracy_passed=accuracy_passed(float(history['y'][best_idx]),
                                                reference['reference_value'], value_tolerance_pct),
                value_tolerance_pct=value_tolerance_pct,
                convergence_iters=convergence if convergence is not None else '',
                bo_iters=len(history['best_value']) - 1, n_init_actual=n_init_actual,
                total_evaluations=len(history['y']),
                best_found_iter=max(0, best_idx - n_init_actual + 1),
                best_value=float(history['y'][best_idx]),
                **{f'best_{axis}': float(value)
                   for axis, value in zip(('x', 'y', 'z'), history['X'][best_idx])},
                **{f'best_{name}': float(history[name][best_idx])
                   for name in ('ed', 'pf', 'ed_pf_ratio')},
                history_file=f'histories/{case_id}.npz',
                iterations_file=f'iterations/{case_id}.csv')
            row.update(calculate_total_distances(history))
            deviations = iteration_deviations(
                case, history, reference, value_tolerance_pct)
            final = deviations[-1]
            row.update({key: final[key] for key in (
                'reference_method', 'reference_value', 'reference_x', 'reference_y', 'reference_z',
                'value_abs_error', 'value_bias_pct', 'value_relative_error_pct',
                *POSITION_FIELDS)})
            np.savez_compressed(history_dir / f'{case_id}.npz', **history)
            write_csv(iterations_dir / f'{case_id}.csv', ITERATION_FIELDS, deviations)
        except Exception as exc:
            row.update(status='error', error=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter() - start
        records[case_id] = row
        write_reports(directory, plan, records)
        print(f"[{index}/{len(plan)}] {case_id}: {row['status']} | "
              f"BO={row['bo_iters']}, 收敛轮数={row['convergence_iters']} | "
              f"Best={row['best_value']}, ED={row['best_ed']}, PF={row['best_pf']}, "
              f"ED/PF={row['best_ed_pf_ratio']}"
              + (f" | {row['error']}" if row['error'] else ''), flush=True)
    summary = write_reports(directory, plan, records)
    merge_iterations(directory, plan, records)
    generate_excel_and_plots(directory)
    errors = sum(row['error_runs'] for row in summary)
    print(f'批测完成：{len(records)} 条结果，失败 {errors} 次。'
          'comparison.csv 为均值 +/- 样本标准差（收敛次数/有效运行数）；'
          'summary.csv 含上限次数、失败次数及统计明细。', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', type=int, choices=DATA_MODELS, default=list(DATA_MODELS))
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--base-seed', type=int, default=BASE_SEED)
    parser.add_argument('--max-iter', type=int, default=MAX_ITER)
    parser.add_argument('--stop-no-improve', type=int, default=STOP_NO_IMPROVE)
    parser.add_argument('--output-dir', default='batch_results')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='打印每轮 ED、PF、ED/PF')
    parser.add_argument('--plan-only', action='store_true', help='只生成测试计划，不运行寻优')
    parser.add_argument('--reference-grid', type=int, default=REFERENCE_GRID)
    parser.add_argument('--reference-starts', type=int, default=REFERENCE_STARTS)
    parser.add_argument('--reference-seed', type=int, default=REFERENCE_SEED)
    parser.add_argument('--reference-file', help='已知理论值 CSV：data_model,target,value,x,y,z')
    parser.add_argument('--value-tolerance-pct', type=float, default=VALUE_TOLERANCE_PCT,
                        help='收敛时允许的最佳值相对偏差百分数，默认 10')
    report_mode = parser.add_mutually_exclusive_group()
    report_mode.add_argument('--report-only', action='store_true', help='已有 CSV 生成 Excel 和图，不重新批测')
    report_mode.add_argument('--plot-excel', help='只从指定 xlsx 提取 Summary 数据并绘图')
    args = parser.parse_args()
    if args.plot_excel:
        for path in plot_convergence_results(args.plot_excel):
            print(path)
        return 0
    if args.report_only:
        generate_excel_and_plots(args.output_dir)
        return 0
    summary = run_batch(models=tuple(dict.fromkeys(args.models)), repeats=args.repeats,
                        base_seed=args.base_seed, max_iter=args.max_iter,
                        patience=args.stop_no_improve, output_dir=args.output_dir,
                        resume=args.resume, verbose=args.verbose, plan_only=args.plan_only,
                        reference_grid=args.reference_grid, reference_starts=args.reference_starts,
                        reference_seed=args.reference_seed, reference_file=args.reference_file,
                        value_tolerance_pct=args.value_tolerance_pct)
    return int(any(row['error_runs'] for row in summary))


if __name__ == '__main__':
    raise SystemExit(main())
