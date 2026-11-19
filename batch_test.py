"""上层批测：6 个 DATA_MODEL × 2 个目标 × 4 种初始点数 × 多个种子。

只调用现有 search.run_search()，不修改或替换寻优函数。
默认每个模型 10 个种子，共 480 次；统一使用 fitted + rbf，最多 200 轮。
同一模型的两种目标和四种初始点数复用同一组种子，便于配对比较。
收敛次数指现有“连续 stop_no_improve 轮无显著改进”条件触发的轮次，
包括等待确认的轮数；不代表已证明达到全局最优。到达预算不计作收敛。

运行：python3 batch_test.py
续跑：python3 batch_test.py --resume
预览：python3 batch_test.py --plan-only
短流程验证：python3 batch_test.py --repeats 1 --max-iter 2 --output-dir smoke_results

结果：plan.csv、results.csv、summary.csv、comparison.csv、run_config.json，
以及 reference_optima.csv、iteration_results.csv、iterations/ 和 histories/。
每轮记录最佳值/位置，以及 X、Y、Z 各轴的绝对距离偏差和相对位置偏差。
各轴相对位置偏差 = |当前最佳坐标 - 参考坐标| / 该轴搜索跨度 × 100%。
值相对偏差 = |当前最佳值 - 参考值| / |参考值| × 100%；参考值为 0 时留空，
同时保留值的绝对偏差。value_bias_pct 保留正负号，可识别观测值高于参考值。
iteration=0 表示初始扫描。
默认参考值是无抖动 RBF 模型的多起点数值参考最优，不保证是全局解析最优。
--reference-file 可提供已知理论值，CSV 列为 data_model,target,value,x,y,z，
每个模型分别填写 ed 和 ed_pf_ratio 两行。CSV 可直接用 Excel 打开。
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

# 在导入 NumPy/SciPy/Matplotlib 前设置；已有环境设置仍优先。
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import scipy
from scipy.optimize import minimize
import search
from fitter import PF_BASELINE
from measurement_data import MODEL_NAMES


DATA_MODELS = (1, 2, 3, 4, 5, 6)
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

PLAN_FIELDS = ['case_id', 'data_model', 'model_name', 'target', 'init_points', 'repeat', 'seed']
POSITION_FIELDS = [f'{axis}_{metric}' for axis in ('x', 'y', 'z')
                   for metric in ('abs_distance', 'relative_position_error_pct')]
RESULT_FIELDS = PLAN_FIELDS + [
    'status', 'convergence_iters', 'bo_iters', 'n_init_actual', 'total_evaluations',
    'best_found_iter', 'best_value', 'best_x', 'best_y', 'best_z',
    'best_ed', 'best_pf', 'best_ed_pf_ratio',
    'reference_method', 'reference_value', 'reference_x', 'reference_y', 'reference_z',
    'value_abs_error', 'value_bias_pct', 'value_relative_error_pct',
    *POSITION_FIELDS,
    'elapsed_seconds', 'history_file', 'iterations_file', 'error',
]
ITERATION_FIELDS = PLAN_FIELDS + [
    'iteration', 'total_evaluations', 'best_value', 'best_x', 'best_y', 'best_z',
    'reference_method', 'reference_value', 'reference_x', 'reference_y', 'reference_z',
    'value_abs_error', 'value_bias_pct', 'value_relative_error_pct',
    *POSITION_FIELDS,
]
REFERENCE_FIELDS = ['data_model', 'model_name', 'target',
                    'reference_method', 'reference_value', 'reference_x', 'reference_y', 'reference_z']
SUMMARY_FIELDS = [
    'data_model', 'model_name', 'target', 'init_points', 'planned_runs',
    'attempted_runs', 'valid_runs', 'converged_runs', 'max_iter_runs', 'error_runs',
    'convergence_rate', 'convergence_iters_mean', 'convergence_iters_std',
    'convergence_iters_median', 'convergence_iters_min', 'convergence_iters_max',
    'bo_iters_mean', 'total_evaluations_mean', 'best_value_mean', 'elapsed_seconds_mean',
    'value_relative_error_pct_mean',
] + [field + '_mean' for field in POSITION_FIELDS]


def make_plan(models=DATA_MODELS, repeats=REPEATS, base_seed=BASE_SEED):
    """默认种子分组：模型1=42..51，模型2=52..61，依次至模型6=92..101。"""
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


def iteration_deviations(case, history, reference):
    """值偏差用 |参考值| 归一化；XYZ 分别用各轴搜索跨度归一化。"""
    widths = np.ptp(np.asarray(search.BOUNDS, dtype=float), axis=1)
    reference_position = np.array([reference[f'reference_{axis}'] for axis in ('x', 'y', 'z')])
    reference_value = float(reference['reference_value'])
    rows = []
    for iteration, (value, position) in enumerate(zip(history['best_value'], history['best_x'])):
        value = float(value)
        distances = np.abs(np.asarray(position) - reference_position)
        bias = 100.0 * (value - reference_value) / abs(reference_value) if reference_value != 0 else ''
        rows.append(dict(
            **case, iteration=iteration, total_evaluations=int(history['n_init']) + iteration,
            best_value=value, **{f'best_{axis}': float(v)
                               for axis, v in zip(('x', 'y', 'z'), position)},
            **{name: reference[name] for name in REFERENCE_FIELDS[3:]},
            value_abs_error=abs(value - reference_value), value_bias_pct=bias,
            value_relative_error_pct=abs(bias) if bias != '' else '',
            **{f'{axis}_abs_distance': float(v)
               for axis, v in zip(('x', 'y', 'z'), distances)},
            **{f'{axis}_relative_position_error_pct': float(v)
               for axis, v in zip(('x', 'y', 'z'), 100.0 * distances / widths)},
        ))
    return rows


def get_convergence_iteration(history, patience):
    """按现有停止公式复核，区分恰好在最后一轮收敛与单纯达到上限。

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
        if i - last_iter >= patience:
            return i + 1
    return None


def write_csv(path, fields, rows):
    """每次原子替换，避免中断时破坏已经完成的明细。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(plan, records):
    """仅对已触发停止条件的运行统计收敛次数；上限与失败分别计数。"""
    groups = {}
    for case in plan:
        key = (case['data_model'], case['target'], case['init_points'])
        group = groups.setdefault(key, {'planned': 0, 'rows': []})
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
            data_model=model, model_name=MODEL_NAMES[model], target=target,
            init_points=n_init, planned_runs=group['planned'], attempted_runs=len(rows),
            valid_runs=len(valid), converged_runs=len(converged),
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
                                                 if row[column] != '')
               if any(row[column] != '' for row in valid) else ''
               for column in ('bo_iters', 'total_evaluations', 'best_value', 'elapsed_seconds',
                              'value_relative_error_pct', *POSITION_FIELDS)},
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
    write_csv(directory / 'comparison.csv',
              ['data_model', 'model_name', 'target'] + [f'init_{n}' for n in INIT_POINTS],
              comparison.values())
    return summary


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


def make_config(models, repeats, base_seed, max_iter, patience,
                reference_grid, reference_starts, reference_seed, reference_file):
    root = Path(search.__file__).resolve().parent
    sources = [root / name for name in (
        'search.py', 'GPR.py', 'kernel.py', 'EI.py', 'LHS.py',
        'fitter.py', 'formula_models.py', 'measurement_data.py')]
    sources.extend(sorted((root / 'fit_models').glob('*.py')))
    config = dict(
        format_version=3, models=list(models), targets=list(TARGETS),
        init_points=list(INIT_POINTS), repeats=repeats, base_seed=base_seed,
        source='fitted', fit_method=FIT_METHOD, bounds=np.asarray(search.BOUNDS).tolist(),
        bo_parameters={**search.BO_PARAMETERS, 'max_iter': max_iter,
                       'stop_no_improve': patience},
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
              reference_seed=REFERENCE_SEED, reference_file=None):
    if repeats < 1 or max_iter < 1 or patience < 1:
        raise ValueError('repeats、max_iter、stop_no_improve 必须大于 0')
    if not 0 <= base_seed <= 2**32 - len(DATA_MODELS) * repeats:
        raise ValueError('种子范围必须适用于 np.random.seed')
    if reference_grid < 2 or reference_starts < 1:
        raise ValueError('reference_grid 至少为 2，reference_starts 至少为 1')
    plan = make_plan(models, repeats, base_seed)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = make_config(models, repeats, base_seed, max_iter, patience,
                         reference_grid, reference_starts, reference_seed, reference_file)
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
          f' = {len(plan)} 次；RBF；上限 {max_iter} 轮；无显著改进 {patience} 轮停止。', flush=True)
    print(f'输出目录：{directory}', flush=True)
    if plan_only:
        return []

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
            history = search.run_search(
                source='fitted', data_model=case['data_model'], fit_method=FIT_METHOD,
                target=case['target'], seed=case['seed'], init_points=case['init_points'],
                max_iter=max_iter, stop_no_improve=patience, show=False, verbose=verbose)
            convergence = get_convergence_iteration(history, patience)
            best_idx = int(np.argmax(history['y']))
            n_init_actual = int(history['n_init'])
            row.update(
                status='converged' if convergence is not None else 'max_iter',
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
            deviations = iteration_deviations(
                case, history, references[(case['data_model'], case['target'])])
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
    args = parser.parse_args()
    summary = run_batch(models=tuple(dict.fromkeys(args.models)), repeats=args.repeats,
                        base_seed=args.base_seed, max_iter=args.max_iter,
                        patience=args.stop_no_improve, output_dir=args.output_dir,
                        resume=args.resume, verbose=args.verbose, plan_only=args.plan_only,
                        reference_grid=args.reference_grid, reference_starts=args.reference_starts,
                        reference_seed=args.reference_seed, reference_file=args.reference_file)
    return int(any(row['error_runs'] for row in summary))


if __name__ == '__main__':
    raise SystemExit(main())
