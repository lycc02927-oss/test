"""遗传算法配置 search.run_search；详见 GA_TUNING.md。"""
import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import NamedTuple

# batch_test 在导入科学计算库前设置无界面绘图和默认单线程 BLAS。
import batch_test as batch
import numpy as np
import search
from fitter import EDFitter, PF_BASELINE


class Genes(NamedTuple):
    init_points: int
    xi_mode: str
    global_prob: float
    r_frac_min: float
    sigma3_rel: float
    stop_no_improve: int
    ls_bounds: tuple

    def parameters(self, budget):
        p = self._asdict()
        del p['xi_mode']
        xi = (None, None) if self.xi_mode == 'auto' else (0.01, 0.0)
        p.update(max_iter=budget-self.init_points, xi_init=xi[0], xi_final=xi[1])
        return dict(p)


def gene_space(budget):
    initial = [v for v in (10, 20, 30, 40) if v < budget]
    if not initial:
        raise ValueError('总量测预算至少为 11，须给 BO 留出至少一轮')
    return (initial, ('fixed', 'auto'), (0.1, 0.4, 0.7), (0.025, 0.05, 0.1),
            (0., 0.03, 0.1), (10, 20, 30), ((20., 300.), (40., 600.), (40., 2000.)))


def baseline(space):
    return Genes(max(space[0]), 'fixed', 0.4, 0.05, 0., 30, (40., 2000.))


def draw(space, rng):
    return Genes(*(rng.choice(choices) for choices in space))


def next_population(ranked, scores, space, rng, mutation):
    """保留冠军；锦标赛选父母、逐基因均匀交叉，再按概率突变。"""
    def parent():
        return min(rng.sample(ranked, min(3, len(ranked))), key=scores.__getitem__)

    population = [ranked[0]]
    for _ in range(len(ranked) * 30):
        if len(population) == len(ranked):
            break
        a, b = parent(), parent()
        values = []
        for x, y, choices in zip(a, b, space):
            v = rng.choice((x, y))
            if len(choices) > 1 and rng.random() < mutation:
                v = rng.choice([other for other in choices if other != v])
            values.append(v)
        child = Genes(*values)
        if child not in population:
            population.append(child)
    while len(population) < len(ranked):
        child = draw(space, rng)
        if child not in population:
            population.append(child)
    return population


@lru_cache(maxsize=6)
def scoring_surface(model):
    return EDFitter(model, batch.FIT_METHOD)


def evaluate(job):
    genes, model, target, seed, budget, ref, tolerance = job
    h = search.run_search(source='fitted', data_model=model, fit_method=batch.FIT_METHOD,
                          target=target, seed=seed, show=False, verbose=False,
                          **genes.parameters(budget))
    x = h['best_x'][-1]
    clean = scoring_surface(model).predict(x)
    if target == 'ed_pf_ratio':
        clean /= PF_BASELINE
    if not np.isfinite([ref, clean]).all() or ref == 0:
        raise ValueError(f'模型 {model} 的参考值或回测值不适用于相对误差评分')
    error = 100. * abs(clean-ref) / abs(ref)
    return dict(model=model, target=target, seed=seed, evaluations=len(h['y']),
                bo_iters=len(h['y'])-h['n_init'], observed_best=float(h['best_value'][-1]),
                best_x=float(x[0]), best_y=float(x[1]), best_z=float(x[2]),
                clean_value=clean, reference=ref, error_pct=error, hit=error <= tolerance,
                **batch.calculate_total_distances(h))


def metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['target']].append(row)
    rates = [mean(r['hit'] for r in group) for group in groups.values()]
    return dict(worst_hit_rate=min(rates), mean_hit_rate=mean(rates),
                mean_evaluations=mean(r['evaluations'] for r in rows),
                mean_error_pct=mean(r['error_pct'] for r in rows))


def rank(rows):
    m = metrics(rows)
    return (-m['worst_hit_rate'], -m['mean_hit_rate'], m['mean_evaluations'], m['mean_error_pct'])


def write_rows(path, rows, append=False):
    if not rows:
        return
    header = not append or not path.exists()
    with path.open('a' if append else 'w', newline='', encoding='utf-8' if not header else 'utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if header:
            writer.writeheader()
        writer.writerows(rows)


def run(args):
    space = gene_space(args.budget)
    count = int(np.prod([len(d) for d in space]))
    if not 4 <= args.population <= count or args.generations < 1 or args.workers < 1:
        raise ValueError(f'种群须为 4..{count}，代数与进程数须为正数')
    if not 0 < args.mutation <= 1 or not 0 <= args.tolerance <= 100:
        raise ValueError('mutation 须在 (0,1]，tolerance 须在 [0,100]')
    for vals in (args.models, args.targets, args.train_seeds, args.valid_seeds):
        if len(set(vals)) != len(vals):
            raise ValueError('各列表内部不能含重复项')
    if set(args.train_seeds) & set(args.valid_seeds):
        raise ValueError('开发种子与验证种子必须分开')
    if any(s < 0 or s >= 2**32 for s in args.train_seeds + args.valid_seeds):
        raise ValueError('内层随机种子须在 [0, 2**32) 内')
    output = Path(args.output) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')
    references = batch.make_references(args.models)
    write_rows(output/'references.csv', references)
    refs = {(r['data_model'], r['target']): r['reference_value'] for r in references}
    scenarios = [(m, t) for m in args.models for t in args.targets]
    rng = random.Random(args.ga_seed)
    control = baseline(space)
    population = [control]
    while len(population) < args.population:
        g = draw(space, rng)
        if g not in population:
            population.append(g)

    train, identifiers = {}, {}
    validation = []
    with (ProcessPoolExecutor(args.workers) if args.workers > 1 else nullcontext()) as pool:
        def calculate(requests, seeds, stage, generation):
            jobs = ((g, m, t, seed, args.budget, refs[m,t], args.tolerance)
                    for g, cases in requests for m, t in cases for seed in seeds)
            # 任务列表保留返回值与染色体的一一对应，进程之间隔离内层 NumPy 随机状态。
            jobs = list(jobs)
            results = pool.map(evaluate, jobs) if pool else map(evaluate, jobs)
            grouped = defaultdict(list)
            for job, row in zip(jobs, results):
                g = job[0]
                row = dict(stage=stage, generation=generation, candidate=identifiers[g], **row)
                grouped[g].append(row)
                write_rows(output/'trials.csv', [row], append=True)
            return grouped

        for generation in range(1, args.generations+1):
            fresh = [g for g in population if g not in train]
            for g in fresh:
                identifiers[g] = len(identifiers)+1
            write_rows(output/'candidates.csv', [dict(candidate=identifiers[g], **g._asdict())
                                                 for g in fresh], append=True)
            train.update(calculate([(g, scenarios) for g in fresh], args.train_seeds, 'train', generation))
            scores = {g: rank(train[g]) for g in population}
            ranked = sorted(population, key=scores.__getitem__)
            write_rows(output/'generations.csv', [dict(generation=generation, candidate=identifiers[g],
                        **metrics(train[g])) for g in ranked], append=True)
            print(f'第 {generation} 代，累计 {len(train)} 套，当前冠军 {identifiers[ranked[0]]}', flush=True)
            if generation < args.generations:
                population = next_population(ranked, scores, space, rng, args.mutation)

        # 只用开发集挑选。专用方案从所有已评估候选中按该模型/目标重新排名。
        common = min(train, key=lambda g: rank(train[g]))
        assignments = {}
        for m,t in scenarios:
            def case_score(g):
                return rank([r for r in train[g] if (r['model'], r['target']) == (m,t)])
            assignments[m,t] = dict(baseline=control, common=common, specialist=min(train, key=case_score))
        requests = defaultdict(set)
        for case, roles in assignments.items():
            for g in roles.values():
                requests[g].add(case)
        held = calculate([(g, sorted(cases)) for g, cases in requests.items()],
                         args.valid_seeds, 'validation', '',)
        for (m,t), roles in assignments.items():
            for role,g in roles.items():
                rows = [r for r in held[g] if (r['model'],r['target']) == (m,t)]
                validation.append(dict(model=m, target=t, role=role, candidate=identifiers[g],
                    **g._asdict(), max_iter=args.budget-g.init_points, valid_runs=len(rows), **metrics(rows)))
    write_rows(output/'validation.csv', validation)
    result = dict(budget=args.budget, common=common.parameters(args.budget),
                  specialists=[dict(model=m, target=t, parameters=roles['specialist'].parameters(args.budget))
                               for (m,t),roles in assignments.items()])
    (output/'selected.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('结果目录', output, flush=True)
    return output


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=int, nargs='+', choices=batch.DATA_MODELS, default=list(batch.DATA_MODELS))
    p.add_argument('--targets', nargs='+', choices=batch.TARGETS, default=list(batch.TARGETS))
    p.add_argument('--budget', type=int, default=80)
    p.add_argument('--population', type=int, default=8)
    p.add_argument('--generations', type=int, default=3, help='包含初始代')
    p.add_argument('--mutation', type=float, default=.2, help='每个基因独立突变概率')
    p.add_argument('--ga-seed', type=int, default=2026)
    p.add_argument('--train-seeds', type=int, nargs='+', default=[42,43])
    p.add_argument('--valid-seeds', type=int, nargs='+', default=[142,143,144])
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--tolerance', type=float, default=10., help='回测误差容限，单位百分比')
    p.add_argument('--output', default='ga_results')
    return p.parse_args()


if __name__ == '__main__':
    run(arguments())
