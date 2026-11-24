"""从仓库根目录执行 python -m ga_tuning.run --plan pilot。"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# 每个进程同时执行一个 BO。避免 8 个进程再各自启动一批 BLAS 线程。
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

from .design import GA_SEED, MODELS, PLANS
from .encoding import BASELINE, encode
from .genetic import evolve
from .objective import rank_trials
from .simulation import simulate
from .report import Recorder


def main():
    parser = argparse.ArgumentParser(description="对四种带抖动 ED 模型搜索一组通用参数")
    parser.add_argument("--plan", choices=PLANS, default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("ga_results"))
    args = parser.parse_args()
    plan = PLANS[args.plan]
    log = Recorder(args.out, args.plan, plan)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def evaluate(bits_list, stage="train"):
            seeds = plan.train if stage == "train" else plan.validation
            tasks = [(bits, model, seed) for bits in bits_list for model in MODELS for seed in seeds]
            grouped = {bits: [] for bits in bits_list}
            for task, (trial, points) in zip(tasks, pool.map(simulate, tasks)):
                bits, model, seed = task
                log.add(bits, stage, trial, points)
                grouped[bits].append(trial)
            return {bits: rank_trials(rows) for bits, rows in grouped.items()}

        winner, generations, members = evolve(plan, GA_SEED, evaluate)
        baseline = encode(BASELINE)
        evaluate(list(dict.fromkeys([winner, baseline])), "validation")
    log.finish(winner, baseline, generations, members)
    print(f"最佳训练配方 {log.recipes[winner]}；结果已保存到 {args.out}", flush=True)


if __name__ == "__main__":
    main()
