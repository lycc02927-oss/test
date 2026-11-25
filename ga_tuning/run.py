"""在仓库根目录执行 python -m ga_tuning.run。"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

from .design import GA_SEED, MODELS, Standard
from .encoding import decode
from .genetic import evolve
from .objective import recipe_means
from .report import Recorder
from .simulation import default_recipe, simulate


def run(settings, workers, directory):
    default = default_recipe()
    log = Recorder(directory, settings)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate(key, recipe, generation, stage="train"):
            seeds = settings.train if stage == "train" else settings.validation
            tasks = [(recipe, model, seed) for model in MODELS for seed in seeds]
            trial_rows = []
            number = log.register(key, recipe)
            for trial, points in pool.map(simulate, tasks):
                log.add(number, stage, trial, points)
                trial_rows.append(trial)
            score = recipe_means(trial_rows)
            label = f"训练 第 {generation} 代" if stage == "train" else "独立验证"
            print(f"[{label}] 个体 {number}  loss={score['loss']:.6f}  "
                  f"基础 loss={score['base_loss']:.6f}  30轮失败率={score['patience_failed']:.1%}",
                  flush=True)
            return score

        winner, history, members = evolve(settings, GA_SEED, default, evaluate)
        evaluate("default", default, None, "validation")
        evaluate(winner, decode(winner), None, "validation")
    log.finish(winner, default, history, members)
    print(f"GA 入选个体 {log.recipes[winner]}，曲线和工作簿已保存到 {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="standard GA：优化三种单峰 ED 工况")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="结果目录，默认以开始时间命名")
    args = parser.parse_args()
    directory = args.out or Path("ga_results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run(Standard(), args.workers, directory)


if __name__ == "__main__":
    main()
