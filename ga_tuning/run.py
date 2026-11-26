"""在仓库根目录执行 python -m ga_tuning.run。"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

from .design import GA_SEED, Standard, fitted, formula, peak
from .encoding import decode
from .genetic import evolve
from .objective import recipe_means
from .report import Recorder
from .simulation import default_recipe, simulate


# ===== 在这里选择本次 GA 的工况 =====
# 字典左边是自定义工况名。保留要运行的行；每个工况等权，可混用、可只选一个。
# 数据编号：1 P3 试验，2 LT 整机仿真，3 中心高斯，4 偏置高斯，5 多峰高斯。
# 拟合方法：rbf、manual_gpr、linear、polynomial、svm。
MODELS = {
    "LT 整机仿真": fitted(2, "rbf"),
    "中心高斯数据": fitted(3, "rbf"),
    "偏置高斯数据": fitted(4, "rbf"),
    # "自定义偏置峰": formula("formula_offset_gaussian",
    #     peak(peak_ed=0.90, center=(80, -40, 20), sigma=(60, 80, 50))),
    # "双峰": formula("formula_multi_gaussian",
    #     peak(peak_ed=0.70, center=(-80, 0, 30), sigma=(50, 60, 40)),
    #     peak(peak_ed=1.00, center=(110, 40, -20), sigma=(70, 50, 60))),
    # "中心高斯公式": formula("formula_gaussian"),
    # "渐变平台": formula("formula_plateau"),
    # "渐变截断峰": formula("formula_truncated_peak"),
}
# 每个 peak 代表一座峰，center 和 sigma 均按 (X, Y, Z) 填写，单位 µm。
# formula 只填模型名时沿用 fitter 的峰参数。


def run(settings, workers, directory, models=None):
    models = MODELS if models is None else models
    default = default_recipe()
    log = Recorder(directory, settings, models)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate(key, recipe, generation, stage="train"):
            seeds = settings.train if stage == "train" else settings.validation
            tasks = [(recipe, name, case, seed) for name, case in models.items() for seed in seeds]
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
    parser = argparse.ArgumentParser(description="standard GA：在选定 ED 工况上优化 BO 参数")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="结果目录，默认以开始时间命名")
    args = parser.parse_args()
    directory = args.out or Path("ga_results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run(Standard(), args.workers, directory)


if __name__ == "__main__":
    main()
