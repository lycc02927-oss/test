"""保存可直接筛选的原始 CSV；试验逐场落盘。汇总和 Excel 在所有评估结束后生成。"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .design import BUDGET, ED_TOL, GA_SEED, MODELS, PATIENCE, reference
from .encoding import GRID, WIDTH, decode
from .objective import WEIGHTS


def csv_rows(path, rows, append=False):
    if not rows:
        return
    new = not append or not path.exists()
    with path.open("w" if new else "a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def overall_summary(rows):
    """每种工况先等权汇总，再按 GA 的真实优先次序给所有配方排名。"""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], int(row["recipe"]))].append(row)
    results = []
    for (stage, recipe), models in groups.items():
        rates = [float(r["success_rate"]) for r in models]
        result = dict(stage=stage, recipe=recipe, runs=sum(int(r["runs"]) for r in models),
                      hits=sum(int(r["hits"]) for r in models),
                      worst_failure_rate=1 - min(rates), mean_failure_rate=1 - mean(rates))
        for metric in ("mean_error", "mean_samples", "mean_bo", "mean_travel", "mean_loss"):
            result[metric] = mean(float(r[metric]) for r in models)
        results.append(result)
    ranked = []
    for stage in ("train", "validation"):
        subset = [r for r in results if r["stage"] == stage]
        subset.sort(key=lambda r: (r["worst_failure_rate"], r["mean_failure_rate"], r["mean_loss"]))
        for rank, result in enumerate(subset, 1):
            ranked.append(dict(rank=rank, **result))
    return ranked


class Recorder:
    def __init__(self, directory, name, plan):
        self.dir = Path(directory)
        # 已有结果不覆写，以免把两次实验的轨迹混在一起。
        self.dir.mkdir(parents=True, exist_ok=True)
        if any(self.dir.glob("*.csv")):
            raise FileExistsError(f"请指定新的 --out 目录；{directory} 已有结果。")
        self.recipes, self.trials = {}, []
        import search
        span = max(hi - lo for lo, hi in search.BOUNDS)
        settings = dict(plan=name, population=plan.population, generations=plan.generations,
                        ga_seed=GA_SEED, patience=PATIENCE, error_limit=ED_TOL, sample_cap=BUDGET,
                        axis_span=span, distance_cap=BUDGET * span, weight_error=WEIGHTS[0],
                        weight_samples=WEIGHTS[1], weight_travel=WEIGHTS[2])
        self.write("settings", [settings])
        self.write("seeds", [dict(stage=stage, seed=s) for stage, seeds in
                             (("train", plan.train), ("validation", plan.validation)) for s in seeds])
        models = []
        for m, label in MODELS.items():
            x, y, z, ed = reference(m)
            models.append(dict(model=m, name=label, design_ed=ed, x=x, y=y, z=z))
        self.write("models", models)
        self.write("grid", [dict(parameter=k, bits=WIDTH[k], index=i, value=v)
                            for k, levels in GRID.items() for i, v in enumerate(levels)])

    def write(self, name, rows, append=False):
        csv_rows(self.dir / (name + ".csv"), rows, append)

    def add(self, bits, stage, trial, points):
        if bits not in self.recipes:
            self.recipes[bits] = len(self.recipes) + 1
            self.write("recipes", [dict(recipe=self.recipes[bits], chromosome=bits, **decode(bits))], True)
        trial_id = len(self.trials) + 1
        record = dict(trial=trial_id, stage=stage, recipe=self.recipes[bits], **trial)
        self.trials.append(record)
        self.write("trials", [record], True)
        self.write("points", [dict(trial=trial_id, **p) for p in points], True)

    def finish(self, winner, baseline, generations, members):
        self.write("generations", [dict(row, champion=self.recipes[row["champion"]]) for row in generations])
        self.write("members", [dict(generation=r["generation"], rank=r["rank"],
                                    recipe=self.recipes[r["chromosome"]]) for r in members])
        groups = defaultdict(list)
        for trial in self.trials:
            key = (trial["stage"], trial["recipe"], trial["model"])
            groups[key].append(trial)
        summary = []
        for (stage, recipe, model), trials in groups.items():
            hits = sum(t["success"] for t in trials)
            row = dict(stage=stage, recipe=recipe, model=model, runs=len(trials),
                       hits=hits, success_rate=hits / len(trials))
            for metric in ("best_ed", "error", "samples", "bo", "travel", "loss"):
                row["mean_" + metric] = mean(t[metric] for t in trials)
            summary.append(row)
        self.write("summary", summary)
        self.write("overall", overall_summary(summary))
        self.write("selection", [dict(role=role, recipe=self.recipes[bits], **decode(bits))
                                 for role, bits in (("GA", winner), ("baseline", baseline))])
        from .excel import export
        export(self.dir)
