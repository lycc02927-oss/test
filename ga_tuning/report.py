"""保存可直接筛选的原始 CSV；试验逐场落盘。汇总和 Excel 在所有评估结束后生成。"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .design import BUDGET, ED_TOL, GA_SEED, PATIENCE
from .encoding import GRID, WIDTH, decode
from .objective import FAILURE_DIVISOR, WEIGHTS, composite_loss


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
    """按工况等权汇总，用同一个综合 loss 排名。"""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], int(row["recipe"]))].append(row)
    results = []
    for (stage, recipe), models in groups.items():
        result = dict(stage=stage, recipe=recipe, runs=sum(r["runs"] for r in models),
                      hits=sum(r["hits"] for r in models))
        for metric in ("failure_rate", "mean_base_loss", "mean_error",
                       "mean_samples", "mean_bo", "mean_travel", "mean_sum_step_max"):
            result[metric] = mean(r[metric] for r in models)
        result["mean_loss"] = composite_loss(result["mean_base_loss"], result["failure_rate"])
        results.append(result)
    ranked = []
    for stage in ("train", "validation"):
        subset = [r for r in results if r["stage"] == stage]
        subset.sort(key=lambda r: r["mean_loss"])
        for rank, result in enumerate(subset, 1):
            ranked.append(dict(rank=rank, **result))
    return ranked


class Recorder:
    def __init__(self, directory, plan, models):
        self.dir = Path(directory)
        self.population = plan.population
        # 已有结果不覆写，以免把两次实验的轨迹混在一起。
        self.dir.mkdir(parents=True, exist_ok=True)
        if any(self.dir.glob("*.csv")):
            raise FileExistsError(f"请指定新的 --out 目录；{directory} 已有结果。")
        self.recipes, self.trials = {}, []
        import search
        span = max(hi - lo for lo, hi in search.BOUNDS)
        settings = dict(plan="standard", population=plan.population, generations=plan.generations,
                        ga_seed=GA_SEED, patience=PATIENCE, error_limit=ED_TOL, sample_cap=BUDGET,
                        axis_span=span, distance_cap=BUDGET * span, weight_error=WEIGHTS[0],
                        weight_samples=WEIGHTS[1], weight_travel=WEIGHTS[2], distance_metric="sum_step_max",
                        failure_divisor=FAILURE_DIVISOR)
        self.write("settings", [settings])
        self.write("seeds", [dict(stage=stage, seed=s) for stage, seeds in
                             (("train", plan.train), ("validation", plan.validation)) for s in seeds])
        model_rows, peak_rows = [], []
        for name, case in models.items():
            options = case.options
            x, y, z, ed = case.reference
            model_rows.append(dict(model=name, source=options["source"], data_model=options.get("data_model"),
                                   fit_method=options.get("fit_method"), formula_model=options.get("formula_model"),
                                   design_ed=ed, x=x, y=y, z=z))
            for index, peak in enumerate(options.get("peaks", []), 1):
                cx, cy, cz = peak["center"]
                sx, sy, sz = peak["sigma"]
                peak_rows.append(dict(model=name, peak_index=index, peak_ed=peak["peak_ed"],
                                      center_x=cx, center_y=cy, center_z=cz, sigma_x=sx, sigma_y=sy, sigma_z=sz))
        self.write("models", model_rows)
        self.write("peaks", peak_rows)
        self.write("grid", [dict(parameter=k, bits=WIDTH[k], index=i, value=v)
                            for k, levels in GRID.items() for i, v in enumerate(levels)])

    def write(self, name, rows, append=False):
        csv_rows(self.dir / (name + ".csv"), rows, append)

    def register(self, key, recipe):
        if key not in self.recipes:
            number = self.recipes[key] = len(self.recipes) + 1
            self.write("recipes", [dict(recipe=number, chromosome=key, **recipe)], True)
        return self.recipes[key]

    def add(self, number, stage, trial, points):
        trial_id = len(self.trials) + 1
        record = dict(trial=trial_id, stage=stage, recipe=number, **trial)
        self.trials.append(record)
        self.write("trials", [record], True)
        self.write("points", [dict(trial=trial_id, **p) for p in points], True)

    def finish(self, winner, default, generations, members):
        self.write("generations", [dict(row, champion=self.recipes[row["champion"]]) for row in generations])
        self.write("members", [dict(generation=r["generation"], rank=r["rank"],
                                    recipe=self.recipes[r["chromosome"]], mean_base_loss=r["mean_base_loss"],
                                    failure_rate=r["failure_rate"], loss=r["loss"]) for r in members])
        groups = defaultdict(list)
        for trial in self.trials:
            key = (trial["stage"], trial["recipe"], trial["model"])
            groups[key].append(trial)
        summary = []
        for (stage, recipe, model), trials in groups.items():
            hits = sum(t["success"] for t in trials)
            row = dict(stage=stage, recipe=recipe, model=model, runs=len(trials),
                       hits=hits, success_rate=hits / len(trials),
                       failure_rate=mean(t["patience_failed"] for t in trials))
            for metric in ("best_ed", "error", "samples", "bo", "travel", "sum_step_max", "base_loss"):
                row["mean_" + metric] = mean(t[metric] for t in trials)
            row["mean_loss"] = composite_loss(row["mean_base_loss"], row["failure_rate"])
            summary.append(row)
        self.write("summary", summary)
        self.write("overall", overall_summary(summary))
        self.write("selection", [dict(role="GA", recipe=self.recipes[winner], **decode(winner)),
                                 dict(role="baseline", recipe=self.recipes["default"], **default)])
        plot_progress(generations, self.dir / "ga_progress.png", self.population)
        from .excel import export
        export(self.dir)


def plot_progress(rows, path, population):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    generations = [r["generation"] for r in rows]
    best = [r["mean_loss"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.axhline(best[0], ls="--", lw=1, color="0.5", label="Default BO")
    ax.plot(generations[1:], [r["population_mean_loss"] for r in rows[1:]],
            color="#aa6733", ls=":", marker=".", label="Population mean")
    ax.plot(generations, best, color="#006c78", marker="o", label="Best training loss")
    ax.scatter(0, best[0], s=75, marker="D", color="#252b35", zorder=3)
    ax.set(title=f"GA training  |  {population} individuals, {generations[-1]} generations",
           xlabel="GA generation (0 = default BO)", ylabel="Composite loss (lower is better)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
