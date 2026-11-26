"""定义试验规模、BO 停机条件，以及 run.py 选用的工况。"""
from dataclasses import dataclass

from fitter import FORMULA_CONFIGS
from measurement_data import MODEL_DATA

PATIENCE = 30
ED_TOL = 0.10
BUDGET = 90  # 初始点和后续 BO 点共用此预算。
GA_SEED = 20261124


@dataclass(frozen=True)
class Standard:
    population: int = 20
    generations: int = 15
    train: tuple = (42, 43, 44)
    validation: tuple = tuple(range(1001, 1011))


@dataclass(frozen=True)
class Scenario:
    options: dict       # 原样传给 search.run_search 建立空间模型。
    reference: tuple    # 无抖动设计参考 (x, y, z, ED)。


def fitted(data_model, fit_method="rbf"):
    rows = MODEL_DATA[data_model]
    reference = tuple(rows[rows[:, 3].argmax(), :4])
    return Scenario(dict(source="fitted", data_model=data_model, fit_method=fit_method), reference)


def peak(peak_ed, center, sigma):
    return dict(peak_ed=peak_ed, center=center, sigma=sigma)


def formula(formula_model, *peaks):
    # 省略峰参数时使用 fitter 自带的配置；多峰工况以最高设计峰为参考。
    peaks = list(peaks) if peaks else FORMULA_CONFIGS[formula_model]["peaks"]
    highest = max(peaks, key=lambda p: p["peak_ed"])
    options = dict(source="formula", formula_model=formula_model, peaks=peaks)
    return Scenario(options, (*highest["center"], highest["peak_ed"]))
