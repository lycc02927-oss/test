"""只在此处指定模型、独立种子、试验规模和 BO 联合停机门槛。"""
from dataclasses import dataclass
from measurement_data import MODEL_DATA, MODEL_NAMES

MODELS = {i: MODEL_NAMES[i] for i in (2, 3, 4)}
PATIENCE = 30
ED_TOL = 0.10
BUDGET = 90  # 包含初始采样。每个 recipe 用同样的总预算。
GA_SEED = 20261124


@dataclass(frozen=True)
class Standard:
    population: int = 20
    generations: int = 15
    train: tuple = (42, 43, 44)
    validation: tuple = tuple(range(1001, 1011))


def reference(model):
    """设计参考取原始数据表的最高 ED 数据点；与 RBF 插值面最高值可有差异。"""
    rows = MODEL_DATA[model]
    return rows[rows[:, 3].argmax(), :4].tolist()
