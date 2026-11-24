"""只在此处指定模型、独立种子、试验规模和 BO 联合停机门槛。"""
from dataclasses import dataclass
from measurement_data import MODEL_DATA

MODELS = {2: "LT 仿真", 3: "LT 精细仿真", 4: "中心高斯", 5: "偏置高斯"}
PATIENCE = 30
ED_TOL = 0.10
BUDGET = 90  # 包含初始采样。每个 recipe 用同样的总预算。
GA_SEED = 20261124


@dataclass(frozen=True)
class Plan:
    population: int
    generations: int
    train: tuple
    validation: tuple
    spread_initial: bool = False


# 四工况等权。同一阶段所有候选用相同种子，独立验证只比较冠军与基准。
PLANS = {
    "pilot": Plan(4, 3, (42,), (501, 502)),
    "study": Plan(4, 4, (42, 43), tuple(range(501, 506))),
    "balanced": Plan(6, 3, (42, 43), tuple(range(901, 906)), True),
    "standard": Plan(12, 8, (42, 43, 44), tuple(range(1001, 1011)), True),
}


def reference(model):
    """设计参考取原始数据表的最高 ED 数据点；与 RBF 插值面最高值可有差异。"""
    rows = MODEL_DATA[model]
    return rows[rows[:, 3].argmax(), :4].tolist()
