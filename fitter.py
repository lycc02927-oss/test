"""GPR 与 Mock 的单发 ED/PF 数据模型，坐标输入为 (x, y, z)。"""

from functools import lru_cache

import numpy as np

import measurement_data
from fit_models import create_model
from formula_models import build_formula_ed_model

PF_BASELINE = 6.0          # PF 定值，与 ED 空间模型的能量尺度无关。
JITTER_PROBABILITY = 0.10  # 每轮先抽签，10% 概率同步影响 ED 和 PF。
PF_JITTER = 0.05          # 触发后 PF 在基值的 ±5% 内变化。
ED_PF_RESPONSE = 0.5      # ED 跟随 PF 相对变化的 50%。
ED_EXTRA_JITTER = 0.005   # 触发时 ED 另有 ±0.5% 独立小扰动。


class TimeJitter:
    """保留原 Mock 使用的 ED 参数和随机流；抖动计算在 run_pipeline 中。"""

    def __init__(self, *, pf_response=ED_PF_RESPONSE, extra_jitter=ED_EXTRA_JITTER, seed=2026):
        self.pf_response = pf_response
        self.extra_jitter = extra_jitter
        self.rng = np.random.default_rng(seed)


class PFTimeJitter:
    """保留原 Mock 使用的 PF 参数和随机流。"""

    def __init__(self, *, baseline=PF_BASELINE, jump_probability=JITTER_PROBABILITY,
                 jitter_relative=PF_JITTER, seed=2026):
        self.baseline = baseline
        self.jump_probability = jump_probability
        self.jitter_relative = jitter_relative
        self.rng = np.random.default_rng(seed)


class EDFitter:
    """选择一份试验数据，训练 XYZ 到 ED 的空间模型。"""

    def __init__(self, model_id=1, method="rbf", parameters=None):
        data = measurement_data.MODEL_DATA[model_id]
        self.model = create_model(method, parameters)
        self.model.fit(data[:, :3], data[:, 3])

    def predict(self, position):
        return float(self.model.predict([position])[0])


class EDPFModel:
    """每次调用生成一发 ED/PF；复用对象使随机数流持续推进。"""

    def __init__(self, spatial_model, *, seed=42, position_half_width=10.0,
                 ed_time_parameters=None, pf_time_parameters=None):
        self.spatial_model = spatial_model
        self.position_half_width = position_half_width
        position_seed, ed_seed, pf_seed = np.random.SeedSequence(seed).spawn(3)
        self.position_rng = np.random.default_rng(position_seed)
        self.ed_time = TimeJitter(seed=ed_seed, **(ed_time_parameters or {}))
        self.pf_time = PFTimeJitter(seed=pf_seed, **(pf_time_parameters or {}))

    @classmethod
    def from_formula(cls, mode, **sampling_parameters):
        """例如 EDPFModel.from_formula("formula_plateau", seed=42)。"""
        return cls(build_formula_ed_model(mode), **sampling_parameters)

    def measure(self, position):
        result = run_pipeline(position, self.position_half_width, self.position_rng,
                              self.spatial_model, ed_time_model=self.ed_time,
                              pf_model=self.pf_time)
        return result["ed"], result["pf"]


@lru_cache(maxsize=6)
def _get_default_fitter(model_id):
    return EDFitter(model_id)


def run_pipeline(center, half_width=10.0, random_seed=42,
                 fitter=None, *, ed_time_model=None, pf_model=None):
    """返回一发 ED/PF。fitter 可传空间模型或公式名，如 "formula_truncated_peak"。"""
    rng = np.random.default_rng(random_seed)
    if isinstance(fitter, str):
        fitter = build_formula_ed_model(fitter)
    spatial_model = _get_default_fitter(measurement_data.MODEL_ID) if fitter is None else fitter
    ed_time_model = TimeJitter(seed=rng.integers(0, 2**63)) if ed_time_model is None else ed_time_model
    pf_model = PFTimeJitter(seed=rng.integers(0, 2**63)) if pf_model is None else pf_model
    triggered = rng.random() < pf_model.jump_probability
    position = np.asarray(center) + rng.uniform(-half_width, half_width, 3)
    ed = spatial_model.predict(position)
    pf = pf_model.baseline
    if triggered:
        pf_relative = pf_model.rng.uniform(-pf_model.jitter_relative, pf_model.jitter_relative)
        pf = pf_model.baseline * (1 + pf_relative)
        ed *= 1 + ed_time_model.pf_response * pf_relative + ed_time_model.rng.uniform(
            -ed_time_model.extra_jitter, ed_time_model.extra_jitter)
    return {"ed": ed, "pf": pf, "ed_pf_ratio": ed / pf}
