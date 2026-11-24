"""GPR 与 Mock 的单发 ED/PF 数据模型，坐标输入为 (x, y, z)。"""

from functools import lru_cache

import numpy as np

import measurement_data
from fit_models import create_model

PF_BASELINE = 6.0          # PF 定值，与 ED 空间模型的能量尺度无关。
JITTER_PROBABILITY = 0.10  # 每轮先抽签，10% 概率同步影响 ED 和 PF。
PF_JITTER = 0.05          # 触发后 PF 在基值的 ±5% 内变化。
ED_PF_RESPONSE = 0.5      # ED 跟随 PF 相对变化的 50%。
ED_EXTRA_JITTER = 0.005   # 触发时 ED 另有 ±0.5% 独立小扰动。


# 公式模型的设置。每个字典是一座峰；增删多峰列表中的字典即可改变峰数。
# peak_ed 是无抖动的峰值，center 和 sigma 均按 X、Y、Z 排列。
FORMULA_CONFIGS = {
    "formula_gaussian": {
        "peaks": [
            {"peak_ed": 1.0, "center": (0, 0, 0), "sigma": (100, 100, 100)},
        ],
    },
    "formula_offset_gaussian": {
        "peaks": [
            {"peak_ed": 1.0, "center": (120, -80, 80), "sigma": (90, 90, 90)},
        ],
    },
    "formula_multi_gaussian": {
        "peaks": [
            {"peak_ed": 0.2200, "center": (160, 80, 80), "sigma": (70, 70, 70)},
            {"peak_ed": 0.0800, "center": (-160, -80, -80), "sigma": (70, 70, 70)},
            {"peak_ed": 0.0373, "center": (0, 0, 0), "sigma": (100, 100, 100)},
        ],
    },
    "formula_plateau": {
        "baseline": 0.2,
        "peaks": [{"peak_ed": 1.0, "center": (0, 0, 0), "sigma": (100, 80, 60)}],
        "plateau_outer_radius": 1.25,  # 相对三个椭球半轴的距离，在这里降到底值
    },
    "formula_truncated_peak": {
        "baseline": 0.2,
        "peaks": [{"peak_ed": 1.0, "center": (120, 0, 0), "sigma": (80, 90, 70)}],
        "cutoff_width": 20.0,  # 从峰顶沿 +X 下降至底值的宽度
    },
}


def smooth_fall(t):
    """从 1 连续降到 0，两端斜率为零。"""
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t) ** 2 * (1.0 + 2.0 * t)


class FormulaModel:
    def __init__(self, baseline=0.0, components=None, plateau_edge=None, cutoff_x=None,
                 *, peaks=None, plateau_outer_radius=None, cutoff_width=None):
        # 旧版直接构造模型时以相对背景的 amplitude 描述高斯分量。
        self.add_components = components is not None
        if peaks is None:
            peaks = [dict(c, peak_ed=baseline + c["amplitude"]) for c in components]
        self.baseline = baseline
        self.centers = np.array([p["center"] for p in peaks], dtype=float)
        self.sigmas = np.array([p["sigma"] for p in peaks], dtype=float)
        self.heights = np.array([p["peak_ed"] for p in peaks], dtype=float)
        self.plateau_outer_radius = plateau_outer_radius
        self.cutoff_width = cutoff_width
        if plateau_edge is not None:
            self.plateau_outer_radius = 1.0 + plateau_edge / 2
        if cutoff_x is not None:
            self.centers[0, 0] = min(self.centers[0, 0], cutoff_x)
            self.cutoff_width = self.sigmas[0, 0] / 4
        self.ed_limits = None
        if self.plateau_outer_radius is not None or self.cutoff_width is not None:
            self.ed_limits = (baseline, self.heights[0])

    def predict(self, position):
        offset = np.asarray(position) - self.centers
        distance = offset / self.sigmas
        r2 = np.sum(distance ** 2, axis=1)

        if self.plateau_outer_radius is not None:
            radius = np.sqrt(r2[0]) / self.plateau_outer_radius
            response = smooth_fall(min(radius, 1.0) ** 3)
            return float(self.baseline + (self.heights[0] - self.baseline) * response)

        response = np.exp(-0.5 * r2)
        if self.cutoff_width is not None:
            response[0] *= smooth_fall(offset[0, 0] / self.cutoff_width)
        values = self.baseline + (self.heights - self.baseline) * response
        if len(values) == 1:
            return float(values[0])
        if self.add_components:
            return float(self.baseline + (values - self.baseline).sum())

        # 在每个中心精确取该峰的高度；中心之间平滑混合，避免叠加推高或移动峰顶。
        nearest = int(np.argmin(r2))
        if r2[nearest] == 0.0:
            return float(self.heights[nearest])
        weights = (r2[nearest] / r2) ** 2
        return float(np.average(values, weights=weights))

    def single_peak_position(self, bounds):
        """默认单峰、平台中心与渐变截断峰的峰位，用于搜索结果标注。"""
        if len(self.centers) != 1:
            return None
        lower, upper = np.asarray(bounds, dtype=float).T
        peak = np.clip(self.centers[0], lower, upper)
        if self.ed_limits is not None and self.predict(peak) <= self.baseline:
            return None
        return peak


def build_formula_ed_model(mode="formula_gaussian", *, peaks=None):
    """peaks 可覆盖本次高斯峰配置；默认值在本文件的 FORMULA_CONFIGS 中。"""
    config = FORMULA_CONFIGS[mode]
    if peaks is not None:
        config = dict(config, peaks=peaks)
    return FormulaModel(**config)


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
    def from_formula(cls, mode, *, peaks=None, **sampling_parameters):
        """例如 EDPFModel.from_formula("formula_plateau", seed=42)。"""
        return cls(build_formula_ed_model(mode, peaks=peaks), **sampling_parameters)

    def measure(self, position):
        result = run_pipeline(position, self.position_half_width, self.position_rng,
                              self.spatial_model, ed_time_model=self.ed_time,
                              pf_model=self.pf_time)
        return result["ed"], result["pf"]


@lru_cache(maxsize=6)
def _get_default_fitter(model_id):
    return EDFitter(model_id)


def run_pipeline(center, half_width=10.0, random_seed=42,
                 fitter=None, *, ed_time_model=None, pf_model=None, peaks=None):
    """返回一发 ED/PF。fitter 可传空间模型或公式名，如 "formula_truncated_peak"。"""
    rng = np.random.default_rng(random_seed)
    if isinstance(fitter, str):
        fitter = build_formula_ed_model(fitter, peaks=peaks)
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
    limits = getattr(spatial_model, "ed_limits", None)
    if limits is not None:
        ed = float(np.clip(ed, *limits))
    return {"ed": ed, "pf": pf, "ed_pf_ratio": ed / pf}
