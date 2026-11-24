"""数学公式 ED 空间模型，坐标输入为 (x, y, z)。"""

import numpy as np


FORMULA_CONFIGS = {
    "formula_gaussian": {
        "baseline": 0.0,
        "components": [
            {"amplitude": 1.0, "center": (0.0, 0.0, 0.0), "sigma": (100.0, 100.0, 100.0)},
        ],
    },
    "formula_offset_gaussian": {
        "baseline": 0.0,
        "components": [
            {"amplitude": 1.0, "center": (120.0, -80.0, 80.0), "sigma": (90.0, 90.0, 90.0)},
        ],
    },
    "formula_multi_gaussian": {
        "baseline": 0.0,
        "components": [
            {"amplitude": 0.2200, "center": (160.0, 80.0, 80.0), "sigma": (70.0, 70.0, 70.0)},
            {"amplitude": 0.0800, "center": (-160.0, -80.0, -80.0), "sigma": (70.0, 70.0, 70.0)},
            {"amplitude": 0.0373, "center": (0.0, 0.0, 0.0), "sigma": (100.0, 100.0, 100.0)},
        ],
    },
    "formula_plateau": {
        "baseline": 0.0,
        "components": [
            {"amplitude": 1.0, "center": (0, 0, 0), "sigma": (100, 80, 60)},
        ],
        "plateau_edge": 0.5,  # sigma 为平顶椭球半轴，边缘宽度用半轴的比例表示
    },
    "formula_truncated_peak": {
        "baseline": 0.0,
        "components": [
            {"amplitude": 1.0, "center": (160, 0, 0), "sigma": (80, 90, 70)},
        ],
        "cutoff_x": 120.0,  # 保留 x <= 120 的高斯侧翼，另一侧降至背景
    },
}


class FormulaModel:
    def __init__(self, baseline, components, plateau_edge=None, cutoff_x=None):
        self.baseline = baseline
        self.plateau_edge = plateau_edge
        self.cutoff_x = cutoff_x
        self.amplitudes = np.array([part["amplitude"] for part in components])
        self.centers = np.array([part["center"] for part in components])
        self.sigmas = np.array([part["sigma"] for part in components])

    def predict(self, position):
        if self.cutoff_x is not None and position[0] > self.cutoff_x:
            return float(self.baseline)
        distance = (np.asarray(position) - self.centers) / self.sigmas
        radius_squared = np.sum(distance ** 2, axis=1)
        if self.plateau_edge is not None:
            outside = np.maximum(np.sqrt(radius_squared) - 1.0, 0.0)
            radius_squared = (outside / self.plateau_edge) ** 2
        gaussian = np.exp(-0.5 * radius_squared)
        return float(self.baseline + gaussian @ self.amplitudes)

    def single_peak_position(self, bounds):
        """单峰在当前搜索框内的峰位，供绘图标记。"""
        if len(self.centers) != 1 or self.plateau_edge is not None:
            return None
        lower, upper = np.asarray(bounds, dtype=float).T
        peak = np.clip(self.centers[0], lower, upper)
        if self.cutoff_x is not None:
            if self.cutoff_x < lower[0]:
                return None
            peak[0] = min(peak[0], self.cutoff_x)
        return peak


def build_formula_ed_model(mode="formula_gaussian"):
    return FormulaModel(**FORMULA_CONFIGS[mode])
