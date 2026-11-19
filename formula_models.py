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
}


class FormulaModel:
    def __init__(self, baseline, components):
        self.baseline = baseline
        self.amplitudes = np.array([part["amplitude"] for part in components])
        self.centers = np.array([part["center"] for part in components])
        self.sigmas = np.array([part["sigma"] for part in components])

    def predict(self, position):
        distance = (np.asarray(position) - self.centers) / self.sigmas
        gaussian = np.exp(-0.5 * np.sum(distance ** 2, axis=1))
        return float(self.baseline + gaussian @ self.amplitudes)


def build_formula_ed_model(mode="formula_gaussian"):
    return FormulaModel(**FORMULA_CONFIGS[mode])
