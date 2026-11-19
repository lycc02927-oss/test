"""径向基函数 ED 插值模型，factory.py 用 rbf 创建。
继承 base.py 的标准化接口，使用 SciPy RBFInterpolator 训练和预测。
kernel 选择核函数；smoothing 越大通常越平滑；neighbors 指定邻近训练点数量。
neighbors=None 使用全部训练点；return_std=True 返回基类的经验标准差。"""
import numpy as np
from scipy.interpolate import RBFInterpolator

from .base import ScaledModelBase


class RBFModel(ScaledModelBase):
    """在标准化坐标上使用 SciPy 的 RBF 插值器预测 ED。"""
    def __init__(self, kernel="thin_plate_spline", smoothing=1e-6, neighbors=None):
        """保存 kernel、smoothing、neighbors，参数作用见本文件开头。"""
        super().__init__()
        self.kernel = kernel
        self.smoothing = smoothing
        self.neighbors = neighbors

    def _fit_scaled(self, X_scaled, y_scaled):
        """使用标准化 X_scaled、y_scaled 和保存的参数建立 RBF 插值器。"""
        self.interpolator = RBFInterpolator(
            X_scaled,
            y_scaled,
            kernel=self.kernel,
            smoothing=self.smoothing,
            neighbors=self.neighbors,
        )

    def _predict_mean_scaled(self, query_scaled):
        """将 query_scaled 交给插值器，返回一维标准化预测数组。"""
        return np.asarray(self.interpolator(query_scaled), dtype=float).reshape(-1)
