"""线性插值 ED 模型，factory.py 用名称 linear 创建。
继承 base.py 的 fit/predict，训练线性与最近邻插值器；凸包外用最近邻补齐结果。
本模型无额外参数；return_std=True 可返回基类计算的经验标准差。"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .base import ScaledModelBase


class LinearInterpolationModel(ScaledModelBase):
    """先线性插值，凸包外无结果的位置用最近邻 ED 补齐。"""

    def _fit_scaled(self, X_scaled, y_scaled):
        """用标准化 X_scaled、y_scaled 建立线性和最近邻两个插值器。"""
        self.linear = LinearNDInterpolator(X_scaled, y_scaled, fill_value=np.nan)
        self.nearest = NearestNDInterpolator(X_scaled, y_scaled)

    def _predict_mean_scaled(self, query_scaled):
        """计算 query_scaled 的线性结果，补齐无效位置后返回一维预测数组。"""
        values = np.asarray(self.linear(query_scaled), dtype=float).reshape(-1)
        missing = ~np.isfinite(values)
        if np.any(missing):
            values[missing] = np.asarray(
                self.nearest(query_scaled[missing]),
                dtype=float,
            ).reshape(-1)
        return values
