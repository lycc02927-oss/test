"""拟合模型共用的标准化和预测接口，由五种模型继承。
fit(X, y) 先标准化三维坐标与 ED，再训练子类；predict(X) 将结果还原为 ED。
predict(X, return_std=True) 同时返回标准差；GPR 用后验标准差，其它模型用经验估计。
X 每行一个三维坐标，y 为对应 ED；常量列和极小残差的稳定处理保留。"""
import numpy as np
from scipy.spatial.distance import cdist


class ManualStandardScaler:
    """保存训练均值和尺度，完成标准化与原始数值还原。"""
    def fit(self, values):
        """计算 values 每列均值和标准差，常量列尺度用 1；返回当前标准化器。"""
        array = np.asarray(values, dtype=float)
        self.mean_ = np.mean(array, axis=0)
        scale = np.std(array, axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, values):
        """用已保存的均值和尺度标准化 values，返回同形状数组。"""
        array = np.asarray(values, dtype=float)
        return (array - self.mean_) / self.scale_

    def inverse_transform(self, values):
        """把标准化 values 还原为原始尺度，返回同形状数组。"""
        array = np.asarray(values, dtype=float)
        return array * self.scale_ + self.mean_


class ScaledModelBase:
    """统一处理 X、y 标准化和预测结果还原，具体训练算法由子类提供。"""
    def __init__(self):
        """创建三维坐标与 ED 的两个标准化器。"""
        self.x_scaler = ManualStandardScaler()
        self.y_scaler = ManualStandardScaler()

    def fit(self, X, y):
        """标准化 X、y 后训练子类，记录残差标准差；返回当前模型。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        X_scaled = self.x_scaler.fit(X).transform(X)
        y_scaled = self.y_scaler.fit(y.reshape(-1, 1)).transform(
            y.reshape(-1, 1)
        ).ravel()
        self.X_train_scaled = X_scaled
        self._fit_scaled(X_scaled, y_scaled)
        fitted = np.asarray(self._predict_mean_scaled(X_scaled), dtype=float).reshape(-1)
        self.residual_std_scaled = max(
            float(np.sqrt(np.mean((y_scaled - fitted) ** 2))),
            1e-6,
        )
        return self

    def predict(self, X, return_std=False):
        """返回 X 对应 ED；return_std=True 时同时返回非负标准差。"""
        X_scaled = self.x_scaler.transform(np.atleast_2d(X))
        mean_scaled, std_scaled = self._predict_scaled(X_scaled, return_std)
        mean = self.y_scaler.inverse_transform(
            np.asarray(mean_scaled, dtype=float).reshape(-1, 1)
        ).ravel()
        if not return_std:
            return mean
        std = np.asarray(std_scaled, dtype=float).reshape(-1) * self.y_scaler.scale_[0]
        return mean, np.maximum(std, 0.0)

    def _empirical_std_scaled(self, query_scaled):
        """按训练残差和最近训练点距离，返回 query_scaled 的经验标准差。"""
        nearest = np.min(cdist(query_scaled, self.X_train_scaled), axis=1)
        return self.residual_std_scaled * (1.0 + nearest)

    def _predict_scaled(self, query_scaled, return_std):
        """返回标准化均值；需要标准差时附加经验估计，否则附加 None。"""
        mean = self._predict_mean_scaled(query_scaled)
        if not return_std:
            return mean, None
        return mean, self._empirical_std_scaled(query_scaled)
