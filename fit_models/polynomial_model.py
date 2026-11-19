"""多项式 ED 拟合模型，factory.py 用 polynomial 创建。
继承 base.py；fit 建立特征并求解系数，predict 用同样特征计算 ED。
degree=2 包含平方与交叉项，其它值只含常数和一次项。
regularization 控制系数惩罚，越大通常越平滑；截距不受惩罚。"""
import numpy as np

from .base import ScaledModelBase


class PolynomialModel(ScaledModelBase):
    """用一次或二次特征和正则化线性方程拟合 ED 趋势。"""
    def __init__(self, degree=2, regularization=1e-8):
        """保存 degree 与 regularization，参数作用见本文件开头。"""
        super().__init__()
        self.degree = degree
        self.regularization = regularization

    def _features(self, X):
        """把 X 转成常数、一次及可选二次特征；返回设计矩阵。"""
        columns = [np.ones(len(X))]
        columns.extend(X[:, index] for index in range(X.shape[1]))
        if self.degree == 2:
            for left in range(X.shape[1]):
                for right in range(left, X.shape[1]):
                    columns.append(X[:, left] * X[:, right])
        return np.column_stack(columns)

    def _fit_scaled(self, X_scaled, y_scaled):
        """由标准化 X_scaled、y_scaled 求解正则化系数，截距不加惩罚。"""
        design = self._features(X_scaled)
        penalty = np.eye(design.shape[1]) * self.regularization
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ y_scaled,
        )

    def _predict_mean_scaled(self, query_scaled):
        """用 query_scaled 的特征乘以已拟合系数，返回标准化预测均值。"""
        return self._features(query_scaled) @ self.coefficients
