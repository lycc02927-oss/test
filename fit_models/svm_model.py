"""不依赖 sklearn 的 RBF 支持向量回归，factory.py 用 svm 创建。
继承 base.py；fit 使用 L-BFGS-B 拟合并筛出支持点，predict 用支持点计算 ED。
C 越大越重视拟合误差；gamma 越大影响范围越小；epsilon 是无惩罚误差宽度。
maxiter 控制最大优化次数；参数均用于标准化数据，return_std=True 返回经验标准差。"""
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .base import ScaledModelBase


class SVMModel(ScaledModelBase):
    """用 RBF 核和平方 epsilon 不敏感损失拟合支持向量回归。"""

    def __init__(self, C=10.0, gamma=1.0, epsilon=0.05, maxiter=1000):
        """保存 C、gamma、epsilon、maxiter，参数作用见本文件开头。"""
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.maxiter = maxiter

    def _kernel(self, X1, X2):
        """根据 X1、X2 的平方距离与 gamma，返回 RBF 核矩阵。"""
        distance = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-self.gamma * distance)

    def _fit_scaled(self, X_scaled, y_scaled):
        """分解训练核矩阵并优化权重，保存有效支持点、系数和偏置。"""
        kernel = self._kernel(X_scaled, X_scaled)
        # K = L @ L.T；在变换后的权重空间优化，避免直接优化病态核系数。
        factor = cholesky(kernel + np.eye(len(kernel)) * 1e-10, lower=True)

        def loss_and_gradient(parameters):
            """由权重和偏置 parameters 计算损失与解析梯度，只惩罚 epsilon 管外残差。"""
            weights = parameters[:-1]
            fitted = factor @ weights
            residual = fitted + parameters[-1] - y_scaled
            outside = np.sign(residual) * np.maximum(np.abs(residual) - self.epsilon, 0)
            loss = 0.5 * weights @ weights + 0.5 * self.C * np.sum(outside ** 2)
            gradient = np.r_[weights + self.C * factor.T @ outside,
                             self.C * np.sum(outside)]
            return loss, gradient

        result = minimize(
            loss_and_gradient, np.zeros(len(y_scaled) + 1), jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": 1e-12, "gtol": 1e-6},
        )
        # 在最优解处，系数等于误差管外部分乘以 -C。
        residual = factor @ result.x[:-1] + result.x[-1] - y_scaled
        weights = -self.C * np.sign(residual) * np.maximum(np.abs(residual) - self.epsilon, 0)
        support = np.abs(weights) > 1e-7
        self.support_X = X_scaled[support]
        self.coefficients = weights[support]
        self.bias = float(result.x[-1])

    def _predict_mean_scaled(self, query_scaled):
        """用 query_scaled 到支持点的核值计算均值，并加上偏置。"""
        return self._kernel(query_scaled, self.support_X) @ self.coefficients + self.bias
