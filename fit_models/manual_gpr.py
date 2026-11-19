"""不依赖 sklearn 的 Matern 5/2 GPR 拟合模型，factory.py 用 manual_gpr 创建。
继承 base.py 的标准化接口；fit 优化核参数并分解核矩阵，predict 计算均值和标准差。
length_scale：初始相关长度，越大越平滑；signal_variance：初始信号方差。
noise：初始噪声；optimize_hyperparameters：是否自动优化；maxiter：最大优化次数。
random_state 保留调用参数，当前模型无随机步骤，因此不影响拟合结果。"""
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .base import ScaledModelBase


class ManualGPRModel(ScaledModelBase):
    """通过 Matern 核和 Cholesky 分解计算 GPR 均值与后验标准差。"""
    def __init__(
        self,
        length_scale=1.0,
        signal_variance=1.0,
        noise=1e-4,
        optimize_hyperparameters=True,
        maxiter=80,
        random_state=42,
    ):
        """保存核参数初值及优化设置，参数作用见本文件开头。"""
        super().__init__()
        self.initial_length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise = noise
        self.optimize_hyperparameters = optimize_hyperparameters
        self.maxiter = maxiter

    @staticmethod
    def _matern52(X1, X2, length_scale, signal_variance):
        """按 length_scale 缩放 X1、X2 的距离，返回 Matern 5/2 核矩阵。"""
        distances = cdist(X1 / length_scale, X2 / length_scale)
        root5_distance = np.sqrt(5.0) * distances
        return signal_variance * (
            1.0 + root5_distance + (root5_distance ** 2) / 3.0
        ) * np.exp(-root5_distance)

    def _factor(self, parameters):
        """用长度尺度、方差、噪声构造训练核矩阵，返回 Cholesky 下三角矩阵。"""
        length_scale, signal_variance, noise = parameters
        kernel = self._matern52(
            self.X_train_scaled,
            self.X_train_scaled,
            length_scale,
            signal_variance,
        )
        kernel[np.diag_indices_from(kernel)] += noise + 1e-10
        return cholesky(kernel, lower=True, check_finite=False)

    def _negative_log_likelihood(self, log_parameters):
        """将 log_parameters 还原为正参数，返回负对数似然；分解失败时给大惩罚。"""
        parameters = np.exp(log_parameters)
        try:
            factor = self._factor(
                (parameters[:3], parameters[3], parameters[4])
            )
        except np.linalg.LinAlgError:
            return 1e30
        alpha = cho_solve(
            (factor, True),
            self.y_train_scaled,
            check_finite=False,
        )
        return float(
            0.5 * self.y_train_scaled @ alpha
            + np.sum(np.log(np.diag(factor)))
            + 0.5 * len(self.y_train_scaled) * np.log(2.0 * np.pi)
        )

    def _fit_scaled(self, X_scaled, y_scaled):
        """优化标准化训练数据的核参数，并保存核分解和预测系数。"""
        self.y_train_scaled = np.asarray(y_scaled, dtype=float)
        initial_length = np.ones(3) * self.initial_length_scale

        initial = np.log(
            np.r_[initial_length, self.signal_variance, self.noise]
        )
        if self.optimize_hyperparameters:
            result = minimize(
                self._negative_log_likelihood,
                initial,
                method="L-BFGS-B",
                bounds=[(-4.0, 4.0)] * 3 + [(-6.0, 6.0), (-14.0, 0.0)],
                options={"maxiter": self.maxiter},
            )
            optimum = result.x if np.isfinite(result.fun) else initial
        else:
            optimum = initial

        values = np.exp(optimum)
        self.length_scale_ = values[:3]
        self.signal_variance_ = float(values[3])
        self.noise_ = float(values[4])
        self.L = self._factor(
            (self.length_scale_, self.signal_variance_, self.noise_)
        )
        self.alpha_vector = cho_solve(
            (self.L, True),
            self.y_train_scaled,
            check_finite=False,
        )

    def _predict_mean_scaled(self, query_scaled):
        """用查询点与训练点的交叉核计算 query_scaled 的标准化预测均值。"""
        cross_kernel = self._matern52(
            self.X_train_scaled,
            query_scaled,
            self.length_scale_,
            self.signal_variance_,
        )
        return cross_kernel.T @ self.alpha_vector

    def _predict_scaled(self, query_scaled, return_std):
        """返回标准化均值，以及请求时的非负后验标准差。"""
        cross_kernel = self._matern52(
            self.X_train_scaled,
            query_scaled,
            self.length_scale_,
            self.signal_variance_,
        )
        mean = cross_kernel.T @ self.alpha_vector
        if not return_std:
            return mean, None
        solved = solve_triangular(
            self.L,
            cross_kernel,
            lower=True,
            check_finite=False,
        )
        variance = self.signal_variance_ - np.sum(solved ** 2, axis=0)
        return mean, np.sqrt(np.maximum(variance, 0.0))
