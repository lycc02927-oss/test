"""Matérn 5/2、白噪声及组合核，超参数在对数空间优化。"""

import numpy as np
from scipy.spatial.distance import cdist


class MaternKernel:
    """theta = [log(length_scale), log(constant_value)]。"""

    def __init__(self, length_scale=20.0, constant_value=1.0,
                 length_scale_bounds=(5.0, 500.0),
                 constant_value_bounds=(1e-3, 1e3)):
        self.length_scale = float(length_scale)
        self.constant_value = float(constant_value)
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.constant_value_bounds = tuple(constant_value_bounds)

    @property
    def theta(self):
        return np.array([np.log(self.length_scale), np.log(self.constant_value)])

    @theta.setter
    def theta(self, theta):
        self.length_scale = float(np.exp(theta[0]))
        self.constant_value = float(np.exp(theta[1]))

    @property
    def bounds(self):
        return np.array([np.log(self.length_scale_bounds),
                         np.log(self.constant_value_bounds)])

    def __call__(self, X1, X2=None, eval_gradient=False):
        X1 = np.asarray(X1, dtype=float)
        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)
        r = cdist(X1 / self.length_scale, X2 / self.length_scale)
        sq5 = np.sqrt(5.0)
        exp_term = np.exp(-sq5 * r)
        base = (1.0 + sq5 * r + (5.0 / 3.0) * r ** 2) * exp_term
        K = self.constant_value * base
        if not eval_gradient:
            return K
        dK_dlogls = (self.constant_value * (5.0 / 3.0) * r ** 2
                     * (1.0 + sq5 * r) * exp_term)
        return K, np.stack([dK_dlogls, K.copy()], axis=-1)

    def diag(self, X):
        return self.constant_value * np.ones(X.shape[0])


class WhiteKernel:
    """noise_level 表示标准化空间的噪声方差。"""

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-6, 1e-2)):
        self.noise_level = float(noise_level)
        self.noise_level_bounds = tuple(noise_level_bounds)

    @property
    def theta(self):
        return np.array([np.log(self.noise_level)])

    @theta.setter
    def theta(self, theta):
        self.noise_level = float(np.exp(theta[0]))

    @property
    def bounds(self):
        return np.array([np.log(self.noise_level_bounds)])

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient requires Y=None.")
        if Y is not None:
            return np.zeros((X.shape[0], Y.shape[0]))
        K = self.noise_level * np.eye(X.shape[0])
        if eval_gradient:
            return K, K[:, :, None]
        return K

    def diag(self, X):
        return self.noise_level * np.ones(X.shape[0])


class SumKernel:
    def __init__(self, k1, k2):
        self.k1, self.k2 = k1, k2

    @property
    def theta(self):
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, theta):
        n1 = self.k1.theta.size
        self.k1.theta = theta[:n1]
        self.k2.theta = theta[n1:]

    @property
    def bounds(self):
        return np.vstack([self.k1.bounds, self.k2.bounds])

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            K1, dK1 = self.k1(X, Y, eval_gradient=True)
            K2, dK2 = self.k2(X, Y, eval_gradient=True)
            return K1 + K2, np.concatenate([dK1, dK2], axis=-1)
        return self.k1(X, Y) + self.k2(X, Y)

    def diag(self, X):
        return self.k1.diag(X) + self.k2.diag(X)

    def diag_noise_free(self, X):
        return self.k1.diag(X)

    @property
    def length_scale(self):
        return self.k1.length_scale
