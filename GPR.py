"""新版 GP 回归器：fit/LML 显式添加 alpha，Cholesky 只补 jitter。"""

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize


class GaussianProcessRegressor:
    def __init__(self, kernel=None, alpha=1e-10, normalize_y=True,
                 n_restarts_optimizer=10):
        self.kernel = kernel
        self.alpha = float(alpha)
        self.normalize_y = bool(normalize_y)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.y_mean, self.y_std = 0.0, 1.0
        self.X_train = None
        self.L = None
        self.alpha_vec = None
        self.log_marginal_likelihood_value_ = None

    def _cholesky(self, K):
        """只加 jitter，不加 alpha。"""
        for j in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            try:
                return cholesky(K + j * np.eye(K.shape[0]), lower=True)
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError("Cholesky failed")

    def _neg_lml_and_grad(self, theta, X, y):
        self.kernel.theta = theta
        K, dK = self.kernel(X, eval_gradient=True)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = self._cholesky(K)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros_like(theta)
        n = len(X)
        alpha_vec = cho_solve((L, True), y)
        lml = (-0.5 * y @ alpha_vec
               - np.sum(np.log(np.diag(L)))
               - 0.5 * n * np.log(2 * np.pi))
        L_inv = solve_triangular(L, np.eye(n), lower=True)
        K_inv = L_inv.T @ L_inv
        grad = np.zeros_like(theta)
        for i in range(theta.size):
            dKi = dK[:, :, i]
            grad[i] = 0.5 * (alpha_vec @ dKi @ alpha_vec
                             - np.sum(K_inv * dKi))
        return -float(lml), -grad

    def _optimize_hyperparameters(self, X, y):
        bounds = self.kernel.bounds
        theta0 = np.clip(self.kernel.theta, bounds[:, 0], bounds[:, 1])
        starts = [theta0]
        for _ in range(self.n_restarts_optimizer):
            starts.append(np.random.uniform(bounds[:, 0], bounds[:, 1]))
        best_theta, best_nll = theta0.copy(), np.inf
        for t0 in starts:
            try:
                res = minimize(self._neg_lml_and_grad, t0, args=(X, y),
                               bounds=bounds, method='L-BFGS-B', jac=True)
            except Exception:
                continue
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll = float(res.fun)
                best_theta = res.x.copy()
        self.kernel.theta = best_theta
        self.log_marginal_likelihood_value_ = -best_nll

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self.normalize_y:
            self.y_mean = float(y.mean())
            std = float(y.std())
            self.y_std = std if std > 1e-12 else 1.0
            y_norm = (y - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0.0, 1.0
            y_norm = y.copy()
        self.X_train = X
        self._optimize_hyperparameters(X, y_norm)
        K = self.kernel(X)
        K[np.diag_indices_from(K)] += self.alpha
        self.L = self._cholesky(K)
        self.alpha_vec = cho_solve((self.L, True), y_norm)
        return self

    def predict_normalized(self, X, return_std=False):
        X = np.atleast_2d(X)
        K_trans = self.kernel(self.X_train, X)
        y_mean = K_trans.T @ self.alpha_vec
        if not return_std:
            return y_mean
        v = cho_solve((self.L, True), K_trans)
        if hasattr(self.kernel, 'diag_noise_free'):
            k_xx = self.kernel.diag_noise_free(X)
        elif hasattr(self.kernel, 'diag'):
            k_xx = self.kernel.diag(X)
        else:
            k_xx = np.diag(self.kernel(X))
        y_var = np.maximum(k_xx - np.einsum('ij,ij->j', K_trans, v), 0.0)
        return y_mean, np.sqrt(y_var)

    def predict(self, X, return_std=False):
        if return_std:
            mu_n, s_n = self.predict_normalized(X, return_std=True)
            return mu_n * self.y_std + self.y_mean, s_n * self.y_std
        return self.predict_normalized(X) * self.y_std + self.y_mean
