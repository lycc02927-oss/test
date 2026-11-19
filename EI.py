"""最大化目标的负 EI，供 L-BFGS-B 最小化。"""

import numpy as np
from scipy.stats import norm


def expected_improvement(X, gpr, best_norm, xi=0.0, eps=1e-9):
    """best_norm、xi、预测均值及标准差均处于标准化空间。"""
    X = np.atleast_2d(X)
    mu, sigma = gpr.predict_normalized(X, return_std=True)
    mu = np.asarray(mu).ravel()
    sigma = np.asarray(sigma).ravel()
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return 1e25
    sigma = np.maximum(sigma, eps)
    imp = mu - best_norm - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = np.maximum(ei, 0.0)
    ei[ei < 1e-12] = 0.0
    return -ei if ei.size > 1 else -float(ei[0])
