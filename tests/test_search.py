"""新版数值计算及模型接入的回归检查；不运行批量寻优实验。"""

import contextlib
import io
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve
from scipy.spatial.distance import cdist

import search
from GPR import GaussianProcessRegressor
from kernel import MaternKernel, WhiteKernel, SumKernel


class NumericalTests(unittest.TestCase):
    def test_kernel_and_lml_gradients(self):
        X = np.array([[0., 0., 0.], [1., 2., 0.], [-2., 0., 1.], [2., 1., 1.]])
        kernel = SumKernel(MaternKernel(2.0, 1.3), WhiteKernel(0.003))
        theta = kernel.theta.copy()
        _, gradient = kernel(X, eval_gradient=True)
        gp = GaussianProcessRegressor(kernel, alpha=1e-8)
        y = np.array([-1.0, 0.5, 1.1, -0.1])
        _, lml_grad = gp._neg_lml_and_grad(theta, X, y)
        step = 1e-5
        for index in range(len(theta)):
            delta = np.eye(len(theta))[index] * step
            kernel.theta = theta + delta
            plus = kernel(X)
            kernel.theta = theta - delta
            minus = kernel(X)
            np.testing.assert_allclose(gradient[:, :, index], (plus - minus) / (2 * step),
                                       rtol=1e-5, atol=1e-7)
            f_plus = gp._neg_lml_and_grad(theta + delta, X, y)[0]
            f_minus = gp._neg_lml_and_grad(theta - delta, X, y)[0]
            self.assertAlmostEqual(lml_grad[index], (f_plus - f_minus) / (2 * step), places=5)
        kernel.theta = theta

    def test_alpha_once_and_noise_free_prediction(self):
        X = np.array([[0., 0., 0.], [1., 1., 0.], [2., 0., 1.]])
        gp = GaussianProcessRegressor(
            SumKernel(MaternKernel(2., length_scale_bounds=(0.5, 5.)), WhiteKernel(.003)),
            alpha=.02, n_restarts_optimizer=0).fit(X, [1., 2., 1.5])
        K = gp.kernel(X) + gp.alpha * np.eye(len(X))
        np.testing.assert_allclose(gp.L @ gp.L.T, K, atol=1e-10)
        query = np.array([[.5, .5, .5], [3., 2., 1.]])
        cross = gp.kernel(X, query)
        expected = gp.kernel.k1.diag(query) - np.sum(cross * cho_solve((gp.L, True), cross), axis=0)
        mu, std = gp.predict_normalized(query, return_std=True)
        np.testing.assert_allclose(std ** 2, np.maximum(expected, 0), atol=1e-10)
        raw_mu, raw_std = gp.predict(query, return_std=True)
        np.testing.assert_allclose(raw_mu, mu * gp.y_std + gp.y_mean)
        np.testing.assert_allclose(raw_std, std * gp.y_std)


class IntegrationTests(unittest.TestCase):
    @staticmethod
    def run_case(source='formula', target='ed', **overrides):
        parameters = dict(source=source, target=target, show=False, seed=42,
                          max_iter=3, init_points=6, n_restarts_ei=2,
                          verbose=False, stop_no_improve=10,
                          sampling_parameters={'pf_time_parameters': {'jump_probability': 1.0}})
        parameters.update(overrides)
        return search.run_search(**parameters)

    def check_history(self, history, target):
        self.assertEqual(len(history['y']), history['n_init'] + len(history['xi_used']))
        for key in ('X', 'ed', 'pf', 'ed_pf_ratio'):
            self.assertEqual(len(history[key]), len(history['y']))
            self.assertTrue(np.all(np.isfinite(history[key])))
        np.testing.assert_array_equal(history['y'], history[target])
        np.testing.assert_allclose(history['ed_pf_ratio'], history['ed'] / history['pf'])
        np.testing.assert_array_equal(history['best_value'],
                                      np.maximum.accumulate(history['y'])[history['n_init'] - 1:])
        self.assertTrue(np.all(history['X'] >= search.BOUNDS[:, 0]))
        self.assertTrue(np.all(history['X'] <= search.BOUNDS[:, 1]))
        for i, distance in enumerate(history['distance_to_nearest'], start=history['n_init']):
            self.assertAlmostEqual(distance, np.min(cdist([history['X'][i]], history['X'][:i])))

    def test_formula_and_fitted_both_targets(self):
        for source in ('formula', 'fitted'):
            for target in ('ed', 'ed_pf_ratio'):
                with self.subTest(source=source, target=target):
                    history = self.run_case(source, target,
                                            sigma3_rel=.03 if target == 'ed_pf_ratio' else 0.)
                    self.check_history(history, target)
                    self.assertTrue(np.all(history['noise_level'] > 0)
                                    if target == 'ed_pf_ratio'
                                    else np.all(history['noise_level'] == 0))

    def test_seed_reproduces_measurements_and_points(self):
        first = self.run_case()
        second = self.run_case()
        for key in ('X', 'y', 'ed', 'pf', 'ed_pf_ratio', 'xi_used'):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(first['xi_used'], [.01, .005, 0.])

    def test_printing_does_not_measure_twice(self):
        calls = []
        original = search.EDPFModel.measure

        def measure(model, position):
            calls.append(np.array(position))
            return original(model, position)

        output = io.StringIO()
        with patch.object(search.EDPFModel, 'measure', measure), contextlib.redirect_stdout(output):
            history = self.run_case(verbose=True)
        self.assertEqual(len(calls), len(history['y']))
        lines = [line for line in output.getvalue().splitlines()
                 if line.startswith(('Init ', 'Iter '))]
        self.assertEqual(len(lines), len(history['y']))
        for line in lines:
            self.assertIn('ED=', line)
            self.assertIn('PF=', line)
            self.assertIn('ED/PF=', line)

    def test_scalar_objective_exponential_xi_and_plot(self):
        history = search.bayesian_optimization(
            lambda x: float(np.exp(-np.sum(x ** 2) / 10000.)), search.BOUNDS,
            max_iter=3, init_points=6, n_restarts_ei=2,
            xi_init=.01, xi_final=.0001, verbose=False)
        np.testing.assert_allclose(history['xi_used'], [.01, .001, .0001])
        self.assertTrue(np.all(np.isnan(history['ed'])))
        figure = search.plot_optimization_results(history, show=False)
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 6)  # 四张图、xi 副轴、色条。
        plt.close(figure)

if __name__ == '__main__':
    unittest.main()
