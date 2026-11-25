"""回归检查：档位、联合停止入口参数、轴向累计行程和停滞自适应。"""
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import simulation
from .design import BUDGET, ED_TOL, GA_SEED, MODELS, PATIENCE, Standard
from .encoding import BITS, GRID, decode, encode, repair
from .genetic import evolve, rates
from .excel import sheet_rows
from .objective import measure_path, metrics, recipe_means


CANDIDATE = dict(simulation.default_recipe(), ls_hi=300)


class GATest(unittest.TestCase):
    def test_gene_round_trip(self):
        for key, levels in GRID.items():
            for value in levels:
                recipe = dict(CANDIDATE)
                recipe[key] = value
                self.assertEqual(decode(encode(recipe)), recipe)
        rng = random.Random(22)
        for _ in range(400):
            bits = format(rng.getrandbits(BITS), f"0{BITS}b")
            recipe = decode(repair(bits, rng))
            self.assertLessEqual(recipe["xi_final"], recipe["xi_init"])
            self.assertLess(recipe["ls_lo"], recipe["ls_hi"])

    def test_return_and_axis_order(self):
        xyz = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0]])
        axes, largest_axis, simultaneous = measure_path(xyz, best_index=0)
        np.testing.assert_array_equal(axes, [4, 6, 0])
        self.assertEqual((largest_axis, simultaneous), (6, 8))

    def test_loss_penalizes_each_cost(self):
        def score(points, values):
            history = dict(X=np.asarray(points), y=np.asarray(values),
                           n_init=1, stop_reason="converged")
            return metrics(history, peak=1.0, span=600, patience_round=31)

        xyz = [[0, 0, 0], [2, 0, 0], [2, 3, 0]]
        values = [1.0, 0.8, 0.8]
        result = score(xyz, values)
        self.assertEqual(result["travel"], 6)
        self.assertEqual(result["sum_step_max"], 8)
        expected = (5 * 3 / BUDGET + 2 * 8 / (BUDGET * 600)) / 10
        self.assertAlmostEqual(result["base_loss"], expected)
        expected *= 0.5
        self.assertAlmostEqual(result["loss"], expected)
        # 控制另外两项不变，分别增大路程、点数和 ED 偏差。
        self.assertGreater(score(np.asarray(xyz) * 2, values)["loss"], expected)
        self.assertGreater(score(xyz + [xyz[-1]], values + [0.8])["loss"], expected)
        self.assertGreater(score(xyz, [0.95, 0.8, 0.8])["loss"], expected)

    def test_failure_is_half_of_loss_without_priority(self):
        def score(base, failures):
            rows = []
            for model, fail in zip(MODELS, failures):
                hist = dict(X=np.array([[0, 0, 0]]), y=np.array([1.0 - base]),
                            n_init=1, stop_reason="max_iter")
                row = metrics(hist, 1.0, 600.0, None if fail else 31)
                rows.append(dict(model=model, **row))
            result = recipe_means(rows)
            expected = 0.5 * result["base_loss"] + 0.5 * sum(failures) / 3
            self.assertAlmostEqual(result["loss"], expected)
            return result["loss"]
        self.assertLess(score(0.01, [1, 0, 0]), score(0.20, [0, 0, 0]))
        self.assertGreater(score(0.01, [1, 0, 0]), score(0.01, [0, 0, 0]))
        self.assertEqual(tuple(MODELS), (2, 3, 4))

    def test_patience_without_ed_accuracy(self):
        # 常数轨迹第 31 个 BO 点达到停滞门槛，但离设计峰 (1.0) 仍有 40%。
        check = simulation.batch_test.get_convergence_iteration
        h = dict(y=np.full(32, .6), best_value=np.full(32, .6), sigma_abs_used=np.zeros(31), n_init=1)
        self.assertEqual(check(h, 30, 1.0, np.inf), 31)
        self.assertIsNone(check(h, 30, 1.0, 10.0))
        self.assertIsNone(check(h, 31, 1.0, np.inf))

    def test_excel_scores_the_requested_distance(self):
        with TemporaryDirectory() as path:
            directory = Path(path)
            trial = "best_ed,design_ed,samples,travel_x,travel_y,travel_z,travel,sum_step_max\n"
            (directory / "trials.csv").write_text(trial + "1,1,3,4,6,0,6,8\n")
            for settings, expected in [
                ("distance_metric\nsum_step_max\n", "H2"),  # 本版
                ("plan\nstandard\n", "G2"),                    # 旧批次的历史 loss
            ]:
                (directory / "settings.csv").write_text(settings)
                _, row = sheet_rows(directory, "trials")
                self.assertIn("+" + expected + "/", row[-1])

    def test_adaptation(self):
        self.assertEqual(rates(0), rates(1))
        self.assertTrue(all(b > a for a, b in zip(rates(1), rates(2))))
        self.assertEqual(rates(8), rates(50))

    def test_curve_starts_at_true_default_and_tracks_elite(self):
        calls = []
        default = simulation.default_recipe()
        def evaluate(bits, recipe, generation):
            calls.append((bits, recipe, generation))
            loss = recipe["init_points"] / 100
            return dict(loss=loss, base_loss=2 * loss, patience_failed=0)
        settings = Standard(population=6, generations=3)
        winner, history, members = evolve(settings, GA_SEED, default, evaluate)
        self.assertEqual(calls[0], ("default", default, 0))
        self.assertEqual(default["ls_hi"], simulation.search.BO_PARAMETERS["ls_bounds"][1])
        self.assertEqual([r["generation"] for r in history], [0, 1, 2, 3])
        self.assertEqual(history[0]["mean_loss"], default["init_points"] / 100)
        self.assertEqual(len(calls), len({key for key, _, _ in calls}))
        for row in history[1:]:
            losses = [m["loss"] for m in members if m["generation"] == row["generation"]]
            self.assertEqual(row["mean_loss"], min(losses))
            self.assertAlmostEqual(row["population_mean_loss"], sum(losses) / len(losses))
        self.assertEqual(winner, history[-1]["champion"])

    def test_both_restarts_reach_original_bo(self):
        recipe = dict(CANDIDATE, n_restarts=70, n_restarts_ei=40)
        original_params = simulation.search.BO_PARAMETERS.copy()

        def observe(case, reference, rounds, patience, percent):
            self.assertEqual((case["data_model"], case["target"]), (2, "ed"))
            self.assertEqual((rounds, patience, percent), (BUDGET - recipe["init_points"], PATIENCE, 100 * ED_TOL))
            self.assertEqual(simulation.search.BO_PARAMETERS["n_restarts_ei"], 40)
            simulation.search.fit_gp(n_restarts=10)
            # 到这里已测到适配参数，使用测试异常退出以检查上下文复位。
            raise StopIteration

        with patch.object(simulation.search, "fit_gp") as gp:
            with patch.object(simulation.batch_test, "run_search_with_accuracy", observe):
                with self.assertRaises(StopIteration):
                    simulation.simulate((recipe, 2, 42))
            gp.assert_called_once_with(n_restarts=70)
            self.assertIs(simulation.search.fit_gp, gp)
        self.assertEqual(simulation.search.BO_PARAMETERS, original_params)


if __name__ == "__main__":
    unittest.main()
