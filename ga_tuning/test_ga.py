"""回归检查：档位、联合停止入口参数、轴向累计行程和停滞自适应。"""
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import simulation
from .design import BUDGET, ED_TOL, GA_SEED, PATIENCE, PLANS
from .encoding import BASELINE, BITS, GRID, decode, encode, repair
from .genetic import evolve, rates
from .excel import sheet_rows
from .objective import RecipeScore, measure_path, metrics, rank_trials


class GATest(unittest.TestCase):
    def test_gene_round_trip(self):
        for key, levels in GRID.items():
            for value in levels:
                recipe = dict(BASELINE)
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
            return metrics(history, peak=1.0, span=600)

        xyz = [[0, 0, 0], [2, 0, 0], [2, 3, 0]]
        values = [1.0, 0.8, 0.8]
        result = score(xyz, values)
        self.assertEqual(result["travel"], 6)
        self.assertEqual(result["sum_step_max"], 8)
        expected = (5 * 3 / BUDGET + 2 * 8 / (BUDGET * 600)) / 10
        self.assertAlmostEqual(result["loss"], expected)
        # 控制另外两项不变，分别增大路程、点数和 ED 偏差。
        self.assertGreater(score(np.asarray(xyz) * 2, values)["loss"], expected)
        self.assertGreater(score(xyz + [xyz[-1]], values + [0.8])["loss"], expected)
        self.assertGreater(score(xyz, [0.95, 0.8, 0.8])["loss"], expected)

    def test_minimize_loss_within_convergence_constraint(self):
        def candidate(loss, successes):
            return rank_trials([dict(model=m, loss=loss, success=s)
                                for m, s in zip((2, 3, 4, 5), successes)])
        self.assertLess(candidate(0.2, [1, 1, 1, 1]), candidate(0.4, [1, 1, 1, 1]))
        self.assertLess(candidate(0.4, [1, 1, 1, 1]), candidate(0.2, [0, 1, 1, 1]))
        self.assertLess(candidate(0.2, [0, 0, 1, 1]), candidate(0.4, [0, 1, 1, 1]))

    def test_excel_scores_the_requested_distance(self):
        with TemporaryDirectory() as path:
            directory = Path(path)
            trial = "best_ed,design_ed,samples,travel_x,travel_y,travel_z,travel,sum_step_max\n"
            (directory / "trials.csv").write_text(trial + "1,1,3,4,6,0,6,8\n")
            for settings, expected in [
                ("distance_metric\nsum_step_max\n", "H2"),  # 本版
                ("plan\npilot\n", "G2"),                    # 旧批次的历史 loss
            ]:
                (directory / "settings.csv").write_text(settings)
                _, row = sheet_rows(directory, "trials")
                self.assertIn("+" + expected + "/", row[-1])

    def test_adaptation(self):
        self.assertEqual(rates(0), rates(1))
        self.assertTrue(all(b > a for a, b in zip(rates(1), rates(2))))
        self.assertEqual(rates(8), rates(50))

    def test_balanced_first_generation(self):
        # 上轮试验漏采低点数，本测试锁定初代范围与同参数配对对照。
        generations = []
        def evaluate(bits):
            generations.append([decode(b) for b in bits])
            return {b: RecipeScore(False, decode(b)["init_points"] / 100, 0, 0) for b in bits}
        evolve(PLANS["balanced"], GA_SEED, evaluate)
        initial = generations[0]
        self.assertEqual([r["init_points"] for r in initial], [30, 10, 10, 23, 37, 50])
        self.assertEqual(initial[1], dict(BASELINE, init_points=10))

    def test_both_restarts_reach_original_bo(self):
        recipe = dict(BASELINE, n_restarts=70, n_restarts_ei=40)
        original_params = simulation.search.BO_PARAMETERS.copy()

        def observe(case, reference, rounds, patience, percent):
            self.assertEqual((case["data_model"], case["target"]), (3, "ed"))
            self.assertEqual((rounds, patience, percent), (BUDGET - 30, PATIENCE, 100 * ED_TOL))
            self.assertEqual(simulation.search.BO_PARAMETERS["n_restarts_ei"], 40)
            simulation.search.fit_gp(n_restarts=10)
            # 到这里已测到适配参数，使用测试异常退出以检查上下文复位。
            raise StopIteration

        with patch.object(simulation.search, "fit_gp") as gp:
            with patch.object(simulation.batch_test, "run_search_with_accuracy", observe):
                with self.assertRaises(StopIteration):
                    simulation.simulate((encode(recipe), 3, 42))
            gp.assert_called_once_with(n_restarts=70)
            self.assertIs(simulation.search.fit_gp, gp)
        self.assertEqual(simulation.search.BO_PARAMETERS, original_params)


if __name__ == "__main__":
    unittest.main()
