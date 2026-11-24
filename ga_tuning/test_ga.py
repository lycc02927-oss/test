"""回归检查：档位、联合停止入口参数、轴向累计行程和停滞自适应。"""
import random
import unittest
from unittest.mock import patch

import numpy as np

from . import simulation
from .design import BUDGET, ED_TOL, GA_SEED, PATIENCE, PLANS
from .encoding import BASELINE, BITS, GRID, decode, encode, repair
from .genetic import evolve, rates
from .objective import measure_path


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

    def test_adaptation(self):
        self.assertEqual(rates(0), rates(1))
        self.assertTrue(all(b > a for a, b in zip(rates(1), rates(2))))
        self.assertEqual(rates(8), rates(50))

    def test_balanced_first_generation(self):
        # 上轮试验漏采低点数，本测试锁定初代范围与同参数配对对照。
        generations = []
        def evaluate(bits):
            generations.append([decode(b) for b in bits])
            return {b: (0, 0, decode(b)["init_points"] / 100) for b in bits}
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
