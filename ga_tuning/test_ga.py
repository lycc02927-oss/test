"""回归检查：档位、联合停止入口参数、轴向累计行程和停滞自适应。"""
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import simulation
from .design import BUDGET, ED_TOL, GA_SEED, PATIENCE, Standard, fitted, formula, peak
from .encoding import BITS, GRID, decode, encode, repair
from .genetic import evolve, rates
from .excel import sheet_rows
from .report import Recorder
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
        self.assertAlmostEqual(result["loss"], expected)
        # 控制另外两项不变，分别增大路程、点数和 ED 偏差。
        self.assertGreater(score(np.asarray(xyz) * 2, values)["loss"], expected)
        self.assertGreater(score(xyz + [xyz[-1]], values + [0.8])["loss"], expected)
        self.assertGreater(score(xyz, [0.95, 0.8, 0.8])["loss"], expected)

    def test_failure_adds_one_sixth_without_priority(self):
        def score(base, failures):
            rows = []
            for model, fail in enumerate(failures):
                hist = dict(X=np.array([[0, 0, 0]]), y=np.array([1.0 - base]),
                            n_init=1, stop_reason="max_iter")
                row = metrics(hist, 1.0, 600.0, None if fail else 31)
                rows.append(dict(model=model, **row))
            result = recipe_means(rows)
            expected = result["base_loss"] + sum(failures) / len(failures) / 6
            self.assertAlmostEqual(result["loss"], expected)
            return result["loss"]
        self.assertLess(score(0.01, [1, 0, 0]), score(0.20, [0, 0, 0]))
        self.assertGreater(score(0.01, [1, 0, 0]), score(0.01, [0, 0, 0]))
        # 旧 50/50 算法会将零失败者排在前；按新公式，低基础成本者胜出。
        self.assertLess(score(0.01, [1, 0, 0]), score(0.05, [0, 0, 0]))
        self.assertAlmostEqual(score(0.01, [1]) - score(0.01, [0]), 1 / 6)

    def test_search_records_patience_separately_from_joint_stop(self):
        # 压缩到 2 轮以逐点核对：首次停滞、ED 门槛和恰在停滞轮出现的新高。
        search = simulation.search
        point = np.zeros(3)
        with patch.object(search, "get_initial_points", return_value=point[None, :]), \
             patch.object(search, "fit_gp") as gp, \
             patch.object(search, "get_next_sample", return_value=(point, 0.0)):
            gp.return_value.y_mean = 0.
            gp.return_value.y_std = 1.
            gp.return_value.kernel = object()

            def run(values, reference=1.0):
                readings = iter(values)
                return search.bayesian_optimization(
                    lambda _: next(readings), search.BOUNDS, init_points=1, max_iter=len(values) - 1,
                    stop_no_improve=2, reference_value=reference, relative_tolerance=.1, verbose=False)

            low = run([.6] * 6)
            self.assertEqual((low["patience_round"], low["stop_reason"]), (3, "max_iter"))
            late_peak = run([.6, .6, .6, .6, 1., 1., 1., 1.])
            self.assertEqual((late_peak["patience_round"], late_peak["stop_reason"]), (3, "converged"))
            self.assertEqual(len(late_peak["xi_used"]), 7)
            new_peak = run([.6, .6, .6, 1., 1., 1., 1.])
            self.assertEqual(new_peak["patience_round"], 6)
            self.assertEqual(len(new_peak["xi_used"]), 6)
            self.assertEqual(len(run([.6] * 6, reference=None)["xi_used"]), 3)

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

    def test_excel_reexports_old_and_new_scoring(self):
        with TemporaryDirectory() as path:
            directory = Path(path)
            header = "best_ed,design_ed,samples,travel_x,travel_y,travel_z,travel,sum_step_max,patience_failed"
            (directory / "trials.csv").write_text(header + "\n1,1,3,4,6,0,6,8,1\n")
            for parameter, value in [("failure_divisor", 6), ("failure_weight", 0.5)]:
                (directory / "settings.csv").write_text(f"distance_metric,{parameter}\nsum_step_max,{value}\n")
                _, row = sheet_rows(directory, "trials")
                if parameter == "failure_divisor":
                    self.assertTrue(row[-1].endswith("+I2/'试验设置'!$N$2"))
                else:
                    self.assertIn("(1-'试验设置'!$N$2)", row[-1])

    def test_custom_models_forwarded_and_recorded_with_distinct_names(self):
        from fitter import FORMULA_CONFIGS
        a = peak(0.65, (-50, 20, 10), (50, 60, 40))
        b = peak(0.90, (80, -30, 10), (90, 60, 50))
        models = {"同表 RBF": fitted(2), "同表线性": fitted(2, "linear"),
                  "双峰甲": formula("formula_multi_gaussian", a, b),
                  "双峰乙": formula("formula_multi_gaussian", b, a),
                  "平台": formula("formula_plateau")}
        self.assertEqual(models["平台"].options["peaks"], FORMULA_CONFIGS["formula_plateau"]["peaks"])
        self.assertEqual(models["双峰甲"].reference, (*b["center"], b["peak_ed"]))
        self.assertEqual(models["同表 RBF"].reference, models["同表线性"].reference)
        x = np.array([[0, 0, 0]])
        with patch.object(simulation.search, "run_search", return_value=dict(
                X=x, y=[.86], n_init=1, ed=[.86], pf=[6.0], patience_round=None,
                stop_reason="max_iter")) as entry:
            for name, case in models.items():
                result, _ = simulation.simulate((CANDIDATE, name, case, 42))
                self.assertEqual(result["model"], name)
                self.assertEqual(result["design_ed"], case.reference[3])
                args = entry.call_args.kwargs
                self.assertEqual(args["reference_value"], case.reference[3])
                for k, v in case.options.items():
                    self.assertEqual(args[k], v)
        with TemporaryDirectory() as path:
            directory = Path(path)
            Recorder(directory, Standard(), models)
            fields, *rows = sheet_rows(directory, "models")
            self.assertEqual([row[fields.index("model")] for row in rows], list(models))
            fields, *rows = sheet_rows(directory, "peaks")
            records = [dict(zip(fields, row)) for row in rows]
            self.assertEqual(len(records), 5)
            self.assertEqual(records[1]["peak_ed"], b["peak_ed"])
            self.assertEqual(records[1]["sigma_y"], b["sigma"][1])

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
            failed = 0.5 if recipe["init_points"] % 2 == 0 else 0
            return dict(loss=loss + failed / 6, base_loss=loss, patience_failed=failed)
        settings = Standard(population=6, generations=3)
        winner, history, members = evolve(settings, GA_SEED, default, evaluate)
        self.assertEqual(calls[0], ("default", default, 0))
        self.assertEqual(default["ls_hi"], simulation.search.BO_PARAMETERS["ls_bounds"][1])
        self.assertEqual([r["generation"] for r in history], [0, 1, 2, 3])
        self.assertEqual(history[0]["mean_loss"], default["init_points"] / 100 + 0.5 / 6)
        self.assertEqual(len(calls), len({key for key, _, _ in calls}))
        for row in history[1:]:
            losses = [m["loss"] for m in members if m["generation"] == row["generation"]]
            self.assertEqual(losses, sorted(losses))
            self.assertEqual(row["mean_loss"], min(losses))
            expected = row["population_mean_base_loss"] + row["population_failure_rate"] / 6
            self.assertAlmostEqual(row["population_mean_loss"], expected)
            self.assertAlmostEqual(row["population_mean_loss"], sum(losses) / len(losses))
        self.assertEqual(winner, history[-1]["champion"])

    def test_both_restarts_reach_original_bo(self):
        recipe = dict(CANDIDATE, n_restarts=70, n_restarts_ei=40)
        original_params = simulation.search.BO_PARAMETERS.copy()

        search = simulation.search
        with patch.object(search, "run_search", wraps=search.run_search) as entry, \
             patch.object(search, "fit_gp") as gp, \
             patch.object(search, "get_next_sample", side_effect=StopIteration) as ei:
            gp.return_value.y_mean = 0.
            gp.return_value.y_std = 1.
            with self.assertRaises(StopIteration):
                simulation.simulate((recipe, "LT", fitted(2), 42))
        options = entry.call_args.kwargs
        self.assertEqual((options["data_model"], options["seed"]), (2, 42))
        self.assertEqual(options["max_iter"] + options["init_points"], BUDGET)
        self.assertEqual(options["stop_no_improve"], PATIENCE)
        self.assertEqual(options["relative_tolerance"], ED_TOL)
        self.assertEqual(options["reference_value"], fitted(2).reference[3])
        self.assertEqual(gp.call_args.kwargs["n_restarts"], 70)
        self.assertEqual(ei.call_args.kwargs["n_restarts_ei"], 40)
        self.assertEqual(simulation.search.BO_PARAMETERS, original_params)


if __name__ == "__main__":
    unittest.main()
