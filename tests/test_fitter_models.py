import unittest

import numpy as np

from fitter import EDFitter, EDPFModel, FORMULA_CONFIGS, build_formula_ed_model, run_pipeline
from measurement_data import MODEL_DATA, MODEL_NAMES, set_model


class TestEDModels(unittest.TestCase):
    def test_fitted_model_ids(self):
        self.assertEqual(set(MODEL_DATA), {1, 3, 4, 5, 6})
        self.assertEqual(MODEL_DATA.keys(), MODEL_NAMES.keys())
        self.assertEqual(MODEL_NAMES[3], "LT 精细仿真")
        with self.assertRaises(KeyError):
            set_model(2)
        with self.assertRaises(KeyError):
            EDFitter(2)

    def test_gradual_edges(self):
        plateau = build_formula_ed_model("formula_plateau")
        cut = build_formula_ed_model("formula_truncated_peak")
        for model, peak_x, floor_x in [(plateau, 0, 125), (cut, 120, 140)]:
            xs = np.linspace(peak_x, floor_x, 51)
            values = [model.predict((x, 0, 0)) for x in xs]
            self.assertEqual(values[0], 1.0)
            self.assertEqual(values[-1], 0.2)
            self.assertTrue(np.all(np.diff(values) < 0))
            self.assertEqual(model.predict((floor_x + 1, 0, 0)), 0.2)
            self.assertAlmostEqual(model.predict((floor_x - 0.0001, 0, 0)), 0.2, places=8)
        self.assertGreater(plateau.predict((50, 0, 0)), 0.98)
        self.assertLess(plateau.predict((50, 0, 0)), 1.0)
        self.assertAlmostEqual(cut.predict((120.0001, 0, 0)), 1.0, places=8)
        self.assertGreater(cut.predict((119.9, 0, 0)), cut.predict((120.1, 0, 0)))
        plateau_drop = plateau.predict((90, 0, 0)) - plateau.predict((110, 0, 0))
        self.assertLess(plateau_drop, cut.predict((120, 0, 0)) - cut.predict((140, 0, 0)))

    def test_custom_peak_heights_centers_and_widths(self):
        peaks = [
            {"peak_ed": 0.90, "center": (10, 20, 30), "sigma": (30, 40, 50)},
            {"peak_ed": 0.35, "center": (15, 18, 32), "sigma": (60, 50, 40)},
        ]
        one = build_formula_ed_model("formula_gaussian", peaks=peaks[:1])
        for axis, sigma in enumerate(peaks[0]["sigma"]):
            point = list(peaks[0]["center"])
            point[axis] += sigma
            self.assertAlmostEqual(one.predict(point), 0.90 * np.exp(-0.5))
        multi = build_formula_ed_model("formula_multi_gaussian", peaks=peaks)
        for peak in peaks:
            center = np.array(peak["center"])
            self.assertEqual(multi.predict(center), peak["peak_ed"])
            for d in np.eye(3) * 0.001:
                self.assertLess(multi.predict(center + d), peak["peak_ed"])
        direct = run_pipeline((10, 20, 30), fitter="formula_multi_gaussian", peaks=peaks)
        self.assertEqual(direct, run_pipeline((10, 20, 30), fitter=multi))

    def test_all_models_use_position_and_time_jitter(self):
        fitted = [EDFitter(i) for i in MODEL_DATA]
        formulas = [build_formula_ed_model(name) for name in FORMULA_CONFIGS]
        point = (75, 10, 5)
        for model in fitted + formulas:
            def sensor(width, probability):
                return EDPFModel(model, seed=42, position_half_width=width,
                                 pf_time_parameters={"jump_probability": probability})
            clean = model.predict(point)
            self.assertEqual(sensor(0, 0).measure(point), (clean, 6.0))
            moved = sensor(10, 0)
            readings = [moved.measure(point)[0] for _ in range(12)]
            self.assertGreater(np.ptp(readings), 0)
            timed_ed, timed_pf = sensor(0, 1).measure(point)
            self.assertNotEqual(timed_pf, 6.0)
            self.assertNotEqual(timed_ed, clean)
            repeated = sensor(10, 0)
            self.assertEqual(readings, [repeated.measure(point)[0] for _ in range(12)])

    def test_time_jitter_respects_ed_limits_and_ratio(self):
        for name, peak_x in [("formula_plateau", 0), ("formula_truncated_peak", 120)]:
            sensor = EDPFModel.from_formula(name, seed=42, position_half_width=0,
                                            pf_time_parameters={"jump_probability": 1})
            for x, limit in [(peak_x, 1.0), (300, 0.2)]:
                readings = [sensor.measure((x, 0, 0))[0] for _ in range(80)]
                self.assertIn(limit, readings)
                self.assertTrue(all(0.2 <= ed <= 1.0 for ed in readings))
                self.assertGreater(np.ptp(readings), 0)
                row = run_pipeline((x, 0, 0), 0, 42, name,
                                   pf_model=sensor.pf_time, ed_time_model=sensor.ed_time)
                self.assertEqual(row["ed_pf_ratio"], row["ed"] / row["pf"])


if __name__ == "__main__":
    unittest.main()
