"""联合收敛条件、数值步骤一致性和 Excel 结果提取的回归检查。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import batch_test as batch
import search


class ConvergenceTests(unittest.TestCase):
    def run_values(self, new_values, initial=.5, patience=2):
        values = iter([initial] * 10 + new_values)

        def objective(position):
            value = next(values)
            objective.last_measurement = dict(ed=value, pf=1., ed_pf_ratio=value)
            return value

        gp = SimpleNamespace(kernel=SimpleNamespace(), y_mean=0., y_std=1.)
        case = dict(data_model=1, target='ed', init_points=10, seed=42)
        with patch.object(search, 'set_ed_model'), \
                patch.object(search, 'get_initial_points', return_value=np.zeros((10, 3))), \
                patch.object(search, 'make_objective', return_value=objective), \
                patch.object(search, 'fit_gp', return_value=gp), \
                patch.object(search, 'get_next_sample', return_value=(np.zeros(3), 0.)):
            return batch.run_search_with_accuracy(case, {'reference_value': 1.},
                                                   len(new_values), patience)

    def test_stagnation_above_ten_percent_continues_to_budget(self):
        history = self.run_values([.5] * 8)
        self.assertEqual(len(history['best_value']) - 1, 8)
        self.assertEqual(history['stop_reason'], 'max_iter')
        self.assertIsNone(batch.get_convergence_iteration(history, 2, 1.))

    def test_stagnation_within_ten_percent_stops(self):
        history = self.run_values([.9] * 8, initial=.9)
        self.assertEqual(len(history['best_value']) - 1, 3)
        self.assertEqual(history['stop_reason'], 'converged')
        self.assertEqual(batch.get_convergence_iteration(history, 2, 1.), 3)

    def test_accuracy_alone_does_not_stop(self):
        history = self.run_values([.91, .92, .93, .94, .95], patience=1)
        self.assertEqual(len(history['best_value']) - 1, 5)
        self.assertEqual(history['stop_reason'], 'max_iter')

    def test_new_improvement_resets_wait_for_convergence(self):
        history = self.run_values([.5, .5, .5, .95, .95, .95, .95, .95])
        self.assertEqual(len(history['best_value']) - 1, 7)
        self.assertEqual(batch.get_convergence_iteration(history, 2, 1.), 7)

    def test_relative_error_is_absolute_and_inclusive(self):
        self.assertTrue(batch.accuracy_passed(.9, 1.))
        self.assertTrue(batch.accuracy_passed(1.1, 1.))
        self.assertFalse(batch.accuracy_passed(1.2, 1.))
        self.assertFalse(batch.accuracy_passed(.8, 1.))
        self.assertTrue(batch.accuracy_passed(0., 0.))
        self.assertFalse(batch.accuracy_passed(.1, 0.))

    def test_existing_numerical_steps_are_preserved(self):
        for target in batch.TARGETS:
            for sigma in (0., .03):
                with self.subTest(target=target, sigma=sigma), patch.dict(
                        search.BO_PARAMETERS, {'sigma3_rel': sigma, 'xi_init': None, 'xi_final': None}):
                    case = dict(data_model=3, target=target, init_points=6, seed=42)
                    original = search.run_search(source='fitted', data_model=3, fit_method='rbf',
                                                 target=target, seed=42, init_points=6,
                                                 max_iter=3, stop_no_improve=10,
                                                 show=False, verbose=False)
                    updated = batch.run_search_with_accuracy(case, {'reference_value': 1.},
                                                              3, 10, 1000.)
                    for key in ('X', 'y', 'best_value', 'best_x', 'ed', 'pf', 'ed_pf_ratio',
                                'noise_level', 'sigma_abs_used', 'xi_used'):
                        np.testing.assert_array_equal(original[key], updated[key], err_msg=key)


class ReportTests(unittest.TestCase):
    def test_typed_excel_and_plots_read_excel_values(self):
        openpyxl = batch._excel_support()
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = []
            for model in batch.DATA_MODELS:
                for target in batch.TARGETS:
                    for n in batch.INIT_POINTS:
                        row = {field: '' for field in batch.SUMMARY_FIELDS}
                        converged = 0 if model == 6 or (model == 2 and n == 20) else 2
                        row.update(data_model=model, model_name=str(model), target=target,
                                   init_points=n, valid_runs=2, converged_runs=converged,
                                   convergence_rate=converged / 2.,
                                   convergence_iters_mean=30 + model + n / 2 if converged else '',
                                   convergence_iters_std=2. if converged else '',
                                   value_relative_error_pct_mean=5.)
                        rows.append(row)
            batch.write_csv(directory / 'summary.csv', batch.SUMMARY_FIELDS, rows)
            (directory / 'run_config.json').write_text(json.dumps(
                {'format_version': 4, 'value_tolerance_pct': 10., 'reference_file_sha256': None}))
            excel_file = batch.export_excel_results(directory)
            workbook = openpyxl.load_workbook(excel_file)
            sheet = workbook['Summary']
            fields = [cell.value for cell in sheet[1]]
            mean_column = fields.index('convergence_iters_mean') + 1
            percentage_column = fields.index('value_relative_error_pct_mean') + 1
            self.assertAlmostEqual(sheet.cell(2, percentage_column).value, .05)
            self.assertEqual(sheet.cell(2, percentage_column).number_format, '0.00%')
            self.assertEqual(sheet.freeze_panes, 'A2')
            sheet.cell(2, mean_column, 123.)  # CSV 不变，检查图确实读取 Excel。
            workbook.save(excel_file)
            workbook.close()
            calls = []
            original = batch.plt.Axes.errorbar

            def capture(axis, x, y, *args, **kwargs):
                calls.append((list(y), kwargs['color']))
                return original(axis, x, y, *args, **kwargs)

            with patch.object(batch.plt.Axes, 'errorbar', capture):
                plots = batch.plot_convergence_results(excel_file)
            self.assertEqual(len(plots), 2)
            self.assertTrue(all(path.stat().st_size > 1000 for path in plots))
            self.assertEqual(calls[0][0][0], 123.)
            self.assertTrue(np.isnan(calls[1][0][1]))
            self.assertTrue(np.all(np.isnan(calls[5][0])))
            self.assertEqual(len({color for _, color in calls[:6]}), 6)


if __name__ == '__main__':
    unittest.main()
