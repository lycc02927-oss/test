import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ga_tune as ga


class GATests(unittest.TestCase):
    def test_genetic_operators_preserve_domains_elite_and_budget(self):
        space = ga.gene_space(80)
        rng = random.Random(20)
        pop = [ga.baseline(space)]
        while len(pop) < 6:
            g = ga.draw(space, rng)
            if g not in pop:
                pop.append(g)
        new = ga.next_population(pop, {g: i for i, g in enumerate(pop)}, space, rng, 1.)
        self.assertEqual(new[0], pop[0])
        self.assertEqual(len(set(new)), len(pop))
        self.assertTrue(any(g not in pop for g in new))
        for g in new:
            self.assertTrue(all(v in choices for v, choices in zip(g, space)))
            p = g.parameters(80)
            self.assertEqual(p['max_iter']+p['init_points'], 80)
            self.assertEqual(p['xi_init'] is None, g.xi_mode == 'auto')
        self.assertEqual(ga.gene_space(20)[0], [10])

    def test_rank_favors_worst_case_before_saving_samples(self):
        def row(m, passed, n):
            return dict(model=m, target='ed', hit=passed, evaluations=n, error_pct=2.)
        self.assertLess(ga.rank([row(1, True, 80), row(2, True, 80)]),
                        ga.rank([row(1, True, 20), row(2, False, 20)]))

    def test_holdout_never_selects_winner_and_roles_share_trials(self):
        calls = []
        def fake(job):
            g,m,t,s,*_ = job
            calls.append((g,m,t,s))
            control = g == ga.baseline(ga.gene_space(80))
            hit = control if s == 42 else not control
            return dict(model=m,target=t,hit=hit,evaluations=50,error_pct=0. if hit else 30.)
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(budget=80, population=4, generations=2, workers=1, mutation=.3,
                tolerance=10., models=[1,2], targets=['ed'], train_seeds=[42], valid_seeds=[142],
                output=d, ga_seed=8)
            refs=[dict(data_model=m, target='ed', reference_value=1.) for m in args.models]
            with patch.object(ga.batch,'make_references',return_value=refs), patch.object(ga,'evaluate',fake):
                folder=ga.run(args)
            rec=json.loads((Path(folder)/'selected.json').read_text())
            self.assertEqual(rec['common']['init_points'],40)
            self.assertEqual(rec['common']['xi_init'],0.01)
            self.assertEqual(len(set(calls)),len(calls))
            self.assertEqual(sum(c[-1]==142 for c in calls),2)
            args.valid_seeds=[42]
            with self.assertRaisesRegex(ValueError,'种子'):
                ga.run(args)
