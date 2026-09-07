import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from composite_terrain.compare_runs import compare, summarize


class CompareRunsTest(unittest.TestCase):
    def make_run(self, root, name, seed=1):
        path = root / name
        path.mkdir()
        metrics = dict(num_envs=2, difficulty=.5, seed=seed, route_offset=0,
            lower_sha256='frozen', lower_steps=6000, physics_contract='same',
            success_contract='strict', reset_region_prob=0,
            successes_per_env=[1, 0], successes=1, falls=1, timeouts=0,
            right_censored_episodes=2)
        (path / 'metrics.json').write_text(json.dumps(metrics))
        np.savez(path / 'completed_episodes.npz', env_id=[0, 0], episode_id=[0, 1],
                 success=[False, True], fall=[True, False], timeout=[False, False],
                 ticks=[500, 1900])
        return path

    def test_retried_success_is_not_first_attempt_success(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(Path(temp), 'run')
            _, stats, _ = summarize(run)
            self.assertEqual(stats['maps_completed'], 1)
            self.assertEqual(stats['first_attempt_success'], 0)
            self.assertEqual(stats['first_attempt_fall'], 1)
            self.assertEqual(stats['first_attempt_censored'], 1)
            self.assertEqual(compare(run, run)['map_completion_pairs']['both'], 1)

    def test_mismatched_layouts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'seed'):
                compare(self.make_run(root, 'a', 1), self.make_run(root, 'b', 2))


if __name__ == '__main__':
    unittest.main()
