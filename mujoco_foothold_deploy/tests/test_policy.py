import unittest
from pathlib import Path

import numpy as np

from deploy.policy import FootholdPolicy


ROOT = Path(__file__).resolve().parents[1]


class PolicyTest(unittest.TestCase):
    def test_checkpoint_contract_and_deterministic_inference(self):
        policy = FootholdPolicy(ROOT / "checkpoints" / "model_7000.pt")
        observation = np.zeros(30, dtype=np.float32)
        goal = np.zeros(16, dtype=np.float32)
        goal[[3, 10, 14]] = 1.0
        first = policy.infer(observation, goal)
        second = policy.infer(observation, goal)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertTrue(np.isfinite(a).all())
        self.assertEqual(first[0].shape, (8,))
        self.assertEqual(first[1].shape, (30,))
        self.assertEqual(first[2].shape, (16,))


if __name__ == "__main__":
    unittest.main()
