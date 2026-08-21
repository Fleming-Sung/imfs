import io
import unittest

import torch

from foothold.ppo import Normalizer, RunningMeanStd


class RunningMeanStdCheckpointTest(unittest.TestCase):
    def test_normalizer_round_trip_preserves_outputs(self):
        torch.manual_seed(7)
        source = Normalizer(5, 3, 7, gamma=0.995, device="cpu", obs_clip=10.0)

        for _ in range(4):
            obs = torch.randn(32, 5) * 2.0 + 1.5
            goal = torch.randn(32, 3) * 0.2 - 0.1
            critic = torch.randn(32, 7) * 3.0
            reward = torch.randn(32)
            done = torch.rand(32) < 0.1
            source.observations(obs, goal, critic, update=True)
            source.rewards(reward, done, update=True)

        restored = Normalizer(5, 3, 7, gamma=0.9, device="cpu", obs_clip=None)
        restored.load_state_dict(source.state_dict())
        self.assertEqual(restored.gamma, source.gamma)
        self.assertEqual(restored.obs_clip, source.obs_clip)

        probe = (torch.randn(8, 5), torch.randn(8, 3), torch.randn(8, 7))
        expected = source.observations(*probe, update=False)
        actual = restored.observations(*probe, update=False)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

        for name in ("actor_obs", "goal", "critic_obs", "return_rms"):
            expected_rms = getattr(source, name)
            actual_rms = getattr(restored, name)
            torch.testing.assert_close(actual_rms.mean, expected_rms.mean, rtol=0, atol=0)
            torch.testing.assert_close(actual_rms.var, expected_rms.var, rtol=0, atol=0)
            torch.testing.assert_close(actual_rms.count, expected_rms.count, rtol=0, atol=0)

    def test_rejects_wrong_shape(self):
        rms = RunningMeanStd((3,), "cpu")
        state = rms.state_dict()
        state["mean"] = torch.zeros(4)
        with self.assertRaisesRegex(ValueError, "形状不匹配"):
            rms.load_state_dict(state)

    def test_state_survives_torch_checkpoint_serialization(self):
        source = Normalizer(2, 2, 3, gamma=0.99, device="cpu", obs_clip=7.0)
        source.observations(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            update=True)

        buffer = io.BytesIO()
        torch.save({"format_version": 2, "normalizer": source.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu")

        restored = Normalizer(2, 2, 3, gamma=0.5, device="cpu", obs_clip=None)
        restored.load_state_dict(checkpoint["normalizer"])
        torch.testing.assert_close(restored.actor_obs.mean, source.actor_obs.mean, rtol=0, atol=0)
        self.assertEqual(restored.gamma, 0.99)
        self.assertEqual(restored.obs_clip, 7.0)


if __name__ == "__main__":
    unittest.main()
