"""Direct TD-MPC-style latent/reward/value training with an EMA target model."""

import copy

import torch
import torch.nn.functional as F


class WorldModelTrainer:
    def __init__(self, model, learning_rate=3e-4, gamma=0.99, ema=0.99,
                 consistency_coef=1.0, reward_coef=1.0, value_coef=0.5,
                 event_coef=0.5, depth_coef=0.25, future_depth_coef=1.0,
                 reward_scale=10.0,
                 balanced_events=False, max_event_pos_weight=20.0):
        self.model = model
        self.target = copy.deepcopy(model).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.ema = ema
        self.coefs = (consistency_coef, reward_coef, value_coef,
                      event_coef, depth_coef, future_depth_coef)
        self.reward_scale = float(reward_scale)
        self.balanced_events = bool(balanced_events)
        self.max_event_pos_weight = float(max_event_pos_weight)
        self.updates = 0

    def _event_loss(self, logits, target):
        if not self.balanced_events:
            return F.binary_cross_entropy_with_logits(logits, target)
        positives = target.sum()
        negatives = target.numel() - positives
        if positives < 1 or negatives < 1:
            return F.binary_cross_entropy_with_logits(logits, target)
        pos_weight = (negatives / positives).clamp(1.0, self.max_event_pos_weight)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight)

    @staticmethod
    def _balanced_accuracy(logits, target):
        positive = target > 0.5
        negative = ~positive
        predicted = logits > 0
        if positive.any() and negative.any():
            return 0.5 * (predicted[positive].float().mean()
                          + (~predicted[negative]).float().mean())
        return (predicted == positive).float().mean()

    def losses(self, batch):
        latent = self.model.encode(batch["depth"], batch["proprio"])
        predicted_next = self.model.next(latent, batch["action"])
        predicted_reward = self.model.predict_reward(latent, batch["action"])
        predicted_value = self.model.predict_value(latent)
        collision_logit, fall_logit = self.model.predict_event_logits(
            latent, batch["action"])
        component = self.model.predict_task_components(latent, batch["action"])
        reconstructed_depth = self.model.reconstruct_depth(latent)
        predicted_next_depth = self.model.reconstruct_depth(predicted_next)
        with torch.no_grad():
            target_next = self.target.encode(batch["next_depth"], batch["next_proprio"])
            normalized_reward = batch["reward"] / self.reward_scale
            target_value = batch["return_target"]
        alive = 1.0 - batch["done"]
        consistency_per_item = (predicted_next - target_next).square().mean(dim=-1)
        consistency = (consistency_per_item * alive).sum() / alive.sum().clamp_min(1.0)
        reward = F.smooth_l1_loss(predicted_reward, normalized_reward)
        value = F.smooth_l1_loss(predicted_value, target_value)
        collision = self._event_loss(collision_logit, batch["collision"])
        fall_event = self._event_loss(fall_logit, batch["fall"])
        goal_event = self._event_loss(component["goal_logit"], batch["success"])
        off_support_event = self._event_loss(
            component["off_support_logit"], batch["off_support"])
        progress = F.smooth_l1_loss(
            component["progress"], batch["progress"] / self.reward_scale)
        event = collision + fall_event + goal_event + off_support_event + progress
        output_size = reconstructed_depth.shape[-2:]
        depth_target = F.adaptive_avg_pool2d(batch["depth"], output_size)
        next_depth_target = F.adaptive_avg_pool2d(batch["next_depth"], output_size)
        depth_reconstruction = F.smooth_l1_loss(reconstructed_depth, depth_target)
        future_depth_per_item = F.smooth_l1_loss(
            predicted_next_depth, next_depth_target, reduction="none").mean((1, 2, 3))
        future_depth = (future_depth_per_item * alive).sum() / alive.sum().clamp_min(1.0)
        total = (self.coefs[0] * consistency + self.coefs[1] * reward
                 + self.coefs[2] * value + self.coefs[3] * event
                 + self.coefs[4] * depth_reconstruction
                 + self.coefs[5] * future_depth)
        return total, {
            "loss_total": total, "loss_consistency": consistency,
            "loss_reward": reward, "loss_value": value,
            "loss_event": event, "loss_depth_reconstruction": depth_reconstruction,
            "loss_future_depth": future_depth,
            "loss_progress": progress, "loss_goal_event": goal_event,
            "loss_off_support_event": off_support_event,
            "prediction_reward_mae": ((predicted_reward - normalized_reward).abs().mean()
                                      * self.reward_scale),
            "prediction_value_mae": (predicted_value - target_value).abs().mean(),
            "prediction_collision_accuracy": (
                (collision_logit > 0) == (batch["collision"] > 0.5)).float().mean(),
            "prediction_fall_accuracy": (
                (fall_logit > 0) == (batch["fall"] > 0.5)).float().mean(),
            "prediction_goal_accuracy": (
                (component["goal_logit"] > 0) == (batch["success"] > 0.5)).float().mean(),
            "prediction_off_support_accuracy": (
                (component["off_support_logit"] > 0)
                == (batch["off_support"] > 0.5)).float().mean(),
            "prediction_collision_balanced_accuracy": self._balanced_accuracy(
                collision_logit, batch["collision"]),
            "prediction_fall_balanced_accuracy": self._balanced_accuracy(
                fall_logit, batch["fall"]),
            "prediction_goal_balanced_accuracy": self._balanced_accuracy(
                component["goal_logit"], batch["success"]),
            "prediction_off_support_balanced_accuracy": self._balanced_accuracy(
                component["off_support_logit"], batch["off_support"]),
            "target_collision_positive_rate": batch["collision"].mean(),
            "target_fall_positive_rate": batch["fall"].mean(),
            "target_goal_positive_rate": batch["success"].mean(),
            "target_off_support_positive_rate": batch["off_support"].mean(),
            "prediction_progress_mae": (
                (component["progress"] - batch["progress"] / self.reward_scale)
                .abs().mean() * self.reward_scale),
            "depth_reconstruction_mae": (reconstructed_depth - depth_target).abs().mean(),
            "future_depth_mae": ((predicted_next_depth - next_depth_target).abs()
                                 .mean((1, 2, 3)) * alive).sum()
                                / alive.sum().clamp_min(1.0),
            "latent_abs_mean": latent.abs().mean(),
        }

    def train_step(self, batch):
        self.model.train()
        total, metrics = self.losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.model.parameters()):
                target.mul_(self.ema).add_(online, alpha=1.0 - self.ema)
        self.updates += 1
        output = {name: float(value.detach()) for name, value in metrics.items()}
        output["gradient_norm"] = float(gradient)
        return output

    def losses_sequence(self, batch, temporal_decay=0.8):
        """Unroll predicted latent state over a real contiguous macro sequence."""
        horizon = batch["action"].shape[1]
        latent = self.model.encode(batch["depth"][:, 0], batch["proprio"][:, 0])
        sums = {name: latent.new_zeros(()) for name in (
            "consistency", "reward", "value", "collision", "fall", "goal", "off_support",
            "progress", "depth", "future_depth", "reward_abs", "value_abs", "progress_abs",
            "depth_abs", "future_depth_abs", "collision_correct", "fall_correct", "goal_correct",
            "off_support_correct", "collision_balanced", "fall_balanced", "goal_balanced",
            "off_support_balanced", "collision_rate", "fall_rate", "goal_rate",
            "off_support_rate", "latent_abs")}
        weight_sum = 0.0
        consistency_weight = latent.new_zeros(())
        for step in range(horizon):
            weight = float(temporal_decay) ** step
            action = batch["action"][:, step]
            predicted_next = self.model.next(latent, action)
            predicted_reward = self.model.predict_reward(latent, action)
            predicted_value = self.model.predict_value(latent)
            collision_logit, fall_logit = self.model.predict_event_logits(latent, action)
            component = self.model.predict_task_components(latent, action)
            reconstructed_depth = self.model.reconstruct_depth(latent)
            predicted_next_depth = self.model.reconstruct_depth(predicted_next)
            with torch.no_grad():
                target_next = self.target.encode(
                    batch["next_depth"][:, step], batch["next_proprio"][:, step])
                normalized_reward = batch["reward"][:, step] / self.reward_scale
                target_value = batch["return_target"][:, step]
            alive = 1.0 - batch["done"][:, step]
            consistency_item = (predicted_next - target_next).square().mean(dim=-1)
            sums["consistency"] += weight * (consistency_item * alive).sum()
            consistency_weight += weight * alive.sum()
            sums["reward"] += weight * F.smooth_l1_loss(
                predicted_reward, normalized_reward, reduction="mean")
            sums["value"] += weight * F.smooth_l1_loss(
                predicted_value, target_value, reduction="mean")
            collision_target = batch["collision"][:, step]
            fall_target = batch["fall"][:, step]
            goal_target = batch["success"][:, step]
            off_support_target = batch["off_support"][:, step]
            sums["collision"] += weight * self._event_loss(
                collision_logit, collision_target)
            sums["fall"] += weight * self._event_loss(fall_logit, fall_target)
            sums["goal"] += weight * self._event_loss(
                component["goal_logit"], goal_target)
            sums["off_support"] += weight * self._event_loss(
                component["off_support_logit"], off_support_target)
            target_progress = batch["progress"][:, step] / self.reward_scale
            sums["progress"] += weight * F.smooth_l1_loss(
                component["progress"], target_progress)
            output_size = reconstructed_depth.shape[-2:]
            depth_target = F.adaptive_avg_pool2d(
                batch["depth"][:, step], output_size)
            next_depth_target = F.adaptive_avg_pool2d(
                batch["next_depth"][:, step], output_size)
            sums["depth"] += weight * F.smooth_l1_loss(
                reconstructed_depth, depth_target)
            future_depth_item = F.smooth_l1_loss(
                predicted_next_depth, next_depth_target,
                reduction="none").mean((1, 2, 3))
            alive_count = alive.sum().clamp_min(1.0)
            sums["future_depth"] += weight * (
                future_depth_item * alive).sum() / alive_count
            sums["depth_abs"] += weight * (
                reconstructed_depth - depth_target).abs().mean()
            sums["future_depth_abs"] += weight * (
                (predicted_next_depth - next_depth_target).abs().mean((1, 2, 3))
                * alive).sum() / alive_count
            sums["reward_abs"] += weight * (
                predicted_reward - normalized_reward).abs().mean()
            sums["value_abs"] += weight * (
                predicted_value - target_value).abs().mean()
            sums["progress_abs"] += weight * (
                component["progress"] - target_progress).abs().mean()
            sums["collision_correct"] += weight * (
                (collision_logit > 0) == (batch["collision"][:, step] > 0.5)
            ).float().mean()
            sums["fall_correct"] += weight * (
                (fall_logit > 0) == (batch["fall"][:, step] > 0.5)
            ).float().mean()
            sums["goal_correct"] += weight * (
                (component["goal_logit"] > 0) == (batch["success"][:, step] > 0.5)
            ).float().mean()
            sums["off_support_correct"] += weight * (
                (component["off_support_logit"] > 0)
                == (batch["off_support"][:, step] > 0.5)).float().mean()
            sums["collision_balanced"] += weight * self._balanced_accuracy(
                collision_logit, collision_target)
            sums["fall_balanced"] += weight * self._balanced_accuracy(
                fall_logit, fall_target)
            sums["goal_balanced"] += weight * self._balanced_accuracy(
                component["goal_logit"], goal_target)
            sums["off_support_balanced"] += weight * self._balanced_accuracy(
                component["off_support_logit"], off_support_target)
            sums["collision_rate"] += weight * collision_target.mean()
            sums["fall_rate"] += weight * fall_target.mean()
            sums["goal_rate"] += weight * goal_target.mean()
            sums["off_support_rate"] += weight * off_support_target.mean()
            sums["latent_abs"] += weight * latent.abs().mean()
            weight_sum += weight
            # Crucial difference from one-step training: later predictions use
            # the model's own latent, not a fresh encoding of the real frame.
            latent = predicted_next

        consistency = sums["consistency"] / consistency_weight.clamp_min(1.0)
        normalized = {name: value / weight_sum for name, value in sums.items()
                      if name != "consistency"}
        event = (normalized["collision"] + normalized["fall"]
                 + normalized["goal"] + normalized["off_support"]
                 + normalized["progress"])
        total = (self.coefs[0] * consistency + self.coefs[1] * normalized["reward"]
                 + self.coefs[2] * normalized["value"] + self.coefs[3] * event
                 + self.coefs[4] * normalized["depth"]
                 + self.coefs[5] * normalized["future_depth"])
        return total, {
            "loss_total": total, "loss_consistency": consistency,
            "loss_reward": normalized["reward"], "loss_value": normalized["value"],
            "loss_event": event, "loss_depth_reconstruction": normalized["depth"],
            "loss_future_depth": normalized["future_depth"],
            "loss_progress": normalized["progress"], "loss_goal_event": normalized["goal"],
            "loss_off_support_event": normalized["off_support"],
            "prediction_reward_mae": normalized["reward_abs"] * self.reward_scale,
            "prediction_value_mae": normalized["value_abs"],
            "prediction_collision_accuracy": normalized["collision_correct"],
            "prediction_fall_accuracy": normalized["fall_correct"],
            "prediction_goal_accuracy": normalized["goal_correct"],
            "prediction_off_support_accuracy": normalized["off_support_correct"],
            "prediction_collision_balanced_accuracy": normalized["collision_balanced"],
            "prediction_fall_balanced_accuracy": normalized["fall_balanced"],
            "prediction_goal_balanced_accuracy": normalized["goal_balanced"],
            "prediction_off_support_balanced_accuracy": normalized["off_support_balanced"],
            "target_collision_positive_rate": normalized["collision_rate"],
            "target_fall_positive_rate": normalized["fall_rate"],
            "target_goal_positive_rate": normalized["goal_rate"],
            "target_off_support_positive_rate": normalized["off_support_rate"],
            "prediction_progress_mae": normalized["progress_abs"] * self.reward_scale,
            "depth_reconstruction_mae": normalized["depth_abs"],
            "future_depth_mae": normalized["future_depth_abs"],
            "latent_abs_mean": normalized["latent_abs"],
        }

    def train_step_sequence(self, batch, temporal_decay=0.8):
        self.model.train()
        total, metrics = self.losses_sequence(batch, temporal_decay)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.model.parameters()):
                target.mul_(self.ema).add_(online, alpha=1.0 - self.ema)
        self.updates += 1
        output = {name: float(value.detach()) for name, value in metrics.items()}
        output["gradient_norm"] = float(gradient)
        return output
