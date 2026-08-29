"""Task-predictive latent training without future observation reconstruction."""

import copy

import torch
import torch.nn.functional as F


class TaskWorldModelTrainer:
    """Train multi-step outcome predictions on the model's own latent rollout."""

    def __init__(self, model, learning_rate=3e-4, gamma=0.99, ema=0.99,
                 event_coef=1.0, margin_coef=1.0,
                 task_state_coef=1.0, value_coef=0.25, regularization_coef=0.01,
                 reward_scale=10.0, balanced_events=True,
                 max_event_pos_weight=20.0):
        self.model = model
        self.target = copy.deepcopy(model).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.ema = float(ema)
        self.reward_scale = float(reward_scale)
        self.balanced_events = bool(balanced_events)
        self.max_event_pos_weight = float(max_event_pos_weight)
        self.coefs = {
            "event": float(event_coef),
            "margin": float(margin_coef),
            "task_state": float(task_state_coef),
            "value": float(value_coef),
            "regularization": float(regularization_coef),
        }
        self.updates = 0

    def _event_loss(self, logits, target):
        positives = target.sum()
        negatives = target.numel() - positives
        if (not self.balanced_events or positives < 1 or negatives < 1):
            return F.binary_cross_entropy_with_logits(logits, target)
        positive_weight = (negatives / positives).clamp(
            1.0, self.max_event_pos_weight)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=positive_weight)

    @staticmethod
    def _balanced_accuracy(logits, target):
        positive = target > 0.5
        negative = ~positive
        predicted = logits > 0
        if positive.any() and negative.any():
            return 0.5 * (predicted[positive].float().mean()
                          + (~predicted[negative]).float().mean())
        return (predicted == positive).float().mean()

    @staticmethod
    def _latent_regularization(latent):
        centered = latent - latent.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.square().mean(dim=0) + 1e-4)
        variance = F.relu(0.20 - std).mean()
        if latent.shape[0] < 2:
            return variance
        covariance = centered.T @ centered / float(latent.shape[0] - 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        return variance + off_diagonal.square().mean()

    def losses_sequence(self, batch, temporal_decay=0.8):
        horizon = batch["action"].shape[1]
        latent = self.model.encode(batch["depth"][:, 0], batch["proprio"][:, 0])
        sums = {name: latent.new_zeros(()) for name in (
            "progress", "heading", "collision", "fall", "goal",
            "off_support", "collision_force", "stability_margin", "support_fraction",
            "touchdown_error", "task_state", "value", "regularization",
            "progress_abs", "heading_abs", "collision_force_abs", "stability_abs",
            "support_abs", "touchdown_abs", "task_state_abs", "value_abs",
            "collision_balanced", "fall_balanced", "goal_balanced",
            "off_support_balanced", "collision_rate", "fall_rate", "goal_rate",
            "off_support_rate", "latent_abs")}
        weight_sum = 0.0

        for step in range(horizon):
            weight = float(temporal_decay) ** step
            valid = (batch["valid"][:, step]
                     if "valid" in batch
                     else torch.ones_like(batch["done"][:, step]))
            valid_bool = valid > 0.5
            if not valid_bool.any():
                break

            def masked_mean(value):
                if value.ndim > 1:
                    value = value.flatten(1).mean(dim=1)
                return (value * valid).sum() / valid.sum().clamp_min(1.0)

            action = batch["action"][:, step]
            predicted_next = self.model.next(latent, action)
            component = self.model.predict_task_components(latent, action)
            predicted_value = self.model.predict_value(latent)
            alive = (1.0 - batch["done"][:, step]) * valid

            def alive_mean(value):
                if value.ndim > 1:
                    value = value.flatten(1).mean(dim=1)
                return (value * alive).sum() / alive.sum().clamp_min(1.0)

            progress_target = batch["progress"][:, step] / self.reward_scale
            heading_target = batch["heading_progress"][:, step] / torch.pi
            collision_force_target = torch.clamp(
                batch["collision_force"][:, step] / 100.0, 0.0, 5.0)
            touchdown_target = torch.clamp(
                batch["touchdown_error"][:, step] / 0.10, 0.0, 5.0)
            stability_target = batch["stability_margin"][:, step]
            support_target = batch["support_fraction"][:, step]
            target_task_state = batch["macro_state"][:, step]
            predicted_task_state = self.model.predict_task_state(predicted_next)

            regression = {
                "progress": (component["progress"], progress_target),
                "heading": (component["heading_progress"], heading_target),
                "collision_force": (component["collision_force"], collision_force_target),
                "stability_margin": (component["stability_margin"], stability_target),
                "support_fraction": (component["support_fraction"], support_target),
                "touchdown_error": (component["touchdown_error"], touchdown_target),
                "value": (predicted_value, batch["return_target"][:, step]),
            }
            for name, (prediction, target) in regression.items():
                item = F.smooth_l1_loss(prediction, target, reduction="none")
                sums[name] += weight * masked_mean(item)
            macro_item = F.smooth_l1_loss(
                predicted_task_state, target_task_state, reduction="none")
            sums["task_state"] += weight * alive_mean(macro_item)

            event_targets = {
                "collision": batch["collision"][:, step],
                "fall": batch["fall"][:, step],
                "goal": batch["success"][:, step],
                "off_support": batch["off_support"][:, step],
            }
            event_logits = {
                "collision": component["collision_logit"],
                "fall": component["fall_logit"],
                "goal": component["goal_logit"],
                "off_support": component["off_support_logit"],
            }
            for name in event_targets:
                sums[name] += weight * self._event_loss(
                    event_logits[name][valid_bool], event_targets[name][valid_bool])
                sums[name + "_balanced"] += weight * self._balanced_accuracy(
                    event_logits[name][valid_bool], event_targets[name][valid_bool])
                sums[name + "_rate"] += weight * masked_mean(event_targets[name])

            sums["regularization"] += weight * self._latent_regularization(
                latent[valid_bool])
            sums["progress_abs"] += weight * masked_mean(
                (component["progress"] - progress_target).abs())
            sums["heading_abs"] += weight * masked_mean(
                (component["heading_progress"] - heading_target).abs())
            sums["collision_force_abs"] += weight * masked_mean(
                (component["collision_force"] - collision_force_target).abs())
            sums["stability_abs"] += weight * masked_mean(
                (component["stability_margin"] - stability_target).abs())
            sums["support_abs"] += weight * masked_mean(
                (component["support_fraction"] - support_target).abs())
            sums["touchdown_abs"] += weight * masked_mean(
                (component["touchdown_error"] - touchdown_target).abs())
            sums["task_state_abs"] += weight * alive_mean(
                (predicted_task_state - target_task_state).abs())
            sums["value_abs"] += weight * masked_mean(
                (predicted_value - batch["return_target"][:, step]).abs())
            sums["latent_abs"] += weight * masked_mean(latent.abs())
            weight_sum += weight
            latent = predicted_next

        normalized = {name: value / weight_sum for name, value in sums.items()}
        event = sum(normalized[name] for name in (
            "collision", "fall", "goal", "off_support"))
        margin = sum(normalized[name] for name in (
            "progress", "heading", "collision_force", "stability_margin",
            "support_fraction", "touchdown_error"))
        total = (self.coefs["event"] * event
                 + self.coefs["margin"] * margin
                 + self.coefs["task_state"] * normalized["task_state"]
                 + self.coefs["value"] * normalized["value"]
                 + self.coefs["regularization"] * normalized["regularization"])
        return total, {
            "loss_total": total,
            "loss_event": event,
            "loss_task_margin": margin,
            "loss_task_state": normalized["task_state"],
            "loss_value": normalized["value"],
            "loss_latent_regularization": normalized["regularization"],
            # The target was divided by reward_scale above, so it is already
            # expressed as physical forward progress in metres.
            "prediction_progress_mae_m": normalized["progress_abs"],
            "prediction_heading_progress_mae_rad": normalized["heading_abs"] * torch.pi,
            "prediction_collision_force_mae_n": normalized["collision_force_abs"] * 100.0,
            "prediction_stability_margin_mae": normalized["stability_abs"],
            "prediction_support_fraction_mae": normalized["support_abs"],
            "prediction_touchdown_error_mae_m": normalized["touchdown_abs"] * 0.10,
            "prediction_task_state_mae": normalized["task_state_abs"],
            "prediction_value_mae": normalized["value_abs"],
            "prediction_collision_balanced_accuracy": normalized["collision_balanced"],
            "prediction_fall_balanced_accuracy": normalized["fall_balanced"],
            "prediction_goal_balanced_accuracy": normalized["goal_balanced"],
            "prediction_off_support_balanced_accuracy": normalized["off_support_balanced"],
            "target_collision_positive_rate": normalized["collision_rate"],
            "target_fall_positive_rate": normalized["fall_rate"],
            "target_goal_positive_rate": normalized["goal_rate"],
            "target_off_support_positive_rate": normalized["off_support_rate"],
            "latent_abs_mean": normalized["latent_abs"],
        }

    def losses(self, batch):
        sequence = {name: value[:, None] for name, value in batch.items()}
        return self.losses_sequence(sequence, temporal_decay=1.0)

    def _step(self, total, metrics):
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

    def train_step(self, batch):
        self.model.train()
        total, metrics = self.losses(batch)
        return self._step(total, metrics)

    def train_step_sequence(self, batch, temporal_decay=0.8):
        self.model.train()
        total, metrics = self.losses_sequence(batch, temporal_decay)
        return self._step(total, metrics)
