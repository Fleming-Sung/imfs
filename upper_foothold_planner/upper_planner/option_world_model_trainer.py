"""TD training for the semi-Markov foothold-option world model."""

import torch
import torch.nn.functional as F

from .task_world_model_trainer import TaskWorldModelTrainer


class OptionWorldModelTrainer(TaskWorldModelTrainer):
    """Task prediction plus twin-Q semi-Markov temporal-difference learning."""

    def __init__(self, model, learning_rate=2e-4, gamma=0.99, ema=0.99,
                 reward_scale=10.0, nominal_option_ticks=25.0,
                 q_coef=1.0, duration_coef=0.25, continuation_coef=0.5,
                 ranking_coef=0.25, policy_coef=0.1, policy_q_coef=0.0,
                 policy_safe_only=False, max_event_pos_weight=50.0):
        super().__init__(
            model, learning_rate=learning_rate, gamma=gamma, ema=ema,
            value_coef=0.0, regularization_coef=0.01,
            reward_scale=reward_scale, balanced_events=False,
            max_event_pos_weight=max_event_pos_weight)
        self.nominal_option_ticks = float(nominal_option_ticks)
        self.option_coefs = {
            "q": float(q_coef),
            "duration": float(duration_coef),
            "continuation": float(continuation_coef),
            "ranking": float(ranking_coef),
            "policy": float(policy_coef),
            "policy_q": float(policy_q_coef),
        }
        self.policy_safe_only = bool(policy_safe_only)

    @staticmethod
    def _group_ranking_loss(score, outcome, group):
        """Pairwise ranking for true same-state counterfactual action groups."""
        losses = []
        for group_id in torch.unique(group[group >= 0]):
            mask = group == group_id
            if int(mask.sum()) < 2:
                continue
            group_score = score[mask]
            group_outcome = outcome[mask]
            target_difference = group_outcome[:, None] - group_outcome[None, :]
            valid = target_difference.abs() > 1e-3
            if valid.any():
                score_difference = group_score[:, None] - group_score[None, :]
                sign = target_difference.sign()
                losses.append(F.softplus(-sign[valid] * score_difference[valid]).mean())
        return (torch.stack(losses).mean() if losses
                else score.new_zeros(()))

    def losses_sequence(self, batch, temporal_decay=0.8):
        base_total, metrics = super().losses_sequence(batch, temporal_decay)
        horizon = batch["action"].shape[1]
        latent = self.model.encode(batch["depth"][:, 0], batch["proprio"][:, 0])
        q_loss = latent.new_zeros(())
        duration_loss = latent.new_zeros(())
        continuation_loss = latent.new_zeros(())
        ranking_loss = latent.new_zeros(())
        policy_loss = latent.new_zeros(())
        policy_abs = latent.new_zeros(())
        policy_q_loss = latent.new_zeros(())
        policy_weight = latent.new_zeros(())
        q_abs = latent.new_zeros(())
        duration_abs = latent.new_zeros(())
        continuation_accuracy = latent.new_zeros(())
        horizon_metrics = {}
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
            if "from_planner" in batch:
                planner_steps = (batch["from_planner"][:, step] > 0.5) * valid_bool
            else:
                planner_steps = torch.zeros_like(valid_bool)
            if self.policy_safe_only:
                safe = (~(batch["fall"][:, step] > 0.5)
                        & ~(batch["collision"][:, step] > 0.5)
                        & ~(batch["off_support"][:, step] > 0.5))
                planner_steps = planner_steps & safe
            if planner_steps.any():
                policy_action = self.model.policy_action(latent)
                policy_item = (policy_action - action).square().mean(dim=-1)
                policy_loss += weight * (policy_item * planner_steps).sum()
                policy_abs += weight * (
                    (policy_action - action).abs().mean(dim=-1)
                    * planner_steps).sum()
                if self.option_coefs.get("policy_q", 0.0) != 0.0:
                    q_values = self.model.predict_q(latent, policy_action)
                    min_q = q_values.min(dim=0).values
                    policy_q_loss += weight * (-min_q * planner_steps).sum()
                policy_weight += weight * planner_steps.sum()
            component = self.model.predict_task_components(latent, action)
            predicted_q = self.model.predict_q(latent, action)
            duration_target = torch.clamp(
                batch["option_duration_ticks"][:, step]
                / self.nominal_option_ticks, 0.04, 4.0)
            continuation_target = 1.0 - batch["done"][:, step]

            # CEM is the policy improvement operator in this project; there is
            # no separately trained actor that can provide an in-distribution
            # bootstrap action.  Regress the action value to the measured
            # finite-horizon return instead of maximizing Q over arbitrary
            # random actions and feeding that extrapolation back as a target.
            q_target = batch["return_target"][:, step].detach()

            q_item = F.smooth_l1_loss(
                predicted_q, q_target.unsqueeze(0).expand_as(predicted_q),
                reduction="none").mean(dim=0)
            q_loss += weight * masked_mean(q_item)
            duration_loss += weight * masked_mean(F.smooth_l1_loss(
                component["option_duration"], duration_target, reduction="none"))
            continuation_loss += weight * masked_mean(
                F.binary_cross_entropy_with_logits(
                    component["continuation_logit"], continuation_target,
                    reduction="none"))
            q_abs += weight * masked_mean(
                (predicted_q.mean(0) - q_target).abs())
            duration_abs += weight * masked_mean(
                (component["option_duration"] - duration_target).abs())
            continuation_accuracy += weight * masked_mean(
                (component["continuation_logit"] > 0)
                == (continuation_target > 0.5)).float()
            horizon_metrics.update({
                "prediction_q_mae_h{}".format(step + 1): (
                    masked_mean((predicted_q.mean(0) - q_target).abs())),
                "prediction_q_mean_h{}".format(step + 1): masked_mean(
                    predicted_q.mean(0)),
                "target_return_mean_h{}".format(step + 1): masked_mean(q_target),
                "prediction_progress_mae_h{}_m".format(step + 1): (
                    masked_mean((component["progress"]
                    - batch["progress"][:, step] / self.reward_scale).abs())),
                "prediction_progress_mean_h{}_m".format(step + 1): (
                    masked_mean(component["progress"])),
                "target_progress_mean_h{}_m".format(step + 1): (
                    masked_mean(batch["progress"][:, step]) / self.reward_scale),
                "target_duration_mean_h{}_ticks".format(step + 1): (
                    masked_mean(batch["option_duration_ticks"][:, step])),
                "target_fall_rate_h{}".format(step + 1): (
                    masked_mean(batch["fall"][:, step])),
                "target_off_support_rate_h{}".format(step + 1): (
                    masked_mean(batch["off_support"][:, step])),
                "prediction_fall_brier_h{}".format(step + 1): (
                    masked_mean((torch.sigmoid(component["fall_logit"])
                    - batch["fall"][:, step]).square())),
                "prediction_off_support_brier_h{}".format(step + 1): (
                    masked_mean((torch.sigmoid(component["off_support_logit"])
                    - batch["off_support"][:, step]).square())),
            })

            groups = batch["counterfactual_group"][:, step].clone()
            groups[~valid_bool] = -1
            ranking_loss += weight * self._group_ranking_loss(
                predicted_q.mean(0), q_target, groups)
            weight_sum += weight
            latent = self.model.next(latent, action)

        q_loss = q_loss / weight_sum
        duration_loss = duration_loss / weight_sum
        continuation_loss = continuation_loss / weight_sum
        ranking_loss = ranking_loss / weight_sum
        policy_loss = policy_loss / policy_weight.clamp_min(1.0)
        policy_abs = policy_abs / policy_weight.clamp_min(1.0)
        policy_q_loss = policy_q_loss / policy_weight.clamp_min(1.0)
        total = (base_total
                 + self.option_coefs["q"] * q_loss
                 + self.option_coefs["duration"] * duration_loss
                 + self.option_coefs["continuation"] * continuation_loss
                 + self.option_coefs["ranking"] * ranking_loss
                 + self.option_coefs["policy"] * policy_loss
                 + self.option_coefs["policy_q"] * policy_q_loss)
        metrics.update({
            "loss_total": total,
            "loss_q": q_loss,
            "loss_option_duration": duration_loss,
            "loss_continuation": continuation_loss,
            "loss_counterfactual_ranking": ranking_loss,
            "loss_policy_prior": policy_loss,
            "prediction_policy_mae": policy_abs,
            "loss_policy_q": policy_q_loss,
            "prediction_q_mae": q_abs / weight_sum,
            "prediction_option_duration_mae_ticks": (
                duration_abs / weight_sum * self.nominal_option_ticks),
            "prediction_continuation_accuracy": (
                continuation_accuracy / weight_sum),
        })
        metrics.update(horizon_metrics)
        return total, metrics
