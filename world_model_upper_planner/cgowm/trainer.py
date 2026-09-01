"""Sequence training for the candidate-grounded option model."""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 3e-4
    discount: float = 0.99
    ema: float = 0.99
    temporal_decay: float = 0.8
    consistency_coef: float = 2.0
    q_coef: float = 1.0
    policy_coef: float = 0.2
    outcome_coef: float = 1.0
    privileged_geometry_coef: float = 1.0
    geometry_ranking_coef: float = 1.0
    geometry_policy_coef: float = 1.0
    geometry_temperature: float = 0.5
    geometry_alignment_weight: float = 2.0
    max_grad_norm: float = 10.0


class WorldModelTrainer:
    """Uses real next encodings as anchors so imagined latents cannot drift free."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or TrainerConfig()
        self.target = copy.deepcopy(model).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.updates = 0

    @torch.no_grad()
    def update_target(self):
        for target, online in zip(self.target.parameters(), self.model.parameters()):
            target.lerp_(online, 1.0 - self.config.ema)

    def loss(self, batch):
        """Batch convention: observations T+1; actions and labels T."""
        depth, proprio, action = batch["depth"], batch["proprio"], batch["action"]
        horizon = action.shape[1]
        latent = self.model.encode(depth[:, 0], proprio[:, 0])
        total = latent.new_zeros(())
        metrics = {}
        weight_sum = 0.0
        for step in range(horizon):
            weight = self.config.temporal_decay ** step
            valid = batch.get("valid", torch.ones_like(batch["done"]))[:, step].float()
            denominator = valid.sum().clamp_min(1.0)

            def mean(value):
                if value.ndim > 1:
                    value = value.flatten(1).mean(-1)
                return (value * valid).sum() / denominator

            chosen = self.model.predict_action(latent, action[:, step])
            with torch.no_grad():
                target_next = self.target.encode(depth[:, step + 1], proprio[:, step + 1])
                target_all = self.target.predict_candidates(target_next)
                next_q = target_all["q"].min(0).values.max(-1).values
                q_target = (batch["reward"][:, step]
                            + self.config.discount * (1.0 - batch["done"][:, step]) * next_q)

            predicted_next = self.model.next(latent, action[:, step])
            consistency = mean((predicted_next - target_next).square())
            reward_loss = mean(F.smooth_l1_loss(
                chosen["reward"], batch["reward"][:, step], reduction="none"))
            progress_loss = mean(F.smooth_l1_loss(
                chosen["progress"], batch["progress"][:, step], reduction="none"))
            support_loss = mean(F.smooth_l1_loss(
                chosen["support"], batch["support"][:, step], reduction="none"))
            touchdown_loss = mean(F.smooth_l1_loss(
                chosen["touchdown_error"], batch["touchdown_error"][:, step],
                reduction="none"))
            event_loss = mean(
                F.binary_cross_entropy_with_logits(
                    chosen["fall_logit"], batch["fall"][:, step], reduction="none")
                + F.binary_cross_entropy_with_logits(
                    chosen["collision_logit"], batch["collision"][:, step], reduction="none")
                + F.binary_cross_entropy_with_logits(
                    chosen["continuation_logit"], 1.0 - batch["done"][:, step],
                    reduction="none"))
            q_loss = mean(F.smooth_l1_loss(
                chosen["q"], q_target.unsqueeze(0).expand_as(chosen["q"]),
                reduction="none").mean(0))

            # The policy prior only proposes in-distribution candidates. It is
            # trained from a soft pessimistic-Q target, not from a rare goal head.
            all_prediction = self.model.predict_candidates(latent)
            soft_target = torch.softmax(
                all_prediction["q"].detach().min(0).values, dim=-1)
            policy_loss = mean(-(soft_target * F.log_softmax(
                all_prediction["policy_logits"], dim=-1)).sum(-1))
            # When exact simulator geometry labels are available, ground the
            # representation on every candidate at the same state. Dynamic
            # outcomes remain supervised only for the physically executed
            # option; no counterfactual physics is fabricated.
            if "candidate_support" in batch:
                support_all = F.smooth_l1_loss(
                    all_prediction["support"],
                    batch["candidate_support"][:, step], reduction="none")
                support_all_loss = mean(support_all)
                candidate_valid = batch.get("candidate_valid")
                if candidate_valid is None:
                    candidate_valid = torch.ones_like(support_all)
                else:
                    candidate_valid = candidate_valid[:, step].float()
                progress_all = F.smooth_l1_loss(
                    all_prediction["progress"],
                    batch["candidate_progress"][:, step], reduction="none")
                progress_all = ((progress_all * candidate_valid).sum(-1)
                                / candidate_valid.sum(-1).clamp_min(1.0))
                progress_all_loss = mean(progress_all)
                target_score = (
                    10.0 * batch["candidate_progress"][:, step]
                    - 3.0 * (1.0 - batch["candidate_support"][:, step]))
                if "candidate_alignment" in batch:
                    target_score = target_score + (
                        self.config.geometry_alignment_weight
                        * batch["candidate_alignment"][:, step])
                target_score = target_score.masked_fill(
                    candidate_valid < 0.5, -30.0)
                target_distribution = torch.softmax(
                    target_score / self.config.geometry_temperature, dim=-1)
                predicted_score = (
                    10.0 * all_prediction["progress"]
                    - 3.0 * (1.0 - all_prediction["support"]))
                # A disconnected/harder-than-action-set state can have no
                # valid candidate. Such a row is useful negative support and
                # outcome data, but has no well-defined ranking target.
                ranking_valid = valid * (candidate_valid.sum(-1) > 0).float()
                ranking_denominator = ranking_valid.sum().clamp_min(1.0)
                ranking_ce = -(
                    target_distribution * F.log_softmax(
                        predicted_score / self.config.geometry_temperature,
                        dim=-1)).sum(-1)
                policy_ce = -(
                    target_distribution * F.log_softmax(
                        all_prediction["policy_logits"], dim=-1)).sum(-1)
                geometry_ranking_loss = (
                    ranking_ce * ranking_valid).sum() / ranking_denominator
                geometry_policy_loss = (
                    policy_ce * ranking_valid).sum() / ranking_denominator
                geometry_loss = support_all_loss + progress_all_loss
            else:
                support_all_loss = support_loss
                progress_all_loss = progress_loss
                geometry_loss = support_loss + progress_loss
                geometry_ranking_loss = latent.new_zeros(())
                geometry_policy_loss = latent.new_zeros(())
            outcome_loss = reward_loss + touchdown_loss + event_loss
            step_loss = (
                self.config.consistency_coef * consistency
                + self.config.q_coef * q_loss
                + self.config.policy_coef * policy_loss
                + self.config.outcome_coef * outcome_loss
                + self.config.privileged_geometry_coef * geometry_loss
                + self.config.geometry_ranking_coef * geometry_ranking_loss
                + self.config.geometry_policy_coef * geometry_policy_loss)
            total = total + weight * step_loss
            weight_sum += weight
            metrics.update({
                f"consistency_h{step + 1}": consistency.detach(),
                f"q_h{step + 1}": q_loss.detach(),
                f"reward_h{step + 1}": reward_loss.detach(),
                f"support_h{step + 1}": support_loss.detach(),
                f"candidate_support_h{step + 1}": support_all_loss.detach(),
                f"candidate_progress_h{step + 1}": progress_all_loss.detach(),
                f"geometry_ranking_h{step + 1}": geometry_ranking_loss.detach(),
                f"geometry_policy_h{step + 1}": geometry_policy_loss.detach(),
            })
            latent = predicted_next
        total = total / max(weight_sum, 1e-6)
        metrics["loss_total"] = total.detach()
        return total, metrics

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.update_target()
        self.updates += 1
        result = {name: float(value) for name, value in metrics.items()}
        result.update({"grad_norm": float(grad_norm), "updates": self.updates})
        return result
