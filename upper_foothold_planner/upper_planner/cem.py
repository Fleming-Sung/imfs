"""CEM action-sequence search in the learned latent dynamics."""

import torch

from .contracts import UPPER_ACTION_DIM


def _apply_task_risks(model, latent, action, reward, collision_force_risk=0.0,
                      stability_risk=0.0, support_risk=0.0,
                      touchdown_risk=0.0):
    """Add continuous task risks predicted by TaskLatentWorldModel.

    collision_force is normalized by 100 N and touchdown_error by 0.10 m in
    the model, while stability_margin and support_fraction are dimensionless.
    """
    if not any((collision_force_risk, stability_risk, support_risk,
                touchdown_risk)):
        return reward
    if not hasattr(model, "predict_task_components"):
        return reward
    component = model.predict_task_components(latent, action)
    return (reward
            - float(collision_force_risk) * component["collision_force"]
            - float(stability_risk) * torch.relu(-component["stability_margin"])
            - float(support_risk) * (1.0 - component["support_fraction"])
            - float(touchdown_risk) * component["touchdown_error"])


@torch.no_grad()
def plan(model, latent, horizon=5, candidates=512, elites=64, iterations=5,
         discount=0.99, min_std=0.05, collision_risk=0.0, fall_risk=0.0,
         reward_cfg=None, reward_scale=10.0, action_l2=0.0,
         collision_force_risk=0.0, stability_risk=0.0, support_risk=0.0,
         touchdown_risk=0.0, terminal_value_coef=1.0,
         terminal_q_aggregation="min", freeze_latent_rollout=False,
         future_action_mode="free", policy=None, policy_rollout=False,
         policy_std=0.3):
    device = latent.device
    batch = latent.shape[0]
    action_dim = UPPER_ACTION_DIM
    mean = torch.zeros(batch, horizon, action_dim, device=device)
    std = torch.ones_like(mean) * 0.6
    if policy is not None:
        with torch.no_grad():
            prior = policy(latent).clamp(-1.0, 1.0)
        mean = prior.unsqueeze(1).expand(-1, horizon, -1).clone()

    for _ in range(iterations):
        actions = mean[:, None] + std[:, None] * torch.randn(
            batch, candidates, horizon, action_dim, device=device)
        actions.clamp_(-1.0, 1.0)
        if future_action_mode == "neutral":
            actions[:, :, 1:] = 0.0
        elif future_action_mode == "repeat":
            actions[:, :, 1:] = actions[:, :, :1]
        elif future_action_mode != "free":
            raise ValueError("future action mode must be free, neutral, or repeat")
        z = latent[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        returns = torch.zeros(batch, candidates, device=device)
        cumulative_discount = torch.ones(
            batch, candidates, device=device)
        for step in range(horizon):
            flat_action = actions[:, :, step].reshape(batch * candidates, action_dim)
            step_reward = (model.predict_task_reward(
                z, flat_action, reward_cfg, reward_scale)
                if reward_cfg is not None else model.predict_reward(z, flat_action))
            if collision_risk or fall_risk:
                collision_logit, fall_logit = model.predict_event_logits(z, flat_action)
                step_reward = (step_reward
                               - float(collision_risk) * torch.sigmoid(collision_logit)
                               - float(fall_risk) * torch.sigmoid(fall_logit))
            step_reward = _apply_task_risks(
                model, z, flat_action, step_reward,
                collision_force_risk=collision_force_risk,
                stability_risk=stability_risk, support_risk=support_risk,
                touchdown_risk=touchdown_risk)
            step_reward = step_reward - float(action_l2) * flat_action.square().mean(-1)
            returns += cumulative_discount * step_reward.view(batch, candidates)
            if hasattr(model, "option_duration"):
                option_component = model.predict_task_components(z, flat_action)
                duration = option_component["option_duration"].clamp(0.04, 4.0)
                continuation = torch.sigmoid(
                    option_component["continuation_logit"])
                cumulative_discount *= (
                    float(discount) ** duration * continuation).view(
                        batch, candidates)
            else:
                cumulative_discount *= float(discount)
            if not freeze_latent_rollout:
                z = model.next(z, flat_action)
                if policy_rollout and policy is not None and step < horizon - 1:
                    with torch.no_grad():
                        prior_next = policy(z)
                    nxt = (prior_next
                           + float(policy_std) * torch.randn_like(prior_next)
                           ).clamp(-1.0, 1.0)
                    actions[:, :, step + 1] = nxt.view(
                        batch, candidates, action_dim)
        if hasattr(model, "predict_terminal_value"):
            # Preserve compatibility with compact test/dummy models whose
            # terminal-value method predates the option-model Q ensemble.
            terminal = (model.predict_terminal_value(z)
                        if terminal_q_aggregation == "min"
                        else model.predict_terminal_value(
                            z, aggregation=terminal_q_aggregation))
        else:
            terminal = model.predict_value(z)
        returns += (float(terminal_value_coef) * cumulative_discount
                    * terminal.view(batch, candidates))
        elite_index = returns.topk(elites, dim=1).indices
        gather_index = elite_index[:, :, None, None].expand(-1, -1, horizon, action_dim)
        elite_actions = torch.gather(actions, 1, gather_index)
        mean = elite_actions.mean(1)
        std = elite_actions.std(1, unbiased=False).clamp_min(min_std)
    action = mean[:, 0].clamp(-1.0, 1.0)
    return (action[0] if batch == 1 else action), {"mean": mean, "std": std}


@torch.no_grad()
def plan_ensemble(models, latents, horizon=5, candidates=512, elites=64,
                  iterations=5, discount=0.99, min_std=0.05,
                  collision_risk=0.0, fall_risk=0.0, reward_cfg=None,
                  reward_scale=10.0, uncertainty_coef=0.0, action_l2=0.0,
                  collision_force_risk=0.0, stability_risk=0.0,
                  support_risk=0.0, touchdown_risk=0.0):
    """CEM over mean return minus calibrated ensemble return disagreement."""
    if len(models) != len(latents) or not models:
        raise ValueError("models and latents must be non-empty and have equal length")
    device = latents[0].device
    batch = latents[0].shape[0]
    action_dim = UPPER_ACTION_DIM
    mean = torch.zeros(batch, horizon, action_dim, device=device)
    std = torch.ones_like(mean) * 0.6
    powers = discount ** torch.arange(horizon, device=device)
    final_return_mean = final_return_std = None

    for _ in range(iterations):
        actions = mean[:, None] + std[:, None] * torch.randn(
            batch, candidates, horizon, action_dim, device=device)
        actions.clamp_(-1.0, 1.0)
        member_latents = [
            latent[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
            for latent in latents]
        member_returns = torch.zeros(
            len(models), batch, candidates, device=device)
        for step in range(horizon):
            flat_action = actions[:, :, step].reshape(batch * candidates, action_dim)
            for member, model in enumerate(models):
                z = member_latents[member]
                step_reward = (model.predict_task_reward(
                    z, flat_action, reward_cfg, reward_scale)
                    if reward_cfg is not None else model.predict_reward(z, flat_action))
                if collision_risk or fall_risk:
                    collision_logit, fall_logit = model.predict_event_logits(z, flat_action)
                    step_reward = (step_reward
                                   - float(collision_risk) * torch.sigmoid(collision_logit)
                                   - float(fall_risk) * torch.sigmoid(fall_logit))
                step_reward = _apply_task_risks(
                    model, z, flat_action, step_reward,
                    collision_force_risk=collision_force_risk,
                    stability_risk=stability_risk, support_risk=support_risk,
                    touchdown_risk=touchdown_risk)
                step_reward = step_reward - float(action_l2) * flat_action.square().mean(-1)
                member_returns[member] += powers[step] * step_reward.view(batch, candidates)
                member_latents[member] = model.next(z, flat_action)
        for member, model in enumerate(models):
            member_returns[member] += ((discount ** horizon)
                                       * model.predict_value(member_latents[member])
                                       .view(batch, candidates))
        final_return_mean = member_returns.mean(0)
        final_return_std = member_returns.std(0, unbiased=False)
        score = final_return_mean - float(uncertainty_coef) * final_return_std
        elite_index = score.topk(elites, dim=1).indices
        gather_index = elite_index[:, :, None, None].expand(
            -1, -1, horizon, action_dim)
        elite_actions = torch.gather(actions, 1, gather_index)
        mean = elite_actions.mean(1)
        std = elite_actions.std(1, unbiased=False).clamp_min(min_std)
    action = mean[:, 0].clamp(-1.0, 1.0)
    return (action[0] if batch == 1 else action), {
        "mean": mean, "std": std,
        "candidate_return_mean": final_return_mean,
        "candidate_return_std": final_return_std,
    }


@torch.no_grad()
def plan_anchored_ensemble(nominal_model, nominal_latent, models, latents,
                           horizon=5, candidates=512, elites=64, iterations=5,
                           discount=0.99, min_std=0.05, collision_risk=0.0,
                           fall_risk=0.0, reward_cfg=None, reward_scale=10.0,
                           uncertainty_coef=0.0, action_l2=0.0,
                           collision_force_risk=0.0, stability_risk=0.0,
                           support_risk=0.0, touchdown_risk=0.0):
    """Keep the validated nominal objective; use ensemble only as a risk signal."""
    if len(models) != len(latents) or not models:
        raise ValueError("models and latents must be non-empty and have equal length")
    device = nominal_latent.device
    batch = nominal_latent.shape[0]
    action_dim = UPPER_ACTION_DIM
    mean = torch.zeros(batch, horizon, action_dim, device=device)
    std = torch.ones_like(mean) * 0.6
    powers = discount ** torch.arange(horizon, device=device)
    final_nominal = final_uncertainty = None

    for _ in range(iterations):
        actions = mean[:, None] + std[:, None] * torch.randn(
            batch, candidates, horizon, action_dim, device=device)
        actions.clamp_(-1.0, 1.0)
        nominal_z = nominal_latent[:, None].expand(
            -1, candidates, -1).reshape(batch * candidates, -1)
        member_z = [latent[:, None].expand(
            -1, candidates, -1).reshape(batch * candidates, -1) for latent in latents]
        nominal_return = torch.zeros(batch, candidates, device=device)
        # Calibration validates disagreement of accumulated decomposed task
        # reward, not decoder error or bootstrapped value disagreement.
        member_task_returns = torch.zeros(
            len(models), batch, candidates, device=device)
        for step in range(horizon):
            action = actions[:, :, step].reshape(batch * candidates, action_dim)
            nominal_reward = (nominal_model.predict_task_reward(
                nominal_z, action, reward_cfg, reward_scale)
                if reward_cfg is not None
                else nominal_model.predict_reward(nominal_z, action))
            if collision_risk or fall_risk:
                collision_logit, fall_logit = nominal_model.predict_event_logits(
                    nominal_z, action)
                nominal_reward = (nominal_reward
                                  - float(collision_risk) * torch.sigmoid(collision_logit)
                                  - float(fall_risk) * torch.sigmoid(fall_logit))
            nominal_reward = _apply_task_risks(
                nominal_model, nominal_z, action, nominal_reward,
                collision_force_risk=collision_force_risk,
                stability_risk=stability_risk, support_risk=support_risk,
                touchdown_risk=touchdown_risk)
            nominal_reward = (nominal_reward
                              - float(action_l2) * action.square().mean(-1))
            nominal_return += powers[step] * nominal_reward.view(batch, candidates)
            nominal_z = nominal_model.next(nominal_z, action)
            for member, model in enumerate(models):
                task_reward = (model.predict_task_reward(
                    member_z[member], action, reward_cfg, reward_scale)
                    if reward_cfg is not None
                    else model.predict_reward(member_z[member], action))
                task_reward = (task_reward
                               - float(action_l2) * action.square().mean(-1))
                member_task_returns[member] += (
                    powers[step] * task_reward.view(batch, candidates))
                member_z[member] = model.next(member_z[member], action)
        nominal_return += ((discount ** horizon)
                           * nominal_model.predict_value(nominal_z).view(batch, candidates))
        final_nominal = nominal_return
        final_uncertainty = member_task_returns.std(0, unbiased=False)
        score = nominal_return - float(uncertainty_coef) * final_uncertainty
        elite_index = score.topk(elites, dim=1).indices
        gather_index = elite_index[:, :, None, None].expand(
            -1, -1, horizon, action_dim)
        elite_actions = torch.gather(actions, 1, gather_index)
        mean = elite_actions.mean(1)
        std = elite_actions.std(1, unbiased=False).clamp_min(min_std)
    action = mean[:, 0].clamp(-1.0, 1.0)
    return (action[0] if batch == 1 else action), {
        "mean": mean, "std": std, "candidate_nominal_return": final_nominal,
        "candidate_task_return_std": final_uncertainty,
    }
