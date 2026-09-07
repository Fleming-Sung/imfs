"""Bounded H1 action refinement using the same learned physical objective."""
import torch
from .interface import world_targets, geometry
from .anchored import crop_map, physical_goal_progress


@torch.no_grad()
def refine_root(env, ids, initial, image, proprio, cache, planner):
    # Fine XY offsets stay inside the old action range. Retain the original
    # action as candidate zero; the model predicts the actual continuous action,
    # never a nearest coarse index standing in for a different command.
    x, y = torch.meshgrid(initial.new_tensor([-.02, 0, .02]),
                          initial.new_tensor([-.02, 0, .02]), indexing='ij')
    proposed = initial[:, None].expand(-1, 9, -1).clone()
    proposed[..., 0] = (proposed[..., 0] + x.flatten()/.09).clamp(-1, 1)
    proposed[..., 1] = (proposed[..., 1] + y.flatten()/.06).clamp(-1, 1)
    xyz, _ = world_targets(env, ids, proposed)
    stance_z = env.foot_positions[ids, 1-env.sampler.swing_foot[ids], 2]
    # Height is an already observed geometric constraint, not a future outcome.
    proposed[..., 2] = (env.sample_height(xyz[..., :2])-stance_z[:, None])/.08
    proposals = torch.cat((initial[:, None], proposed), 1)
    valid = geometry(env, ids, proposals)['candidate_valid'] & (proposals.abs() <= 1.00001).all(-1)
    empty = ~valid.any(-1); valid[empty, 0] = True
    world, model, cfg = planner.world, planner.motion, planner.config
    latent = world.encode(image, proprio)
    prediction = world.predict_candidates(latent, proposals)
    fall_probability,collision_probability=planner.risk_probabilities(latent,proprio,proposals,prediction)
    score = cfg.progress_weight*prediction['progress']-cfg.support_weight*(1-prediction['support'])
    score -= cfg.fall_weight*fall_probability
    score -= cfg.collision_weight*collision_probability
    if cfg.reward_weight:
        score = (1-cfg.reward_weight)*score+cfg.reward_weight*prediction['reward']
    score -= cfg.uncertainty_weight*prediction['q'].std(0, unbiased=False)
    count = proposals.shape[1]
    features = latent[:, None].expand(-1, count, -1).flatten(0, 1)
    states = proprio[:, None].expand(-1, count, -1).flatten(0, 1)
    motion, next_state, uncertainty = model.predict(features, states, proposals.flatten(0, 1))
    if planner.goal_cost_radius:
        progress,distance=physical_goal_progress(states,motion)
        replacement=progress.view_as(score)-prediction['progress']
        score+=(1-cfg.reward_weight)*cfg.progress_weight*replacement*(distance.view_as(score)<planner.goal_cost_radius)
    score -= .5*uncertainty.view_as(score)
    if cfg.terminal_value_weight:
        maps = cache[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        feature = world.encode(crop_map(maps, motion)[0], next_state)
        value = world.predict_candidates(feature)['q'].min(0).values.max(-1).values.view_as(score)
        score += cfg.terminal_value_weight*cfg.discount*torch.sigmoid(prediction['continuation_logit'])*value
    score = score.masked_fill(~valid, -torch.inf)
    return proposals[torch.arange(len(ids), device=initial.device), score.argmax(-1)]
