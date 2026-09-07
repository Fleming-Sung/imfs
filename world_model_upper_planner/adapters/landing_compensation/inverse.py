"""Choose a lower command whose predicted landing matches a desired option."""

import torch


@torch.no_grad()
def compensate_candidate_indices(env, ids, candidates, bounds,
                                 desired_index, landing_prediction,
                                 yaw_weight_m=0.04):
    """Invert expected frozen-lower landing over the discrete action grid.

    The planner selects a desired foothold candidate. For every executable
    command candidate, the landing critic predicts target-frame XY residual.
    We select the command whose expected actual foothold is closest to the
    desired nominal foothold, while weakly retaining desired foot yaw.
    """
    device = desired_index.device
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    candidates = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    swing = env.sampler.swing_foot[ids]
    actions = candidates[None].expand(len(ids), -1, -1)
    local = bounds.decode(actions, swing[:, None].expand(-1, len(candidates)))
    residual = landing_prediction["mean"].mean(0)
    yaw_delta = local[..., 3]
    cosine, sine = torch.cos(yaw_delta), torch.sin(yaw_delta)
    expected = local[..., :2].clone()
    expected[..., 0] += cosine * residual[..., 0] - sine * residual[..., 1]
    expected[..., 1] += sine * residual[..., 0] + cosine * residual[..., 1]
    row = torch.arange(len(ids), device=device)
    desired_xy = local[row, desired_index, :2]
    desired_yaw = local[row, desired_index, 3]
    position_cost = (expected - desired_xy[:, None]).square().sum(-1)
    yaw_cost = yaw_weight_m ** 2 * (
        local[..., 3] - desired_yaw[:, None]).square()
    return (position_cost + yaw_cost).argmin(-1)
