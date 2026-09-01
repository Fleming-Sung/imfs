"""Fixed Cartesian candidate set shared by training, planning and baselines."""

import torch


FORWARD_LEVELS = (
    -1.0, -0.818182, -0.636364, -0.454545, -0.272727, -0.090909,
    0.090909, 0.272727, 0.454545, 0.636364, 0.818182, 1.0)
LATERAL_LEVELS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
YAW_LEVELS = (-0.5, 0.0, 0.5)


def make_candidates(bounds, device="cpu", minimum_radial_m=0.12,
                    maximum_radial_m=0.35):
    values = torch.tensor([
        (forward, lateral, yaw)
        for forward in FORWARD_LEVELS
        for lateral in LATERAL_LEVELS
        for yaw in YAW_LEVELS
    ], dtype=torch.float32, device=device)
    swing_left = torch.zeros(len(values), dtype=torch.long, device=device)
    decoded = bounds.decode(values, swing_left)
    radial = torch.linalg.norm(decoded[:, :2], dim=-1)
    keep = ((radial >= float(minimum_radial_m))
            & (radial <= float(maximum_radial_m)))
    return values[keep]

