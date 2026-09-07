"""Turn intent adapter that leaves the frozen lower policy untouched.

The frozen policy tracks Cartesian swing-foot positions reliably but its foot
yaw target produces essentially no base-yaw response.  This adapter maps the
same upper yaw coordinate to *curvature*: it rotates the next stance-frame
placement vector before handing the ordinary target observation to the frozen
policy.  Inner/outer step-width asymmetry is therefore physical and observable.
"""

import torch

from adapters.frozen_lower_env.target_interface import UpperFootholdTargetInterface


class _CurvatureBounds:
    def __init__(self, bounds, gain, minimum_lateral_abs_m):
        self.bounds = bounds
        self.gain = float(gain)
        self.minimum_lateral_abs_m = float(minimum_lateral_abs_m)

    def decode(self, normalized_action, swing_foot):
        local = self.bounds.decode(normalized_action, swing_foot).clone()
        angle = self.gain * local[..., 3]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        forward = cosine * local[..., 0] - sine * local[..., 1]
        lateral = sine * local[..., 0] + cosine * local[..., 1]
        # Preserve the lower policy's trained non-crossing support.  Curvature
        # appears as a narrower inner step and wider outer step.
        side = torch.where(torch.as_tensor(
            swing_foot, device=local.device) == 0, 1.0, -1.0).to(local.dtype)
        lateral = side * torch.maximum(
            side * lateral, lateral.new_full((), self.minimum_lateral_abs_m))
        local[..., 0] = forward
        local[..., 1] = lateral
        return local


class CurvatureFootholdTargetInterface(UpperFootholdTargetInterface):
    def __init__(self, bounds, ground_height_m=0.0, curvature_gain=3.0,
                 minimum_lateral_abs_m=0.06):
        self.original_bounds = bounds
        super().__init__(_CurvatureBounds(
            bounds, curvature_gain, minimum_lateral_abs_m), ground_height_m)
