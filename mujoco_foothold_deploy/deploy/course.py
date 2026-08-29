"""Fixed support layouts and the rule-based live foothold target scheduler."""

from dataclasses import dataclass
from pathlib import Path
import re

import mujoco
import numpy as np

from .math_utils import canonicalize, quat_conjugate, quat_multiply, rotate_inverse, yaw_quaternion


COURSE_MARKER = "<!-- COURSE_BODIES: deploy.course replaces this marker with fixed pile bodies. -->"
FOOT_SITE_NAMES = ("foot_L_site", "foot_R_site")
SUPPORT_CODES = {"platform": 0, "pile": 1, "flat": 2, "transition": 3}


@dataclass(frozen=True)
class Foothold:
    index: int
    foot: int
    position: np.ndarray
    support: str

    @property
    def site_name(self):
        return f"foothold_{self.index:03d}"


@dataclass(frozen=True)
class CourseLayout:
    scenario: str
    footholds: tuple
    support_height: float
    platform_start_x: float
    platform_end_x: float
    platform_half_width: float


def nominal_foot_xy(robot_xml, reset_joint_angles, support_height):
    """Use the official robot FK to locate both soles in the reset joint pose."""
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, support_height + 0.663, 1.0, 0.0, 0.0, 0.0]
    for name, value in reset_joint_angles.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)
    return np.stack([
        data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name), :2].copy()
        for name in FOOT_SITE_NAMES
    ])


def _limits(cfg, name):
    low, high = map(float, cfg[name])
    if not 0.0 < low <= high:
        raise ValueError(f"{name} must satisfy 0 < low <= high")
    return low, high


def _append_alternating_targets(
        footholds, count, x, spacing, cfg, rng, support, lateral_jitter=None):
    half_width = float(cfg["nominal_half_width"])
    jitter = float(cfg["lateral_jitter"] if lateral_jitter is None else lateral_jitter)
    height = float(cfg["support_height"])
    first_foot = str(cfg.get("first_new_foot", "right")).lower()
    if first_foot not in ("left", "right"):
        raise ValueError("first_new_foot must be 'left' or 'right'")
    first_foot_index = 0 if first_foot == "left" else 1
    for _ in range(count):
        index = len(footholds)
        foot = (first_foot_index + index - 2) % 2
        x += rng.uniform(*spacing)
        side = 1.0 if foot == 0 else -1.0
        y = side * half_width + rng.uniform(-jitter, jitter)
        footholds.append(Foothold(index, foot, np.array([x, y, height]), support))
    return x


def generate_layout(cfg, initial_foot_xy):
    """Generate a deterministic flat course or a platform-to-piles course."""
    scenario = cfg.get("scenario", "plum_piles")
    if scenario not in ("plum_piles", "flat"):
        raise ValueError(f"unsupported scenario {scenario!r}")
    rng = np.random.default_rng(int(cfg["seed"]))
    height = float(cfg["support_height"])
    if height <= 0.0:
        raise ValueError("support_height must be positive")
    initial_support = "platform" if scenario == "plum_piles" else "flat"
    footholds = [
        Foothold(0, 0, np.array([*initial_foot_xy[0], height]), initial_support),
        Foothold(1, 1, np.array([*initial_foot_xy[1], height]), initial_support),
    ]
    x = float(max(initial_foot_xy[:, 0]))

    if scenario == "flat":
        count = int(cfg["num_footholds"])
        if count < 4:
            raise ValueError("num_footholds must be at least 4")
        x = _append_alternating_targets(
            footholds, count - 2, x, _limits(cfg, "longitudinal_spacing"), cfg, rng, "flat")
        margin = float(cfg.get("platform_margin", 1.0))
        return CourseLayout(
            scenario, tuple(footholds), height,
            float(min(initial_foot_xy[:, 0]) - margin), x + margin,
            float(cfg["platform_half_width"]))

    warmup_steps = int(cfg["warmup_steps"])
    if warmup_steps < 2 or warmup_steps % 2:
        raise ValueError("warmup_steps must be a positive even number")
    x = _append_alternating_targets(
        footholds, warmup_steps, x, _limits(cfg, "warmup_longitudinal_spacing"),
        cfg, rng, "platform", float(cfg.get("warmup_lateral_jitter", 0.0)))
    platform_start_x = float(cfg["platform_start_x"])
    platform_end_x = float(cfg["platform_end_x"])
    if not platform_start_x < float(initial_foot_xy[:, 0].min()) < x < platform_end_x:
        raise ValueError("warmup footholds must fit strictly inside the start platform")

    num_piles = int(cfg["num_piles"])
    radius = float(cfg["pile_radius"])
    clearance = float(cfg.get("minimum_pile_clearance", 0.0))
    if num_piles < 2 or radius <= 0.0 or clearance < 0.0:
        raise ValueError("num_piles/radius/clearance are invalid")
    entry_low, entry_high = _limits(cfg, "pile_entry_gap")
    x = float(cfg.get(
        "first_pile_x", platform_end_x + radius + rng.uniform(entry_low, entry_high)))
    pile_spacing = _limits(cfg, "longitudinal_spacing")
    pile_points = []
    half_width = float(cfg["nominal_half_width"])
    center_y = float(cfg.get("route_center_y", 0.0))
    jitter = float(cfg["lateral_jitter"])
    transition_targets = int(cfg.get("transition_targets", 0))
    if transition_targets < 0 or transition_targets > num_piles:
        raise ValueError("transition_targets must be between zero and num_piles")
    for pile_number in range(num_piles):
        index = len(footholds)
        first_foot = str(cfg.get("first_new_foot", "right")).lower()
        first_foot_index = 0 if first_foot == "left" else 1
        foot = (first_foot_index + index - 2) % 2
        if pile_number:
            x += rng.uniform(*pile_spacing)
        side = 1.0 if foot == 0 else -1.0
        support = "transition" if pile_number < transition_targets else "pile"
        point = Foothold(
            index, foot,
            np.array([x, center_y + side * half_width + rng.uniform(-jitter, jitter), height]), support)
        footholds.append(point)
        if support == "pile":
            pile_points.append(point)

    pile_xy = np.stack([point.position[:2] for point in pile_points])
    pairwise = np.linalg.norm(pile_xy[:, None] - pile_xy[None, :], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    required = 2.0 * radius + clearance
    if pairwise.min() < required - 1e-9:
        raise ValueError(
            f"generated piles overlap: min center distance={pairwise.min():.3f}, required={required:.3f}")
    if pile_xy[:, 0].min() - radius < platform_end_x + entry_low - 1e-9:
        raise AssertionError("first pile overlaps the start platform")
    return CourseLayout(
        scenario, tuple(footholds), height, platform_start_x, platform_end_x,
        float(cfg["platform_half_width"]))


def _platform_xml(layout):
    center_x = 0.5 * (layout.platform_start_x + layout.platform_end_x)
    half_length = 0.5 * (layout.platform_end_x - layout.platform_start_x)
    name = "start_platform" if layout.scenario == "plum_piles" else "flat_validation_platform"
    sites = []
    for point in layout.footholds:
        if point.support == "pile":
            continue
        local_x = point.position[0] - center_x
        sites.append(
            f'<site name="{point.site_name}" pos="{local_x:.9f} {point.position[1]:.9f} '
            f'{layout.support_height:.9f}" size="0.006" rgba="1 0.85 0.1 1"/>')
    return (
        f'<body name="{name}" pos="{center_x:.9f} 0 0">\n'
        f'  <geom name="{name}_geom" type="box" size="{half_length:.9f} '
        f'{layout.platform_half_width:.9f} {0.5 * layout.support_height:.9f}" '
        f'pos="0 0 {0.5 * layout.support_height:.9f}" rgba="0.24 0.29 0.34 1"/>\n'
        f'  ' + "\n  ".join(sites) + "\n"
        f'</body>')


def _pile_xml(point, radius, height):
    x, y, _ = point.position
    rgba = "0.58 0.34 0.16 1" if point.foot == 0 else "0.48 0.25 0.11 1"
    return (
        f'<body name="pile_{point.index:03d}" pos="{x:.9f} {y:.9f} 0">\n'
        f'  <geom name="pile_geom_{point.index:03d}" type="cylinder" '
        f'size="{radius:.9f} {0.5 * height:.9f}" pos="0 0 {0.5 * height:.9f}" '
        f'rgba="{rgba}"/>\n'
        f'  <site name="{point.site_name}" pos="0 0 {height:.9f}" size="0.006" '
        f'rgba="1 0.85 0.1 1"/>\n'
        f'</body>')


def build_scene(robot_xml, output_xml, cfg, reset_joint_angles):
    """Inject one fixed course into the copied official robot MJCF."""
    robot_xml = Path(robot_xml)
    initial_xy = nominal_foot_xy(
        robot_xml, reset_joint_angles, float(cfg["support_height"]))
    layout = generate_layout(cfg, initial_xy)
    source = robot_xml.read_text()
    if source.count(COURSE_MARKER) != 1:
        raise ValueError(f"scene template must contain exactly one {COURSE_MARKER!r}")
    mesh_match = re.search(r'meshdir="([^"]+)"', source)
    if mesh_match is None:
        raise ValueError("robot XML is missing compiler meshdir")
    mesh_dir = (robot_xml.parent / mesh_match.group(1)).resolve()
    source = source.replace(mesh_match.group(0), f'meshdir="{mesh_dir}"', 1)
    bodies = [_platform_xml(layout)]
    if layout.scenario == "plum_piles":
        bodies.extend(
            _pile_xml(point, float(cfg["pile_radius"]), layout.support_height)
            for point in layout.footholds if point.support == "pile")
    output_xml = Path(output_xml)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    output_xml.write_text(source.replace(COURSE_MARKER, "\n    ".join(bodies)))
    return layout


def build_training_demo_scene(robot_xml, output_xml):
    """Build the unbounded flat-floor scene used to replay the training goal sampler."""
    robot_xml = Path(robot_xml)
    source = robot_xml.read_text()
    if source.count(COURSE_MARKER) != 1:
        raise ValueError(f"scene template must contain exactly one {COURSE_MARKER!r}")
    mesh_match = re.search(r'meshdir="([^"]+)"', source)
    if mesh_match is None:
        raise ValueError("robot XML is missing compiler meshdir")
    mesh_dir = (robot_xml.parent / mesh_match.group(1)).resolve()
    source = source.replace(mesh_match.group(0), f'meshdir="{mesh_dir}"', 1)
    output_xml = Path(output_xml)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    macro_path = (
        '<body name="movement_path" mocap="true">\n'
        '  <geom name="movement_path_line" type="capsule" size="0.008 4.0" '
        'pos="4 0 0" quat="0.7071068 0 0.7071068 0" rgba="0.05 0.85 1 0.55" '
        'contype="0" conaffinity="0" group="2"/>\n'
        '  <geom name="movement_path_origin" type="sphere" size="0.025" '
        'rgba="0.05 0.85 1 0.8" contype="0" conaffinity="0" group="2"/>\n'
        '  <geom name="movement_path_end" type="sphere" size="0.045" pos="8 0 0" '
        'rgba="0.05 0.85 1 0.8" contype="0" conaffinity="0" group="2"/>\n'
        '</body>')
    output_xml.write_text(source.replace(COURSE_MARKER, macro_path))
    return CourseLayout("training_demo", tuple(), 0.0, 0.0, 0.0, 0.0)


class FootholdTargetPlanner:
    """Read live support sites and express targets in the current stance-foot frame."""

    def __init__(self, model, data, layout, frequency, initial_phase=0.0):
        self.model = model
        self.data = data
        self.layout = layout
        self.footholds = layout.footholds
        self.frequency = float(frequency)
        self.initial_phase = float(initial_phase) % 1.0
        self.phase = 0.0
        self.swing_foot = 0
        self.target_index = [0, 1]
        self.target_quat = np.tile(yaw_quaternion(0.0), (2, 1))
        self.foot_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES
        ]
        self.foot_body_ids = [model.site_bodyid[i] for i in self.foot_site_ids]
        self.foothold_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, point.site_name)
            for point in self.footholds
        ]
        self.reset()

    @property
    def target_pile(self):  # backward-compatible display/trace field
        return self.target_index

    def reset(self):
        self.phase = self.initial_phase
        self.swing_foot = int(self.phase >= 0.5)
        self.target_index[:] = [0, 1]
        self.next_by_foot = {
            foot: [point.index for point in self.footholds[2:] if point.foot == foot]
            for foot in (0, 1)
        }
        self.target_quat[:] = self.data.xquat[self.foot_body_ids]

    def foothold_positions(self):
        # Sites are read every tick, so later height-dynamic bodies require no Actor change.
        return self.data.site_xpos[self.foothold_site_ids].copy()

    def pile_positions(self):  # compatibility alias for older analysis scripts
        return self.foothold_positions()

    def target_positions(self):
        return self.foothold_positions()[self.target_index]

    def target_quaternions(self):
        return self.target_quat.copy()

    def target_supports(self):
        return [self.footholds[index].support for index in self.target_index]

    def advance(self, policy_dt):
        old_half = int(np.floor(self.phase * 2.0))
        self.phase = (self.phase + policy_dt * self.frequency) % 1.0
        new_half = int(np.floor(self.phase * 2.0))
        if new_half != old_half:
            self.swing_foot = 1 if self.phase >= 0.5 else 0
            queue = self.next_by_foot[self.swing_foot]
            if queue:
                self.target_index[self.swing_foot] = queue.pop(0)
                self.target_quat[self.swing_foot] = yaw_quaternion(0.0)
            return True
        return False

    def observation(self):
        stance = 1 - self.swing_foot
        origin = self.data.site_xpos[self.foot_site_ids[stance]]
        stance_quat = self.data.xquat[self.foot_body_ids[stance]]
        targets = self.target_positions()
        parts = []
        for foot in (0, 1):
            rel_pos = rotate_inverse(stance_quat, targets[foot] - origin)
            rel_quat = canonicalize(quat_multiply(
                quat_conjugate(stance_quat), self.target_quat[foot]))
            parts.extend((rel_pos, rel_quat))
        phase = np.array([
            np.cos(2.0 * np.pi * self.phase), np.sin(2.0 * np.pi * self.phase)])
        return np.concatenate((*parts, phase)).astype(np.float32)


def _quaternion_yaw(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RegionalFootholdTargetPlanner(FootholdTargetPlanner):
    """Anchor training-shaped live targets inside fixed physical support regions.

    The terrain and each support center stay fixed. At a phase transition the
    planner forms the same deterministic candidate geometry as the training
    sampler, then projects only the commanded point into the safe inner region
    of the assigned support. This compensates accumulated landing error without
    moving the robot or the terrain.
    """

    def __init__(self, model, data, layout, frequency, cfg):
        self.cfg = cfg
        self.target_pos = np.zeros((2, 3), dtype=np.float64)
        self.planning_history = []
        super().__init__(
            model, data, layout, frequency, float(cfg.get("initial_phase", 0.0)))

    def reset(self):
        super().reset()
        self.target_pos[:] = self.data.site_xpos[self.foot_site_ids]
        self.planning_history = []

    def target_positions(self):
        return self.target_pos.copy()

    def _live_candidate(self, swing, support):
        stance = 1 - swing
        stance_pos = self.data.site_xpos[self.foot_site_ids[stance]].copy()
        stance_yaw = _quaternion_yaw(self.data.xquat[self.foot_body_ids[stance]])
        key = ("warmup_planner_step_distance" if support == "platform"
               else "pile_planner_step_distance")
        distance = float(self.cfg[key])
        direction = np.deg2rad(float(self.cfg.get("route_direction_deg", 0.0)))
        target = stance_pos.copy()
        target[:2] += distance * np.array([np.cos(direction), np.sin(direction)])

        c, s = np.cos(stance_yaw), np.sin(stance_yaw)
        delta = target - stance_pos
        local = np.array([
            c * delta[0] + s * delta[1],
            -s * delta[0] + c * delta[1],
            delta[2],
        ])
        separation = float(self.cfg["planner_lateral_separation"])
        local[1] = max(local[1], separation) if swing == 0 else min(local[1], -separation)
        target[0] = stance_pos[0] + c * local[0] - s * local[1]
        target[1] = stance_pos[1] + s * local[0] + c * local[1]
        return target, stance_pos, stance_yaw

    def _select_target(self, swing, index):
        point = self.footholds[index]
        centers = self.foothold_positions()
        center = centers[index]
        candidate, stance_pos, stance_yaw = self._live_candidate(swing, point.support)
        radius_key = {
            "platform": "platform_target_adjustment_radius",
            "transition": "transition_target_adjustment_radius",
            "pile": "pile_target_adjustment_radius",
        }[point.support]
        radius = float(self.cfg[radius_key])
        offset = candidate[:2] - center[:2]
        requested_offset = float(np.linalg.norm(offset))
        if requested_offset > radius:
            offset *= radius / requested_offset
        selected = center.copy()
        selected[:2] += offset
        self.target_pos[swing] = selected
        self.target_quat[swing] = yaw_quaternion(
            np.deg2rad(float(self.cfg.get("target_world_yaw_deg", 0.0))))
        self.planning_history.append({
            "time_s": float(self.data.time),
            "foot": "left" if swing == 0 else "right",
            "foothold_index": int(index),
            "support": point.support,
            "support_center": center.tolist(),
            "live_unconstrained_candidate": candidate.tolist(),
            "selected_target": selected.tolist(),
            "requested_center_offset_m": requested_offset,
            "selected_center_offset_m": float(np.linalg.norm(offset)),
            "adjustment_radius_m": radius,
            "candidate_inside_safe_region": bool(requested_offset <= radius + 1e-9),
            "stance_position": stance_pos.tolist(),
            "stance_yaw_deg": float(np.degrees(stance_yaw)),
        })

    def advance(self, policy_dt):
        old_half = int(np.floor(self.phase * 2.0))
        self.phase = (self.phase + policy_dt * self.frequency) % 1.0
        new_half = int(np.floor(self.phase * 2.0))
        if new_half == old_half:
            return False
        self.swing_foot = 1 if self.phase >= 0.5 else 0
        queue = self.next_by_foot[self.swing_foot]
        if queue:
            index = queue.pop(0)
            self.target_index[self.swing_foot] = index
            self._select_target(self.swing_foot, index)
        return True
