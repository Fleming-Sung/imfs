"""MuJoCo rollout loop, rich state logging, rendering and video recording."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from .course import (
    FOOT_SITE_NAMES, SUPPORT_CODES, FootholdTargetPlanner, RegionalFootholdTargetPlanner, build_scene,
    build_training_demo_scene,
)
from .math_utils import rotate_inverse
from .policy import FootholdPolicy
from .training_sampler import TrainingGoalSampler


JOINT_NAMES = (
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint",
)
ROBOT_BODY_NAMES = (
    "base_Link", "abad_L_Link", "hip_L_Link", "knee_L_Link", "ankle_L_Link",
    "abad_R_Link", "hip_R_Link", "knee_R_Link", "ankle_R_Link",
)


class VideoWriter:
    """Minimal raw-RGB ffmpeg writer, avoiding an optional Python video backend."""

    def __init__(self, path, width, height, fps):
        # The isaacgym conda environment ships an ffmpeg build without libx264.
        # Prefer the system build when it is present so recording behaves the same
        # inside and outside that environment.
        system_ffmpeg = Path("/usr/bin/ffmpeg")
        executable = str(system_ffmpeg) if system_ffmpeg.is_file() else shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required for --record-video")
        command = [
            executable, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
        ]
        child_env = os.environ.copy()
        if executable == str(system_ffmpeg):
            # Conda's FFmpeg libraries are ABI-incompatible with /usr/bin/ffmpeg.
            child_env.pop("LD_LIBRARY_PATH", None)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, env=child_env)

    def append_data(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def _id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"MuJoCo model is missing {name!r}")
    return value


class TraceRecorder:
    def __init__(self, model, planner):
        self.model = model
        self.planner = planner
        self.body_ids = np.array([
            _id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ROBOT_BODY_NAMES
        ])
        self.foot_site_ids = np.array([
            _id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES
        ])
        self.values = {}

    def append(self, key, value):
        self.values.setdefault(key, []).append(np.asarray(value).copy())

    def record(self, data, raw_obs, norm_obs, raw_goal, norm_goal, raw_action, clipped_action,
               target_qpos, torque, command_target_position, command_target_foothold_index,
               command_target_quaternion, command_phase, command_swing_foot,
               foot_contact_force):
        body_vel = np.zeros((len(self.body_ids), 6), dtype=np.float64)
        for row, body_id in enumerate(self.body_ids):
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), body_vel[row], 0)
        site_vel = np.zeros((len(self.foot_site_ids), 6), dtype=np.float64)
        for row, site_id in enumerate(self.foot_site_ids):
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_SITE, int(site_id), site_vel[row], 0)
        self.append("time", data.time)
        self.append("qpos", data.qpos)
        self.append("qvel", data.qvel)
        self.append("qacc", data.qacc)
        self.append("actuator_force", data.actuator_force)
        self.append("ctrl_torque", torque)
        self.append("joint_target", target_qpos)
        self.append("body_position", data.xpos[self.body_ids])
        self.append("body_quaternion_wxyz", data.xquat[self.body_ids])
        self.append("body_velocity_world_ang_lin", body_vel)
        self.append("foot_site_position", data.site_xpos[self.foot_site_ids])
        self.append("foot_site_velocity_world_ang_lin", site_vel)
        self.append("foot_contact_force", foot_contact_force)
        self.append("foothold_position", self.planner.foothold_positions())
        self.append("target_position", self.planner.target_positions())
        self.append("target_quaternion_wxyz", self.planner.target_quaternions())
        self.append("target_foothold_index", self.planner.target_index)
        self.append("target_support_code", [
            SUPPORT_CODES[name] for name in self.planner.target_supports()])
        self.append("phase", self.planner.phase)
        self.append("swing_foot", self.planner.swing_foot)
        # These four fields are the command actually consumed by raw_goal/action on this step.
        # The unprefixed planner fields above are the post-step state for the next action.
        self.append("command_target_position", command_target_position)
        self.append("command_target_foothold_index", command_target_foothold_index)
        self.append("command_target_quaternion_wxyz", command_target_quaternion)
        self.append("command_phase", command_phase)
        self.append("command_swing_foot", command_swing_foot)
        self.append("observation_raw", raw_obs)
        self.append("observation_normalized", norm_obs)
        self.append("goal_raw", raw_goal)
        self.append("goal_normalized", norm_goal)
        self.append("action_raw", raw_action)
        self.append("action_clipped", clipped_action)

    def save(self, path):
        arrays = {key: np.stack(value) for key, value in self.values.items()}
        np.savez_compressed(path, **arrays)
        return arrays


class Deployment:
    def __init__(self, root, course_cfg, output_dir):
        self.root = Path(root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.course_cfg = course_cfg
        self.policy = FootholdPolicy(self.root / "checkpoints" / "model_7000.pt")
        train_cfg = self.policy.checkpoint_config

        scene_path = self.output_dir / "scene.xml"
        robot_xml = self.root / "assets" / "SF_TRON1A" / "xml" / "robot_deploy.xml"
        if course_cfg.get("scenario") == "training_demo":
            self.layout = build_training_demo_scene(robot_xml, scene_path)
        else:
            self.layout = build_scene(
                robot_xml, scene_path, course_cfg, train_cfg["init"]["reset_joint_angles"])
        self.footholds = self.layout.footholds
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self._validate_model(train_cfg)

        self.policy_dt = float(train_cfg["env"]["dt"]) * int(train_cfg["env"]["decimation"])
        ratio = self.policy_dt / self.model.opt.timestep
        self.decimation = int(round(ratio))
        if not np.isclose(ratio, self.decimation) or self.decimation <= 0:
            raise ValueError("policy dt must be an integer multiple of MuJoCo timestep")

        self.joint_ids = np.array([
            _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
        ])
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]
        self.joint_ranges = self.model.jnt_range[self.joint_ids]
        control_cfg = train_cfg["control"]
        self.default_qpos = np.array([
            train_cfg["init"]["default_joint_angles"][name] for name in JOINT_NAMES
        ])
        self.reset_qpos = np.array([
            train_cfg["init"]["reset_joint_angles"][name] for name in JOINT_NAMES
        ])
        self.kp = np.array([control_cfg["stiffness"][name] for name in JOINT_NAMES])
        self.kd = np.array([control_cfg["damping"][name] for name in JOINT_NAMES])
        self.torque_limits = np.array([80.0, 80.0, 80.0, 20.0] * 2)
        self.action_scale = float(control_cfg["action_scale"])
        self.action_clip = float(train_cfg["normalization"]["clip_actions"])

        self.base_id = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self.foot_body_ids = np.array([
            _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ankle_L_Link"),
            _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ankle_R_Link"),
        ])
        self.foot_site_ids = np.array([
            _id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES
        ])
        self.marker_mocap_ids = np.array([
            self.model.body_mocapid[_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_L")],
            self.model.body_mocapid[_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_R")],
        ])
        self.movement_mocap_id = None
        if self.layout.scenario == "training_demo":
            self.planner = TrainingGoalSampler(
                self.model, self.data, train_cfg, course_cfg, int(course_cfg["seed"]))
            movement_body = _id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "movement_path")
            self.movement_mocap_id = int(self.model.body_mocapid[movement_body])
        else:
            if course_cfg.get("planner_mode") == "regional":
                self.planner = RegionalFootholdTargetPlanner(
                    self.model, self.data, self.layout,
                    float(course_cfg["gait_frequency"]), course_cfg)
            else:
                self.planner = FootholdTargetPlanner(
                    self.model, self.data, self.layout, float(course_cfg["gait_frequency"]),
                    float(course_cfg.get("initial_phase", 0.0)))
        self.previous_action = np.zeros(8, dtype=np.float32)
        self._reset()

    def _validate_model(self, train_cfg):
        if self.model.nq != 15 or self.model.nv != 14 or self.model.nu != 8:
            raise ValueError(
                f"unexpected model dimensions nq/nv/nu={self.model.nq}/{self.model.nv}/{self.model.nu}")
        actual = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).replace("_motor", "_Joint")
            for i in range(self.model.nu)
        )
        if actual != JOINT_NAMES:
            raise ValueError(f"actuator order {actual} does not match policy order {JOINT_NAMES}")
        if train_cfg["asset"]["name"] != "SF_TRON1A":
            raise ValueError("checkpoint was not trained for SF_TRON1A")

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        top = self.layout.support_height
        self.data.qpos[:7] = [0.0, 0.0, top + 0.663, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.reset_qpos
        self.data.qvel[:] = 0.0
        self.previous_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Isaac Gym tolerates the reset URDF pose initially intersecting a plane and resolves it
        # during the environment's settling substeps.  On an isolated pile that penetration would
        # create a large, simulator-specific impulse.  Translate the spawn pose once so the copied
        # sole sites touch (but do not penetrate) the two support tops; no pose is forced after t=0.
        support_z = np.full(2, top)
        sole_z = self.data.site_xpos[self.foot_site_ids, 2]
        self.data.qpos[2] += float(np.mean(support_z - sole_z))
        mujoco.mj_forward(self.model, self.data)
        self.planner.reset()
        initial_error = np.linalg.norm(
            self.data.site_xpos[self.foot_site_ids] - self.planner.target_positions(), axis=1)
        if np.max(initial_error) > 2e-3:
            raise ValueError(f"initial feet do not align with support surface: errors={initial_error}")
        self._update_markers()
        if self.movement_mocap_id is not None:
            self.data.mocap_pos[self.movement_mocap_id] = self.planner.movement_origin
            self.data.mocap_quat[self.movement_mocap_id] = self.planner.movement_quaternion()

    def _observation(self):
        base_quat = self.data.xquat[self.base_id]
        projected_gravity = rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        body_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, body_velocity, 0)
        joint_pos = self.data.qpos[self.qpos_adr]
        joint_vel = self.data.qvel[self.dof_adr]
        # Training actor order: gravity, q, WORLD angular velocity, 0.1*dq, previous raw action.
        return np.concatenate((
            projected_gravity, joint_pos, body_velocity[:3], 0.1 * joint_vel,
            self.previous_action,
        )).astype(np.float32)

    def _update_markers(self):
        targets = self.planner.target_positions()
        self.data.mocap_pos[self.marker_mocap_ids] = targets + np.array([0.0, 0.0, 0.035])
        self.data.mocap_quat[self.marker_mocap_ids] = self.planner.target_quaternions()

    def _foot_contact_force(self):
        result = np.zeros((2, 6), dtype=np.float64)
        contact_force = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = (self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2])
            mujoco.mj_contactForce(self.model, self.data, index, contact_force)
            magnitude = np.linalg.norm(contact_force[:3])
            for foot, body_id in enumerate(self.foot_body_ids):
                if body_id in bodies:
                    result[foot, :3] += magnitude * contact.frame[:3]
                    result[foot, 3:] += contact_force[3:]
        return result

    def step(self):
        command_target_position = self.planner.target_positions()
        command_target_foothold_index = np.asarray(self.planner.target_index, dtype=np.int64).copy()
        command_target_quaternion = self.planner.target_quaternions()
        command_phase = float(self.planner.phase)
        command_swing_foot = int(self.planner.swing_foot)
        raw_obs = self._observation()
        raw_goal = self.planner.observation()
        raw_action, norm_obs, norm_goal = self.policy.infer(raw_obs, raw_goal)
        clipped_action = np.clip(raw_action, -self.action_clip, self.action_clip)
        target = np.clip(
            self.default_qpos + self.action_scale * clipped_action,
            self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        torque = self.kp * (target - self.data.qpos[self.qpos_adr]) - self.kd * self.data.qvel[self.dof_adr]
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)

        switched = self.planner.advance(self.policy_dt)
        if switched:
            self._update_markers()
        self.data.ctrl[:] = torque
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.previous_action = raw_action.astype(np.float32)
        return (raw_obs, norm_obs, raw_goal, norm_goal, raw_action, clipped_action, target, torque,
                command_target_position, command_target_foothold_index, command_target_quaternion,
                command_phase, command_swing_foot)

    def run(self, duration_s, interactive=False, realtime=False, video_path=None, camera_cfg=None):
        viewer = None
        if interactive:
            from mujoco import viewer as mj_viewer
            viewer = mj_viewer.launch_passive(self.model, self.data)

        renderer = writer = camera = None
        if video_path is not None:
            width, height = int(camera_cfg["video_width"]), int(camera_cfg["video_height"])
            writer = VideoWriter(video_path, width, height, int(camera_cfg["video_fps"]))
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            camera = mujoco.MjvCamera()
            camera.distance = float(camera_cfg["camera_distance"])
            camera.azimuth = float(camera_cfg["camera_azimuth_deg"])
            camera.elevation = float(camera_cfg["camera_elevation_deg"])

        recorder = TraceRecorder(self.model, self.planner)
        num_steps = int(np.ceil(float(duration_s) / self.policy_dt))
        first_fall_time = None
        first_excessive_tilt_time = None
        wall_start = time.monotonic()
        try:
            for step_index in range(num_steps):
                state = self.step()
                contact = self._foot_contact_force()
                recorder.record(self.data, *state, contact)

                if (first_fall_time is None
                        and self.data.xpos[self.base_id, 2] < self.layout.support_height + 0.30):
                    first_fall_time = float(self.data.time)
                # Keep this separate from the height heuristic: neither signal alone proves a
                # physical fall. The saved video and full body/link trajectory remain the source
                # of truth. xmat[2, 2] is the cosine between base-up and world-up.
                base_tilt_deg = float(np.degrees(np.arccos(np.clip(
                    self.data.xmat[self.base_id].reshape(3, 3)[2, 2], -1.0, 1.0))))
                if first_excessive_tilt_time is None and base_tilt_deg > 60.0:
                    first_excessive_tilt_time = float(self.data.time)

                if writer is not None:
                    camera.lookat[:] = self.data.xpos[self.base_id]
                    renderer.update_scene(self.data, camera=camera)
                    writer.append_data(renderer.render())
                if viewer is not None:
                    viewer.cam.lookat[:] = self.data.xpos[self.base_id]
                    viewer.cam.distance = float(camera_cfg["camera_distance"])
                    viewer.cam.azimuth = float(camera_cfg["camera_azimuth_deg"])
                    viewer.cam.elevation = float(camera_cfg["camera_elevation_deg"])
                    viewer.sync()
                if realtime:
                    deadline = wall_start + (step_index + 1) * self.policy_dt
                    time.sleep(max(0.0, deadline - time.monotonic()))
                if step_index % max(1, int(round(1.0 / self.policy_dt))) == 0:
                    targets = self.planner.target_positions()
                    feet = self.data.site_xpos[self.foot_site_ids]
                    errors = np.linalg.norm(feet[:, :2] - targets[:, :2], axis=1)
                    print(
                        f"t={self.data.time:6.2f}s base_z={self.data.xpos[self.base_id, 2]:.3f} "
                        f"phase={self.planner.phase:.2f} swing={self.planner.swing_foot} "
                        f"target={self.planner.target_index} support={self.planner.target_supports()} "
                        f"foot_xy_error={errors.round(3).tolist()}",
                        flush=True)
        finally:
            if writer is not None:
                writer.close()
            if renderer is not None:
                renderer.close()
            if viewer is not None:
                viewer.close()

        arrays = recorder.save(self.output_dir / "trajectory.npz")
        feet_xy_error = np.linalg.norm(
            arrays["foot_site_position"][..., :2] - arrays["target_position"][..., :2], axis=-1)
        swing = arrays["swing_foot"].astype(np.int64)
        switches = np.r_[True, swing[1:] != swing[:-1]]
        if first_fall_time is not None:
            switches &= arrays["time"] <= first_fall_time
        switch_rows = np.flatnonzero(switches)
        landing_records = []
        for row in switch_rows[1:]:  # the t=0 initial support is not a commanded landing
            stance = 1 - int(swing[row])
            # The first half-cycle merely keeps the reset foot on its initial support.
            target_index = int(arrays["target_foothold_index"][row, stance])
            if target_index < 2:
                continue
            support = ("flat" if self.layout.scenario == "training_demo"
                       else self.footholds[target_index].support)
            tolerance = (float(self.course_cfg["pile_radius"]) if support == "pile"
                         else float(self.course_cfg.get("landing_tolerance", 0.10)))
            error = float(feet_xy_error[row, stance])
            landing_records.append({
                "time_s": float(arrays["time"][row]),
                "foot": "left" if stance == 0 else "right",
                "target_index": target_index,
                "support": support,
                "xy_error_m": error,
                "within_tolerance": error <= tolerance,
            })
        pile_code = SUPPORT_CODES["pile"]
        pile_rows = np.flatnonzero((arrays["target_support_code"] == pile_code).any(axis=1))
        pile_entry_time = float(arrays["time"][pile_rows[0]]) if pile_rows.size else None
        grouped = {}
        for support in ("platform", "transition", "pile", "flat"):
            records = [record for record in landing_records if record["support"] == support]
            grouped[support] = {
                "commanded": len(records),
                "within_tolerance": sum(record["within_tolerance"] for record in records),
                "xy_error_m": [record["xy_error_m"] for record in records],
            }
        metrics = {
            "scenario": self.layout.scenario,
            "duration_requested_s": float(duration_s),
            "duration_simulated_s": float(self.data.time),
            "control_steps": int(num_steps),
            "first_fall_time_s": first_fall_time,
            "fall_detection_note": (
                "first_fall_time_s is only the first base-height threshold crossing; "
                "confirm falls from video plus full body/link and foot-contact trajectories"),
            "first_excessive_base_tilt_time_s": first_excessive_tilt_time,
            "final_base_position": self.data.xpos[self.base_id].tolist(),
            "mean_live_foot_xy_error_m": feet_xy_error.mean(axis=0).tolist(),
            "max_live_foot_xy_error_m": feet_xy_error.max(axis=0).tolist(),
            "pile_entry_time_s": pile_entry_time,
            "fall_relative_to_pile_entry_s": (
                None if first_fall_time is None or pile_entry_time is None
                else first_fall_time - pile_entry_time),
            "landing_records_before_fall": landing_records,
            "landing_summary_before_fall": grouped,
            "final_target_foothold_index": list(map(int, self.planner.target_index)),
            "checkpoint": "checkpoints/model_7000.pt",
            "course_seed": int(self.course_cfg["seed"]),
        }
        if self.layout.scenario == "training_demo":
            metrics["training_demo_settings"] = self.planner.effective_settings()
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        layout = [
            {"index": point.index, "foot": "left" if point.foot == 0 else "right",
             "support": point.support, "position": point.position.tolist()}
            for point in self.footholds
        ]
        (self.output_dir / "foothold_layout.json").write_text(json.dumps(layout, indent=2))
        if self.layout.scenario == "training_demo":
            (self.output_dir / "sampled_target_events.json").write_text(
                json.dumps(self.planner.target_history, indent=2))
            (self.output_dir / "training_demo_settings.json").write_text(
                json.dumps(self.planner.effective_settings(), indent=2))
        if hasattr(self.planner, "planning_history"):
            (self.output_dir / "planning_events.json").write_text(
                json.dumps(self.planner.planning_history, indent=2))
        trace_schema = {
            "robot_body_names": list(ROBOT_BODY_NAMES),
            "joint_and_actuator_order": list(JOINT_NAMES),
            "foot_order": ["left", "right"],
            "quaternion_order": "wxyz",
            "body_velocity_order": ["angular_x", "angular_y", "angular_z", "linear_x", "linear_y", "linear_z"],
            "command_fields": "pre-step values consumed by the recorded action",
            "unprefixed_planner_fields": "post-step values for the next action",
            "support_codes": SUPPORT_CODES,
        }
        (self.output_dir / "trajectory_schema.json").write_text(json.dumps(trace_schema, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
