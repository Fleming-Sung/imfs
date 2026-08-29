"""env.py —— 基于 Isaac Gym 的 SF_TRON1A 平地落足跟踪环境。

只使用 gymapi/gymtorch 底层接口。控制采用论文的 PD（EFFORT 模式手算力矩），
终止条件为高度-only（论文 paper_termination_path），观测/动作/奖励均按论文。
"""

import math
import os
import sys

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import to_torch, quat_rotate_inverse, torch_rand_float
import torch

from .sampler import FootholdSampler, rigid_body_site_state
from .rewards import Rewards


def make_sim_params(cfg, args):
    sim = gymapi.SimParams()
    sim.use_gpu_pipeline = args.use_gpu_pipeline
    sim.physx.use_gpu = args.use_gpu
    sim.physx.num_subscenes = args.subscenes
    gymutil.parse_sim_config({
        "dt": cfg.env.dt,
        "gravity": cfg.env.gravity,
        "up_axis": cfg.env.up_axis,
        "physx": {
            "num_threads": 0, "solver_type": 1,
            "num_position_iterations": 4, "num_velocity_iterations": 0,
            "contact_offset": 0.01, "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.5,
            "max_depenetration_velocity": 1.0,
        },
    }, sim)
    return sim


class FootholdEnv:
    def __init__(self, cfg, sim_params, sim_device, headless=False):
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.sim_params = sim_params
        self.headless = headless
        sim_type, self.sim_device_id = gymutil.parse_device_str(sim_device)
        self.device = sim_device if (sim_type == "cuda" and sim_params.use_gpu_pipeline) else "cpu"

        self.num_envs = cfg.env.num_envs
        self.dt = cfg.env.decimation * cfg.env.dt
        self.max_episode_length = int(np.ceil(cfg.env.episode_length_s / self.dt))
        self.fail_to_terminal = int(np.ceil(cfg.env.fail_to_terminal_time_s / self.dt))
        # Reverse-curriculum reset: with probability respawn at a supported
        # mid-course position instead of the course start.
        self.reset_curriculum_spawns = None   # (num_envs, K, 2) local xy candidates
        self.reset_curriculum_prob = 0.0

        # ---- 建仿真（平地）----
        camera_cfg = getattr(cfg, "camera", None)
        self.camera_enabled = bool(camera_cfg and getattr(camera_cfg, "enabled", False))
        self.graphics_device_id = self.sim_device_id if (not headless or self.camera_enabled) else -1
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       gymapi.SIM_PHYSX, self.sim_params)
        self._create_ground()
        self._create_envs()
        self._create_camera_sensors()
        self.gym.prepare_sim(self.sim)

        # ---- 张量 ----
        root = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof = self.gym.acquire_dof_state_tensor(self.sim)
        cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        rb = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(root).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof).view(self.num_envs, -1, 2)
        self.contact_forces = gymtorch.wrap_tensor(cf).view(self.num_envs, -1, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rb).view(self.num_envs, -1, 13)

        self.base_position = self.root_states[:, 0, :3]
        self.base_quat = self.root_states[:, 0, 3:7]
        self.dof_pos = self.dof_state[:, :, 0]
        self.dof_vel = self.dof_state[:, :, 1]
        self.base_lin_vel = torch.zeros_like(self.root_states[:, 0, 7:10])
        self.base_ang_vel = torch.zeros_like(self.root_states[:, 0, 10:13])
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.root_acceleration = torch.zeros_like(self.root_states[:, 0, 7:13])
        g = torch.tensor(cfg.env.gravity, device=self.device)
        self.gravity_vec = (g / g.norm()).repeat(self.num_envs, 1)

        # ---- 缓冲 ----
        self.actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.policy_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.last_actions = torch.zeros(self.num_envs, self.num_dof, 2, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_velocity = self.root_states[:, 0, 7:13].clone()
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.fail_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_sums = {}
        self.extras = {}

        # ---- 足端 site ----
        offsets = torch.tensor(self.cfg.asset.foot_site_offsets, dtype=torch.float32, device=self.device)
        self.foot_site_offsets = offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        self._compute_foot_state()

        # ---- 目标采样器与奖励 ----
        self.sampler = FootholdSampler(self.num_envs, self.cfg.foothold, self.device)
        self.rewards = Rewards(self, self.cfg)
        for name in self.rewards.names:
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        # ---- 观测 ----
        self.num_obs = 6 + 3 * self.num_dof
        self.num_critic_obs = 9 + 3 * self.num_dof
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device)
        self.goal_buf = torch.zeros(self.num_envs, self.cfg.foothold.goal_dim, device=self.device)
        self.goal_reset_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---- 可视化 ----
        self.enable_viewer_sync = True   # 按 V 键可在"渲染/不渲染"之间切换（加速训练）
        self.viewer = None
        if not headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # 必须显式订阅键盘事件，否则 V 键在 query_viewer_action_events 里不会出现
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.viewer_camera_look_at(self.viewer, None,
                                           gymapi.Vec3(2.5, -1.5, 1.5), gymapi.Vec3(0, 0, 0.6))

        # ---- 重力随机化 ----
        self.rigid_body_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self.rigid_body_torques = torch.zeros_like(self.rigid_body_forces)
        if self.cfg.domain_rand.randomize_gravity:
            lo, hi = self.cfg.domain_rand.gravity_magnitude_range
            self.gravity_magnitude = torch_rand_float(lo, hi, (self.num_envs, 1), device=self.device).squeeze(1)

        # ---- 初始 reset ----
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()

    def _create_ground(self):
        """Create the reference plane or an explicitly supplied tiled heightfield."""
        cfg = self.cfg.terrain
        samples = getattr(cfg, "height_samples", None)
        if samples is None:
            if bool(getattr(cfg, "add_ground_plane", True)):
                plane_params = gymapi.PlaneParams()
                plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
                plane_params.static_friction = cfg.static_friction
                plane_params.dynamic_friction = cfg.dynamic_friction
                plane_params.restitution = cfg.restitution
                # Plane equation uses n.x + distance = 0, so z=-0.4 m
                # requires distance=+0.4 rather than -0.4.
                plane_params.distance = -float(getattr(cfg, "ground_height_m", 0.0))
                self.gym.add_ground(self.sim, plane_params)
            vertices = getattr(cfg, "mesh_vertices", None)
            triangles = getattr(cfg, "mesh_triangles", None)
            if vertices is not None:
                vertices = np.asarray(vertices, dtype=np.float32)
                triangles = np.asarray(triangles, dtype=np.uint32)
                mesh = gymapi.TriangleMeshParams()
                mesh.nb_vertices = vertices.shape[0]
                mesh.nb_triangles = triangles.shape[0]
                mesh.static_friction = cfg.static_friction
                mesh.dynamic_friction = cfg.dynamic_friction
                mesh.restitution = cfg.restitution
                self.gym.add_triangle_mesh(
                    self.sim, vertices.flatten(order="C"),
                    triangles.flatten(order="C"), mesh)
            return
        samples = np.asarray(samples, dtype=np.int16)
        if samples.ndim != 2:
            raise ValueError("terrain.height_samples must be a 2-D integer array")
        params = gymapi.HeightFieldParams()
        params.row_scale = float(cfg.horizontal_scale)
        params.column_scale = float(cfg.horizontal_scale)
        params.vertical_scale = float(cfg.vertical_scale)
        # Our research arrays use [x,y], while Isaac's flattened heightfield uses
        # [row=y,column=x]. Pass an explicit transpose; changing only dimensions
        # either creates periodic fences or swaps the corridor and spawn axes.
        api_samples = np.ascontiguousarray(samples.T)
        params.nbRows = api_samples.shape[0]
        params.nbColumns = api_samples.shape[1]
        origin = cfg.heightfield_origin_xy
        params.transform.p.x = float(origin[0])
        params.transform.p.y = float(origin[1])
        params.transform.p.z = 0.0
        params.static_friction = cfg.static_friction
        params.dynamic_friction = cfg.dynamic_friction
        params.restitution = cfg.restitution
        self.gym.add_heightfield(self.sim, api_samples.flatten(order="C"), params)

    def _create_camera_sensors(self):
        self.camera_handles = []
        self.camera_tensors = []
        if not self.camera_enabled:
            return
        cfg = self.cfg.camera
        props = gymapi.CameraProperties()
        props.width = int(cfg.width)
        props.height = int(cfg.height)
        props.horizontal_fov = float(cfg.horizontal_fov_deg)
        props.enable_tensors = True
        xyz = cfg.mount_xyz_base_m
        roll, pitch, yaw = cfg.mount_rpy_base_rad
        mount = gymapi.Transform(
            p=gymapi.Vec3(*[float(value) for value in xyz]),
            r=gymapi.Quat.from_euler_zyx(float(yaw), float(pitch), float(roll)))
        for env, actor in zip(self.envs, self.actor_handles):
            camera = self.gym.create_camera_sensor(env, props)
            base = self.gym.find_actor_rigid_body_handle(env, actor, "base_Link")
            self.gym.attach_camera_to_body(camera, env, base, mount, gymapi.FOLLOW_TRANSFORM)
            image = self.gym.get_camera_image_gpu_tensor(
                self.sim, env, camera, gymapi.IMAGE_DEPTH)
            self.camera_handles.append(camera)
            self.camera_tensors.append(gymtorch.wrap_tensor(image))

    @torch.no_grad()
    def capture_depth(self):
        """Render synchronized raw Isaac depth as (num_envs,H,W) on the sim device."""
        if not self.camera_enabled:
            raise RuntimeError("camera sensors are not enabled")
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        depth = torch.stack([tensor.clone() for tensor in self.camera_tensors])
        self.gym.end_access_image_tensors(self.sim)
        return depth

    # ------------------------------------------------------------------ 建环境

    def _create_envs(self):
        cfg = self.cfg
        asset_path = cfg.asset.file
        options = gymapi.AssetOptions()
        options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode   # 3 = EFFORT
        options.collapse_fixed_joints = True
        options.fix_base_link = cfg.asset.fix_base_link
        options.disable_gravity = cfg.asset.disable_gravity
        options.density = cfg.asset.density
        asset = self.gym.load_asset(self.sim, os.path.dirname(asset_path),
                                    os.path.basename(asset_path), options)
        segment_cfg = getattr(cfg.terrain, "support_segments", None)
        static_boxes = getattr(cfg.terrain, "static_boxes", None)
        obstacles_xy = getattr(cfg.terrain, "obstacles_xy", None)
        segment_asset = reset_asset = None
        obstacle_asset = None
        if segment_cfg is not None:
            terrain_options = gymapi.AssetOptions()
            terrain_options.fix_base_link = True
            terrain_options.disable_gravity = True
            height = float(cfg.terrain.pit_depth_m)
            segment_asset = self.gym.create_box(
                self.sim, float(segment_cfg.segment_length_m),
                float(segment_cfg.segment_width_m), height, terrain_options)
            reset_asset = self.gym.create_box(
                self.sim, float(segment_cfg.reset_size_xy_m[0]),
                float(segment_cfg.reset_size_xy_m[1]), height, terrain_options)
        if obstacles_xy is not None:
            obstacle_options = gymapi.AssetOptions()
            obstacle_options.fix_base_link = True
            obstacle_options.disable_gravity = True
            size = cfg.terrain.obstacle_size_xyz_m
            obstacle_asset = self.gym.create_box(
                self.sim, float(size[0]), float(size[1]), float(size[2]),
                obstacle_options)
        self.num_dof = self.gym.get_asset_dof_count(asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        body_names = self.gym.get_asset_rigid_body_names(asset)

        self.feet_indices = None
        dof_props = self.gym.get_asset_dof_properties(asset)
        self._build_dof_tensors(dof_props)

        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        spacing = getattr(cfg.env, "env_spacing_xy", [3.0, 3.0])
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_origins[:, 0] = ((torch.arange(self.num_envs, device=self.device) % num_per_row).float()
                                  * float(spacing[0]))
        self.env_origins[:, 1] = ((torch.arange(self.num_envs, device=self.device) // num_per_row).float()
                                  * float(spacing[1]))

        base_state = cfg.init.pos + cfg.init.rot + [0, 0, 0, 0, 0, 0]
        spawn_xy = getattr(cfg.init, "spawn_xy", None)
        if spawn_xy is not None:
            base_state[0] += float(spawn_xy[0])
            base_state[1] += float(spawn_xy[1])
        self.base_init_state = to_torch(base_state, device=self.device)

        self.envs = []
        self.actor_handles = []
        self.robot_actor_indices = []
        self.terrain_actor_handles = []
        self.rigid_body_masses = torch.zeros(self.num_envs, self.num_bodies, device=self.device)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), num_per_row)
            start = gymapi.Transform()
            pos = self.env_origins[i].clone()
            spawn_xy = getattr(cfg.init, "spawn_xy", None)
            if spawn_xy is None:
                pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            else:
                pos[:2] += torch.as_tensor(spawn_xy, dtype=pos.dtype, device=self.device)
            pos[2] = cfg.init.pos[2]  # 出生即站姿高度，避免脚深插地面（URDF 默认伸直腿）
            start.p = gymapi.Vec3(*pos.tolist())

            actor = self.gym.create_actor(env, asset, start, cfg.asset.name, i,
                                          int(cfg.asset.self_collisions), 0)
            self.robot_actor_indices.append(
                self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))
            # 出生即用名义关节角（弯腿站姿），让 prepare_sim 的 FK 使脚贴地而非穿透
            init_dof = np.zeros(self.num_dof, dtype=gymapi.DofState)
            init_dof["pos"] = self.default_dof_pos[i].detach().cpu().numpy()
            self.gym.set_actor_dof_states(env, actor, init_dof, gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env, actor, dof_props)

            # 关节内在属性随机化（REPLACE 而非乘）。注意：EFFORT 模式下 dof_props 的
            # stiffness/damping 是 POS/VEL 驱动的增益，对本项目无效（no-op）；
            # armature 是转子惯量(kg·m²)，加入会使有效惯量略增，属合理扰动。
            if (cfg.domain_rand.randomize_joint_damping or cfg.domain_rand.randomize_joint_armature
                    or cfg.domain_rand.randomize_joint_friction):
                actor_dof_props = self.gym.get_actor_dof_properties(env, actor)
                for j in range(self.num_dof):
                    if cfg.domain_rand.randomize_joint_damping:
                        actor_dof_props["damping"][j] = float(np.random.uniform(*cfg.domain_rand.joint_damping_range))
                    if cfg.domain_rand.randomize_joint_armature:
                        actor_dof_props["armature"][j] = float(np.random.uniform(*cfg.domain_rand.joint_armature_range))
                    if cfg.domain_rand.randomize_joint_friction:
                        # friction 是无量纲系数(0=自由, 1=锁死)，非 MuJoCo 的 N·m。
                        # [0.0, 0.01] 对齐真实 ankle 摩擦 0.01(等效 ≈0.5~1.5 N·m)，
                        # 计算依据见 config.py 中 joint_friction_range 的标定注释。
                        actor_dof_props["friction"][j] = float(np.random.uniform(*cfg.domain_rand.joint_friction_range))
                self.gym.set_actor_dof_properties(env, actor, actor_dof_props)

            # 摩擦随机化（每 env 独立）。论文随机化 floor 摩擦；IsaacGym 地面全局共享，
            # 这里以机器人形状摩擦近似，实现每 env 独立摩擦多样性
            if cfg.domain_rand.randomize_friction:
                shape_props = self.gym.get_actor_rigid_shape_properties(env, actor)
                f = float(np.random.uniform(*cfg.domain_rand.friction_range))
                for p in shape_props:
                    p.friction = f
                self.gym.set_actor_rigid_shape_properties(env, actor, shape_props)

            # 质量 / COM 随机化
            body_props = self.gym.get_actor_rigid_body_properties(env, actor)
            for b, p in enumerate(body_props):
                if (b == 0 and cfg.domain_rand.randomize_base_mass) or \
                   (b > 0 and cfg.domain_rand.randomize_link_mass):
                    lim = cfg.domain_rand.base_mass_multiplier if b == 0 else cfg.domain_rand.link_mass_multiplier
                    p.mass *= float(np.random.uniform(*lim))
                self.rigid_body_masses[i, b] = p.mass
            if cfg.domain_rand.randomize_base_com:
                c = cfg.domain_rand.rand_com_vec
                body_props[0].com += gymapi.Vec3(*[float(np.random.uniform(-v, v)) for v in c])
            self.gym.set_actor_rigid_body_properties(env, actor, body_props, recomputeInertia=False)

            self.envs.append(env)
            self.actor_handles.append(actor)
            terrain_handles = []
            if static_boxes is not None:
                box_options = gymapi.AssetOptions()
                box_options.fix_base_link = True
                box_options.disable_gravity = True
                for box_id in range(static_boxes.centers_xyz_m.shape[1]):
                    size = static_boxes.sizes_xyz_m[i, box_id]
                    box_asset = self.gym.create_box(
                        self.sim, float(size[0]), float(size[1]), float(size[2]),
                        box_options)
                    pose = gymapi.Transform()
                    if bool(static_boxes.active[i, box_id]):
                        center = (self.env_origins[i].cpu().numpy()
                                  + np.asarray(static_boxes.centers_xyz_m[i, box_id]))
                        pose.p = gymapi.Vec3(*[float(value) for value in center])
                    else:
                        pose.p = gymapi.Vec3(
                            float(self.env_origins[i, 0]),
                            float(self.env_origins[i, 1]), -10.0)
                    handle = self.gym.create_actor(
                        env, box_asset, pose, "course_{:02d}".format(box_id), i, 0)
                    color = (0.80, 0.18, 0.08) if bool(
                        static_boxes.obstacle[i, box_id]) else (0.18, 0.32, 0.42)
                    self.gym.set_rigid_body_color(
                        env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                        gymapi.Vec3(*color))
                    terrain_handles.append(handle)
            if segment_cfg is not None:
                height = float(cfg.terrain.pit_depth_m)
                reset_pose = gymapi.Transform()
                reset_xy = self.env_origins[i, :2].cpu().numpy() + np.asarray(
                    segment_cfg.reset_center_xy_m)
                reset_pose.p = gymapi.Vec3(float(reset_xy[0]), float(reset_xy[1]), -0.5 * height)
                terrain_handles.append(self.gym.create_actor(
                    env, reset_asset, reset_pose, "reset_support", i, 0))
                for segment_id in range(len(segment_cfg.yaw_rad[i])):
                    pose = gymapi.Transform()
                    if bool(segment_cfg.active[i][segment_id]):
                        center = (self.env_origins[i, :2].cpu().numpy()
                                  + np.asarray(segment_cfg.centers_xy_m[i][segment_id]))
                        pose.p = gymapi.Vec3(float(center[0]), float(center[1]), -0.5 * height)
                    else:
                        pose.p = gymapi.Vec3(float(self.env_origins[i, 0]),
                                             float(self.env_origins[i, 1]), -10.0)
                    pose.r = gymapi.Quat.from_euler_zyx(
                        float(segment_cfg.yaw_rad[i][segment_id]), 0.0, 0.0)
                    terrain_handles.append(self.gym.create_actor(
                        env, segment_asset, pose,
                        "support_{:03d}".format(segment_id), i, 0))
            if obstacles_xy is not None:
                size = cfg.terrain.obstacle_size_xyz_m
                for obstacle_id, local_xy in enumerate(obstacles_xy[i]):
                    pose = gymapi.Transform()
                    center = self.env_origins[i, :2].cpu().numpy() + np.asarray(local_xy)
                    pose.p = gymapi.Vec3(float(center[0]), float(center[1]), 0.5 * float(size[2]))
                    handle = self.gym.create_actor(
                        env, obstacle_asset, pose,
                        "obstacle_{:02d}".format(obstacle_id), i, 0)
                    self.gym.set_rigid_body_color(
                        env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                        gymapi.Vec3(0.75, 0.12, 0.08))
                    terrain_handles.append(handle)
            self.terrain_actor_handles.append(terrain_handles)

        self.robot_actor_indices = torch.tensor(
            self.robot_actor_indices, dtype=torch.int32, device=self.device)

        # 足端 body 索引
        foot_idx = []
        for name in body_names:
            if cfg.asset.foot_name in name:
                foot_idx.append(self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name))
        self.feet_indices = torch.tensor(foot_idx, dtype=torch.long, device=self.device)

        # 触地惩罚 body 索引（膝盖/大腿等：abad/hip/knee 链接）
        penalty_idx = []
        penalty_names = list(getattr(cfg.rewards, "contact_penalty_bodies", []) or [])
        for name in body_names:
            if any(key in name for key in penalty_names):
                penalty_idx.append(self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name))
        self.body_contact_indices = torch.tensor(penalty_idx, dtype=torch.long, device=self.device)
        nonfoot_idx = [index for index in range(len(body_names)) if index not in foot_idx]
        self.nonfoot_indices = torch.tensor(nonfoot_idx, dtype=torch.long, device=self.device)

    def _build_dof_tensors(self, dof_props):
        cfg = self.cfg
        n = self.num_dof
        self.torque_limits = to_torch([float(dof_props["effort"][i]) for i in range(n)], device=self.device)
        self.torque_limits = torch.min(self.torque_limits,
                                       torch.full_like(self.torque_limits, cfg.control.user_torque_limit))
        self.p_gains = torch.zeros(n, device=self.device)
        self.d_gains = torch.zeros(n, device=self.device)
        self.default_dof_pos = torch.zeros(n, device=self.device)
        self.reset_dof_pos = torch.zeros(n, device=self.device)
        self.dof_pos_limits = torch.zeros(n, 2, device=self.device)
        for i, name in enumerate(self.dof_names):
            self.p_gains[i] = cfg.control.stiffness[name]
            self.d_gains[i] = cfg.control.damping[name]
            self.default_dof_pos[i] = cfg.init.default_joint_angles[name]
            self.reset_dof_pos[i] = cfg.init.reset_joint_angles[name]
            self.dof_pos_limits[i, 0] = float(dof_props["lower"][i])
            self.dof_pos_limits[i, 1] = float(dof_props["upper"][i])
        self.p_gains = self.p_gains.unsqueeze(0).repeat(self.num_envs, 1)
        self.d_gains = self.d_gains.unsqueeze(0).repeat(self.num_envs, 1)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.reset_dof_pos = self.reset_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self._base_p_gains = self.p_gains.clone()
        self._base_d_gains = self.d_gains.clone()
        self._randomize_pd_gains(torch.arange(self.num_envs, device=self.device))

    def _randomize_pd_gains(self, ids):
        """论文：PD 增益噪声 p = p*(1+0.15·U(-1,1))、d = d*(1+0.5·U(-1,1))，每 episode 重采样。"""
        if ids.numel() == 0:
            return
        cfg = self.cfg
        self.p_gains[ids] = self._base_p_gains[ids]
        self.d_gains[ids] = self._base_d_gains[ids]
        if cfg.domain_rand.randomize_Kp:
            lo, hi = cfg.domain_rand.Kp_range
            self.p_gains[ids] *= torch_rand_float(lo, hi, (ids.numel(), self.num_dof), device=self.device)
        if cfg.domain_rand.randomize_Kd:
            lo, hi = cfg.domain_rand.Kd_range
            self.d_gains[ids] *= torch_rand_float(lo, hi, (ids.numel(), self.num_dof), device=self.device)

    # ------------------------------------------------------------------ 步进

    def get_observations(self):
        return self.obs_buf, self.goal_buf, self.critic_obs_buf

    @torch.no_grad()
    def reset_task_episodes(self, ids):
        """Reset episodes terminated by the upper task and resynchronize state.

        Lower-policy failures are reset inside :meth:`_post_step`, but reaching
        the navigation goal is only known after the upper macro reward has been
        evaluated.  This explicit path gives goal success the same physical
        episode boundary without changing the frozen lower-controller code or
        fabricating any robot state.
        """
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        self._reset_idx(ids)

        # _reset_idx advances two zero-torque settling substeps.  Refresh every
        # derived tensor before the next policy inference; otherwise base/link
        # observations can still describe the pre-reset terminal pose.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_position = self.root_states[:, 0, :3]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
        self.dof_acc[ids] = 0.0
        self.root_acceleration[ids] = 0.0
        self._compute_foot_state()
        self.done_buf[ids] = 0
        self.time_out_buf[ids] = False
        self._compute_observations()
        self.last_dof_vel[ids] = self.dof_vel[ids]
        self.last_root_velocity[ids] = self.root_states[ids, 0, 7:13]

    def step(self, actions):
        # 动作：先存原始采样值（供观测/action_rate），再裁剪到 [-1,1]
        self.policy_actions.copy_(actions)
        self.actions = torch.clamp(actions, -self.cfg.normalization.clip_actions,
                                   self.cfg.normalization.clip_actions)

        self._render()
        self.sampler.step(self.dt, self.foot_positions,
                          self.rigid_body_states[:, self.feet_indices, 3:7])
        self._kick()

        for _ in range(self.cfg.env.decimation):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.domain_rand.randomize_gravity:
                self._apply_gravity_randomization()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self._post_step()
        return self.obs_buf, self.rew_buf, self.done_buf, self.extras, self.goal_buf, self.critic_obs_buf

    def _compute_torques(self, actions):
        target = torch.clamp(actions * self.cfg.control.action_scale + self.default_dof_pos,
                             self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        tau = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(tau, -self.torque_limits, self.torque_limits)

    def _apply_gravity_randomization(self):
        self.rigid_body_forces[:] = 0.0
        self.rigid_body_forces[:, :, 2] = self.rigid_body_masses * (9.81 - self.gravity_magnitude).unsqueeze(1)
        self.rigid_body_torques[:] = 0.0
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.rigid_body_forces),
            gymtorch.unwrap_tensor(self.rigid_body_torques), gymapi.ENV_SPACE)

    def _kick(self):
        if not self.cfg.domain_rand.kick_robots:
            return
        mask = torch.rand(self.num_envs, device=self.device) < self.cfg.domain_rand.kick_probability
        ids = mask.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        lo, hi = self.cfg.domain_rand.kick_velocity_range
        kick = torch_rand_float(-hi, hi, (ids.numel(), 3), device=self.device)
        kick = torch.sign(kick) * torch.clamp(kick.abs(), min=lo, max=hi)
        self.root_states[ids, 0, 7:10] += kick
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self.robot_actor_indices[ids]), ids.numel())

    # ------------------------------------------------------------------ 后处理

    def _post_step(self):
        self.extras = {}
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1

        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_position = self.root_states[:, 0, :3]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.root_acceleration = (self.root_states[:, 0, 7:13] - self.last_root_velocity) / self.dt

        self._compute_foot_state()
        if self.goal_reset_pending.any():
            ids = self.goal_reset_pending.nonzero(as_tuple=False).flatten()
            self.sampler.reset(ids, self.foot_positions,
                               self.rigid_body_states[:, self.feet_indices, 3:7])
            self.goal_reset_pending[ids] = False

        self._check_termination()
        self.done_buf[:] = self.reset_buf          # 在 reset 清零前保存 done 掩码
        self._compute_reward()
        # absorbing = 本步因"连续失败超过宽限期"而终止（非 timeout）
        self.extras["absorbing"] = (self.fail_buf > self.fail_to_terminal).clone()
        self.extras["time_outs"] = self.time_out_buf.clone()

        # 终止帧：reset 前捕获，用于非 absorbing（timeout）的正确 bootstrap
        terminal_obs, terminal_critic_obs = self._compute_observations_raw()
        terminal_goal = self.sampler.observation(
            self.foot_positions, self.rigid_body_states[:, self.feet_indices, 3:7])

        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if ids.numel():
            # Preserve the actual terminal physics before _reset_idx overwrites
            # tensors. Evaluation must never infer a fall from the last healthy
            # frame followed by a reset jump.
            self.extras["terminal_physics"] = {
                "ids": ids.clone(),
                "root": self.root_states[ids, 0].clone(),
                "rigid_body": self.rigid_body_states[ids].clone(),
                "contact_force": self.contact_forces[ids].clone(),
                "dof_pos": self.dof_pos[ids].clone(),
                "dof_vel": self.dof_vel[ids].clone(),
                "torque": self.torques[ids].clone(),
                "fail_count": self.fail_buf[ids].clone(),
                "timeout": self.time_out_buf[ids].clone(),
                "termination_height": self.extras["termination_reasons"]["height"][ids].clone(),
                "height_above_lower_reference_limit": self.extras[
                    "termination_reasons"]["height_above_lower_reference_limit"][ids].clone(),
                "termination_tilt": self.extras["termination_reasons"]["tilt"][ids].clone(),
                "termination_nonfoot_contact": self.extras[
                    "termination_reasons"]["nonfoot_contact"][ids].clone(),
            }
        self._reset_idx(ids)
        self._compute_observations()
        if ids.numel():
            self.obs_buf[ids] = terminal_obs[ids]
            self.critic_obs_buf[ids] = terminal_critic_obs[ids]
            self.goal_buf[ids] = terminal_goal[ids]

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0].clone()
        self.last_actions[:, :, 0] = self.policy_actions.clone()
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_velocity[:] = self.root_states[:, 0, 7:13]

    def _check_termination(self):
        # 论文 paper_termination_path：只按 base 高度判定。
        # fail_buf 累积计数（对齐 tron1_RL），连续失败超过 fail_to_terminal 步才终止，
        # 避免出生/瞬时扰动导致 1 帧回合；fail_buf 在 _reset_idx 中清零。
        lo, hi = self.cfg.env.healthy_height_range
        height = self.base_position[:, 2] - self.env_origins[:, 2]
        failed_height_low = (height < lo) | ~torch.isfinite(height)
        failed_height_high = height > hi
        upper_mode = getattr(self.cfg.env, "termination_mode", "lower_reference") == "upper_joint"
        if upper_mode:
            # The copied lower-paper upper bound (0.80 m) is not a fall test for
            # this robot: ordinary leg extension reaches 0.806--0.82 m while the
            # base is upright and only the ankles contact the ground. Keep exact
            # lower-reference semantics in the other branch, but do not reset a
            # physically healthy upper-task rollout merely for standing tall.
            failed_height = failed_height_low
            max_tilt = math.radians(float(self.cfg.env.max_tilt_deg))
            # projected_gravity.z=-cos(tilt) for an upright base.
            failed_tilt = self.projected_gravity[:, 2] > -math.cos(max_tilt)
            threshold = float(self.cfg.env.nonfoot_terminal_force_n)
            failed_contact = (torch.norm(
                self.contact_forces[:, self.nonfoot_indices], dim=-1).max(dim=-1).values
                              > threshold)
            failed = failed_height | failed_tilt | failed_contact
            # Upper task termination uses genuinely consecutive failure time.
            self.fail_buf[:] = torch.where(failed, self.fail_buf + 1,
                                           torch.zeros_like(self.fail_buf))
            self.extras["termination_reasons"] = {
                "height": failed_height.clone(),
                "height_above_lower_reference_limit": failed_height_high.clone(),
                "tilt": failed_tilt.clone(),
                "nonfoot_contact": failed_contact.clone(),
            }
        else:
            failed_height = failed_height_low | failed_height_high
            failed = failed_height
            self.fail_buf += failed.long()
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = ((self.fail_buf > self.fail_to_terminal) | self.time_out_buf).long()

    def _compute_reward(self):
        self.rew_buf, raw, weighted = self.rewards.compute()
        self.extras["reward_terms_raw"] = raw
        self.extras["reward_terms_weighted"] = {k: v.float().mean().item() for k, v in weighted.items()}
        for name in self.rewards.names:
            self.episode_sums[name] += weighted[name]

    def _compute_foot_state(self):
        foot = self.rigid_body_states[:, self.feet_indices]
        self.foot_positions, self.foot_velocities = rigid_body_site_state(foot, self.foot_site_offsets)

    # ------------------------------------------------------------------ 观测

    def _compute_observations(self):
        self.obs_buf, self.critic_obs_buf = self._compute_observations_raw()
        self.goal_buf = self.sampler.observation(
            self.foot_positions, self.rigid_body_states[:, self.feet_indices, 3:7])
        if self.goal_reset_pending.any():
            pending = self.goal_reset_pending
            self.goal_buf[pending] = 0.0
            self.goal_buf[pending, 3] = 1.0
            self.goal_buf[pending, 10] = 1.0
        if self.cfg.noise.add_noise:
            self.obs_buf = self.obs_buf + torch.randn_like(self.obs_buf) * self._noise_vec()

    def _compute_observations_raw(self):
        sc = self.cfg.normalization.obs_scales
        # Actor：投影重力(3) + 关节角(8) + 世界系角速度(3) + 关节角速度(8) + 上一步动作(8)
        obs = torch.cat([
            self.projected_gravity,
            self.dof_pos * sc.dof_pos,
            self.root_states[:, 0, 10:13] * sc.ang_vel,
            self.dof_vel * sc.dof_vel,
            self.policy_actions,
        ], dim=-1)
        # Critic：比 Actor 多世界系线速度，且全部用原始单位
        critic_obs = torch.cat([
            self.projected_gravity,
            self.dof_pos,
            self.root_states[:, 0, 7:10],
            self.root_states[:, 0, 10:13],
            self.dof_vel,
            self.policy_actions,
        ], dim=-1)
        return obs, critic_obs

    def _noise_vec(self):
        ns = self.cfg.noise.scales
        sc = self.cfg.normalization.obs_scales
        n = self.num_dof
        v = torch.zeros(self.num_obs, device=self.device)
        v[0:3] = ns.gravity * self.cfg.noise.noise_level
        v[3:3 + n] = ns.dof_pos * self.cfg.noise.noise_level * sc.dof_pos
        v[3 + n:6 + n] = ns.ang_vel * self.cfg.noise.noise_level * sc.ang_vel
        v[6 + n:6 + 2 * n] = ns.dof_vel * self.cfg.noise.noise_level * sc.dof_vel
        return v

    # ------------------------------------------------------------------ 重置

    def configure_reset_curriculum(self, spawns, prob):
        """Reverse-curriculum reset: respawn at a random supported mid-course
        position with probability `prob`.  `spawns` is (num_envs, K, 2) local xy."""
        self.reset_curriculum_spawns = torch.as_tensor(
            spawns, dtype=torch.float32, device=self.device)
        self.reset_curriculum_prob = float(prob)

    def _reset_idx(self, ids):
        if len(ids) == 0:
            return
        self.dof_pos[ids] = self.reset_dof_pos[ids]
        self.dof_vel[ids] = 0.0
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.robot_actor_indices[ids]), ids.numel())

        self.root_states[ids, 0] = self.base_init_state
        self.root_states[ids, 0, :3] += self.env_origins[ids]
        self.root_states[ids, 0, 7:13] = 0.0
        if (self.reset_curriculum_spawns is not None
                and self.reset_curriculum_prob > 0.0):
            use_mid = (torch.rand(ids.numel(), device=self.device)
                       < self.reset_curriculum_prob)
            if use_mid.any():
                mid_ids = ids[use_mid]
                k = torch.randint(
                    0, self.reset_curriculum_spawns.shape[1],
                    (mid_ids.numel(),), device=self.device)
                self.root_states[mid_ids, 0, :2] = (
                    self.env_origins[mid_ids, :2]
                    + self.reset_curriculum_spawns[mid_ids, k])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self.robot_actor_indices[ids]), ids.numel())

        # 落定（关键）：set_root + set_dof 只改了 base 与关节状态，派生的 link
        # 世界坐标/速度仍是摔倒时的旧值，与新的 base/关节状态不一致，下一次
        # simulate 会在同一帧内既重算 FK 又解动力学，旧速度导致数值爆炸、
        # 机器人 1 帧内又趴回地面（高并行度下尤其明显）。补 2 个零力矩子步让
        # FK 重算、link 状态与关节状态恢复一致。见 scripts/diag_reset_order.py。
        zero_tau = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        for _ in range(2):
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(zero_tau))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # 每 episode 重采样域随机化（论文在 reset 时重采样 PD 增益与重力）
        self._randomize_pd_gains(ids)
        if self.cfg.domain_rand.randomize_gravity:
            lo, hi = self.cfg.domain_rand.gravity_magnitude_range
            self.gravity_magnitude[ids] = torch_rand_float(
                lo, hi, (ids.numel(), 1), device=self.device).squeeze(1)

        self.actions[ids] = 0.0
        self.policy_actions[ids] = 0.0
        self.last_actions[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.reset_buf[ids] = 0
        self.fail_buf[ids] = 0
        self.goal_reset_pending[ids] = True

        for name in self.episode_sums:
            mean = self.episode_sums[name][ids].mean().item()
            self.extras.setdefault("episode", {})[f"rew_{name}"] = mean / self.cfg.env.episode_length_s
            self.episode_sums[name][ids] = 0.0

    # ------------------------------------------------------------------ 渲染

    def _render(self):
        if self.viewer is None:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit("viewer window closed")
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit("viewer quit")
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)
        if not self.enable_viewer_sync:
            # 关闭渲染以加速：仅轮询事件队列，保证再次按 V 能重新开启渲染
            self.gym.poll_viewer_events(self.viewer)
            return
        self.gym.clear_lines(self.viewer)
        self._draw_target(0)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def _draw_target(self, i):
        origin = self.env_origins[i].cpu().numpy()
        pts, cols = [], []
        for f in range(2):
            p = self.sampler.target_pos[i, f].cpu().numpy() - origin
            swing = self.sampler.swing_foot[i].item() == f
            color = (0.0, 1.0, 0.0) if swing else (1.0, 0.25, 0.25)
            z = p[2] + 0.03
            N = 32
            for r in (0.060, 0.065):
                for k in range(N):
                    a0 = 2 * math.pi * k / N
                    a1 = 2 * math.pi * (k + 1) / N
                    pts += [[p[0] + r * math.cos(a0), p[1] + r * math.sin(a0), z],
                            [p[0] + r * math.cos(a1), p[1] + r * math.sin(a1), z]]
                    cols += [color] * 2
            L = 0.09
            pts += [[p[0] - L, p[1], z], [p[0] + L, p[1], z],
                    [p[0], p[1] - L, z], [p[0], p[1] + L, z]]
            cols += [color] * 4
        verts = np.asarray(pts, dtype=np.float32).flatten()
        colors = np.asarray(cols, dtype=np.float32).flatten()
        self.gym.add_lines(self.viewer, self.envs[i], len(pts) // 2, verts, colors)
