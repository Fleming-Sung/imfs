"""config.py —— SF_TRON1A 平地落足跟踪配置。

数值全部来自论文 "Mind Your Steps" 及其发布代码（mind_steps/footstep_configs.py、
configs.py），只有 num_envs 等运行时参数留作命令行覆盖。
"""

import os

ROBOT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")


class AttrDict(dict):
    """可属性访问的 dict：cfg.a.b == cfg['a']['b']。"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def from_nested(cls, d):
        out = cls()
        for k, v in d.items():
            out[k] = cls.from_nested(v) if isinstance(v, dict) else v
        return out


def get_flat_config():
    """SF 平地落足跟踪配置（论文 Appendix F + 发布代码的 SF 映射）。"""
    cfg = {
        "env": {
            "num_envs": 128,
            "episode_length_s": 20.0,
            "fail_to_terminal_time_s": 0.2,  # 摔倒宽限期：连续失败 0.5s 才终止（对齐 tron1_RL）
            "dt": 0.005,          # 论文物理步长 0.005（250 Hz）
            "decimation": 4,       # 论文 n_substeps=4 → 策略 50 Hz
            "up_axis": 1,          # gymapi.UpAxis: 1 = Z 轴向上
            "gravity": [0.0, 0.0, -9.81],
            "healthy_height_range": [0.30, 0.90],  # T1 值；SF 映射待校验
            "render_target_envs": 1,               # 只画第 0 个环境的落足目标
        },
        "terrain": {
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0,
        },
        "asset": {
            "file": os.path.join(ROBOT_ROOT, "SF_TRON1A", "urdf", "robot.urdf"),
            "name": "SF_TRON1A",
            "foot_name": "ankle",
            "foot_site_offsets": [[0.0, 0.0, -0.05990000], [0.0, 0.0, -0.05990148]],
            "default_dof_drive_mode": 3,   # 3 = DOF_MODE_EFFORT（论文用 PD 手算力矩）
            "self_collisions": 0,
            "density": 0.001,
            "fix_base_link": False,
            "disable_gravity": False,
        },
        "init": {
            "pos": [0.0, 0.0, 0.663],
            "rot": [0.0, 0.0, 0.0, 1.0],
            "default_joint_angles": {
                "abad_L_Joint": 0.0, "hip_L_Joint": 0.20, "knee_L_Joint": 0.40, "ankle_L_Joint": -0.25,
                "abad_R_Joint": 0.0, "hip_R_Joint": -0.20, "knee_R_Joint": -0.40, "ankle_R_Joint": -0.25,
            },
            "reset_joint_angles": {
                "abad_L_Joint": 0.0, "hip_L_Joint": 0.20, "knee_L_Joint": 0.50, "ankle_L_Joint": -0.25,
                "abad_R_Joint": 0.0, "hip_R_Joint": -0.20, "knee_R_Joint": -0.50, "ankle_R_Joint": -0.25,
            },
        },
        "control": {
            "stiffness": {
                "abad_L_Joint": 45, "hip_L_Joint": 45, "knee_L_Joint": 45, "ankle_L_Joint": 45,
                "abad_R_Joint": 45, "hip_R_Joint": 45, "knee_R_Joint": 45, "ankle_R_Joint": 45,
            },
            "damping": {
                "abad_L_Joint": 1.5, "hip_L_Joint": 1.5, "knee_L_Joint": 1.5, "ankle_L_Joint": 0.8,
                "abad_R_Joint": 1.5, "hip_R_Joint": 1.5, "knee_R_Joint": 1.5, "ankle_R_Joint": 0.8,
            },
            "action_scale": 1.0,
            "clip_actions": 1.0,
            "user_torque_limit": 80.0,
        },
        "foothold": {
            "goal_dim": 16,
            "step_distance": [0.05, 0.20],
            "step_angle_deg": [-30.0, 30.0],
            "target_yaw_deg": [-30.0, 30.0],
            "movement_direction_deg": [-180.0, 180.0],
            "feet_direction_deg": [0.0, 0.0],
            "z_distance": [0.0, 0.0],
            "minimum_lateral_separation": 0.10,
            "min_swing_distance": 0.05,
            "max_rejection_attempts": 10,
            "gait_frequency": [1.0, 1.0],
            "hold_probability": 0.10,
            "hold_feet_distance": 0.20,
            "swing_height": 0.05,
            "swing_window_half_width": 0.10,
            "goal_height": 0.65,
            "max_num_gaits": 10,
            "step_distance_curriculum": {"start": [0.05, 0.10], "end": [0.05, 0.20], "ramp_iterations": 1000},
        },
        "rewards": {
            "scales": {
                "survival": 0.25,
                "swing_xy": 5.0, "swing_z": 4.0, "swing_yaw": 0.4,
                "stance_xy": 5.0, "stance_z": 0.0, "stance_yaw": 0.4,
                "feet_swing": 6.0, "gait_height": 4.0, "knee_height": 0.0,
                "nominal_joint_pos": 4.0,
                "base_height": -2.0, "action_rate": -3.0, "feet_slip": -3.0,
                "feet_roll": -0.4, "orientation": -5.0, "torques": -2e-4,
                "energy": -2e-3, "ang_vel_xy": -0.2, "dof_vel": -9e-4,
                "dof_acc": -1e-7, "root_acc": -1e-4, "dof_pos_limits": -1.0,
            },
            "sharpness": {"xy": 100.0, "z": 100.0, "yaw": 100.0, "gait_height": 100.0},
            "base_height_target": 0.65,
            "joint_position_limit_scale": 0.98,
            "clip_single_reward": None,
            "clip_reward": None,
        },
        "normalization": {
            "obs_scales": {"ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 0.1},
            "clip_observations": None,
            "clip_actions": 1.0,
            "running_obs_clip": 10.0,
        },
        "noise": {
            "add_noise": True,
            "noise_level": 1.0,
            "scales": {"dof_pos": 0.03, "dof_vel": 0.30, "ang_vel": 0.20, "gravity": 0.015},
        },
        "domain_rand": {
            "randomize_friction": True, "friction_range": [0.5, 1.5],
            "randomize_base_mass": True, "base_mass_multiplier": [0.8, 1.2],
            "randomize_link_mass": True, "link_mass_multiplier": [0.9, 1.1],
            "randomize_base_com": True, "rand_com_vec": [0.05, 0.05, 0.05],
            "randomize_Kp": True, "Kp_range": [0.85, 1.15],
            "randomize_Kd": True, "Kd_range": [0.5, 1.5],
            "randomize_gravity": True, "gravity_magnitude_range": [9.51, 10.11],
            "randomize_joint_damping": True, "joint_damping_range": [0.005, 0.03],
            # ---- 关节摩擦域随机化（真实量级）----
            # IsaacGym 的 dof_props["friction"] 是【无量纲系数】(0=无摩擦, 1=锁死)，
            # 其摩擦扭矩 ∝ 关节当前承载的约束力（负载越大摩擦越大），与 MuJoCo 的
            # dof_frictionloss【恒定库仑干摩擦扭矩, 单位 N·m】是两种不同的物理量。
            # 论文的 joint_friction_loss_range=[0,1] 是 N·m（最大只加 1 N·m 干摩擦，
            # 仅为 80 N·m 扭矩上限的 1.25%），绝不能直接照搬到 IsaacGym 的系数上。
            #
            # 标定依据（scripts/calibrate_friction.py，knee_L，重力下 0.1s 短时窗，
            # 恒扭矩驱动，看 Δq 相对无摩擦时的衰减）：
            #   friction=0.01 → Δq 几乎不变(20 N·m 下 0.933 vs 0.961)，等效摩擦扭矩
            #                   ≈0.5~1.5 N·m，正好落在论文 [0,1] N·m 的真实量级内
            #   friction=0.10 → 等效 ≈10~15 N·m（已是论文上界的 ~10 倍）
            #   friction=0.30 → 50 N·m 下也近乎锁死
            #   friction≥0.50 → 完全锁死
            # 真实 URDF：ankle <dynamics friction="0.01"/>，其余关节=0（IsaacGym 已把
            # 该值读入 asset 默认属性）。故取 [0.0, 0.01]：上界对齐真实 ankle 摩擦 0.01，
            # 物理效果对齐论文 [0,1] N·m。
            "randomize_joint_friction": True, "joint_friction_range": [0.0, 0.01],
            "randomize_joint_armature": True, "joint_armature_range": [0.007, 0.03],
            "kick_robots": True, "kick_probability": 0.004, "kick_velocity_range": [0.1, 0.4],
        },
        "policy": {
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "orthogonal_init": True,
            "init_noise_std": 0.135,
            "min_std": 0.05,
            "max_std": 0.5,
        },
        "ppo": {
            "value_loss_coef": 0.5,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 10,
            "num_mini_batches": 32,
            "learning_rate": 3e-4,
            "min_learning_rate": 1e-6,
            "max_learning_rate": 1e-2,
            "schedule": "adaptive",
            "gamma": 0.995,
            "lam": 0.95,
            "desired_kl": 0.02,
            "kl_margin": 1.5,
            "kl_scale": 1.5,
            "max_grad_norm": 1.0,
            "optimizer": "adamw",
            "adam_epsilon": 1e-5,
            "weight_decay": 0.0,
        },
        "runner": {
            "num_steps_per_env": 50,
            "max_iterations": 2442,
            "save_interval": 100,
            "log_interval": 1,
        },
    }
    return AttrDict.from_nested(cfg)
