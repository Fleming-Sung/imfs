"""diag_reset_knee.py —— 验证"reset 关节角(knee=0.5) ≠ PD 默认目标(knee=0.4)"是否是重置后爆炸元凶。

对比两种 reset 关节角配置下，重置后下一帧"仍趴地"的比例。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
        {"name": "--mode", "type": str, "default": "original",
         "help": "original | reset_eq_default"},
    ]
    args = gymutil.parse_arguments(description="diag reset knee", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def run(mode, args):
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    if mode == "reset_eq_default":
        # 让 reset 关节角 = 默认关节角（消除膝盖 mismatch）
        cfg.init.reset_joint_angles = dict(cfg.init.default_joint_angles)
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)
    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)

    n_restand, n_fallen = 0, 0
    prev_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        if prev_reset.any():
            z = env.base_position[prev_reset, 2] - env.env_origins[prev_reset, 2]
            n_restand += int((z > 0.5).sum().item())
            n_fallen += int((z < 0.3).sum().item())
        prev_reset = done.bool()
    total = n_restand + n_fallen
    rate = n_fallen / total if total else 0.0
    print(f"[{mode:18s}] num_envs={env.num_envs}  重置后下一帧: 站起 {n_restand:6d}  "
          f"仍趴地 {n_fallen:6d}  (趴地率 {rate*100:.1f}%)")
    return rate


def main():
    args = parse_args()
    run(args.mode, args)


if __name__ == "__main__":
    main()
