"""diag_reset_links.py —— 重置后立即检查各 link(脚)的世界坐标是否也回到站姿。

若 base 回到 0.663 但脚仍趴在地面(z≈0)，说明 set_*_tensor_indexed 只移动了 base，
link 世界坐标要等下一次 simulate 的 FK 才重算，导致"拉伸"→ 下一步爆炸回地面。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 4096},
        {"name": "--num_steps", "type": int, "default": 80},
    ]
    args = gymutil.parse_arguments(description="diag reset links", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)
    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)

    print(f"num_envs={env.num_envs}\n")
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        # 立即 refresh 三种状态：base、dof、link(刚体)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        base_z = env.root_states[ids, 0, 2] - env.env_origins[ids, 2]
        foot = env.rigid_body_states[ids][:, env.feet_indices, 2]  # (n, 2)
        foot_z = foot.min(dim=1).values
        print(f"[step {step}] 重置 {ids.numel():5d} 个 → base_z min={base_z.min().item():.3f} "
              f"max={base_z.max().item():.3f} | 脚世界 z min={foot_z.min().item():.3f} "
              f"max={foot_z.max().item():.3f}")
        if step >= 3:
            break


if __name__ == "__main__":
    main()
