"""diag_reset_links2.py —— 污染法验证：reset 后 GPU 中 link(脚) 的世界坐标与速度是否正确。

若脚的位置/速度仍停在摔倒时的旧值（没被 FK 重算），则关节体处于"拉伸/不一致"状态，
下一步 simulate 就会把它拉垮——这才是代码层的真正 bug。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
    ]
    args = gymutil.parse_arguments(description="diag reset links2", headless=True,
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
    fi = env.feet_indices

    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        # 污染 Python 的 rigid_body_states
        env.rigid_body_states[ids] = 12345.0
        torch.cuda.synchronize()
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        torch.cuda.synchronize()
        foot = env.rigid_body_states[ids][:, fi, :]  # (n, 2, 13)
        fz = foot[:, :, 2]            # 脚 z
        fvz = foot[:, :, 9]           # 脚 z 向速度
        print(f"[step {step}] 重置 {ids.numel():4d} 个 → 脚 z min={fz.min().item():.3f} "
              f"max={fz.max().item():.3f} | 脚 vz min={fvz.min().item():.3f} max={fvz.max().item():.3f}")
        if step >= 60:
            break


if __name__ == "__main__":
    main()
