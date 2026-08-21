"""diag_reset_state.py —— 重置后逐步观察：接触力、速度、base_z 在接下来 3 步的演化。"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 512},
        {"name": "--num_steps", "type": int, "default": 120},
    ]
    args = gymutil.parse_arguments(description="diag reset state", headless=True,
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

    # 跑到出现重置
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue

        # 重置刚发生，观察接下来 3 步的状态演化
        wid = ids[0]
        print(f"[step {step}] env {int(wid)} 被重置。接下来 3 步观察：")
        print(f"  {'帧':>3s} {'base_z':>9s} {'base_vel_z':>11s} {'dof_vel_max':>12s} "
              f"{'contact_max':>12s}")
        for k in range(3):
            # 读当前帧（重置后/步进后）
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_net_contact_force_tensor(env.sim)
            bz = env.root_states[wid, 0, 2].item() - env.env_origins[wid, 2].item()
            bv = env.root_states[wid, 0, 9].item()
            dv = env.dof_vel[wid].abs().max().item()
            cf = env.contact_forces[wid].abs().max().item()
            print(f"  {k:3d} {bz:9.4f} {bv:11.4f} {dv:12.4f} {cf:12.2f}")
            # 再走一步
            obs, reward, done, extras, goal, critic_obs = env.step(action)
        print()
        break


if __name__ == "__main__":
    main()
