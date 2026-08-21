"""diag_reset_orient.py —— 检查重置后 base 朝向是否真的回到竖直(单位四元数)。"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 512},
        {"name": "--num_steps", "type": int, "default": 150},
    ]
    args = gymutil.parse_arguments(description="diag reset orient", headless=True,
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

    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        env.gym.refresh_actor_root_state_tensor(env.sim)
        q = env.root_states[ids, 0, 3:7]  # 四元数 [x,y,z,w]
        # w 应≈1（竖直），x/y/z≈0
        w = q[:, 3]
        tilt = (w.abs() < 0.9).sum().item()  # 明显倾斜的 env 数
        print(f"[step {step}] 重置 {ids.numel():5d} 个 → 四元数 w: min={w.min().item():.3f} "
              f"max={w.max().item():.3f} | 明显倾斜(w<0.9) {tilt} 个")
        if step >= 60:
            break


if __name__ == "__main__":
    main()
