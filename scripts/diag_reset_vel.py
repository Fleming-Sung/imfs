"""diag_reset_vel.py —— 污染法验证 reset 后 GPU 中 base 速度、关节速度是否真的为 0。"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
    ]
    args = gymutil.parse_arguments(description="diag reset vel", headless=True,
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
        # 污染
        env.root_states[ids, 0, 7:13] = 99.0
        env.dof_state[ids, :, 1] = 99.0
        torch.cuda.synchronize()
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        torch.cuda.synchronize()
        bv = env.root_states[ids, 0, 7:13].abs().max().item()   # base 线/角速度
        dv = env.dof_state[ids, :, 1].abs().max().item()        # 关节速度
        n_bv = int((env.root_states[ids, 0, 7:13].abs().max(dim=1).values > 0.01).sum())
        n_dv = int((env.dof_state[ids, :, 1].abs().max(dim=1).values > 0.01).sum())
        print(f"[step {step}] 重置 {ids.numel():4d} → base速度 max={bv:.4f} (非零 {n_bv} 个) | "
              f"关节速度 max={dv:.4f} (非零 {n_dv} 个)")
        if step >= 60:
            break


if __name__ == "__main__":
    main()
