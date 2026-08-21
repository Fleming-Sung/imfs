"""diag_reset_sync.py —— 验证"set_*_tensor_indexed 是异步 H2D，下一步 simulate 前未完成"的竞态假设。

做法：monkey-patch env._reset_idx，在 H2D 写之后强制 torch.cuda.synchronize()，
再看"重置后下一帧仍趴地"的比例是否大幅下降。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
        {"name": "--sync", "type": int, "default": 1, "help": "1=重置后强制同步, 0=不"},
    ]
    args = gymutil.parse_arguments(description="diag reset sync", headless=True,
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

    if args.sync:
        orig = env._reset_idx

        def reset_sync(ids):
            orig(ids)
            torch.cuda.synchronize()   # 强制 H2D 完成

        env._reset_idx = reset_sync

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
    print(f"sync={bool(args.sync)}  num_envs={env.num_envs}  重置后下一帧: "
          f"站起 {n_restand:6d}  仍趴地 {n_fallen:6d}  (趴地率 {rate*100:.1f}%)")


if __name__ == "__main__":
    main()
