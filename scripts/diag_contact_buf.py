"""diag_contact_buf.py —— 验证"接触对缓冲区(max_gpu_contact_pairs)默认值在高并行度下不足"假设。

对比：默认参数 vs 加上 legged_gym 的 max_gpu_contact_pairs=2**23 + default_buffer_size_multiplier=5。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 4096},
        {"name": "--num_steps", "type": int, "default": 120},
        {"name": "--bigbuf", "type": int, "default": 1, "help": "1=加大接触缓冲, 0=默认"},
    ]
    args = gymutil.parse_arguments(description="diag contact buf", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    sp = make_sim_params(cfg, args)
    if args.bigbuf:
        sp.physx.max_gpu_contact_pairs = 2 ** 23
        sp.physx.default_buffer_size_multiplier = 5
        print("已设置 max_gpu_contact_pairs=2**23, default_buffer_size_multiplier=5")
    else:
        print(f"默认: max_gpu_contact_pairs={sp.physx.max_gpu_contact_pairs} "
              f"buffer_mult={sp.physx.default_buffer_size_multiplier}")
    env = FootholdEnv(cfg, sp, args.sim_device, True)

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
    print(f"bigbuf={bool(args.bigbuf)}  num_envs={env.num_envs}  重置后下一帧: "
          f"站起 {n_restand:6d}  仍趴地 {n_fallen:6d}  (趴地率 {rate*100:.1f}%)")


if __name__ == "__main__":
    main()
