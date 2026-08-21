"""diag_step_internals.py —— 在 env.step 内部观测终止判定读到的真实高度/NaN。

monkey-patch env._check_termination：在原始判定之前，打印它即将看到的
base 高度 NaN 数、越界数、min/max。对比 step 返回的 done 数，定位误判来源。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 10},
    ]
    args = gymutil.parse_arguments(description="diag step internals", headless=True,
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

    lo, hi = cfg.env.healthy_height_range
    orig_check = env._check_termination

    def patched_check():
        # 在原始判定前，读取它即将看到的高度
        h = env.base_position[:, 2] - env.env_origins[:, 2]
        n_nan = int(torch.isnan(h).sum().item())
        n_inf = int(torch.isinf(h).sum().item())
        n_fail = int(((h < lo) | (h > hi) | ~torch.isfinite(h)).sum().item())
        hf = h[torch.isfinite(h)]
        hmin = hf.min().item() if hf.numel() else float("nan")
        hmax = hf.max().item() if hf.numel() else float("nan")
        print(f"    [check] NaN={n_nan} inf={n_inf} fail={n_fail} "
              f"hmin={hmin:.4f} hmax={hmax:.4f} ep_len_max={env.episode_length_buf.max().item()}")
        return orig_check()

    env._check_termination = patched_check

    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)
    print(f"num_envs={env.num_envs}  healthy=[{lo},{hi}]  init_z={cfg.init.pos[2]}\n")
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        n_done = int(done.sum().item())
        # step 后（已重置）的高度
        h2 = env.base_position[:, 2] - env.env_origins[:, 2]
        print(f"[step {step}] done={n_done}  post_z_min={h2.min().item():.4f} "
              f"post_z_max={h2.max().item():.4f}  ep_len_mean={env.episode_length_buf.float().mean().item():.3f}")


if __name__ == "__main__":
    main()
