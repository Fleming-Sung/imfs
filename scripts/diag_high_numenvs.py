"""diag_high_numenvs.py —— 诊断高并行度(如 2048)下机器人"起始已摔倒/重置不生效"问题。

零动作驱动（PD 目标=名义姿态），跟踪：
  1. 初始(step 0 之前)各 env 的 base 高度分布、被判"摔倒"的数量；
  2. 每个 step 的 done/fail/timeout 数量、base 高度统计；
  3. 对若干 env：跨 step 打印 base 高度、dof_pos、episode_length，判断重置是否真正让机器人站起。
"""
from isaacgym import gymapi, gymtorch, gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 40},
        {"name": "--action", "type": float, "default": 0.0},
    ]
    args = gymutil.parse_arguments(description="diag high numenvs", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def height_stats(env, label):
    h = (env.base_position[:, 2] - env.env_origins[:, 2])
    lo, hi = env.cfg.env.healthy_height_range
    n_fail = int(((h < lo) | (h > hi) | ~torch.isfinite(h)).sum().item())
    print(f"{label:28s} base_z min={h.min().item():.4f} max={h.max().item():.4f} "
          f"mean={h.mean().item():.4f}  越界(摔倒)={n_fail}/{env.num_envs}")


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)

    lo, hi = cfg.env.healthy_height_range
    print(f"num_envs={env.num_envs}  num_bodies={env.num_bodies}  num_dof={env.num_dof}")
    print(f"dt={env.dt:.4f}s  decimation={cfg.env.decimation}  max_episode_length={env.max_episode_length}")
    print(f"healthy_height_range=[{lo}, {hi}]  init_pos_z={cfg.init.pos[2]}")
    print(f"dof_names={env.dof_names}\n")

    # ---- 1) 初始状态（step 0 之前）----
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    height_stats(env, "初始(未步进)")

    # ---- 2) 连续零动作步进，统计每步 done ----
    action = torch.full((env.num_envs, env.num_dof), args.action, device=env.device)
    print(f"\n{'step':>5s} {'done':>6s} {'fail':>6s} {'timeout':>8s} "
          f"{'base_z_min':>11s} {'ep_len_max':>10s} {'ep_len_mean':>11s}")
    # 记录几个"始终 done"的 env 样本
    watch = []
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        h = env.base_position[:, 2] - env.env_origins[:, 2]
        n_done = int(done.sum().item())
        n_fail = int(env.fail_buf.sum().item())
        n_to = int(env.time_out_buf.sum().item())
        print(f"{step:5d} {n_done:6d} {n_fail:6d} {n_to:8d} "
              f"{h.min().item():11.4f} {env.episode_length_buf.max().item():10d} "
              f"{env.episode_length_buf.float().mean().item():11.3f}")
        if step == 0 and n_done > 0:
            watch = done.nonzero(as_tuple=False).flatten()[:3].tolist()
            print(f"  → step0 就 done 的 env 样本: {watch}")

    # ---- 3) 抽查 watch env 的跨步轨迹 ----
    if watch:
        print(f"\n=== 抽查 env {watch} 的跨步状态（步进后） ===")
        print(f"{'step':>5s} {'base_z':>9s} {'ep_len':>7s} {'knee_L':>9s} {'knee_R':>9s}")
        for step in range(args.num_steps):
            obs, reward, done, extras, goal, critic_obs = env.step(action)
            z0 = env.base_position[watch[0], 2].item()
            kl = env.dof_pos[watch[0], env.dof_names.index("knee_L_Joint")].item()
            kr = env.dof_pos[watch[0], env.dof_names.index("knee_R_Joint")].item()
            el = env.episode_length_buf[watch[0]].item()
            if step % 5 == 0 or el <= 1:
                print(f"{step:5d} {z0:9.4f} {el:7d} {kl:9.4f} {kr:9.4f}")


if __name__ == "__main__":
    main()
