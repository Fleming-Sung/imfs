"""diag_reset_nan.py —— 定位高并行度下"重置不生效/1帧回合"的根因。

手动复刻 env.step 里的 decimation 仿真循环（不含重置），在每次 refresh 后、
进入终止判定之前，直接打印 base 高度的 NaN 数量与越界数量。
从而区分：(a) base 高度出现 NaN → 终止判定误判；(b) 重置未真正把机器人站起。
"""
from isaacgym import gymapi, gymtorch, gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 20},
    ]
    args = gymutil.parse_arguments(description="diag reset nan", headless=True,
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
    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)

    def step_physics_only():
        """复刻 env.step 的 decimation 仿真循环（不含 _post_step/重置）。"""
        for _ in range(cfg.env.decimation):
            tau = env._compute_torques(action)
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
            if cfg.domain_rand.randomize_gravity:
                env._apply_gravity_randomization()
            env.gym.simulate(env.sim)
            if env.device == "cpu":
                env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)

    print(f"num_envs={env.num_envs}  decimation={cfg.env.decimation}  "
          f"healthy=[{lo},{hi}]  init_z={cfg.init.pos[2]}\n")

    print(f"{'step':>5s} {'NaN':>6s} {'越界':>6s} {'h_min':>9s} {'h_max':>9s} "
          f"{'dof_vel_max':>11s} {'contact_max':>11s}")
    for step in range(args.num_steps):
        step_physics_only()
        h = env.root_states[:, 0, 2] - env.env_origins[:, 2]
        n_nan = int(torch.isnan(h).sum().item())
        n_fail = int(((h < lo) | (h > hi) | ~torch.isfinite(h)).sum().item())
        # 只看有限值
        hf = h[torch.isfinite(h)]
        hmin = hf.min().item() if hf.numel() else float("nan")
        hmax = hf.max().item() if hf.numel() else float("nan")
        dv = env.dof_vel.abs().max().item()
        env.gym.refresh_net_contact_force_tensor(env.sim)
        cf = env.contact_forces.abs().max().item()
        print(f"{step:5d} {n_nan:6d} {n_fail:6d} {hmin:9.4f} {hmax:9.4f} "
              f"{dv:11.3f} {cf:11.2f}")

        # 手动重置越界的 env（复刻 _reset_idx），继续观察下一轮
        failed = ((h < lo) | (h > hi) | ~torch.isfinite(h))
        ids = failed.nonzero(as_tuple=False).flatten()
        if ids.numel():
            env._reset_idx(ids)


if __name__ == "__main__":
    main()
