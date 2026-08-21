"""diag_reset_height.py —— 判定"重置后 1 步坍塌"是接触残留还是瞬移本身。

把重置高度改成很高(1.5m，远高于地面)，看机器人是否仍 1 步坍塌：
  - 若仍坍塌 → 与接触无关，是关节体瞬移本身的状态不一致
  - 若不坍塌(正常下落) → 与地面接触残留有关
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
        {"name": "--z", "type": float, "default": 0.663, "help": "重置高度"},
    ]
    args = gymutil.parse_arguments(description="diag reset height", headless=True,
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

    orig = env._reset_idx

    def reset_h(ids):
        orig(ids)
        if args.z != 0.663:
            # 把 base 抬到指定高度
            env.root_states[ids, 0, 2] = env.env_origins[ids, 2] + args.z
            from isaacgym import gymtorch
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env.root_states),
                gymtorch.unwrap_tensor(ids.to(torch.int32)), ids.numel())

    env._reset_idx = reset_h
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
    print(f"z={args.z}  num_envs={env.num_envs}  重置后下一帧: "
          f"站起 {n_restand:6d}  仍趴地 {n_fallen:6d}  (趴地率 {rate*100:.1f}%)")


if __name__ == "__main__":
    main()
