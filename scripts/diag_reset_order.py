"""diag_reset_order.py —— 验证 set_root 与 set_dof 的调用顺序是否影响重置后爆炸。

候选顺序：
  1) dof 先、root 后（现状）
  2) root 先、dof 后
  3) 现状 + 重置后再多跑 2 个"落定"simulate
"""
from isaacgym import gymutil, gymtorch  # isaacgym 必须先于 torch 导入
from isaacgym.torch_utils import torch_rand_float
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
        {"name": "--mode", "type": str, "default": "orig",
         "help": "orig|root_first|settle"},
    ]
    args = gymutil.parse_arguments(description="diag reset order", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def make_reset(env, mode):
    def base(ids):
        env.dof_pos[ids] = env.reset_dof_pos[ids]
        env.dof_vel[ids] = 0.0
        env.root_states[ids] = env.base_init_state
        env.root_states[ids, 0, :3] += env.env_origins[ids]
        env.root_states[ids, 0, 7:13] = 0.0
        env._randomize_pd_gains(ids)
        if env.cfg.domain_rand.randomize_gravity:
            lo, hi = env.cfg.domain_rand.gravity_magnitude_range
            env.gravity_magnitude[ids] = torch_rand_float(lo, hi, (ids.numel(), 1), device=env.device).squeeze(1)
        env.actions[ids] = 0.0
        env.policy_actions[ids] = 0.0
        env.last_actions[ids] = 0.0
        env.episode_length_buf[ids] = 0
        env.reset_buf[ids] = 0
        env.fail_buf[ids] = 0
        env.goal_reset_pending[ids] = True
        for name in env.episode_sums:
            mean = env.episode_sums[name][ids].mean().item()
            env.extras.setdefault("episode", {})[f"rew_{name}"] = mean / env.cfg.env.episode_length_s
            env.episode_sums[name][ids] = 0.0

    def dof_then_root(ids):
        base(ids)
        env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state),
                                             gymtorch.unwrap_tensor(ids.to(torch.int32)), ids.numel())
        env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_states),
                                                    gymtorch.unwrap_tensor(ids.to(torch.int32)), ids.numel())

    def root_then_dof(ids):
        base(ids)
        env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_states),
                                                    gymtorch.unwrap_tensor(ids.to(torch.int32)), ids.numel())
        env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state),
                                             gymtorch.unwrap_tensor(ids.to(torch.int32)), ids.numel())

    def settle(ids):
        dof_then_root(ids)
        # 重置后额外落定 2 个 substep（零力矩），让 link 状态与 base/关节状态一致
        tau = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        for _ in range(2):
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
            env.gym.simulate(env.sim)

    return {"orig": dof_then_root, "root_first": root_then_dof, "settle": settle}[mode]


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)
    env._reset_idx = make_reset(env, args.mode)

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
    print(f"mode={args.mode}  num_envs={env.num_envs}  重置后下一帧: "
          f"站起 {n_restand:6d}  仍趴地 {n_fallen:6d}  (趴地率 {rate*100:.1f}%)")


if __name__ == "__main__":
    main()
