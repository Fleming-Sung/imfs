"""diag_reset_flat.py —— 验证"3D view 传给 set_*_tensor_indexed 导致 C++ 误读形状"假设。

legged_gym 保持 root_states/dof_state 为 FLAT(2D)，而 mind-steps 把它们 .view() 成 3D 后传给 C++。
本脚本 monkey-patch _reset_idx，改用 FLAT 视图传参，对比趴地率。
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
        {"name": "--flat", "type": int, "default": 1, "help": "1=flat 传参, 0=现状"},
    ]
    args = gymutil.parse_arguments(description="diag reset flat", headless=True,
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

    if args.flat:
        def reset_flat(ids):
            if len(ids) == 0:
                return
            self = env
            self.dof_pos[ids] = self.reset_dof_pos[ids]
            self.dof_vel[ids] = 0.0
            # FLAT: (num_envs*num_dof, 2)
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state.view(-1, 2)),
                gymtorch.unwrap_tensor(ids.to(torch.int32)), ids.numel())
            self.root_states[ids] = self.base_init_state
            self.root_states[ids, 0, :3] += self.env_origins[ids]
            self.root_states[ids, 0, 7:13] = 0.0
            # FLAT: (num_envs, 13)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states.view(-1, 13)),
                gymtorch.unwrap_tensor(ids.to(torch.int32)), ids.numel())
            # 其余与原 _reset_idx 相同
            self._randomize_pd_gains(ids)
            if self.cfg.domain_rand.randomize_gravity:
                lo, hi = self.cfg.domain_rand.gravity_magnitude_range
                self.gravity_magnitude[ids] = torch_rand_float(lo, hi, (ids.numel(), 1), device=self.device).squeeze(1)
            self.actions[ids] = 0.0
            self.policy_actions[ids] = 0.0
            self.last_actions[ids] = 0.0
            self.episode_length_buf[ids] = 0
            self.reset_buf[ids] = 0
            self.fail_buf[ids] = 0
            self.goal_reset_pending[ids] = True
            for name in self.episode_sums:
                mean = self.episode_sums[name][ids].mean().item()
                self.extras.setdefault("episode", {})[f"rew_{name}"] = mean / self.cfg.env.episode_length_s
                self.episode_sums[name][ids] = 0.0

        env._reset_idx = reset_flat

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
    print(f"flat={bool(args.flat)}  num_envs={env.num_envs}  重置后下一帧: "
          f"站起 {n_restand:6d}  仍趴地 {n_fallen:6d}  (趴地率 {rate*100:.1f}%)")


if __name__ == "__main__":
    main()
