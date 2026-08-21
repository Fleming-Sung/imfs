"""diag_reset_gpu2.py —— 同时用污染法验证 GPU 中 base 与 dof(关节角) 是否都正确。"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
    ]
    args = gymutil.parse_arguments(description="diag reset gpu2", headless=True,
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
    knee = env.dof_names.index("knee_L_Joint")

    n_root_ok = n_root_bad = 0
    n_dof_ok = n_dof_bad = 0
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        # 污染两个 Python 张量
        env.root_states[ids, 0, 2] = -5.0
        env.dof_state[ids, knee, 0] = -7.0
        torch.cuda.synchronize()
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        torch.cuda.synchronize()
        z = env.root_states[ids, 0, 2]
        k = env.dof_state[ids, knee, 0]
        root_ok = (z.abs() - 0.663).abs() < 0.01
        dof_ok = (k - 0.5).abs() < 0.01
        n_root_ok += int(root_ok.sum().item())
        n_root_bad += int((~root_ok).sum().item())
        n_dof_ok += int(dof_ok.sum().item())
        n_dof_bad += int((~dof_ok).sum().item())
        if int((~root_ok | ~dof_ok).sum().item()):
            bad = ids[~root_ok | ~dof_ok][:3]
            print(f"[step {step}] base错 {int((~root_ok).sum()):4d} 个 | dof错 {int((~dof_ok).sum()):4d} 个 "
                  f"| 样本 {bad.tolist()} base_z={z[~root_ok][:3].tolist()} knee={k[~dof_ok][:3].tolist()}")
        if step >= 60:
            break

    print(f"\n汇总: base 正确 {n_root_ok} / 错 {n_root_bad} | dof 正确 {n_dof_ok} / 错 {n_dof_bad}")


if __name__ == "__main__":
    main()
