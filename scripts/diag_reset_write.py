"""diag_reset_write.py —— 区分"重置没写进 GPU" 还是"重置后 1 步内又炸回地面"。

在 env.step 返回后(内部已完成 reset)，立即从 GPU refresh 一次 root/dof 状态，
检查被重置 env 的 base_z 与 dof_pos：
  - base_z≈0.663 且 dof=reset 角 → GPU 写入成功（则下一步趴地是"1 步内又炸"）
  - base_z≈0.14  → GPU 写入失败（set_*_tensor_indexed 没生效）
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 4096},
        {"name": "--num_steps", "type": int, "default": 80},
    ]
    args = gymutil.parse_arguments(description="diag reset write", headless=True,
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

    print(f"num_envs={env.num_envs}  reset 后立即 refresh 检查 GPU 是否真的被写入\n")
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        # 立即从 GPU refresh，覆盖 Python 张量，看 GPU 里被重置 env 的真实状态
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        z = env.root_states[ids, 0, 2] - env.env_origins[ids, 2]
        k = env.dof_pos[ids, knee]
        n_ok = int((z > 0.5).sum().item())
        n_bad = int((z < 0.3).sum().item())
        print(f"[step {step}] 本步重置 {ids.numel():5d} 个 → GPU base_z: "
              f"站起({'>0.5'}) {n_ok:5d}  趴地({'<0.3'}) {n_bad:5d}  "
              f"z_min={z.min().item():.3f} z_max={z.max().item():.3f}  "
              f"knee_L min={k.min().item():.3f} max={k.max().item():.3f}")
        if step >= 3 and n_bad > 0:
            # 打印几个"写入失败"样本的原始值
            bad = ids[z < 0.3][:3]
            print(f"    样本 {bad.tolist()}: base_z={z[z < 0.3][:3].tolist()}")
            break


if __name__ == "__main__":
    main()
