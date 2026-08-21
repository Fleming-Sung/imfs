"""diag_reset_gpu.py —— 用"污染再刷新"法判定重置状态是否真的写进了 GPU。

关键：env.step 内部 reset 后，Python 张量 root_states 已直接被赋成 0.663。
若直接 refresh+读，读到的 0.663 可能来自 Python 赋值而非 GPU。
所以：reset 后先把 Python 张量污染成 -5，再 refresh(D2H)，再读：
  - 读到 0.663      → GPU 确实有 0.663（H2D 生效）
  - 读到 -5         → refresh 异步未完成/未覆盖
  - 读到其它(如0.14)→ GPU 里仍是旧的趴地值（H2D 丢失/延迟）
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--num_steps", "type": int, "default": 120},
    ]
    args = gymutil.parse_arguments(description="diag reset gpu", headless=True,
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

    n_ok = n_stale = n_other = 0
    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        # 污染 Python 张量
        env.root_states[ids, 0, 2] = -5.0
        # 强制同步后从 GPU 读回
        torch.cuda.synchronize()
        env.gym.refresh_actor_root_state_tensor(env.sim)
        torch.cuda.synchronize()
        z = env.root_states[ids, 0, 2]
        ok = (z.abs() - 0.663).abs() < 0.01
        stale = (z + 5.0).abs() < 0.01
        n_ok += int(ok.sum().item())
        n_stale += int(stale.sum().item())
        n_other += int((~ok & ~stale).sum().item())
        if n_other or (step % 20 == 0):
            print(f"[step {step}] 重置 {ids.numel():5d} → GPU 有0.663: {int(ok.sum()):5d} "
                  f"| 仍-5(异步未回读): {int(stale.sum()):5d} | 其它: {int((~ok & ~stale).sum()):5d}")
        if step >= 60:
            break

    print(f"\n汇总: GPU 确实有0.663 = {n_ok}, 异步未回读 = {n_stale}, 其它(旧值/异常) = {n_other}")


if __name__ == "__main__":
    main()
