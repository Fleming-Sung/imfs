"""random_action_test.py —— 用随机/手动动作驱动环境，验证环境正确性（不训练）。

完全复用训练环境 FootholdEnv，仅把并行数量设为 16，不改任何其它配置。
动作值域 [-1, 1]（与训练时 normalization.clip_actions=1.0 一致）。

运行（在 workspace 根目录）：
    python -m scripts.random_action_test                     # 默认 16 环境、随机动作，开启仿真窗口
    python -m scripts.random_action_test --mode manual       # 手动设计动作
    python -m scripts.random_action_test --num_steps 3000
    python -m scripts.random_action_test --env_id 3 --plot_path curves.png  # 记录并绘制指定 env 关节曲线
    python -m scripts.random_action_test --hold_duration 0.3 --data_path data.npz  # 随机动作每 0.3s 重采样并保存原始数据
    python -m scripts.random_action_test --headless          # 无窗口（headless）

动作接口（后续手动设计人工动作只需改 manual_action()）：
    get_action(env, step, mode)  —— 统一入口，按 mode 分发
      ├─ random_action(env)      —— 每个关节在 [-1,1] 均匀随机
      └─ manual_action(env, step)—— 按 step 返回固定/周期动作（示例已给）
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境也能保存图片
import matplotlib.pyplot as plt

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


# --------------------------------------------------------------------------
# 动作接口：后续手动设计动作时，只需修改下面三个函数
# --------------------------------------------------------------------------

def random_action(env):
    """随机动作：每个关节在 [-1, 1] 均匀采样。

    返回 shape (num_envs, num_dof) 的张量，device 与 env 一致。
    """
    return torch.rand(env.num_envs, env.num_dof, device=env.device) * 2.0 - 1.0


def manual_action(env, step):
    """手动动作示例：在这里写你想要的关节目标动作。

    - 返回 shape (num_envs, num_dof) 的张量，值域建议 [-1, 1]。
    - 关节顺序见 env.dof_names（脚本启动时会打印）。
    - 当前示例：恒定输出「左髋 +0.2 / 右髋 -0.2 / 双膝 ±0.4 / 双踝 -0.25」，
      即把 PD 目标固定在名义姿态附近（动作 0 即名义姿态）。
    """
    a = torch.zeros(env.num_envs, env.num_dof, device=env.device)
    names = env.dof_names
    a[:, names.index("hip_L_Joint")] = 0.2
    a[:, names.index("hip_R_Joint")] = -0.2
    a[:, names.index("knee_L_Joint")] = 0.4
    a[:, names.index("knee_R_Joint")] = -0.4
    a[:, names.index("ankle_L_Joint")] = -0.25
    a[:, names.index("ankle_R_Joint")] = -0.25
    return a


def get_action(env, step, mode):
    """统一动作入口：根据 mode 选择动作来源。

    mode:
      "random"  → random_action(env)
      "manual"  → manual_action(env, step)
    """
    if mode == "random":
        return random_action(env)
    if mode == "manual":
        return manual_action(env, step)
    raise ValueError(f"未知 mode: {mode}")


def action_to_target(env, action):
    """动作 → PD 目标转角（与 env._compute_torques 里的换算一致）。"""
    return torch.clamp(action * env.cfg.control.action_scale + env.default_dof_pos,
                       env.dof_pos_limits[:, 0], env.dof_pos_limits[:, 1])


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def parse_args():
    custom = [
        {"name": "--mode", "type": str, "default": "random"},
        {"name": "--num_envs", "type": int, "default": 16},
        {"name": "--num_steps", "type": int, "default": 1000},
        {"name": "--env_id", "type": int, "default": 0},
        {"name": "--plot_path", "type": str, "default": "joint_curves.png"},
        {"name": "--torque_plot_path", "type": str, "default": "joint_torques.png"},
        {"name": "--hold_duration", "type": float, "default": 0.3},
        {"name": "--data_path", "type": str, "default": "joint_data.npz"},
    ]
    args = gymutil.parse_arguments(description="random action env check", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def _plot_joint_curves(env, q, target, env_id, path):
    """绘制指定 env 每个关节的实际转角与目标转角曲线，纵轴固定为各关节物理限位。"""
    n = env.num_dof
    steps = np.arange(q.shape[0]) * env.dt  # 横轴时间（秒）
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows), squeeze=False)
    axes = axes.flatten()
    for j in range(n):
        ax = axes[j]
        y = q[:, j]
        default = float(env.default_dof_pos[env_id, j].detach().cpu())
        lo = float(env.dof_pos_limits[j, 0].detach().cpu())
        hi = float(env.dof_pos_limits[j, 1].detach().cpu())
        ax.plot(steps, y, linewidth=0.8, color="tab:blue", label="actual")
        ax.plot(steps, target[:, j], linewidth=0.8, color="tab:orange", alpha=0.8, label="target")
        ax.axhline(default, color="red", linestyle="--", linewidth=1.0, label="default")  # 零动作默认转角
        ax.set_title(env.dof_names[j])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (rad)")
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Joint angles (rad) — env {env_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n关节曲线已保存到：{path}")


def _plot_torque_curves(env, tau, env_id, path):
    """绘制指定 env 每个关节的实际扭矩曲线，并画出 ±扭矩限位线。"""
    n = env.num_dof
    steps = np.arange(tau.shape[0]) * env.dt  # 横轴时间（秒）
    limits = env.torque_limits.detach().cpu().numpy()  # (num_dof,)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows), squeeze=False)
    axes = axes.flatten()
    for j in range(n):
        ax = axes[j]
        y = tau[:, j]
        lim = limits[j]
        ax.plot(steps, y, linewidth=0.8, color="tab:blue")
        ax.axhline(lim, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(-lim, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"{env.dof_names[j]} (limit ±{lim:.1f})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("torque (N·m)")
        ax.set_ylim(-lim * 1.15, lim * 1.15)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Joint torques (N·m) — env {env_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"关节扭矩曲线已保存到：{path}")


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs  # 仅改并行数量，其余配置不动

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)

    print(f"dof_names: {env.dof_names}")
    print(f"num_dof={env.num_dof}  dt={env.dt:.4f}  action 值域 [-1, 1]  mode={args.mode}\n")

    obs, goal, critic_obs = env.get_observations()
    total_done = 0
    rew_acc = 0.0
    hold_steps = max(1, int(args.hold_duration / env.dt))  # 随机动作保持步数
    joint_history = []   # env_id 实际关节转角（rad）
    torque_history = []  # env_id 实际关节扭矩（N·m）
    target_history = []  # env_id PD 目标转角（rad）
    action_history = []  # env_id 施加的动作

    action = get_action(env, 0, args.mode)
    for step in range(args.num_steps):
        # random 模式：每 hold_steps 步重采样一次，其余步保持上次动作；manual 模式：每步更新
        if args.mode == "random":
            if step % hold_steps == 0:
                action = get_action(env, step, args.mode)
        else:
            action = get_action(env, step, args.mode)

        obs, reward, done, extras, goal, critic_obs = env.step(action)
        rew_acc += float(reward.mean().item())
        ids = done.nonzero(as_tuple=False).flatten()
        total_done += ids.numel()
        joint_history.append(env.dof_pos[args.env_id].detach().cpu().clone())
        torque_history.append(env.torques[args.env_id].detach().cpu().clone())
        target_history.append(action_to_target(env, action)[args.env_id].detach().cpu().clone())
        action_history.append(action[args.env_id].detach().cpu().clone())
        if step % 50 == 0:
            print(f"[step {step:4d}] base_z={env.base_position[:, 2].mean().item():.3f} "
                  f"foot_z_min={env.foot_positions[:, :, 2].min().item():.3f} "
                  f"foot_z_max={env.foot_positions[:, :, 2].max().item():.3f}")

    print(f"\n完成：{args.num_steps} 步 × {args.num_envs} 环境，共 {total_done} 次终止/重置，"
          f"平均每步奖励={rew_acc / args.num_steps:.4f}")

    # ---- 统计 env_id 各关节实际转角范围（rad）并绘图 ----
    q = torch.stack(joint_history).numpy()              # (num_steps, num_dof)
    limits = env.dof_pos_limits.detach().cpu().numpy()  # (num_dof, 2) 物理限位
    print(f"\n=== env {args.env_id} 各关节实际转角（单位 rad）===")
    print(f"{'joint':16s} {'物理下限':>10s} {'物理上限':>10s} {'实际min':>10s} {'实际max':>10s} {'实际幅值':>10s}")
    for j, name in enumerate(env.dof_names):
        lo, hi = limits[j]
        amin, amax = q[:, j].min(), q[:, j].max()
        print(f"{name:16s} {lo:10.4f} {hi:10.4f} {amin:10.4f} {amax:10.4f} {amax - amin:10.4f}")

    target = torch.stack(target_history).numpy()          # (num_steps, num_dof)
    _plot_joint_curves(env, q, target, args.env_id, args.plot_path)

    # ---- 统计 env_id 各关节实际扭矩 vs 扭矩限位（N·m）并绘图 ----
    tau = torch.stack(torque_history).numpy()          # (num_steps, num_dof)
    tlim = env.torque_limits.detach().cpu().numpy()    # (num_dof,)
    print(f"\n=== env {args.env_id} 各关节实际扭矩 vs 扭矩限位（单位 N·m）===")
    print(f"{'joint':16s} {'扭矩限位':>10s} {'max|tau|':>10s} {'平均|tau|':>10s} {'饱和占比':>10s}")
    for j, name in enumerate(env.dof_names):
        lim = tlim[j]
        abs_tau = np.abs(tau[:, j])
        sat = float((abs_tau >= lim * 0.99).mean())
        print(f"{name:16s} {lim:10.2f} {abs_tau.max():10.2f} {abs_tau.mean():10.2f} {sat * 100:9.1f}%")

    _plot_torque_curves(env, tau, args.env_id, args.torque_plot_path)

    # ---- 保存原始数据 ----
    act = torch.stack(action_history).numpy()            # (num_steps, num_dof)
    np.savez(args.data_path,
             time=np.arange(q.shape[0]) * env.dt,
             q=q,
             tau=tau,
             target=target,
             action=act,
             dof_names=np.array(env.dof_names),
             dof_pos_limits=limits,
             torque_limits=tlim,
             default_dof_pos=env.default_dof_pos[args.env_id].detach().cpu().numpy(),
             dt=env.dt)
    print(f"\n原始数据已保存到：{args.data_path}")


if __name__ == "__main__":
    main()
