"""calibrate_friction.py —— 标定 IsaacGym 关节 friction 系数的物理含义。

目的：IsaacGym 的 dof_props["friction"] 是 **无量纲系数**（0=自由，1=锁死），
而论文(MuJoCo)的 dof_frictionloss 是 **库仑干摩擦扭矩，单位 N·m**（[0,1] N·m）。
两者单位不同，不能直接照搬。本脚本通过实验测量：
  (1) asset 默认的 friction/damping/armature（看 IsaacGym 是否读取 URDF <dynamics>）。
  (2) 每个 friction 系数对应的"起动力矩(breakaway torque)"——即关节在零重力下
      需要多大的恒定扭矩才能明显转动。据此反推：MuJoCo [0,1] N·m 对应 IsaacGym
      的 friction 系数范围。
"""
from isaacgym import gymapi, gymtorch, gymutil  # isaacgym 必须先于 torch 导入
import torch
import numpy as np

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [{"name": "--num_envs", "type": int, "default": 4}]
    args = gymutil.parse_arguments(description="calibrate friction", headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)
    dt = env.sim_params.dt
    print(f"sim dt = {dt:.5f} s")

    # ---------- (1) asset 默认属性（随机化之前） ----------
    asset_path = cfg.asset.file
    import os
    options = gymapi.AssetOptions()
    options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
    options.collapse_fixed_joints = True
    options.fix_base_link = cfg.asset.fix_base_link
    options.disable_gravity = cfg.asset.disable_gravity
    options.density = cfg.asset.density
    asset = env.gym.load_asset(env.sim, os.path.dirname(asset_path),
                               os.path.basename(asset_path), options)
    aprops = env.gym.get_asset_dof_properties(asset)
    print("\n=== asset 默认 dof_props（随机化之前，即 URDF <dynamics> 的值）===")
    print(f"{'joint':16s} {'friction':>9s} {'damping':>9s} {'armature':>9s}")
    for i, name in enumerate(env.dof_names):
        print(f"{name:16s} {float(aprops['friction'][i]):9.4f} "
              f"{float(aprops['damping'][i]):9.4f} {float(aprops['armature'][i]):9.4f}")

    # ---------- 准备：零重力 + 抬离地面 ----------
    sp = env.gym.get_sim_params(env.sim)
    sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    env.gym.set_sim_params(env.sim, sp)
    print("\n已把重力置零（用于标定纯摩擦扭矩，排除重力负载与闭链约束干扰）")

    def reset_air():
        env.dof_pos[:] = env.default_dof_pos
        env.dof_vel[:] = 0.0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.root_states[:, 0, 0:2] = env.env_origins[:, 0:2]
        env.root_states[:, 0, 2] = 3.0
        env.root_states[:, 0, 3:7] = 0.0
        env.root_states[:, 0, 6] = 1.0
        env.root_states[:, 0, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)

    knee = env.dof_names.index("knee_L_Joint")

    def set_friction_all_envs(c):
        for e in range(env.num_envs):
            props = env.gym.get_actor_dof_properties(env.envs[e], env.actor_handles[e])
            for k in range(env.num_dof):
                props["friction"][k] = c
            env.gym.set_actor_dof_properties(env.envs[e], env.actor_handles[e], props)

    # ---------- (2) 起动力矩标定：friction 系数 vs 能转动所需最小扭矩 ----------
    # 零重力下，自由关节在恒定扭矩 tau 下恒加速：Δq = 0.5*(tau/I_eff)*t^2。
    # 有库仑摩擦 tau_c 时，tau <= tau_c 关节不动。取"明显转动"阈值判定起动力矩。
    NSTEPS = 400                     # 400 * dt = 2.0 s
    MOVE_THRESH = 0.02               # rad，超过即认为"明显转动"
    tau_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    c_list = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    # 先测自由(c=0)情况，反推有效惯量 I_eff
    set_friction_all_envs(0.0)
    reset_air()
    tau = torch.zeros(env.num_envs, env.num_dof, device=env.device)
    tau[:, knee] = 0.1
    q0 = env.dof_pos[0, knee].item()
    for _ in range(NSTEPS):
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
        env.gym.simulate(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
    dq_free = env.dof_pos[0, knee].item() - q0
    I_eff = 0.5 * 0.1 * (NSTEPS * dt) ** 2 / max(dq_free, 1e-6)
    print(f"\nc=0 自由关节在 0.1 N·m 下 {NSTEPS*dt:.1f}s 转 {dq_free:.4f} rad → 有效惯量 I_eff≈{I_eff:.4f} kg·m²")

    # 主表：行=扭矩，列=系数，值=Δq(rad)
    print(f"\n=== 起动力矩标定：Δq(rad)，零重力，{NSTEPS*dt:.1f}s，阈值 {MOVE_THRESH} rad ===")
    header = f"{'tau(Nm)':>8s} " + "".join(f"{c:>9.4g}" for c in c_list)
    print(header)
    breakaway = {}
    for c in c_list:
        breakaway[c] = None
    for t in tau_list:
        row = f"{t:8.3f} "
        for c in c_list:
            set_friction_all_envs(c)
            reset_air()
            tau = torch.zeros(env.num_envs, env.num_dof, device=env.device)
            tau[:, knee] = t
            q0 = env.dof_pos[0, knee].item()
            for _ in range(NSTEPS):
                env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
                env.gym.simulate(env.sim)
                env.gym.refresh_dof_state_tensor(env.sim)
            dq = env.dof_pos[0, knee].item() - q0
            row += f"{dq:9.4f}"
            if breakaway[c] is None and dq > MOVE_THRESH:
                breakaway[c] = t
        print(row)

    print("\n=== 每个 friction 系数的起动力矩(breakaway) ===")
    for c in c_list:
        tb = breakaway[c]
        if tb is None:
            print(f"friction={c:>6.4f}  起动力矩 > {tau_list[-1]:.2f} N·m（近似锁死）")
        else:
            print(f"friction={c:>6.4f}  起动力矩 ≈ {tb:.3f} N·m")

    # ---------- (2b) 重力恢复，短时窗(0.1s)测重力负载下的摩擦。
    # 短时窗是为了避免整机自由下落翻滚污染 knee 角读数。----------
    sp = env.gym.get_sim_params(env.sim)
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    env.gym.set_sim_params(env.sim, sp)
    print("\n已恢复重力 -9.81 m/s²（空中短时窗，knee 承载小腿+脚负载）")

    SHORT = 20                      # 20 * dt = 0.1 s
    g_c_list = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
    g_tau_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    print(f"\n=== 重力下(真实负载) Δq(rad)，{SHORT*dt:.2f}s ===")
    header = f"{'tau(Nm)':>8s} " + "".join(f"{c:>9.4g}" for c in g_c_list)
    print(header)
    for t in g_tau_list:
        row = f"{t:8.3f} "
        for c in g_c_list:
            set_friction_all_envs(c)
            reset_air()
            tau = torch.zeros(env.num_envs, env.num_dof, device=env.device)
            tau[:, knee] = t
            q0 = env.dof_pos[0, knee].item()
            for _ in range(SHORT):
                env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
                env.gym.simulate(env.sim)
                env.gym.refresh_dof_state_tensor(env.sim)
            dq = env.dof_pos[0, knee].item() - q0
            row += f"{dq:9.4f}"
        print(row)

    # ---------- (3) 结论：MuJoCo [0,1] N·m 对应哪个 friction 系数范围 ----------
    print("\n=== 换算结论 ===")
    print("MuJoCo dof_frictionloss ∈ [0,1] N·m（库仑干摩擦扭矩）")
    print("真实 URDF：ankle friction=0.01 N·m，其余关节=0")
    print("请根据上表把 [0,1] N·m 映射到 IsaacGym friction 系数区间。")


if __name__ == "__main__":
    main()
