"""diag_dof_props.py —— 诊断：运行时每个关节实际生效的 dof 属性，并施加恒定扭矩看关节能否运动。"""
from isaacgym import gymapi, gymtorch, gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [{"name": "--num_envs", "type": int, "default": 16}]
    args = gymutil.parse_arguments(description="diag dof props", headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)

    # 1) 运行时每个关节实际生效的 dof 属性
    props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    print("=== env 0 实际 dof_props ===")
    print(f"{'joint':16s} {'drive':>5s} {'stiff':>8s} {'damp':>8s} {'effort':>7s} "
          f"{'friction':>9s} {'armature':>9s} {'velocity':>9s}")
    for i, name in enumerate(env.dof_names):
        print(f"{name:16s} {int(props['driveMode'][i]):5d} "
              f"{float(props['stiffness'][i]):8.4f} {float(props['damping'][i]):8.4f} "
              f"{float(props['effort'][i]):7.2f} {float(props['friction'][i]):9.4f} "
              f"{float(props['armature'][i]):9.4f} {float(props['velocity'][i]):9.2f}")

    # 1.5) 运行时每个刚体的质量
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    print("\n=== env 0 各刚体质量（kg）===")
    for b, name in enumerate(body_names):
        print(f"{b:2d}: {name:24s} mass={env.rigid_body_masses[0, b].item():.4f}")

    # 2) 空中单关节恒定扭矩测试（抬离地面，避免闭链约束；每次只驱动一个关节）
    def reset_to_nominal_air():
        env.dof_pos[:] = env.default_dof_pos
        env.dof_vel[:] = 0.0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.root_states[:, 0, 0:2] = env.env_origins[:, 0:2]
        env.root_states[:, 0, 2] = 3.0          # 抬到 3m 高空
        env.root_states[:, 0, 3:7] = 0.0
        env.root_states[:, 0, 6] = 1.0          # 单位四元数 (x=0,y=0,z=0,w=1)
        env.root_states[:, 0, 7:13] = 0.0       # 零速度
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)

    print("\n=== 空中单关节 +50 N·m 测试（抬离地面，每次只驱动一个关节，20 子步=0.1s）===")
    print(f"{'joint':16s} {'初始q':>10s} {'Δq':>12s} {'effort限':>8s}")
    for j, name in enumerate(env.dof_names):
        reset_to_nominal_air()
        tau = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        tau[:, j] = 50.0  # 只给第 j 个关节 +50 Nm
        q0 = env.dof_pos[0, j].item()
        for _ in range(20):
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
            env.gym.simulate(env.sim)
            env.gym.refresh_dof_state_tensor(env.sim)
        dq = env.dof_pos[0, j].item() - q0
        print(f"{name:16s} {q0:10.4f} {dq:12.4f} {env.torque_limits[j].item():8.1f}")

    # 3) 对照实验：把 knee_L 的 friction 分别强制设为 0.0 / 0.5 / 0.9，看 Δq 变化
    print("\n=== 对照：knee_L friction 与 Δq 的关系（+50 N·m，20 子步）===")
    print(f"{'friction':>10s} {'Δq':>12s}")
    for fric in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        props = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
        for k in range(env.num_dof):
            props["friction"][k] = fric
        env.gym.set_actor_dof_properties(env.envs[0], env.actor_handles[0], props)
        reset_to_nominal_air()
        tau = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        tau[:, env.dof_names.index("knee_L_Joint")] = 50.0
        q0 = env.dof_pos[0, env.dof_names.index("knee_L_Joint")].item()
        for _ in range(20):
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
            env.gym.simulate(env.sim)
            env.gym.refresh_dof_state_tensor(env.sim)
        dq = env.dof_pos[0, env.dof_names.index("knee_L_Joint")].item() - q0
        print(f"{fric:10.2f} {dq:12.4f}")


if __name__ == "__main__":
    main()
