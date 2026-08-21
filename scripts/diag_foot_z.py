"""diag_foot_z.py —— 检查初始(重置后)足端是否穿透地面。"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [{"name": "--num_envs", "type": int, "default": 128}]
    args = gymutil.parse_arguments(description="diag foot z", headless=True,
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

    # 初始(重置后)足端位置
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    foot = env.rigid_body_states[:, env.feet_indices]
    print("初始(重置后)足端 body 中心 z：")
    for f, name in enumerate(["foot_L", "foot_R"]):
        z = foot[:, f, 2]
        print(f"  {name}: min={z.min().item():.4f} max={z.max().item():.4f} mean={z.mean().item():.4f}")
    # 足底接触点(减去 foot_site_offset 的 z)
    z_contact = foot[:, :, 2] + torch.tensor(cfg.asset.foot_site_offsets, device=env.device)[:, 2]
    print(f"  足底接触点 z: min={z_contact.min().item():.4f} max={z_contact.max().item():.4f}")

    # base z
    env.gym.refresh_actor_root_state_tensor(env.sim)
    print(f"  base z: min={env.root_states[:, 0, 2].min().item():.4f} "
          f"max={env.root_states[:, 0, 2].max().item():.4f}")

    # 步进 1 步后足端/base z
    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)
    obs, reward, done, extras, goal, critic_obs = env.step(action)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    foot = env.rigid_body_states[:, env.feet_indices]
    z_contact = foot[:, :, 2] + torch.tensor(cfg.asset.foot_site_offsets, device=env.device)[:, 2]
    print(f"\nstep0 后 done={int(done.sum().item())}/{env.num_envs}")
    print(f"  足底接触点 z: min={z_contact.min().item():.4f} max={z_contact.max().item():.4f}")


if __name__ == "__main__":
    main()
