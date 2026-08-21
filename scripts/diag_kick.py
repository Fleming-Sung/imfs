"""diag_kick.py —— 单 env 内验证 _kick 是否是首步"爆炸"元凶。"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [{"name": "--num_envs", "type": int, "default": 2048}]
    args = gymutil.parse_arguments(description="diag kick", headless=True,
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
    lo, hi = cfg.env.healthy_height_range
    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)

    # 记录 _kick 每次踢了多少个 env
    orig_kick = env._kick

    def logging_kick():
        n = int((torch.rand(env.num_envs, device=env.device) < env.cfg.domain_rand.kick_probability).sum().item())
        if n:
            print(f"    [_kick] 本步踢了 {n} 个 env")
        return orig_kick()

    env._kick = logging_kick

    print(f"num_envs={env.num_envs} healthy=[{lo},{hi}]\n")
    for step in range(6):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        h = env.base_position[:, 2] - env.env_origins[:, 2]
        print(f"[step {step}] done={int(done.sum().item()):5d}  "
              f"post_hmin={h.min().item():.4f} post_hmax={h.max().item():.4f}")


if __name__ == "__main__":
    main()
