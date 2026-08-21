"""eval.py —— 加载模型做确定性 rollout，打印指标 + 保存轨迹。

用法：
    python -m foothold.eval --checkpoint foothold/logs/<run>/model_100.pt --steps 500
"""

import argparse
import json
import os

import numpy as np
from isaacgym import gymutil
import torch

from .config import get_flat_config
from .env import FootholdEnv, make_sim_params
from .networks import ActorCritic
from .ppo import Normalizer


def cpu(x):
    return x.detach().cpu().numpy().copy()


def parse_args():
    custom = [
        {"name": "--checkpoint", "type": str, "default": None},
        {"name": "--steps", "type": int, "default": 500},
        {"name": "--seed", "type": int, "default": 42},
        {"name": "--output_dir", "type": str, "default": None},
    ]
    args = gymutil.parse_arguments(description="foothold eval", headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise SystemExit("请提供有效的 --checkpoint 路径")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_flat_config()
    cfg.env.num_envs = 1
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)

    ckpt = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    ac = ActorCritic(env.num_obs + cfg.foothold.goal_dim,
                     env.num_critic_obs + cfg.foothold.goal_dim, env.num_dof, cfg).to(env.device)
    ac.load_state_dict(ckpt["actor_critic"])
    ac.eval()

    normalizer = Normalizer(env.num_obs, cfg.foothold.goal_dim, env.num_critic_obs,
                            cfg.ppo.gamma, env.device, cfg.normalization.running_obs_clip)

    obs, goal, critic_obs = env.get_observations()
    obs, goal, critic_obs = normalizer.observations(obs, goal, critic_obs, update=False)

    keys = ["root", "dof_pos", "dof_vel", "torque", "foot_pos", "target_pos", "target_yaw",
            "swing_foot", "phase", "reward", "done", "time"]
    traces = {k: [] for k in keys}
    falls, cur_len, ep_lens = 0, 0, []

    for step in range(args.steps):
        with torch.no_grad():
            action = ac.act_inference(torch.cat((obs, goal), dim=-1))
        next_obs, reward, done, extras, next_goal, next_critic_obs = env.step(action)

        traces["root"].append(cpu(env.root_states[0, 0]))
        traces["dof_pos"].append(cpu(env.dof_pos[0]))
        traces["dof_vel"].append(cpu(env.dof_vel[0]))
        traces["torque"].append(cpu(env.torques[0]))
        traces["foot_pos"].append(cpu(env.foot_positions[0]))
        traces["target_pos"].append(cpu(env.sampler.target_pos[0]))
        traces["target_yaw"].append(cpu(env.sampler.target_yaw[0]))
        traces["swing_foot"].append(cpu(env.sampler.swing_foot)[0])
        traces["phase"].append(cpu(env.sampler.phase)[0])
        traces["reward"].append(cpu(reward)[0])
        traces["done"].append(cpu(done)[0])
        traces["time"].append(step * env.dt)

        cur_len += 1
        if bool(done[0]):
            ep_lens.append(cur_len)
            cur_len = 0
            if not bool(extras.get("time_outs", torch.zeros(1, dtype=torch.bool, device=env.device))[0]):
                falls += 1

        obs, goal, critic_obs = normalizer.observations(next_obs, next_goal, next_critic_obs, update=False)

    arrays = {k: np.stack(v) for k, v in traces.items()}
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(os.path.join(out_dir, "eval"), exist_ok=True)
    np.savez(os.path.join(out_dir, "eval", "trajectory.npz"), **arrays)

    metrics = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "steps": args.steps, "falls": falls,
        "episode_lengths": ep_lens,
        "max_continuous_steps": max(ep_lens) if ep_lens else 0,
        "mean_reward": float(arrays["reward"].sum()),
    }
    with open(os.path.join(out_dir, "eval", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
