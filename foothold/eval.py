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

from .config import AttrDict, get_flat_config
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

    # 不传 weights_only，以同时兼容项目使用的 PyTorch 1.12 和较新版本。
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "normalizer" not in ckpt:
        raise SystemExit(
            "checkpoint 不包含训练期 Normalizer/RMS，无法进行与训练一致的评估。"
            "这是旧格式 checkpoint；请使用包含 normalizer 字段的新 checkpoint，"
            "不要用 eval 轨迹临时重估 RMS。")
    if ckpt.get("format_version") != 2 or "config" not in ckpt:
        raise SystemExit(
            f"不支持的 checkpoint 格式: format_version={ckpt.get('format_version')!r}；"
            "评估需要同时包含训练配置的 version 2 checkpoint。")

    try:
        cfg = AttrDict.from_nested(ckpt["config"])
        runtime_cfg = get_flat_config()
        # 训练配置保存的是绝对路径；跨机器评估时只重定位同型号的本地资产。
        if not os.path.isfile(cfg.asset.file):
            if cfg.asset.name != runtime_cfg.asset.name:
                raise ValueError(
                    f"checkpoint 机器人 {cfg.asset.name!r} 与本地默认资源 "
                    f"{runtime_cfg.asset.name!r} 不一致")
            cfg.asset.file = runtime_cfg.asset.file
        cfg.env.num_envs = 1
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"checkpoint 中的训练配置无效: {exc}") from exc

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)
    ac = ActorCritic(env.num_obs + cfg.foothold.goal_dim,
                     env.num_critic_obs + cfg.foothold.goal_dim, env.num_dof, cfg).to(env.device)
    ac.load_state_dict(ckpt["actor_critic"])
    ac.eval()

    normalizer = Normalizer(env.num_obs, cfg.foothold.goal_dim, env.num_critic_obs,
                            cfg.ppo.gamma, env.device, cfg.normalization.running_obs_clip)
    try:
        normalizer.load_state_dict(ckpt["normalizer"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"checkpoint 中的 Normalizer 状态无效: {exc}") from exc

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
