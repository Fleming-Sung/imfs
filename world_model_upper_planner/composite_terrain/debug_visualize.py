"""交互式调试：可视化最佳模型（H3）的观测与候选评分。

用法（在 world_model_upper_planner 目录）：
    python -m composite_terrain.debug_visualize \
        --lower_checkpoint runs/lower3d_v3_z06/model_8201.pt \
        --checkpoint runs/composite3d_value_h3_v2/h3_best.pt \
        --motion_checkpoint runs/composite3d_value_h3_v2/motion_sequence/model_best.pt \
        --difficulty 0.5 --horizon 3 --value_weight 1.0 --steps 3000

左边子图 = 64x64 局部高度图（即模型的"深度观测"，相对支撑脚高度）；
候选落足点以彩色圆点叠加其上（颜色=评分，越红越高；灰色=几何无效），
选中候选以星号标出。右边子图 = 128x128 全景缓存（多步 rollout 用的静态地图）。

说明：本环境是"无相机"的静态高度图环境，观测 H_t 就是这张 64x64 局部高度图，
不是渲染的深度相机图。若要同时看 3D 场景，不要加 --headless（IsaacGym viewer 会开窗）。
"""
import argparse
import math
from pathlib import Path

import numpy as np
from isaacgym import gymapi
import torch

from composite_terrain.env import create_composite_env
from composite_terrain.interface import (
    candidates, sense, geometry, apply, panorama, world_targets)
from adapters.frozen_lower_env.sampler import quaternion_yaw, wrap_to_pi
from adapters.frozen_lower_env.lower_policy import FrozenLowerPolicy
from cgowm import (CandidateGroundedWorldModel, ModelConfig,
                   PlannerConfig, VectorizedBeamPlanner)
from composite_terrain.anchored import MotionModel, AnchoredPlanner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lower_checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motion_checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("experiments/debug_viz"))
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=9801)
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--proposals_per_beam", type=int, default=8)
    p.add_argument("--value_weight", type=float, default=1.0)
    p.add_argument("--show_top", type=int, default=30,
                   help="how many candidates to draw as colored dots (rest faint)")
    p.add_argument("--save_frames", action="store_true",
                   help="save each decision frame as PNG under --output")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--sim_device", default="cuda:0")
    args = p.parse_args()
    args.use_gpu_pipeline = True
    args.use_gpu = True
    args.subscenes = 0
    return args


def local_xy(base_pos, base_quat, world_xy):
    """World XY -> robot-yaw frame (forward, lateral), same as sense().
    base_pos: [3]; base_quat: [4]; world_xy: [C, 3] -> returns [C],[C]."""
    yaw = quaternion_yaw(base_quat.unsqueeze(0))[0]
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = world_xy[..., 0] - base_pos[0]
    dy = world_xy[..., 1] - base_pos[1]
    return c * dx + s * dy, -s * dx + c * dy


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use('Agg' if args.headless else 'TkAgg')
    import matplotlib.pyplot as plt

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    env = create_composite_env(args)
    lower = FrozenLowerPolicy(args.lower_checkpoint, env.device)
    grid = candidates(env.device)
    n = env.num_envs
    proprio_dim = 75

    ck = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    model = CandidateGroundedWorldModel(
        ck["candidates"], ModelConfig(**ck["model_config"])).to(env.device)
    model.load_state_dict(ck["model"]); model.eval()
    if not torch.equal(model.candidates, grid):
        raise ValueError("候选集不一致：请用同一份代码生成的候选")
    motion_ck = torch.load(args.motion_checkpoint, map_location=env.device,
                           weights_only=False)
    motion = MotionModel(motion_ck["latent_dim"], motion_ck["proprio_dim"]).to(env.device)
    motion.load_state_dict(motion_ck["motion_model"]); motion.eval()
    planner = AnchoredPlanner(
        model, motion,
        PlannerConfig(horizon=args.horizon, beam_width=16,
                      proposals_per_beam=args.proposals_per_beam,
                      support_weight=3.0, value_weight=args.value_weight))

    active = torch.zeros(n, dtype=torch.bool, device=env.device)
    prev = torch.zeros(n, 4, device=env.device)
    episode = torch.zeros(n, dtype=torch.long, device=env.device)
    option = torch.zeros_like(episode)
    durations = torch.zeros_like(episode)
    state_image = torch.zeros(n, 1, 64, 64, device=env.device)
    state_proprio = torch.zeros(n, proprio_dim, device=env.device)
    chosen = torch.zeros_like(episode)
    collided = torch.zeros_like(active)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (register 3d projection)
    fig = plt.figure(figsize=(14, 6.5))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2)
    plt.ion()
    frame_idx = [0]

    def viz(ids, image, proprio, cache, scores, valid, selected):
        ax0.cla(); ax1.cla()
        # 左：3D 高度曲面（高度相对支撑脚，单位米）。台阶/梅花桩就是曲面上的凸起。
        h = image[0, 0].cpu().numpy() * 1.6 - 0.8          # [forward, lateral] -> m
        fwd = np.linspace(-0.5, 2.65, 64)
        lat = np.linspace(-1.575, 1.575, 64)
        X, Y = np.meshgrid(lat, fwd)                        # X=lateral, Y=forward
        ax0.plot_surface(X, Y, h, cmap='terrain', alpha=0.85,
                         linewidth=0, antialiased=True)
        targets, _ = world_targets(env, ids, grid)
        lx, ly = local_xy(env.base_position[ids][0],
                          env.base_quat[ids][0], targets[0])
        dz = 0.06 * grid[:, 2].cpu().numpy()               # 候选命令高度(相对支撑脚)
        order = torch.argsort(scores, descending=True)
        top = order[:args.show_top]
        cmap = plt.cm.coolwarm
        valid_mask = valid[0].bool()
        sc = scores.clone()
        sc[~valid_mask] = -torch.inf
        lo, hi = sc[valid_mask].min(), sc[valid_mask].max()
        norm = (sc - lo) / (hi - lo).clamp_min(1e-6)
        for j in top:
            k = int(j)
            color = cmap(float(norm[k].clamp(0, 1))) if valid_mask[k] else (0.6, 0.6, 0.6)
            ax0.scatter(float(ly[k]), float(lx[k]), float(dz[k]), s=40,
                        color=color, depthshade=False)
        sel = int(selected[0])
        ax0.scatter(float(ly[sel]), float(lx[sel]), float(dz[sel]),
                    s=260, marker='*', color='lime', depthshade=False)
        ax0.set_xlabel('lateral (m)'); ax0.set_ylabel('forward (m)')
        ax0.set_zlabel('height rel stance (m)')
        ax0.set_zlim(-0.3, 0.3)
        ax0.view_init(elev=45, azim=-90)
        ax0.set_title(f"3D heightfield + candidates (green=selected)\n"
                      f"score range [{float(lo):.1f}, {float(hi):.1f}]")
        # 右：128x128 全景缓存（发散色标，红=高于支撑脚，蓝=低于，单位 cm）
        pano = cache[0, 0].cpu().numpy() * 1.6 - 0.8
        ax1.imshow(pano * 100.0, origin='lower', cmap='RdYlBu_r',
                   vmin=-20, vmax=20)
        ax1.set_title("128x128 panorama cache (height rel stance, cm)")
        fig.canvas.draw()
        if args.save_frames:
            fig.savefig(args.output / f"frame_{int(episode[0])}_{frame_idx[0]:05d}.png")
        frame_idx[0] += 1
        plt.pause(0.001)

    with torch.no_grad():
        for tick in range(args.steps):
            will_switch = torch.floor(env.sampler.phase * 2) != torch.floor(
                torch.remainder(env.sampler.phase + env.dt * env.sampler.frequency, 1) * 2)
            ids = (will_switch & active & ~env.goal_reset_pending).nonzero().flatten()
            env.sampler.advance(env.dt, env.foot_positions,
                                env.rigid_body_states[:, env.feet_indices, 3:7])
            option[ids] += 1
            active[ids] = False

            ids = (~active & ~env.goal_reset_pending).nonzero().flatten()
            if len(ids):
                image, proprio = sense(env, prev, proprio_dim, ids)
                cache = panorama(env, ids)
                geo = geometry(env, ids, grid)
                mask = geo["candidate_valid"].clone()
                empty = ~mask.any(-1)
                if empty.any():
                    mask[empty] = geo["candidate_support"][empty] >= \
                        geo["candidate_support"][empty].max(-1, keepdim=True).values
                selection, _ = planner.plan(image, proprio, cache, mask)

                # 复算第一步的全候选评分（与 AnchoredPlanner 第一步公式一致）
                latent = model.encode(image, proprio)
                pred = model.predict_candidates(latent)
                q = pred["q"][:, 0, :]              # [2, C]
                q_min = q.min(0).values
                uncertainty = q.std(0, unbiased=False)
                fall = torch.sigmoid(pred["fall_logit"][0])
                collision = torch.sigmoid(pred["collision_logit"][0])
                score = (10.0 * pred["progress"][0]
                         - 3.0 * (1 - pred["support"][0])
                         + args.value_weight * q_min
                         - 5.0 * fall
                         - 2.0 * collision
                         - 0.5 * uncertainty)
                state_image[ids] = image
                state_proprio[ids] = proprio
                chosen[ids] = selection
                prev[ids] = grid[selection]
                apply(env, ids, prev[ids])
                active[ids] = True
                collided[ids] = False
                durations[ids] = 0
                viz(ids, image, proprio, cache, score, mask, selection)

            obs, goal, _ = env.get_observations()
            action, _ = lower.infer(obs, goal)
            _, _, done, extras, _, _ = env.step(action)
            durations[active] += 1
            collided |= (torch.norm(env.contact_forces[:, env.nonfoot_indices],
                                    dim=-1).max(-1).values > 5)
            ended = done.bool().nonzero().flatten()
            if len(ended):
                for j in ended.cpu().tolist():
                    print(f"tick={tick} env={j} episode={int(episode[j])} "
                          f"fall={bool(extras['absorbing'][j])} reason="
                          f"{ {k: bool(v[j]) for k, v in extras['termination_reasons'].items()} }",
                          flush=True)
                active[ended] = False
                episode[ended] += 1
                option[ended] = 0
    plt.close(fig)


if __name__ == "__main__":
    main()
