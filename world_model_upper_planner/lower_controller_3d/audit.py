"""Deterministic physical tracking audit; no adaptive landing surfaces."""
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from isaacgym import gymapi, gymutil
import torch
from .config import get_flat_config
from .env import FootholdEnv, make_sim_params
from .networks import ActorCritic
from .ppo import Normalizer


def main():
    args = gymutil.parse_arguments(description=__doc__, headless=True, custom_parameters=[
        dict(name="--checkpoint", type=str, default="checkpoints/lower_model_7000.pt"),
        dict(name="--num_envs", type=int, default=64),
        dict(name="--steps", type=int, default=1000),
        dict(name="--height_range", type=float, default=0.02),
        dict(name="--lateral_separation", type=float, default=0.10),
        dict(name="--seed", type=int, default=8201),
        dict(name="--output", type=str, required=True),
        dict(name="--record_video", action="store_true")])
    args.sim_device = "cuda:0" if args.sim_device_type == "cuda" else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = get_flat_config(); cfg.env.num_envs = args.num_envs
    cfg.foothold.z_distance = [-args.height_range, args.height_range]
    cfg.foothold.minimum_lateral_separation=args.lateral_separation
    cfg.noise.add_noise = False
    cfg.domain_rand.kick_robots = False
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)
    ac = ActorCritic(env.num_obs+16, env.num_critic_obs+16, env.num_dof, cfg).to(env.device)
    ckpt = torch.load(args.checkpoint, map_location=env.device)
    ac.load_state_dict(ckpt["actor_critic"]); ac.eval()
    norm = Normalizer(env.num_obs, 16, env.num_critic_obs, cfg.ppo.gamma,
                      env.device, cfg.normalization.running_obs_clip)
    norm.load_state_dict(ckpt["normalizer"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if args.record_video:
        if env.viewer is None:
            raise ValueError("video requires viewer; omit --headless")
        (output / "frames").mkdir(exist_ok=True)
    trace = {k: [] for k in ("root", "feet", "targets", "pillars", "done", "phase", "contact")}
    errors, target_dz, contacts = [], [], []
    falls = torch.zeros(args.num_envs, device=env.device)
    alive_steps = torch.zeros_like(falls); longest = torch.zeros_like(falls)
    height_mismatch = 0.0
    start = time.perf_counter()
    with torch.no_grad():
        for tick in range(args.steps):
            obs, goal, critic = norm.observations(*env.get_observations(), update=False)
            # Score the previous goal immediately BEFORE a scheduled gait switch.
            switching = torch.floor(env.sampler.phase*2) != torch.floor(
                torch.remainder(env.sampler.phase+env.dt*env.sampler.frequency, 1.0)*2)
            ids = (switching & env.pillar_target_active & ~env.goal_reset_pending).nonzero().flatten()
            if len(ids):
                foot = env.sampler.swing_foot[ids]
                errors.append((env.foot_positions[ids, foot]-env.sampler.target_pos[ids, foot]).cpu().numpy())
                target_dz.append((env.sampler.target_pos[ids, foot, 2]-env.support_height()[ids]).cpu().numpy())
                contacts.append((env.contact_forces[ids, env.feet_indices[foot], 2] > 1).cpu().numpy())
            _, _, done, extras, _, _ = env.step(ac.act_inference(torch.cat((obs, goal), -1)))
            active = env.pillar_target_active
            rows = active.nonzero().flatten()
            if len(rows):
                tops = env.root_states[rows, env.target_pillar[rows]+1, 2] + cfg.terrain.pillar_size[2]/2
                height_mismatch = max(height_mismatch, float((tops-env.sampler.target_pos[rows, env.sampler.swing_foot[rows], 2]).abs().max()))
            falls += extras["absorbing"].float()
            alive_steps += 1; longest = torch.maximum(longest, alive_steps)
            alive_steps[done.bool()] = 0
            values = dict(root=extras["terminal_root"][0], feet=extras["terminal_feet"][0],
                          targets=extras["terminal_targets"][0], pillars=env.root_states[0,1:],
                          done=done[0], phase=env.sampler.phase[0],
                          contact=env.contact_forces[0, env.feet_indices, 2])
            for k,v in values.items(): trace[k].append(v.cpu().numpy().copy())
            if args.record_video and tick % 2 == 0:
                p = env.base_position[0].cpu().numpy()
                env.gym.viewer_camera_look_at(env.viewer, None,
                    gymapi.Vec3(float(p[0]+1.8), float(p[1]-2.0), float(p[2]+0.8)),
                    gymapi.Vec3(float(p[0]), float(p[1]), float(p[2]-0.2)))
                env.gym.write_viewer_image_to_file(env.viewer, str(output / "frames" / f"{tick//2:06d}.png"))
    e = np.concatenate(errors) if errors else np.zeros((0,3))
    np.savez_compressed(output/"trajectory.npz", **{k:np.stack(v) for k,v in trace.items()})
    np.savez_compressed(output/"touchdowns.npz", error_xyz=e,
                        target_dz=np.concatenate(target_dz) if target_dz else [],
                        contact=np.concatenate(contacts) if contacts else [])
    metrics = dict(checkpoint=args.checkpoint, seed=args.seed, num_envs=args.num_envs,
        steps=args.steps, height_range_m=args.height_range, falls=int(falls.sum()),
        lateral_separation_m=args.lateral_separation,
        no_fall_env_fraction=float((falls==0).float().mean()),
        longest_seconds_mean=float(longest.mean()*env.dt), touchdown_events=len(e),
        xy_median_m=float(np.median(np.linalg.norm(e[:,:2],axis=-1))) if len(e) else None,
        z_median_m=float(np.median(np.abs(e[:,2]))) if len(e) else None,
        xyz_p90_m=float(np.quantile(np.linalg.norm(e,axis=-1),.9)) if len(e) else None,
        contact_fraction=float(np.concatenate(contacts).mean()) if contacts else None,
        pillar_target_height_max_error_m=height_mismatch, wall_seconds=time.perf_counter()-start)
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2))
    if args.record_video:
        subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","25","-i",
                        str(output/"frames/%06d.png"),"-c:v","mpeg4","-q:v","3","-pix_fmt","yuv420p",
                        str(output/"rollout.mp4")],check=True)
        metrics["video"] = str(output/"rollout.mp4")
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

if __name__ == "__main__": main()
