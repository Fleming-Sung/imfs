"""Asymmetric PPO fine-tune of the behaviour-cloned candidate Actor (Gate E).

Actor  : CandidateActor warm-started from a BC checkpoint, a discrete policy
         over the candidate set, seeing only deployable depth + 36-D proprio.
Critic : privileged MLP over proprio + geodesic distance + support fraction +
         absolute stance/base positions (asymmetric, never deployed).
Reward : geodesic potential shaping r = d_t - gamma * d_{t+1}, plus success /
         collision / fall / off-support penalties.

The frozen lower policy executes every foothold target, so only the upper
candidate policy and privileged critic are trained.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacgym import gymapi  # must precede torch
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from upper_planner.candidate_actor import CandidateActor
from upper_planner.factory import create_upper_system
from upper_planner.ppo_critic import PrivilegedCritic, make_privileged
from upper_planner.privileged_planner import (
    PrivilegedPlannerConfig, PrivilegedTerrainPlanner)
from upper_planner.rollout import UpperRollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--rollout_ticks", type=int, default=600)
    parser.add_argument("--total_updates", type=int, default=200)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # reward weights (match the privileged gate reward configuration)
    parser.add_argument("--reward_progress", type=float, default=1.0)
    parser.add_argument("--reward_goal", type=float, default=10.0)
    parser.add_argument("--reward_collision", type=float, default=-5.0)
    parser.add_argument("--reward_fall", type=float, default=-5.0)
    parser.add_argument("--reward_off_support", type=float, default=-3.0)
    # terrain
    parser.add_argument("--terrain_curriculum", type=str, default="research")
    parser.add_argument("--research_kind", type=str, default="random_composite")
    parser.add_argument("--course_length_m", type=float, default=3.5)
    parser.add_argument("--random_width_min_m", type=float, default=0.50)
    parser.add_argument("--random_width_max_m", type=float, default=1.30)
    parser.add_argument("--random_gap_max_m", type=float, default=0.14)
    parser.add_argument("--random_obstacle_probability", type=float, default=0.0)
    parser.add_argument("--privileged_forward_levels", type=str,
                        default="-1.0,-0.818182,-0.636364,-0.454545,-0.272727,"
                                "-0.090909,0.090909,0.272727,0.454545,0.636364,"
                                "0.818182,1.0")
    parser.add_argument("--privileged_lateral_levels", type=str,
                        default="-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--privileged_yaw_levels", type=str, default="-0.5,0.0,0.5")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--physx", action="store_true", default=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _levels(text):
    return tuple(float(value) for value in text.split(","))


class RolloutBuffer:
    def __init__(self):
        self.env_ids = []
        self.privileged = []
        self.depth = []
        self.proprio = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def push(self, env_ids, privileged, depth, proprio, action, log_prob,
             reward, done):
        self.env_ids.append(env_ids)
        self.privileged.append(privileged)
        self.depth.append(depth)
        self.proprio.append(proprio)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def size(self):
        return sum(env_ids.numel() for env_ids in self.env_ids)

    def clear(self):
        self.__init__()


def main():
    args = parse_args()
    args.action_profile = "cartesian_course"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.actor_checkpoint, map_location="cpu")
    num_candidates = int(checkpoint["num_candidates"])
    candidates_levels = checkpoint["candidates_levels"].to(torch.float32)
    feature_dim = int(checkpoint["feature_dim"])
    gru_hidden = int(checkpoint["gru_hidden"])
    actor = CandidateActor(
        num_candidates, feature_dim=feature_dim, gru_hidden=gru_hidden)
    actor.load_state_dict(checkpoint["state_dict"])

    env, policy, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=1.5,
        randomization=False, cameras=True,
        flat_plane=False, obstacles=False,
        course_length_m=args.course_length_m)
    rollout = UpperRollout(
        env, policy, interface, task, cfg["depth"], capture_depth=True)
    planner = PrivilegedTerrainPlanner(
        tiled, interface.bounds, env.device,
        PrivilegedPlannerConfig(
            forward_levels=_levels(args.privileged_forward_levels),
            lateral_levels=_levels(args.privileged_lateral_levels),
            yaw_levels=_levels(args.privileged_yaw_levels)))

    device = env.device
    actor = actor.to(device)
    candidates_levels = candidates_levels.to(device)
    critic = PrivilegedCritic().to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    # Per-env pending-decision queues survive the rollout's internal overwrite.
    pending = [deque() for _ in range(env.num_envs)]
    buffer = RolloutBuffer()
    episode_counts = {"success": 0, "fall": 0, "timeout": 0}

    def policy_logits(depth, proprio):
        (logits, _, _), _ = actor(depth, proprio)
        return logits

    @torch.no_grad()
    def choose_actions(depth, proprio, ids):
        logits = policy_logits(depth, proprio)
        dist = Categorical(logits=logits)
        index = dist.sample()
        log_prob = dist.log_prob(index)
        privileged = make_privileged(proprio, env, planner, ids)
        swing = env.sampler.swing_foot[ids]
        stance = 1 - swing
        row = torch.arange(ids.numel(), device=device)
        stance_xy = env.foot_positions[ids][row, stance, :2]
        for k, env_id in enumerate(ids.cpu().tolist()):
            pending[env_id].append((index[k], log_prob[k],
                                    privileged[k], stance_xy[k]))
        return candidates_levels[index]

    @torch.no_grad()
    def collect_rollout(ticks):
        buffer.clear()
        for _ in range(ticks):
            transitions = rollout.lower_tick(choose_actions)
            if transitions is None:
                continue
            ids = transitions["ids"]

            # A transition is emitted for stance switches and terminations.  A
            # fall that happened before the env's first decision has no pending
            # action, so it is skipped rather than attributed to a zero action.
            valid_rows = []
            popped = []
            for k, e in enumerate(ids.cpu().tolist()):
                if pending[e]:
                    valid_rows.append(k)
                    popped.append(pending[e].popleft())
            if not valid_rows:
                continue
            rows = torch.as_tensor(valid_rows, dtype=torch.long, device=device)
            ids = ids[rows]
            index = torch.stack([popped[j][0] for j in range(len(valid_rows))])
            log_prob = torch.stack([popped[j][1] for j in range(len(valid_rows))])
            privileged = torch.stack([popped[j][2] for j in range(len(valid_rows))])
            stance_xy = torch.stack([popped[j][3] for j in range(len(valid_rows))])
            n = ids.numel()

            success = transitions["diagnostics"]["success"][rows]
            fall = transitions["diagnostics"]["fall"][rows]
            collision = transitions["diagnostics"]["collision"][rows]
            support = transitions["diagnostics"]["support_fraction"][rows]

            swing_now = env.sampler.swing_foot[ids]
            stance_now = 1 - swing_now
            row = torch.arange(n, device=device)
            stance_now_xy = env.foot_positions[ids][row, stance_now, :2]
            d_t = planner.geodesic_distance(env, ids, stance_xy)
            d_next = planner.geodesic_distance(env, ids, stance_now_xy)
            r_progress = d_t - args.gamma * d_next
            r_progress = torch.where(success, d_t, r_progress)
            r_progress = torch.where(fall, torch.zeros_like(r_progress),
                                     r_progress)
            unsupported = torch.where(fall, torch.zeros_like(support),
                                      1.0 - support)
            reward = (args.reward_progress * r_progress
                      + args.reward_goal * success.float()
                      + args.reward_collision * collision.float()
                      + args.reward_fall * fall.float()
                      + args.reward_off_support * unsupported)

            done = transitions["done"][rows].bool()
            timeout = transitions["next_physics"]["timeout"][rows].bool()
            for i in range(n):
                if not done[i]:
                    continue
                if success[i]:
                    episode_counts["success"] += 1
                elif timeout[i]:
                    episode_counts["timeout"] += 1
                else:
                    episode_counts["fall"] += 1

            buffer.push(env_ids=ids,
                        privileged=privileged,
                        depth=transitions["depth"][rows].clone(),
                        proprio=transitions["proprio"][rows].clone(),
                        action=index,
                        log_prob=log_prob,
                        reward=reward,
                        done=done)

    @torch.no_grad()
    def gae_returns():
        privileged = torch.cat(buffer.privileged, dim=0)
        values = critic(privileged)
        env_ids = torch.cat(buffer.env_ids, dim=0)
        rewards = torch.cat(buffer.rewards, dim=0)
        dones = torch.cat(buffer.dones, dim=0)

        groups = {}
        for i, e in enumerate(env_ids.cpu().tolist()):
            groups.setdefault(e, []).append(i)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        for _, indices in groups.items():
            gae = 0.0
            for j in reversed(range(len(indices))):
                i = indices[j]
                next_value = 0.0 if dones[i] else float(values[indices[j + 1]]
                                                        if j + 1 < len(indices)
                                                        else 0.0)
                delta = rewards[i] + args.gamma * next_value - values[i]
                gae = delta + args.gamma * args.lam * gae
                advantages[i] = gae
                returns[i] = gae + values[i]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return (privileged, torch.cat(buffer.depth, dim=0),
                torch.cat(buffer.proprio, dim=0),
                torch.cat(buffer.actions, dim=0),
                torch.cat(buffer.log_probs, dim=0),
                advantages, returns)

    def ppo_update(data, update):
        privileged, depth, proprio, actions, old_log_probs, advantages, returns = data
        size = len(advantages)
        total_loss = 0.0
        for _ in range(args.update_epochs):
            perm = torch.randperm(size, device=device)
            for start in range(0, size, args.batch_size):
                idx = perm[start:start + args.batch_size]
                logits = policy_logits(depth[idx], proprio[idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.clip_epsilon,
                                    1 + args.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = critic(privileged[idx])
                value_loss = F.mse_loss(values, returns[idx])

                loss = (policy_loss + args.value_coef * value_loss
                        - args.entropy_coef * entropy)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    args.max_grad_norm)
                optimizer.step()
                total_loss += float(loss)
        return total_loss

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for update in range(args.total_updates):
        collect_rollout(args.rollout_ticks)
        if buffer.size() == 0:
            print(f"update {update}: empty buffer; skipping")
            continue
        data = gae_returns()
        loss = ppo_update(data, update)

        total = sum(episode_counts.values())
        success_rate = episode_counts["success"] / max(total, 1)
        fall_rate = episode_counts["fall"] / max(total, 1)
        record = {
            "update": update,
            "loss": loss,
            "buffer_size": buffer.size(),
            "episodes": total,
            "success_rate": success_rate,
            "fall_rate": fall_rate,
        }
        history.append(record)
        print(f"update {update:3d} loss {loss:.4f} "
              f"episodes {total} success {success_rate:.1%} "
              f"fall {fall_rate:.1%}", flush=True)

        if (update + 1) % args.checkpoint_every == 0 or update == args.total_updates - 1:
            torch.save({"actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "candidates_levels": candidates_levels.cpu(),
                        "num_candidates": num_candidates,
                        "feature_dim": feature_dim,
                        "gru_hidden": gru_hidden,
                        "update": update,
                        "config": vars(args)},
                       output / f"ppo_{update + 1}.pt")
    with open(output / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save({"actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "candidates_levels": candidates_levels.cpu(),
                "num_candidates": num_candidates,
                "feature_dim": feature_dim,
                "gru_hidden": gru_hidden,
                "update": args.total_updates,
                "config": vars(args)},
               output / "ppo_final.pt")
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
