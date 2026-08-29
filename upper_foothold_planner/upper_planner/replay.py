"""Compact transition replay; depth is stored as uint8 proximity images."""

import numpy as np
import torch
from collections import deque

from .contracts import MACRO_STATE_DIM


class ReplayBuffer:
    def __init__(self, capacity, depth_shape=(1, 64, 64), proprio_dim=36, action_dim=3,
                 num_envs=None, return_horizon=5, gamma=0.99, reward_scale=10.0,
                 duration_aware_returns=False, nominal_option_ticks=25.0):
        self.capacity = int(capacity)
        self.depth = np.empty((capacity,) + depth_shape, dtype=np.uint8)
        self.next_depth = np.empty_like(self.depth)
        self.proprio = np.empty((capacity, proprio_dim), dtype=np.float32)
        self.next_proprio = np.empty_like(self.proprio)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.collision = np.empty(capacity, dtype=np.bool_)
        self.fall = np.empty(capacity, dtype=np.bool_)
        self.success = np.empty(capacity, dtype=np.bool_)
        self.off_support = np.empty(capacity, dtype=np.bool_)
        self.progress = np.empty(capacity, dtype=np.float32)
        self.heading_progress = np.empty(capacity, dtype=np.float32)
        self.collision_force = np.empty(capacity, dtype=np.float32)
        self.stability_margin = np.empty(capacity, dtype=np.float32)
        self.support_fraction = np.empty(capacity, dtype=np.float32)
        self.touchdown_error = np.empty(capacity, dtype=np.float32)
        self.option_duration_ticks = np.empty(capacity, dtype=np.float32)
        self.macro_state = np.zeros((capacity, MACRO_STATE_DIM), dtype=np.float32)
        self.scene_id = np.full(capacity, -1, dtype=np.int16)
        # Reserved for true same-state multi-action collections. A negative id
        # means that no counterfactual group is available for that transition.
        self.counterfactual_group = np.full(capacity, -1, dtype=np.int64)
        # True when the executed action was chosen by the online planner (used
        # as a policy-prior distillation target), False for random/stratified
        # or perturbed coverage actions.
        self.from_planner = np.zeros(capacity, dtype=np.bool_)
        self.done = np.empty(capacity, dtype=np.bool_)
        self.return_target = np.full(capacity, np.nan, dtype=np.float32)
        # Explicit per-environment temporal links. Batch insertion interleaves
        # environments, so adjacent ring-buffer indices are not a trajectory.
        self.env_id = np.full(capacity, -1, dtype=np.int32)
        self.env_step = np.full(capacity, -1, dtype=np.int64)
        self.next_index = np.full(capacity, -1, dtype=np.int64)
        self.size = 0
        self.position = 0
        self.return_horizon = int(return_horizon)
        self.gamma = float(gamma)
        self.reward_scale = float(reward_scale)
        self.duration_aware_returns = bool(duration_aware_returns)
        self.nominal_option_ticks = float(nominal_option_ticks)
        self.pending = ([deque() for _ in range(int(num_envs))]
                        if num_envs is not None else None)
        self.last_index = (np.full(int(num_envs), -1, dtype=np.int64)
                           if num_envs is not None else None)
        self.env_step_counter = (np.zeros(int(num_envs), dtype=np.int64)
                                 if num_envs is not None else None)
        self.last_new_valid = 0
        self._sequence_cache = {}
        self._padded_sequence_cache = {}
        # Version-based cache: the O(size) sequence tables are rebuilt only
        # every ``_sequence_refresh_interval`` new transitions, not on every
        # batch add.  A small staleness window is harmless for a world model.
        self._data_version = 0
        self._sequence_refresh_interval = 2000

    @property
    def valid_indices(self):
        return np.flatnonzero(np.isfinite(self.return_target[:self.size]))

    @property
    def valid_size(self):
        return int(np.isfinite(self.return_target[:self.size]).sum())

    def _write_oldest_return(self, queue):
        total = 0.0
        elapsed_options = 0.0
        for step, index in enumerate(list(queue)[:self.return_horizon]):
            exponent = elapsed_options if self.duration_aware_returns else float(step)
            total += (self.gamma ** exponent) * float(self.reward[index])
            if self.done[index]:
                break
            elapsed_options += (float(self.option_duration_ticks[index])
                                / self.nominal_option_ticks)
        index = queue.popleft()
        was_invalid = not np.isfinite(self.return_target[index])
        self.return_target[index] = total / self.reward_scale
        return int(was_invalid)

    def add_transition_batch(self, transition):
        if transition is None:
            return
        self._data_version += int(transition["ids"].numel())
        count = transition["ids"].numel()
        indices = (np.arange(count) + self.position) % self.capacity
        if self.last_index is not None:
            for index in indices:
                old_env = int(self.env_id[index])
                if old_env >= 0 and self.last_index[old_env] == index:
                    self.last_index[old_env] = -1
        depth = transition["depth"].detach().cpu().numpy()
        next_depth = transition["next_depth"].detach().cpu().numpy()
        self.depth[indices] = np.uint8(np.clip(depth, 0.0, 1.0) * 255.0 + 0.5)
        self.next_depth[indices] = np.uint8(np.clip(next_depth, 0.0, 1.0) * 255.0 + 0.5)
        self.proprio[indices] = transition["proprio"].detach().cpu().numpy()
        self.next_proprio[indices] = transition["next_proprio"].detach().cpu().numpy()
        self.action[indices] = transition["action"].detach().cpu().numpy()
        self.reward[indices] = transition["reward"].detach().cpu().numpy()
        self.collision[indices] = transition["diagnostics"]["collision"].detach().cpu().numpy()
        self.fall[indices] = transition["diagnostics"]["fall"].detach().cpu().numpy()
        self.success[indices] = transition["diagnostics"]["success"].detach().cpu().numpy()
        self.off_support[indices] = transition["diagnostics"]["off_support"].detach().cpu().numpy()
        self.progress[indices] = transition["terms"]["progress"].detach().cpu().numpy()
        diagnostics = transition["diagnostics"]
        zeros = torch.zeros(count, device=transition["reward"].device)
        ones = torch.ones(count, device=transition["reward"].device)
        self.heading_progress[indices] = diagnostics.get(
            "heading_progress_rad", zeros).detach().cpu().numpy()
        self.collision_force[indices] = diagnostics.get(
            "collision_force_n", zeros).detach().cpu().numpy()
        self.stability_margin[indices] = diagnostics.get(
            "stability_margin", zeros).detach().cpu().numpy()
        self.support_fraction[indices] = diagnostics.get(
            "support_fraction", ones).detach().cpu().numpy()
        self.touchdown_error[indices] = diagnostics.get(
            "touchdown_error_m", zeros).detach().cpu().numpy()
        self.option_duration_ticks[indices] = diagnostics.get(
            "option_duration_ticks", torch.full_like(zeros, 25.0)
        ).detach().cpu().numpy()
        macro_state = diagnostics.get("macro_state")
        if macro_state is not None:
            self.macro_state[indices] = torch.as_tensor(
                macro_state).detach().cpu().numpy()
        else:
            self.macro_state[indices] = 0.0
        scene = transition.get("scene_id")
        if scene is not None:
            self.scene_id[indices] = torch.as_tensor(scene).detach().cpu().numpy()
        else:
            self.scene_id[indices] = -1
        group = transition.get("counterfactual_group")
        if group is not None:
            self.counterfactual_group[indices] = torch.as_tensor(
                group).detach().cpu().numpy()
        else:
            self.counterfactual_group[indices] = -1
        from_planner = transition.get("from_planner")
        if from_planner is not None:
            self.from_planner[indices] = torch.as_tensor(
                from_planner, dtype=torch.bool).detach().cpu().numpy()
        else:
            self.from_planner[indices] = False
        self.done[indices] = transition["done"].detach().cpu().numpy()
        self.return_target[indices] = np.nan
        self.next_index[indices] = -1
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.last_new_valid = 0
        if self.pending is None:
            self.return_target[indices] = self.reward[indices] / self.reward_scale
            self.last_new_valid = count
            return
        env_ids = transition["ids"].detach().cpu().numpy().astype(np.int64)
        for env_id, index in zip(env_ids, indices):
            previous = int(self.last_index[env_id])
            step = int(self.env_step_counter[env_id])
            self.env_id[index] = env_id
            self.env_step[index] = step
            if (previous >= 0 and self.env_id[previous] == env_id
                    and self.env_step[previous] == step - 1
                    and not self.done[previous]):
                self.next_index[previous] = int(index)
            self.last_index[env_id] = int(index)
            self.env_step_counter[env_id] += 1
            queue = self.pending[env_id]
            queue.append(int(index))
            if len(queue) >= self.return_horizon:
                self.last_new_valid += self._write_oldest_return(queue)
            if self.done[index]:
                while queue:
                    self.last_new_valid += self._write_oldest_return(queue)
                self.last_index[env_id] = -1

    def sample(self, batch_size, device, allowed_indices=None):
        valid = self.valid_indices
        if len(valid) < batch_size:
            raise ValueError("not enough replay samples")
        pool = valid if allowed_indices is None else np.asarray(allowed_indices)
        index = np.random.choice(pool, int(batch_size), replace=len(pool) < batch_size)
        tensor = lambda value, dtype=None: torch.as_tensor(value[index], dtype=dtype, device=device)
        return {
            "depth": tensor(self.depth, torch.float32) / 255.0,
            "next_depth": tensor(self.next_depth, torch.float32) / 255.0,
            "proprio": tensor(self.proprio, torch.float32),
            "next_proprio": tensor(self.next_proprio, torch.float32),
            "action": tensor(self.action, torch.float32),
            "reward": tensor(self.reward, torch.float32),
            "collision": tensor(self.collision, torch.float32),
            "fall": tensor(self.fall, torch.float32),
            "success": tensor(self.success, torch.float32),
            "off_support": tensor(self.off_support, torch.float32),
            "progress": tensor(self.progress, torch.float32),
            "heading_progress": tensor(self.heading_progress, torch.float32),
            "collision_force": tensor(self.collision_force, torch.float32),
            "stability_margin": tensor(self.stability_margin, torch.float32),
            "support_fraction": tensor(self.support_fraction, torch.float32),
            "touchdown_error": tensor(self.touchdown_error, torch.float32),
            "option_duration_ticks": tensor(
                self.option_duration_ticks, torch.float32),
            "macro_state": tensor(self.macro_state, torch.float32),
            "scene_id": tensor(self.scene_id, torch.long),
            "counterfactual_group": tensor(self.counterfactual_group, torch.long),
            "from_planner": tensor(self.from_planner, torch.bool),
            "done": tensor(self.done, torch.float32),
            "return_target": tensor(self.return_target, torch.float32),
        }

    def _cached_sequences(self, cache, horizon):
        entry = cache.get(horizon)
        if entry is not None and (
                self._data_version - entry[0] < self._sequence_refresh_interval):
            return entry[1]
        return None

    def sequence_indices(self, horizon):
        """Return valid linked index rows without crossing an episode boundary."""
        horizon = int(horizon)
        cached = self._cached_sequences(self._sequence_cache, horizon)
        if cached is not None:
            return cached
        valid = np.isfinite(self.return_target[:self.size])
        starts = np.flatnonzero(valid).astype(np.int64)
        columns = [starts]
        current = starts
        keep = np.ones(len(starts), dtype=bool)
        for _ in range(1, horizon):
            safe_current = np.clip(current, 0, max(self.size - 1, 0))
            nxt = self.next_index[safe_current]
            in_range = (nxt >= 0) & (nxt < self.size)
            safe_next = np.clip(nxt, 0, max(self.size - 1, 0))
            linked = (in_range & ~self.done[safe_current] & valid[safe_next]
                      & (self.env_id[safe_next] == self.env_id[safe_current])
                      & (self.env_step[safe_next]
                         == self.env_step[safe_current] + 1))
            keep &= linked
            current = safe_next
            columns.append(current)
        result = np.stack(columns, axis=1)[keep]
        self._sequence_cache[horizon] = (self._data_version, result)
        return result

    def padded_sequence_indices(self, horizon):
        """Linked rows with terminal suffix padding and an explicit valid mask.

        A terminal transition may occur at any horizon position. Rows cut short
        merely because newer replay data are unavailable are excluded; only a
        true terminal is allowed to end a sequence before ``horizon``.

        Vectorized: chains are built column-wise with numpy masks instead of a
        per-row Python loop, so it stays fast even at large replay capacities.
        """
        horizon = int(horizon)
        cached = self._cached_sequences(self._padded_sequence_cache, horizon)
        if cached is not None:
            return cached
        size = self.size
        valid_return = np.isfinite(self.return_target[:size])
        starts = np.flatnonzero(valid_return).astype(np.int64)
        if len(starts) == 0:
            result = (np.empty((0, horizon), dtype=np.int64),
                      np.empty((0, horizon), dtype=np.bool_))
            self._padded_sequence_cache[horizon] = (self._data_version, result)
            return result
        done = self.done[:size]
        columns = [starts]
        masks = [np.ones(len(starts), dtype=bool)]
        complete = np.ones(len(starts), dtype=bool)
        current = starts.copy()
        for _ in range(1, horizon):
            is_done = done[current]
            nxt = self.next_index[current]
            safe = np.clip(nxt, 0, size - 1)
            linked = ((nxt >= 0) & (nxt < size) & ~is_done
                      & valid_return[safe]
                      & (self.env_id[safe] == self.env_id[current])
                      & (self.env_step[safe] == self.env_step[current] + 1))
            complete &= (is_done | linked)
            current = np.where(linked, nxt, current)
            columns.append(current)
            masks.append(~is_done)
        rows = np.stack(columns, axis=1)
        mask = np.stack(masks, axis=1)
        result = (rows[complete], mask[complete])
        self._padded_sequence_cache[horizon] = (self._data_version, result)
        return result

    def _balanced_sequence_probabilities(self, rows, valid=None):
        """Inverse-frequency scene/event weights for sequence start rows."""
        start = rows[:, 0]
        weights = np.ones(len(rows), dtype=np.float64)
        scenes = self.scene_id[start]
        for scene in np.unique(scenes):
            if scene >= 0:
                mask = scenes == scene
                weights[mask] *= len(rows) / max(mask.sum(), 1)
        # Rare events are what define the feasibility boundary. Balance each
        # event independently but cap amplification to keep batches diverse.
        for values in (self.collision, self.fall, self.success, self.off_support):
            events = values[rows]
            if valid is not None:
                events = events & valid
            positive = events.any(axis=1)
            count = int(positive.sum())
            if 0 < count < len(rows):
                boost = min((len(rows) - count) / count, 20.0)
                weights[positive] *= boost
        weights = np.minimum(weights, np.percentile(weights, 99.0))
        return weights / weights.sum()

    def sample_sequence(self, batch_size, horizon, device, balanced=False,
                        allow_terminal_padding=False):
        if allow_terminal_padding:
            rows, valid = self.padded_sequence_indices(horizon)
        else:
            rows = self.sequence_indices(horizon)
            valid = np.ones_like(rows, dtype=np.bool_)
        if len(rows) < batch_size:
            raise ValueError("not enough replay sequences")
        if balanced:
            # Subset-then-balance: score only a candidate pool instead of the
            # full sequence table, keeping the cost O(batch) rather than
            # O(replay size) at large replay capacities.
            pool_size = int(min(len(rows), max(8 * batch_size, 1024)))
            pool_idx = np.random.choice(len(rows), pool_size, replace=False)
            pool_rows = rows[pool_idx]
            pool_valid = valid[pool_idx]
            probabilities = self._balanced_sequence_probabilities(
                pool_rows, pool_valid)
            chosen_in_pool = np.random.choice(
                pool_size, int(batch_size), replace=False, p=probabilities)
            chosen = pool_idx[chosen_in_pool]
        else:
            chosen = np.random.choice(
                len(rows), int(batch_size), replace=False)
        index = rows[chosen]
        valid = valid[chosen]
        tensor = lambda value, dtype=None: torch.as_tensor(
            value[index], dtype=dtype, device=device)
        return {
            "depth": tensor(self.depth, torch.float32) / 255.0,
            "next_depth": tensor(self.next_depth, torch.float32) / 255.0,
            "proprio": tensor(self.proprio, torch.float32),
            "next_proprio": tensor(self.next_proprio, torch.float32),
            "action": tensor(self.action, torch.float32),
            "reward": tensor(self.reward, torch.float32),
            "collision": tensor(self.collision, torch.float32),
            "fall": tensor(self.fall, torch.float32),
            "success": tensor(self.success, torch.float32),
            "off_support": tensor(self.off_support, torch.float32),
            "progress": tensor(self.progress, torch.float32),
            "heading_progress": tensor(self.heading_progress, torch.float32),
            "collision_force": tensor(self.collision_force, torch.float32),
            "stability_margin": tensor(self.stability_margin, torch.float32),
            "support_fraction": tensor(self.support_fraction, torch.float32),
            "touchdown_error": tensor(self.touchdown_error, torch.float32),
            "option_duration_ticks": tensor(
                self.option_duration_ticks, torch.float32),
            "macro_state": tensor(self.macro_state, torch.float32),
            "scene_id": tensor(self.scene_id, torch.long),
            "counterfactual_group": tensor(self.counterfactual_group, torch.long),
            "from_planner": tensor(self.from_planner, torch.bool),
            "done": tensor(self.done, torch.float32),
            "return_target": tensor(self.return_target, torch.float32),
            "valid": torch.as_tensor(valid, dtype=torch.float32, device=device),
        }
