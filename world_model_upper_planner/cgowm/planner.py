"""Uncertainty-aware discrete option planning.

There is no continuous CEM loop. Every expanded action belongs to the fixed
reachable candidate set, and ensemble disagreement is penalized explicitly.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 3
    beam_width: int = 32
    proposals_per_beam: int = 12
    discount: float = 0.99
    uncertainty_weight: float = 0.5
    fall_weight: float = 5.0
    collision_weight: float = 2.0
    support_weight: float = 1.0
    reward_weight: float = 0.0
    progress_weight: float = 10.0
    terminal_value_weight: float = 0.1
    feasibility_threshold: float = 0.0


class BeamPlanner:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or PlannerConfig()

    @torch.no_grad()
    def plan(self, latent, static_candidate_mask=None):
        """Return candidate indices and diagnostics for a batch of latents."""
        batch = latent.shape[0]
        chosen, diagnostics = [], []
        for row in range(batch):
            mask = None if static_candidate_mask is None else static_candidate_mask[row]
            index, info = self._plan_one(latent[row:row + 1], mask)
            chosen.append(index)
            diagnostics.append(info)
        return torch.stack(chosen), diagnostics

    def _plan_one(self, initial, static_mask):
        # Each beam stores latent, discounted return, first action and predicted
        # probability that the option sequence is still continuing.
        beams = [(initial, initial.new_zeros(()), None, initial.new_ones(()))]
        expanded_count = 0
        for step in range(self.config.horizon):
            expanded = []
            for latent, accumulated, first, alive in beams:
                prediction = self.model.predict_candidates(latent)
                logits = prediction["policy_logits"][0]
                if static_mask is not None:
                    logits = logits.masked_fill(~static_mask.bool(), -torch.inf)
                available = torch.isfinite(logits).nonzero(as_tuple=False).flatten()
                if not available.numel():
                    continue
                proposal_count = min(self.config.proposals_per_beam,
                                     available.numel())
                local_top = torch.topk(logits[available], proposal_count).indices
                indices = available[local_top]
                q = prediction["q"][:, 0, indices]
                uncertainty = q.std(dim=0, unbiased=False)
                fall = torch.sigmoid(prediction["fall_logit"][0, indices])
                collision = torch.sigmoid(prediction["collision_logit"][0, indices])
                support = prediction["support"][0, indices]
                reward = prediction["reward"][0, indices]
                continuation = torch.sigmoid(
                    prediction["continuation_logit"][0, indices])
                score = (reward
                         - self.config.uncertainty_weight * uncertainty
                         - self.config.fall_weight * fall
                         - self.config.collision_weight * collision
                         - self.config.support_weight * (1.0 - support))
                feasible = support >= self.config.feasibility_threshold
                for local in feasible.nonzero(as_tuple=False).flatten().tolist():
                    candidate_index = indices[local]
                    action = self.model.candidates[candidate_index].unsqueeze(0)
                    next_latent = self.model.next(latent, action)
                    total = accumulated + (
                        self.config.discount ** step) * alive * score[local]
                    first_index = candidate_index if first is None else first
                    expanded.append((next_latent, total, first_index,
                                     alive * continuation[local]))
                expanded_count += int(feasible.sum())
            if not expanded:
                # Static reachable candidates should make this exceptional.
                logits = self.model.predict_candidates(initial)["policy_logits"][0]
                if static_mask is not None:
                    logits = logits.masked_fill(~static_mask.bool(), -torch.inf)
                fallback = torch.argmax(logits)
                return fallback, {"expanded": expanded_count, "fallback": True}
            if step == self.config.horizon - 1:
                scored = []
                for item in expanded:
                    latent, total, _, alive = item
                    terminal = self.model.predict_candidates(latent)["q"].min(0).values.max()
                    scored.append(total + (self.config.discount ** (step + 1))
                                  * alive * terminal)
                order = torch.argsort(torch.stack(scored), descending=True)
            else:
                order = torch.argsort(
                    torch.stack([item[1] for item in expanded]), descending=True)
            beams = [expanded[int(i)] for i in order[:self.config.beam_width].tolist()]
        return beams[0][2], {"expanded": expanded_count, "fallback": False}


class VectorizedBeamPlanner:
    """GPU-batched beam search for asynchronous upper-decision groups."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or PlannerConfig()

    @torch.no_grad()
    def plan(self, initial, static_candidate_mask=None):
        batch, latent_dim = initial.shape
        candidates = self.model.candidates
        beam_latent = initial[:, None]
        beam_return = initial.new_zeros(batch, 1)
        beam_alive = initial.new_ones(batch, 1)
        beam_first = torch.full(
            (batch, 1), -1, dtype=torch.long, device=initial.device)
        expanded = 0
        for step in range(self.config.horizon):
            beams = beam_latent.shape[1]
            flat = beam_latent.reshape(batch * beams, latent_dim)
            prediction = self.model.predict_candidates(flat)
            logits = prediction["policy_logits"].view(batch, beams, -1)
            if static_candidate_mask is not None:
                logits = logits.masked_fill(
                    ~static_candidate_mask[:, None].bool(), -torch.inf)
            proposals = min(self.config.proposals_per_beam, logits.shape[-1])
            index = torch.topk(logits, proposals, dim=-1).indices

            def gather(value):
                value = value.view(batch, beams, -1)
                return torch.gather(value, -1, index)

            q = prediction["q"].view(
                prediction["q"].shape[0], batch, beams, -1)
            q_selected = torch.gather(
                q, -1, index[None].expand(q.shape[0], -1, -1, -1))
            uncertainty = q_selected.std(0, unbiased=False)
            fall = torch.sigmoid(gather(prediction["fall_logit"]))
            collision = torch.sigmoid(gather(prediction["collision_logit"]))
            support = gather(prediction["support"])
            progress = gather(prediction["progress"])
            reward = gather(prediction["reward"])
            continuation = torch.sigmoid(gather(
                prediction["continuation_logit"]))
            score = (
                self.config.reward_weight * reward
                + self.config.progress_weight * progress
                - self.config.uncertainty_weight * uncertainty
                - self.config.fall_weight * fall
                - self.config.collision_weight * collision
                - self.config.support_weight * (1.0 - support))
            # topk has a fixed width for GPU efficiency.  When a static mask
            # contains fewer candidates than that width, topk pads with masked
            # entries; keep those entries impossible during beam selection.
            proposal_allowed = torch.gather(torch.isfinite(logits), -1, index)
            score = score.masked_fill(~proposal_allowed, -torch.inf)
            score = score.masked_fill(
                support < self.config.feasibility_threshold, -torch.inf)
            total = (beam_return[:, :, None]
                     + (self.config.discount ** step)
                     * beam_alive[:, :, None] * score)
            alive = beam_alive[:, :, None] * continuation
            actions = candidates[index.reshape(-1)].view(
                batch * beams * proposals, -1)
            repeated = flat[:, None].expand(-1, proposals, -1).reshape(
                batch * beams * proposals, latent_dim)
            next_latent = self.model.next(repeated, actions).view(
                batch, beams * proposals, latent_dim)
            total = total.reshape(batch, beams * proposals)
            alive = alive.reshape(batch, beams * proposals)
            if step == 0:
                first = index.reshape(batch, beams * proposals)
            else:
                first = beam_first[:, :, None].expand(
                    -1, -1, proposals).reshape(batch, beams * proposals)
            expanded += batch * beams * proposals
            keep = min(self.config.beam_width, total.shape[1])
            values, order = torch.topk(total, keep, dim=-1)
            row = torch.arange(batch, device=initial.device)[:, None]
            beam_latent = next_latent[row, order]
            beam_return = values
            beam_alive = alive[row, order]
            beam_first = first[row, order]
        if self.config.terminal_value_weight:
            flat = beam_latent.reshape(-1, latent_dim)
            terminal = self.model.predict_candidates(flat)["q"].min(0).values.max(-1).values
            terminal = terminal.view_as(beam_return)
            beam_return = (beam_return
                           + self.config.terminal_value_weight
                           * (self.config.discount ** self.config.horizon)
                           * beam_alive * terminal)
        best = beam_return.argmax(-1)
        row = torch.arange(batch, device=initial.device)
        return beam_first[row, best], {
            "expanded": expanded,
            "best_score": beam_return[row, best],
        }
