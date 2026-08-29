"""Minimal independent world-model ensemble utilities."""

import torch

from .world_model import LatentWorldModel


def make_ensemble(size, latent_dim, hidden_dim, device, initial_state=None):
    """Build independent models; an optional state is only an initialization."""
    models = []
    for _ in range(int(size)):
        model = LatentWorldModel(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        if initial_state is not None:
            model.load_state_dict(initial_state)
        models.append(model)
    return models


def load_ensemble_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    states = checkpoint.get("models")
    if states is None:
        states = [checkpoint["model"]]
    cfg = checkpoint["config"]
    models = make_ensemble(
        len(states), cfg["upper_observation"]["latent_dim"],
        cfg["model"]["hidden_dim"], device)
    for model, state in zip(models, states):
        model.load_state_dict(state)
        model.eval()
    return models, checkpoint


@torch.no_grad()
def one_step_predictions(models, depth, proprio, action, reward_cfg, reward_scale):
    """Return member-wise observable predictions, shaped (members,batch,...)."""
    output = {name: [] for name in (
        "reward", "progress", "goal_probability", "collision_probability",
        "fall_probability", "off_support_probability", "next_depth")}
    for model in models:
        latent = model.encode(depth, proprio)
        component = model.predict_task_components(latent, action)
        output["reward"].append(model.predict_task_reward(
            latent, action, reward_cfg, reward_scale))
        output["progress"].append(component["progress"])
        output["goal_probability"].append(torch.sigmoid(component["goal_logit"]))
        output["collision_probability"].append(
            torch.sigmoid(component["collision_logit"]))
        output["fall_probability"].append(torch.sigmoid(component["fall_logit"]))
        output["off_support_probability"].append(
            torch.sigmoid(component["off_support_logit"]))
        output["next_depth"].append(model.reconstruct_depth(model.next(latent, action)))
    return {name: torch.stack(value) for name, value in output.items()}
