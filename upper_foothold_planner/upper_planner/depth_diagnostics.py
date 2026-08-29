"""Save real and model-predicted depth sequences in a directly inspectable form."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def depth_prediction_sequence(model, batch, sample_index=0):
    """Return real/predicted proximity images at the decoder output size."""
    depth = batch["depth"]
    next_depth = batch["next_depth"]
    proprio = batch["proprio"]
    action = batch["action"]
    if depth.ndim == 4:
        depth = depth[:, None]
        next_depth = next_depth[:, None]
        proprio = proprio[:, None]
        action = action[:, None]
    sample = int(sample_index)
    latent = model.encode(depth[sample:sample + 1, 0], proprio[sample:sample + 1, 0])
    first_prediction = model.reconstruct_depth(latent)[0, 0]
    output_shape = first_prediction.shape[-2:]
    real_full = [depth[sample, 0, 0]]
    real_small = [F.adaptive_avg_pool2d(
        depth[sample:sample + 1, 0], output_shape)[0, 0]]
    predicted = [first_prediction]
    for step in range(action.shape[1]):
        latent = model.next(latent, action[sample:sample + 1, step])
        predicted.append(model.reconstruct_depth(latent)[0, 0])
        real_full.append(next_depth[sample, step, 0])
        real_small.append(F.adaptive_avg_pool2d(
            next_depth[sample:sample + 1, step], output_shape)[0, 0])
    real_full = torch.stack(real_full).cpu().numpy()
    real_small = torch.stack(real_small).cpu().numpy()
    predicted = torch.stack(predicted).cpu().numpy()
    absolute_error = np.abs(predicted - real_small)
    return {
        "real_full": real_full,
        "real_16": real_small,
        "predicted_16": predicted,
        "absolute_error_16": absolute_error,
    }


def _stats(images):
    return {
        "min": float(images.min()),
        "max": float(images.max()),
        "mean": float(images.mean()),
        "std": float(images.std()),
        "zero_fraction": float((images <= 1.0 / 255.0).mean()),
        "near_fraction": float((images >= 254.0 / 255.0).mean()),
    }


def save_depth_prediction(path, arrays):
    """Write PNG, NPZ and JSON. Pixel 1 means near; 0 means far/no hit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = arrays["real_16"].shape[0]
    figure, axes = plt.subplots(3, columns, figsize=(2.2 * columns, 6.2), squeeze=False)
    rows = (
        ("real depth", arrays["real_16"], "viridis", 0.0, 1.0),
        ("predicted depth", arrays["predicted_16"], "viridis", 0.0, 1.0),
        ("absolute error", arrays["absolute_error_16"], "magma", 0.0, 0.5),
    )
    for row_id, (name, images, cmap, vmin, vmax) in enumerate(rows):
        for step in range(columns):
            axes[row_id, step].imshow(images[step], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_id, step].set_axis_off()
            if row_id == 0:
                axes[row_id, step].set_title("t+{}".format(step))
            if step == 0:
                axes[row_id, step].set_ylabel(name)
                axes[row_id, step].set_axis_on()
                axes[row_id, step].set_xticks([])
                axes[row_id, step].set_yticks([])
    figure.suptitle("normalized proximity depth: 1=near, 0=far/no hit")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    stem = path.with_suffix("")
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    summary = {
        "real_depth": _stats(arrays["real_full"]),
        "prediction": _stats(arrays["predicted_16"]),
        "mae_by_horizon": arrays["absolute_error_16"].mean((1, 2)).tolist(),
        "real_change_from_t0_by_horizon": np.abs(
            arrays["real_16"] - arrays["real_16"][0]).mean((1, 2)).tolist(),
    }
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def depth_prediction_metrics(model, batch, input_mode="normal"):
    """Evaluate a batch of real sequences, including terrain-boundary pixels."""
    depth = batch["depth"]
    next_depth = batch["next_depth"]
    proprio = batch["proprio"]
    action = batch["action"]
    if depth.ndim != 5:
        raise ValueError("depth_prediction_metrics requires sequence batches")
    initial_depth = depth[:, 0]
    if input_mode == "shuffled":
        initial_depth = initial_depth[torch.roll(
            torch.arange(len(initial_depth), device=initial_depth.device), 1)]
    elif input_mode == "zero":
        initial_depth = torch.zeros_like(initial_depth)
    elif input_mode != "normal":
        raise ValueError("input_mode must be normal, shuffled, or zero")
    latent = model.encode(initial_depth, proprio[:, 0])
    predictions = [model.reconstruct_depth(latent)]
    targets = [F.adaptive_avg_pool2d(depth[:, 0], predictions[0].shape[-2:])]
    for step in range(action.shape[1]):
        latent = model.next(latent, action[:, step])
        predictions.append(model.reconstruct_depth(latent))
        targets.append(F.adaptive_avg_pool2d(
            next_depth[:, step], predictions[-1].shape[-2:]))
    predictions = torch.stack(predictions, dim=1)
    targets = torch.stack(targets, dim=1)
    error = (predictions - targets).abs()
    horizontal = F.pad((targets[..., 1:] - targets[..., :-1]).abs(), (0, 1, 0, 0))
    vertical = F.pad((targets[..., 1:, :] - targets[..., :-1, :]).abs(), (0, 0, 0, 1))
    boundary = torch.maximum(horizontal, vertical) > 0.05
    persistence = targets[:, :1].expand_as(targets)
    def masked_mean(values, mask):
        return float(values[mask].mean()) if mask.any() else float("nan")
    return {
        "mae_by_horizon": error.mean((0, 2, 3, 4)).cpu().tolist(),
        "boundary_mae_by_horizon": [
            masked_mean(error[:, step], boundary[:, step])
            for step in range(error.shape[1])],
        "nonboundary_mae_by_horizon": [
            masked_mean(error[:, step], ~boundary[:, step])
            for step in range(error.shape[1])],
        "persistence_mae_by_horizon": (
            persistence - targets).abs().mean((0, 2, 3, 4)).cpu().tolist(),
        "predicted_change_by_horizon": (
            predictions - predictions[:, :1]).abs().mean((0, 2, 3, 4)).cpu().tolist(),
        "real_change_by_horizon": (
            targets - targets[:, :1]).abs().mean((0, 2, 3, 4)).cpu().tolist(),
        "boundary_fraction": float(boundary.float().mean()),
    }
