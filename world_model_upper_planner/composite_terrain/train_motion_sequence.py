"""Train physical multi-step predictions through one observed static map cache."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from cgowm import CandidateGroundedWorldModel, ModelConfig
from .anchored import MotionModel, compose_pose, crop_map
from cgowm.data import load_arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=128)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(9702)
    rng = np.random.default_rng(9702); start = time.perf_counter()
    ck = torch.load(a.init, map_location='cuda:0', weights_only=False)
    world_path = Path(ck['world_checkpoint'])
    if hashlib.sha256(world_path.read_bytes()).hexdigest() != ck['world_sha256']:
        raise ValueError('changed frozen encoder')
    meta = json.loads((a.dataset.parent / 'metrics.json').read_text())
    if meta['lower_sha256'] != ck['lower_sha256']:
        raise ValueError('changed lower controller')
    wc = torch.load(world_path, map_location='cuda:0', weights_only=False)
    world = CandidateGroundedWorldModel(wc['candidates'], ModelConfig(**wc['model_config'])).cuda()
    world.load_state_dict(wc['model']); world.eval().requires_grad_(False)
    model = MotionModel(ck['latent_dim'], ck['proprio_dim']).cuda()
    model.load_state_dict(ck['motion_model'])
    source = load_arrays(a.dataset) if a.dataset.is_dir() else np.load(a.dataset)
    ids = {k: source[k] for k in ['env_id', 'episode_id', 'option_index', 'done', 'duration']}
    lookup = {tuple(map(int, k)): i for i, k in enumerate(zip(
        ids['env_id'], ids['episode_id'], ids['option_index']))}
    live = ~ids['done'].astype(bool) & (ids['duration'] >= 15)
    sequences = []
    for (env, episode, option), i in lookup.items():
        rows = [lookup.get((env, episode, option + s), -1) for s in range(3)]
        if min(rows) >= 0 and live[rows].all():
            sequences.append(rows)
    sequences = np.asarray(sequences, dtype=np.int64)
    envs = np.unique(ids['env_id']); np.random.default_rng(9701).shuffle(envs)
    heldout = np.isin(ids['env_id'][sequences[:, 0]], envs[:max(1, len(envs)//5)])
    train, val = sequences[~heldout], sequences[heldout]
    if len(train) == 0 or len(val) == 0:
        raise ValueError('need nonterminal three-step sequences in both map splits')
    tensors = {k: torch.as_tensor(source[k], device='cuda:0', dtype=torch.float32)
               for k in ['proprio', 'next_proprio', 'motion', 'action']}
    # Keep compact maps resident on the GPU; convert only the selected batch.
    cache = torch.as_tensor(source['panorama'], device='cuda:0', dtype=torch.uint8)
    depth = source['depth']; latent = []
    with torch.no_grad():
        for offset in range(0, len(depth), 512):
            image = torch.as_tensor(depth[offset:offset+512], device='cuda:0').float()/255
            latent.append(world.encode(image, tensors['proprio'][offset:offset+512]))
    latent = torch.cat(latent); del depth
    validation_rows = val[rng.choice(len(val), min(512, len(val)), replace=False)]

    def rollout(rows, training):
        rows = torch.as_tensor(rows, device='cuda:0')
        prop = tensors['proprio'][rows[:, 0]]
        initial_cache = cache[rows[:, 0]].float()/255
        pose = prop.new_zeros(len(rows), 4); truth = pose.clone()
        loss = prop.new_zeros(()); reports = []
        for s in range(3):
            feature = latent[rows[:, 0]] if s == 0 else world.encode(crop_map(initial_cache, pose)[0], prop)
            motion, states = model(feature, prop, tensors['action'][rows[:, s]])
            predicted_poses = compose_pose(pose[None].expand_as(motion), motion)
            truth = compose_pose(truth, tensors['motion'][rows[:, s]])
            target_state = tensors['next_proprio'][rows[:, s]][None].expand_as(states)
            loss = loss + F.smooth_l1_loss(predicted_poses / model.motion_scale,
                                         (truth / model.motion_scale)[None].expand_as(predicted_poses))
            loss = loss + .25 * F.smooth_l1_loss(states, target_state)
            loss = loss + .25 * F.smooth_l1_loss(states[..., 22:28]/.05, target_state[..., 22:28]/.05)
            pose = predicted_poses.mean(0); prop = states.mean(0)
            if not training:
                reports.append(dict(horizon=s+1, pose_mae=(pose-truth).abs().mean(0).tolist(),
                    feet_state_mae_m=float((prop[:,22:28]-target_state[0,:,22:28]).abs().mean())))
        return loss / 3, reports

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with torch.no_grad():
        _, before = rollout(validation_rows, False)
    # Include the unmodified checkpoint among candidates; no forced promotion.
    def score(report):
        return float(np.mean(np.array(report[-1]['pose_mae']) / np.array([.25,.10,.15,.04])))
    best = score(before)
    initial = dict(ck, sequence_updates=0, sequence_validation=before)
    torch.save(initial, a.output / 'model_best.pt')
    history = []
    for update in range(1, a.updates+1):
        loss, _ = rollout(train[rng.choice(len(train), a.batch_size)], True)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
        if update % 100 == 0 or update == a.updates:
            with torch.no_grad():
                _, report = rollout(validation_rows, False)
            record = dict(update=update, validation=report, score=score(report),
                          wall_seconds=time.perf_counter()-start)
            history.append(record)
            with (a.output/'progress.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            state = dict(ck, motion_model=model.state_dict(), sequence_updates=update,
                sequence_validation=report, sequence_dataset=str(a.dataset), sequence_init=str(a.init))
            torch.save(state, a.output/f'model_{update}.pt')
            if record['score'] < best:
                best = record['score']; torch.save(state, a.output/'model_best.pt')
            print(json.dumps(record), flush=True)
    selected = torch.load(a.output/'model_best.pt', weights_only=False)
    summary = dict(train_sequences=len(train), validation_sequences=len(val),
        evaluated_sequences=len(validation_rows), before=before,
        after=selected['sequence_validation'], selected_updates=selected['sequence_updates'],
        updates=a.updates, new_environment_interactions=0,
        world_sha256=ck['world_sha256'], lower_sha256=ck['lower_sha256'],
        wall_seconds=time.perf_counter()-start)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
