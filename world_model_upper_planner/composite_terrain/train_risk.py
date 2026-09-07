"""Finite-horizon failure risk from observed continuations, including short terminals."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from cgowm import CandidateGroundedWorldModel, ModelConfig, OptionRiskCritic, RiskConfig


def risk_labels(data, horizon=3):
    """No reset crossing and no false safe labels at right-censored tails."""
    keys = list(zip(data['env_id'], data['episode_id'], data['option_index']))
    lookup = {tuple(map(int, key)): i for i, key in enumerate(keys)}
    labels = np.zeros((len(keys), 2), dtype=np.float32)
    known = np.zeros(len(keys), dtype=bool)
    for i, (env, episode, option) in enumerate(keys):
        for step in range(horizon):
            row = lookup.get((int(env), int(episode), int(option)+step))
            if row is None:
                break
            labels[i] = np.maximum(labels[i], [data['fall'][row], data['collision'][row]])
            if data['done'][row]:
                # Success/fall ends the task. An administrative time limit is
                # still censored, not evidence that the future would be safe.
                known[i] = bool(data['fall'][row] or data['success'][row])
                break
            if step == horizon-1:
                known[i] = True
                break
    return labels, known


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--world_checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=3000)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(9710)
    rng = np.random.default_rng(9710); start = time.perf_counter()
    source = np.load(a.dataset)
    data = {k: source[k] for k in ['env_id','episode_id','option_index','fall','collision','done','success']}
    labels, known = risk_labels(data)
    envs = np.unique(data['env_id']); np.random.default_rng(9701).shuffle(envs)
    heldout = np.isin(data['env_id'], envs[:max(1,len(envs)//5)])
    train = np.flatnonzero(known & ~heldout); val = np.flatnonzero(known & heldout)
    if not len(train) or not len(val):
        raise ValueError('need known risk windows in both layout splits')
    ck = torch.load(a.world_checkpoint, map_location='cuda:0', weights_only=False)
    meta = json.loads((a.dataset.parent/'metrics.json').read_text())
    if ck['lower_sha256'] != meta['lower_sha256']:
        raise ValueError('lower mismatch')
    world = CandidateGroundedWorldModel(ck['candidates'], ModelConfig(**ck['model_config'])).cuda()
    world.load_state_dict(ck['model']); world.eval().requires_grad_(False)
    prop = torch.as_tensor(source['proprio'], device='cuda:0', dtype=torch.float32)
    actions = torch.as_tensor(source['action'], device='cuda:0', dtype=torch.float32)
    target = torch.as_tensor(labels, device='cuda:0')
    depth = source['depth']; features = []
    with torch.no_grad():
        for offset in range(0,len(prop),512):
            image = torch.as_tensor(depth[offset:offset+512],device='cuda:0').float()/255
            features.append(torch.cat((world.encode(image,prop[offset:offset+512]), prop[offset:offset+512]),-1))
    features = torch.cat(features); del depth
    cfg = RiskConfig(latent_dim=features.shape[-1], action_dim=4)
    risk = OptionRiskCritic(cfg).cuda()
    optimizer = torch.optim.Adam(risk.parameters(),lr=3e-4)
    best = float('inf')
    validation = val[rng.choice(len(val),min(16384,len(val)),replace=False)]

    @torch.no_grad()
    def validate():
        probabilities = []
        for offset in range(0,len(validation),2048):
            ids = torch.as_tensor(validation[offset:offset+2048],device='cuda:0')
            pred = risk.predict_action(features[ids],actions[ids])
            probabilities.append(torch.stack([pred[k].sigmoid().mean(0) for k in ['fall_logit','collision_logit']],-1))
        probabilities = torch.cat(probabilities); truth = target[validation]
        stats = {}
        for k,name in enumerate(['fall','collision']):
            prob = probabilities[:,k]; real = truth[:,k]; rate = real.mean()
            top = prob.topk(max(1,len(prob)//10)).indices
            stats[name] = dict(rate=float(rate),probability=float(prob.mean()),
                brier=float((prob-real).square().mean()),constant_brier=float(rate*(1-rate)),
                top_decile_event_rate=float(real[top].mean()))
        allrisk = risk.predict_candidates(features[validation[:512]],world.candidates)
        stats['fall_action_std'] = float(allrisk['fall_logit'].sigmoid().mean(0).std(-1,unbiased=False).mean())
        return stats

    for update in range(1,a.updates+1):
        ids = torch.as_tensor(rng.choice(train,2048),device='cuda:0')
        prediction = risk.predict_action(features[ids],actions[ids])
        logits = torch.stack([prediction[k] for k in ['fall_logit','collision_logit']],-1)
        losses = F.binary_cross_entropy_with_logits(logits,target[ids][None].expand_as(logits),reduction='none').mean(-1)
        mask = (torch.rand_like(losses)>.2).float()
        loss = (losses*mask).sum()/mask.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(risk.parameters(),10); optimizer.step()
        if update % 500 == 0 or update == a.updates:
            stats = validate(); score = sum(stats[k]['brier'] for k in ['fall','collision'])
            record = dict(update=update,validation=stats,wall_seconds=time.perf_counter()-start)
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            state = dict(risk=risk.state_dict(),risk_config=asdict(cfg),horizon=3,
                world_sha256=hashlib.sha256(a.world_checkpoint.read_bytes()).hexdigest(),
                lower_sha256=meta['lower_sha256'],world_checkpoint=str(a.world_checkpoint),
                dataset=str(a.dataset),update=update,validation=stats)
            torch.save(state,a.output/f'risk_{update}.pt')
            if score < best:
                best=score;torch.save(state,a.output/'risk_best.pt')
            print(json.dumps(record),flush=True)
    selected=torch.load(a.output/'risk_best.pt',weights_only=False)
    result=dict(rows=len(known),known=int(known.sum()),censored=int((~known).sum()),
        train=len(train),validation=len(val),evaluated=len(validation),updates=a.updates,
        selected_update=selected['update'],metrics=selected['validation'],
        new_environment_interactions=0,wall_seconds=time.perf_counter()-start)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
