"""Fit physical option transitions, freezing the previously learned outcome model."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from cgowm import CandidateGroundedWorldModel,ModelConfig
from .anchored import MotionModel,compose_pose,crop_map
from cgowm.data import load_arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--world_checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=3000);p.add_argument('--seed',type=int,default=9701)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);start=time.perf_counter()
    writer=SummaryWriter(a.output/"tb")
    source=load_arrays(a.dataset) if a.dataset.is_dir() else np.load(a.dataset)
    d={k:source[k] for k in ['depth','proprio','next_proprio','motion','action','done','duration',
                            'env_id','episode_id','option_index']}
    meta=json.loads((a.dataset.parent/'metrics.json').read_text())
    ck=torch.load(a.world_checkpoint,map_location='cuda:0',weights_only=False)
    if ck['lower_sha256']!=meta['lower_sha256']:raise ValueError('lower mismatch')
    world=CandidateGroundedWorldModel(ck['candidates'],ModelConfig(**ck['model_config'])).cuda().eval().requires_grad_(False)
    world.load_state_dict(ck['model'])
    tensors={k:torch.as_tensor(d[k],device='cuda:0',dtype=torch.float32)
             for k in ['proprio','next_proprio','motion','action']}
    latent=[]
    with torch.no_grad():
        for offset in range(0,len(d['env_id']),512):
            image=torch.as_tensor(d['depth'][offset:offset+512],device='cuda:0',dtype=torch.float32)/255
            latent.append(world.encode(image,tensors['proprio'][offset:offset+512]))
    latent=torch.cat(latent)
    envs=np.unique(d['env_id']);rng.shuffle(envs);val_envs=envs[:max(1,len(envs)//5)]
    live=~d['done'].astype(bool)&(d['duration']>=15)
    validation=live&np.isin(d['env_id'],val_envs)
    train=np.flatnonzero(live&~validation);val=np.flatnonzero(validation)
    model=MotionModel(world.latent_dim,world.config.proprio_dim).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    best=float('inf');train_start=time.perf_counter()
    for update in range(1,a.updates+1):
        indices=torch.as_tensor(rng.choice(train,1024),device='cuda:0')
        motion,state=model(latent[indices],tensors['proprio'][indices],tensors['action'][indices])
        ml=F.smooth_l1_loss(motion/model.motion_scale,
            (tensors['motion'][indices]/model.motion_scale)[None].expand_as(motion),reduction='none').mean(-1)
        sl=F.smooth_l1_loss(state,tensors['next_proprio'][indices][None].expand_as(state),reduction='none').mean(-1)
        bootstrap=(torch.rand_like(ml)>.2).float()
        loss=((ml+sl)*bootstrap).sum()/bootstrap.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        if update%100==0:writer.add_scalar('train/loss',float(loss),update)
        if update%500==0 or update==a.updates:
            with torch.no_grad():
                ids=torch.as_tensor(val[:4096],device='cuda:0')
                pm,ps,_=model.predict(latent[ids],tensors['proprio'][ids],tensors['action'][ids])
                mae=(pm-tensors['motion'][ids]).abs().mean(0)
                state_mae=(ps-tensors['next_proprio'][ids]).abs().mean()
                score=float((mae/model.motion_scale).mean()+state_mae)
            record=dict(update=update,motion_mae=mae.tolist(),state_mae=float(state_mae),
                        score=score,wall_seconds=time.perf_counter()-train_start)
            writer.add_scalars('val',{'motion_mae_dx':float(mae[0]),'motion_mae_dy':float(mae[1]),
                'motion_mae_dyaw':float(mae[2]),'motion_mae_dz':float(mae[3]),
                'state_mae':float(state_mae),'score':score},update)
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            state=dict(motion_model=model.state_dict(),world_checkpoint=str(a.world_checkpoint),
                world_sha256=hashlib.sha256(a.world_checkpoint.read_bytes()).hexdigest(),
                lower_sha256=meta['lower_sha256'],proprio_dim=world.config.proprio_dim,
                latent_dim=world.latent_dim,update=update,validation=record)
            if score<best:best=score;torch.save(state,a.output/'model_best.pt')
            torch.save(state,a.output/f'model_{update}.pt')
    model.load_state_dict(torch.load(a.output/'model_best.pt',weights_only=False)['motion_model'])
    lookup={tuple(map(int,k)):i for i,k in enumerate(zip(d['env_id'],d['episode_id'],d['option_index']))}
    sequences=[]
    for i in val:
        key=(int(d['env_id'][i]),int(d['episode_id'][i]),int(d['option_index'][i]))
        rows=[lookup.get((key[0],key[1],key[2]+s),-1) for s in range(3)]
        if min(rows)>=0 and live[rows].all():sequences.append(rows)
    rng.shuffle(sequences);rows=np.asarray(sequences[:256]);ids=rows[:,0]
    cache=torch.as_tensor(source['panorama'][ids],device='cuda:0',dtype=torch.float32)/255
    prop=tensors['proprio'][ids];pose=torch.zeros(len(ids),4,device='cuda:0');truth=pose.clone()
    open_loop=[]
    with torch.no_grad():
        for s in range(3):
            patch,_=crop_map(cache,pose)
            feature=latent[ids] if s==0 else world.encode(patch,prop)
            motion,prop,_=model.predict(feature,prop,tensors['action'][rows[:,s]])
            pose=compose_pose(pose,motion);truth=compose_pose(truth,tensors['motion'][rows[:,s]])
            open_loop.append(dict(horizon=s+1,pose_mae=(pose-truth).abs().mean(0).tolist(),
                state_mae=float((prop-tensors['next_proprio'][rows[:,s]]).abs().mean())))
    summary=dict(train_transitions=len(train),val_transitions=len(val),updates=a.updates,
        best_score=best,open_loop=open_loop,world_checkpoint=str(a.world_checkpoint),
        world_sha256=hashlib.sha256(a.world_checkpoint.read_bytes()).hexdigest(),
        lower_sha256=meta['lower_sha256'],wall_seconds=time.perf_counter()-start)
    writer.close()
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
