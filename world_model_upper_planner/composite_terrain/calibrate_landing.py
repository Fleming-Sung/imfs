"""Measure physical landing residuals on held-out TRAINING layouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from cgowm import CandidateGroundedWorldModel,ModelConfig
from .anchored import MotionModel,predicted_landing


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--motion_checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    ck=torch.load(a.motion_checkpoint,map_location='cuda:0',weights_only=False)
    wc=torch.load(ck['world_checkpoint'],map_location='cuda:0',weights_only=False)
    world=CandidateGroundedWorldModel(wc['candidates'],ModelConfig(**wc['model_config'])).cuda().eval()
    world.load_state_dict(wc['model']);model=MotionModel(ck['latent_dim'],ck['proprio_dim']).cuda().eval()
    model.load_state_dict(ck['motion_model']);source=np.load(a.dataset)
    names=['depth','proprio','action','motion','next_proprio','done','duration','env_id']
    d={k:source[k] for k in names};envs=np.unique(d['env_id']);np.random.default_rng(9701).shuffle(envs)
    ids=np.flatnonzero(np.isin(d['env_id'],envs[:max(1,len(envs)//5)])&~d['done'].astype(bool)&(d['duration']>=15))
    residual=[]
    with torch.no_grad():
        for offset in range(0,len(ids),512):
            rows=ids[offset:offset+512]
            def tensor(k):return torch.as_tensor(d[k][rows],device='cuda:0',dtype=torch.float32)
            latent=world.encode(tensor('depth')/255,tensor('proprio'))
            motion,state=model(latent,tensor('proprio'),tensor('action'))
            predicted=predicted_landing(motion,state).mean(0)
            actual=predicted_landing(tensor('motion'),tensor('next_proprio'))
            residual.append((predicted-actual).cpu().numpy())
    error=np.concatenate(residual)
    result=dict(samples=len(error),xyz_mae_m=abs(error).mean(0).tolist(),
        xy_p90_m=np.quantile(abs(error[:,:2]),.9,axis=0).tolist(),
        motion_sha256=hashlib.sha256(a.motion_checkpoint.read_bytes()).hexdigest(),
        dataset=str(a.dataset),split='motion_train_seed9701_heldout_envs')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
