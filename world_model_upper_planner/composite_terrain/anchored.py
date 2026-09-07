"""Learned closed-loop motion with an explicit, currently observed static map.

The planner receives tensors only. No simulator or map provider is accessible
inside an imagined rollout.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from cgowm.model import mlp


def compose_pose(pose,motion):
    c,s=torch.cos(pose[...,2]),torch.sin(pose[...,2])
    return torch.stack((pose[...,0]+c*motion[...,0]-s*motion[...,1],
        pose[...,1]+s*motion[...,0]+c*motion[...,1],pose[...,2]+motion[...,2],
        pose[...,3]+motion[...,3]),-1)


def physical_goal_progress(proprio, motion):
    """Known task potential evaluated on learned body motion, not target XY."""
    goal = proprio[...,33:35] * proprio.new_tensor([6.,3.])
    distance = goal.norm(dim=-1)
    return distance - (goal-motion[...,:2]).norm(dim=-1), distance


def crop_map(cache,pose):
    """128x128 current panorama -> future 64x64 local observation."""
    f,l=torch.meshgrid(torch.linspace(-.5,2.65,64,device=cache.device),
                       torch.linspace(-1.575,1.575,64,device=cache.device),indexing='ij')
    c,s=torch.cos(pose[:,2,None,None]),torch.sin(pose[:,2,None,None])
    x=pose[:,0,None,None]+c*f-s*l;y=pose[:,1,None,None]+s*f+c*l
    grid=torch.stack((2*(y+3.175)/6.35-1,2*(x+1)/6.35-1),-1)
    known=(grid.abs()<=1).all(-1)[:,None]
    sample=F.grid_sample(cache,grid,align_corners=True,padding_mode='zeros')
    sample=(sample-pose[:,3,None,None,None]/1.6).clamp(0,1)
    return torch.where(known,sample,torch.zeros_like(sample)),known


def predicted_landing(motion,state):
    """Next stance foot in current base-yaw XY / stance-height Z coordinates."""
    feet=state[...,22:28].reshape(*state.shape[:-1],2,3)
    foot=(feet*state[...,41:43].flip(-1)[...,None]).sum(-2)
    gravity=F.normalize(state[...,:3],dim=-1)
    roll=torch.atan2(-gravity[...,1],-gravity[...,2])
    sp=gravity[...,0].clamp(-1,1);cp=(1-sp.square()).clamp_min(0).sqrt()
    sr,cr=torch.sin(roll),torch.cos(roll)
    fx=cp*foot[...,0]+sp*(sr*foot[...,1]+cr*foot[...,2])
    fy=cr*foot[...,1]-sr*foot[...,2]
    c,s=torch.cos(motion[...,2]),torch.sin(motion[...,2])
    return torch.stack((motion[...,0]+c*fx-s*fy,motion[...,1]+s*fx+c*fy,motion[...,3]),-1)


class MotionModel(nn.Module):
    """Three independently fitted predictions of physical option transitions."""
    def __init__(self,latent_dim=128,proprio_dim=75):
        super().__init__();self.proprio_dim=proprio_dim
        self.heads=nn.ModuleList([mlp(latent_dim+proprio_dim+4,256,4+proprio_dim) for _ in range(3)])
        self.register_buffer('motion_scale',torch.tensor([.25,.10,.15,.04]))

    def forward(self,latent,proprio,action):
        features=torch.cat((latent,proprio,action),-1)
        raw=torch.stack([head(features) for head in self.heads])
        motion=raw[...,:4]*self.motion_scale
        state=proprio[None]+raw[...,4:]
        # Exact command bookkeeping and coordinate updates are not relearned.
        state[...,30:33]=action[None,:,[0,1,3]]
        state[...,36]=action[None,:,2]
        state[...,41:43]=proprio[None,:,41:43].flip(-1)
        phase=-proprio[None,:,28:30]
        state[...,28:30]=phase;state[...,57:59]=phase
        goal=proprio[None,:,33:35]*proprio.new_tensor([6.,3.])-motion[...,:2]
        c,s=torch.cos(motion[...,2]),torch.sin(motion[...,2])
        gx=c*goal[...,0]+s*goal[...,1];gy=-s*goal[...,0]+c*goal[...,1]
        state[...,33]=gx/6;state[...,34]=gy/3
        state[...,35]=torch.atan2(gy,gx)/math.pi
        state[...,40]=proprio[None,:,40]-motion[...,3]
        return motion,state

    def predict(self,latent,proprio,action):
        motion,state=self(latent,proprio,action)
        uncertainty=(motion.std(0,unbiased=False)/self.motion_scale).norm(dim=-1)
        mean=motion.mean(0)
        limits=mean.new_tensor([.65,.40,.40,.20])
        return mean.clamp(-limits,limits),state.mean(0),uncertainty


class AnchoredPlanner:
    def __init__(self,world,motion,config,landing_margin=None,goal_cost_radius=0.,risk_model=None):
        self.world=world;self.motion=motion;self.config=config
        self.landing_margin=landing_margin
        self.goal_cost_radius=goal_cost_radius
        self.risk_model=risk_model

    def risk_probabilities(self,latent,proprio,actions,prediction):
        if self.risk_model is None:
            return torch.sigmoid(prediction['fall_logit']),torch.sigmoid(prediction['collision_logit'])
        risk=self.risk_model.predict_candidates(torch.cat((latent,proprio),-1),actions)
        return risk['fall_logit'].sigmoid().mean(0),risk['collision_logit'].sigmoid().mean(0)

    def landing_score(self,latent,proprio,cache):
        count=len(self.world.candidates);batch=len(proprio)
        feature=latent[:,None].expand(-1,count,-1).flatten(0,1)
        state=proprio[:,None].expand(-1,count,-1).flatten(0,1)
        action=self.world.candidates[None].expand(batch,-1,-1).flatten(0,1)
        motion,next_state=self.motion(feature,state,action)
        landing=predicted_landing(motion,next_state).view(3,batch,count,3)
        # Residual margins come from held-out training layouts, not test maps.
        mx,my=self.landing_margin
        ox,oy=torch.meshgrid(landing.new_tensor([-.08-mx,0,.08+mx]),
                             landing.new_tensor([-.035-my,0,.035+my]),indexing='ij')
        xy=landing[...,:2,None].transpose(-1,-2)+torch.stack((ox.flatten(),oy.flatten()),-1)
        grid=torch.stack((2*(xy[...,1]+3.175)/6.35-1,2*(xy[...,0]+1)/6.35-1),-1)
        grid=grid.permute(1,0,2,3,4).reshape(batch,3*count,9,2)
        height=F.grid_sample(cache,grid,mode='nearest',align_corners=True)[:,0]*1.6-.8
        height=height.view(batch,3,count,9).permute(1,0,2,3)
        support=((height-landing[...,2,None]).abs()<.035).float().mean((0,3))
        return -5*(1-support)

    @torch.no_grad()
    def plan(self,image,proprio,cache,root_mask=None):
        cfg=self.config;batch=len(image);device=image.device
        states=proprio[:,None];poses=image.new_zeros(batch,1,4)
        returns=image.new_zeros(batch,1);alive=torch.ones_like(returns)
        first=torch.full((batch,1),-1,device=device,dtype=torch.long)
        initial=self.world.encode(image,proprio)
        landing_bias=self.landing_score(initial,proprio,cache) if self.landing_margin is not None else None
        for step in range(cfg.horizon):
            beams=states.shape[1];flat_state=states.flatten(0,1)
            maps=cache[:,None].expand(-1,beams,-1,-1,-1).flatten(0,1)
            patches,_=crop_map(maps,poses.flatten(0,1))
            latent=initial if step==0 else self.world.encode(patches,flat_state)
            prediction=self.world.predict_candidates(latent)
            logits=prediction['policy_logits'].view(batch,beams,-1)
            if step==0 and landing_bias is not None:logits=logits+landing_bias[:,None]
            if step==0 and root_mask is not None:logits=logits.masked_fill(~root_mask[:,None],-torch.inf)
            count=min(cfg.proposals_per_beam,logits.shape[-1])
            index=logits.topk(count,dim=-1).indices
            def gather(v):return torch.gather(v.view(batch,beams,-1),-1,index)
            q=prediction['q'].view(2,batch,beams,-1)
            qstd=torch.gather(q,-1,index[None].expand(2,-1,-1,-1)).std(0,unbiased=False)
            fall_probability,collision_probability=self.risk_probabilities(latent,flat_state,self.world.candidates,prediction)
            score=cfg.progress_weight*gather(prediction['progress'])-cfg.support_weight*(1-gather(prediction['support']))
            score-=cfg.fall_weight*gather(fall_probability)
            score-=cfg.collision_weight*gather(collision_probability)
            if cfg.reward_weight:
                # Compare geometric surrogate against the learned reward of
                # actual lower-controller execution. The latter already
                # includes progress, falls, collisions and loss of support.
                score=(1-cfg.reward_weight)*score+cfg.reward_weight*gather(prediction['reward'])
            score-=cfg.uncertainty_weight*qstd
            if step==0 and landing_bias is not None:
                score+=torch.gather(landing_bias[:,None],-1,index)
            action=self.world.candidates[index.flatten()]
            repeated_latent=latent[:,None].expand(-1,count,-1).flatten(0,1)
            repeated_state=flat_state[:,None].expand(-1,count,-1).flatten(0,1)
            delta,next_state,uncertainty=self.motion.predict(repeated_latent,repeated_state,action)
            if self.goal_cost_radius:
                progress,distance=physical_goal_progress(repeated_state,delta)
                replacement=(progress.view_as(score)-gather(prediction['progress']))
                score+=(1-cfg.reward_weight)*cfg.progress_weight*replacement*(distance.view_as(score)<self.goal_cost_radius)
            score-=.5*uncertainty.view(batch,beams,count)
            score=score.masked_fill(~torch.gather(torch.isfinite(logits),-1,index),-torch.inf)
            total=(returns[:,:,None]+cfg.discount**step*alive[:,:,None]*score).flatten(1)
            continuation=torch.sigmoid(gather(prediction['continuation_logit']))
            next_alive=(alive[:,:,None]*continuation).flatten(1)
            repeated_pose=poses[:,:,None].expand(-1,-1,count,-1).reshape(-1,4)
            next_pose=compose_pose(repeated_pose,delta).view(batch,-1,4)
            next_state=next_state.view(batch,-1,proprio.shape[-1])
            next_first=index.flatten(1) if step==0 else first[:,:,None].expand(-1,-1,count).flatten(1)
            returns,order=total.topk(min(cfg.beam_width,total.shape[-1]),dim=-1)
            row=torch.arange(batch,device=device)[:,None]
            poses=next_pose[row,order];states=next_state[row,order]
            alive=next_alive[row,order];first=next_first[row,order]
        if cfg.terminal_value_weight:
            beams=states.shape[1];maps=cache[:,None].expand(-1,beams,-1,-1,-1).flatten(0,1)
            patches,_=crop_map(maps,poses.flatten(0,1))
            latent=self.world.encode(patches,states.flatten(0,1))
            value=self.world.predict_candidates(latent)['q'].min(0).values.max(-1).values.view_as(returns)
            returns+=cfg.terminal_value_weight*cfg.discount**cfg.horizon*alive*value
        selected=first[torch.arange(batch,device=device),returns.argmax(-1)]
        return selected,dict(best_score=returns.max(-1).values)
