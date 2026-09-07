"""XYZ+yaw options and current observations for a static terrain map."""
import math
import torch
from adapters.frozen_lower_env.sampler import quaternion_yaw, yaw_quaternion, wrap_to_pi
from adapters.frozen_lower_env.upper_state import build_proprio


def candidates(device):
    axes=[torch.linspace(-1,1,n,device=device) for n in (4,3,7,3)]
    return torch.stack(torch.meshgrid(*axes,indexing="ij"),-1).reshape(-1,4)


def explore_supported(selection, valid, probability):
    """Training-only interventions; unexecuted actions get no physics labels."""
    if not 0 <= probability <= 1:
        raise ValueError('exploration probability must be in [0,1]')
    explored=torch.zeros_like(selection,dtype=torch.bool)
    if probability==0:
        return selection,explored
    explored=(torch.rand(len(selection),device=selection.device)<probability)&valid.any(-1)
    result=selection.clone()
    if explored.any():
        result[explored]=torch.multinomial(valid[explored].float(),1).squeeze(-1)
    return result,explored


def world_targets(env, ids, actions):
    if actions.ndim==2: actions=actions[None].expand(len(ids),-1,-1)
    swing=env.sampler.swing_foot[ids]; row=torch.arange(len(ids),device=env.device)
    stance=env.foot_positions[ids,1-swing]
    yaw=quaternion_yaw(env.rigid_body_states[ids,env.feet_indices[1-swing],3:7])
    dx=.21+.09*actions[...,0]
    dy=(.18+.06*actions[...,1])*torch.where(swing==0,1.,-1.)[:,None]
    angle=actions[...,3]*(math.pi/30)
    # Same curvature semantics as the successful V6 adapter, now with dz.
    dy=dy+1.5*dx*torch.tan(angle)
    dy=torch.where(swing[:,None]==0,dy.clamp_min(.06),dy.clamp_max(-.06))
    xyz=stance[:,None,:].expand(-1,actions.shape[1],-1).clone()
    xyz[...,0]+=torch.cos(yaw)[:,None]*dx-torch.sin(yaw)[:,None]*dy
    xyz[...,1]+=torch.sin(yaw)[:,None]*dx+torch.cos(yaw)[:,None]*dy
    xyz[...,2]+=.06*actions[...,2]
    return xyz,wrap_to_pi(yaw[:,None]+angle)


def apply(env,ids,actions):
    xyz,yaw=world_targets(env,ids,actions[:,None])
    swing=env.sampler.swing_foot[ids]
    env.sampler.target_pos[ids,swing]=xyz[:,0]
    env.sampler.target_yaw[ids,swing]=yaw[:,0]
    env.sampler.target_quat[ids,swing]=yaw_quaternion(yaw[:,0])
    env._compute_observations()


def sense(env,previous,proprio_dim=43,ids=None):
    rows=torch.arange(env.num_envs,device=env.device) if ids is None else ids
    yaw=quaternion_yaw(env.base_quat[rows])
    f,l=torch.meshgrid(torch.linspace(-.5,2.65,64,device=env.device),
                       torch.linspace(-1.575,1.575,64,device=env.device),indexing="ij")
    xy=torch.stack((env.base_position[rows,0,None,None]+torch.cos(yaw)[:,None,None]*f-torch.sin(yaw)[:,None,None]*l,
                    env.base_position[rows,1,None,None]+torch.sin(yaw)[:,None,None]*f+torch.cos(yaw)[:,None,None]*l),-1)
    stance_z=env.foot_positions[rows,1-env.sampler.swing_foot[rows],2]
    height=env.sample_height(xy)-stance_z[:,None,None]
    image=((height.clamp(-.8,.8)+.8)/1.6)[:,None]
    basic=build_proprio(env,env.goals[:,:2],previous[:,[0,1,3]])[rows]
    proprio=torch.cat((basic,previous[rows,2:3],env.base_lin_vel[rows,2:3]*.5,
        env.base_ang_vel[rows,:2]*.25,(env.goals[rows,2:3]-stance_z[:,None]),
        torch.nn.functional.one_hot(env.sampler.swing_foot[rows],2).float()),-1)
    assert proprio.shape[-1]==43
    if proprio_dim==75:
        # The frozen lower responds to BOTH retained foothold targets and
        # action history, not just the freshly issued upper command.
        goal=env.sampler.observation(env.foot_positions,
            env.rigid_body_states[:,env.feet_indices,3:7])[rows]
        proprio=torch.cat((proprio,goal,env.last_actions[rows].flatten(1)),-1)
    assert proprio.shape[-1]==proprio_dim
    return image,proprio


def panorama(env,ids=None):
    """Current static-map cache for anchored rollouts; no future simulator read."""
    rows=torch.arange(env.num_envs,device=env.device) if ids is None else ids
    yaw=quaternion_yaw(env.base_quat[rows])
    f,l=torch.meshgrid(torch.linspace(-1,5.35,128,device=env.device),
                       torch.linspace(-3.175,3.175,128,device=env.device),indexing='ij')
    xy=torch.stack((env.base_position[rows,0,None,None]+torch.cos(yaw)[:,None,None]*f-torch.sin(yaw)[:,None,None]*l,
        env.base_position[rows,1,None,None]+torch.sin(yaw)[:,None,None]*f+torch.cos(yaw)[:,None,None]*l),-1)
    z=env.foot_positions[rows,1-env.sampler.swing_foot[rows],2]
    return ((env.sample_height(xy)-z[:,None,None]).clamp(-.8,.8)+.8)[:,None]/1.6


def geometry(env,ids,grid):
    """Training-only counterfactual static geometry; no fabricated dynamics."""
    xyz,yaw=world_targets(env,ids,grid)
    # The asset box includes bevelled geometry above the sole; this optional
    # guard is conservative, not a claim that the flat contact patch is a box.
    asset_box=getattr(env,'asset_footprint',False)
    sx,sy=torch.meshgrid(torch.tensor([-.09,.01,.11] if asset_box else [-.08,0,.08],device=env.device),
                         torch.tensor([-.0408,0,.0408] if asset_box else [-.035,0,.035],device=env.device),indexing="ij")
    offsets=torch.stack((torch.cos(yaw)[...,None]*sx.flatten()-torch.sin(yaw)[...,None]*sy.flatten(),
                         torch.sin(yaw)[...,None]*sx.flatten()+torch.cos(yaw)[...,None]*sy.flatten()),-1)
    height=env.sample_height(xyz[...,:2,None].transpose(-1,-2)+offsets)
    support=(abs(height-xyz[...,2,None])<.028).float().mean(-1)
    # Goal distance potential measured from the same stance origin as the action.
    stance=env.foot_positions[ids,1-env.sampler.swing_foot[ids],:2]
    goal=env.goals[ids,:2]
    progress=torch.norm(goal-stance,dim=-1)[:,None]-torch.norm(goal[:,None]-xyz[...,:2],dim=-1)
    valid=(support>=7/9)&(env.sample_height(xyz[...,:2])>-.65)
    result=dict(candidate_support=support,candidate_progress=progress,candidate_valid=valid)
    if hasattr(env,'navigation_labels'):
        current=env.navigation_labels.sample(ids,stance)
        future=env.navigation_labels.sample(ids,xyz[...,:2])
        result['candidate_progress']=current[:,None,0]-future[...,0]
        result['candidate_alignment']=torch.cos(yaw)*future[...,1]+torch.sin(yaw)*future[...,2]
    return result
