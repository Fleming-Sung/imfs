"""Collect or evaluate complete static composite-map option trajectories."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time
import numpy as np
from isaacgym import gymapi
import torch
from .env import create_composite_env
from .interface import candidates, sense, geometry, apply, panorama
from adapters.frozen_lower_env.sampler import quaternion_yaw,wrap_to_pi
from adapters.frozen_lower_env.lower_policy import FrozenLowerPolicy
from cgowm import CandidateGroundedWorldModel, ModelConfig, PlannerConfig, VectorizedBeamPlanner


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lower_checkpoint",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--motion_checkpoint",type=Path)
    p.add_argument("--risk_checkpoint",type=Path)
    p.add_argument("--landing_calibration",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--num_envs",type=int,default=64)
    p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--seed",type=int,default=9201)
    p.add_argument("--route_offset",type=int,default=0)
    p.add_argument("--difficulty",type=float,default=.15)
    p.add_argument('--action_profile',choices=['legacy','recovery'],default='legacy')
    p.add_argument('--terrain_profile',choices=['legacy','clearance'],default='legacy')
    p.add_argument('--planning_objective',choices=['legacy','reward_tail'],default='legacy')
    p.add_argument("--difficulty_levels",default='',help='comma-separated training levels, balanced across map layouts')
    p.add_argument("--horizon",type=int,default=3)
    p.add_argument("--proposals_per_beam",type=int,default=8,
                   help='root/action proposal coverage; equals the full grid size to evaluate every candidate')
    p.add_argument("--physical_reward_weight",type=float,default=0.0,
                   help="anchored planner: blend geometric surrogate with learned actual execution reward [0,1]")
    p.add_argument("--value_weight",type=float,default=1.0,
                   help="planner: weight of the learned twin-Q long-term value in the candidate score")
    p.add_argument("--fallback",choices=["minimal","max_support"],default="max_support",
                   help="zero-valid fallback: minimal in-place step or max-support candidate")
    p.add_argument("--success_reward",type=float,default=20.0,
                   help="terminal arrival bonus on a completed episode (value function must learn it dominates)")
    p.add_argument("--time_penalty",type=float,default=0.05,
                   help="per-option standing cost so endless posture adjustment / stalling is not free")
    p.add_argument("--progress_reward",type=float,default=10.0,
                   help="weight of goal-distance progress per option")
    p.add_argument("--goal_cost_radius",type=float,default=0.0,
                   help="within this goal distance, compute progress from learned body motion; zero disables")
    p.add_argument("--root_geometry_guard",action="store_true",
                   help="current observed terrain feasibility only; imagined futures remain learned")
    p.add_argument("--asset_footprint",action="store_true",
                   help="use conservative foot mesh bounding box for current candidate support")
    p.add_argument("--local_refinement",action="store_true",
                   help="bounded continuous XY refinement and observed-surface target height; H1 only")
    p.add_argument("--geodesic_guidance",action="store_true",
                   help="training collection geometry potential, not learned-model inference")
    p.add_argument("--behavior",choices=["diverse","greedy","model"],default="diverse")
    p.add_argument("--collect",action="store_true")
    p.add_argument("--record_motion",action="store_true")
    p.add_argument("--random_action_prob",type=float,default=.12)
    p.add_argument("--model_exploration_prob",type=float,default=0.,
                   help='collection only: sample other currently supported actions')
    p.add_argument("--reset_region_prob",type=float,default=0.0)
    p.add_argument("--record_video",action="store_true")
    p.add_argument("--headless",action="store_true")
    p.add_argument("--sim_device",default="cuda:0")
    args=p.parse_args(); args.use_gpu_pipeline=True; args.use_gpu=True; args.subscenes=0
    if args.action_profile=='recovery':args.proposals_per_beam=max(12,args.proposals_per_beam)
    if args.planning_objective=='reward_tail' and not args.motion_checkpoint:
        raise ValueError('reward-tail planning requires the physical world model')
    if args.action_profile=='recovery' and args.local_refinement:
        raise ValueError('recovery actions use multi-step candidates, not the legacy local refiner')
    if not 0 <= args.physical_reward_weight <= 1:
        raise ValueError('physical_reward_weight must be in [0,1]')
    if args.physical_reward_weight and not args.motion_checkpoint:
        raise ValueError('physical reward experiment requires anchored planner')
    if args.goal_cost_radius < 0 or (args.goal_cost_radius and not args.motion_checkpoint):
        raise ValueError('goal cost requires nonnegative radius and anchored planner')
    if args.risk_checkpoint and (not args.motion_checkpoint or args.physical_reward_weight):
        raise ValueError('risk-tail experiment requires anchored planner and default surrogate score')
    if args.local_refinement and (args.behavior != 'model' or not args.motion_checkpoint or args.horizon != 1 or args.landing_calibration):
        raise ValueError('local refinement requires anchored H1 without landing-calibration bias')
    if not 0 <= args.model_exploration_prob <= 1:
        raise ValueError('model_exploration_prob must be in [0,1]')
    if args.model_exploration_prob and (not args.collect or args.behavior!='model'):
        raise ValueError('model exploration is allowed only in model replay collection')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    snapshot=args.output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    hashes={}
    for folder in ['composite_terrain','cgowm','adapters/frozen_lower_env']:
        for source in (project/folder).glob('*.py'):
            relative=source.relative_to(project);dest=snapshot/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    (args.output/'arguments.json').write_text(json.dumps(vars(args),default=str,indent=2))
    lower_hash=hashlib.sha256(args.lower_checkpoint.read_bytes()).hexdigest()
    started=time.perf_counter(); env=create_composite_env(args)
    env.asset_footprint=args.asset_footprint
    if args.geodesic_guidance:
        if args.behavior=='model' and not args.collect:
            raise ValueError('global navigation labels are training-only')
        from .navigation import NavigationLabels
        env.navigation_labels=NavigationLabels(env)
    lower=FrozenLowerPolicy(args.lower_checkpoint,env.device)
    grid=candidates(env.device,args.action_profile); n=env.num_envs
    # Smallest in-place step (dx=min, dy=min, dz=0, yaw=0): a stable fallback
    # when no candidate is geometrically valid on the current observation.
    min_step=((grid-grid.new_tensor([-1.,-1.,0.,0.])).abs().sum(-1)).argmin()
    proprio_dim=75
    if args.behavior=="model":
        ck=torch.load(args.checkpoint,map_location=env.device)
        if ck.get("lower_sha256",lower_hash)!=lower_hash:
            raise ValueError("world-model and lower checkpoint hashes differ")
        model=CandidateGroundedWorldModel(ck["candidates"],ModelConfig(**ck["model_config"])).to(env.device)
        model.load_state_dict(ck["model"]); model.eval()
        proprio_dim=model.config.proprio_dim
        if not torch.equal(model.candidates,grid): raise ValueError("candidate contract changed")
        if not 1 <= args.proposals_per_beam <= len(grid):
            raise ValueError('proposals_per_beam must be within the fixed action grid')
        planner=VectorizedBeamPlanner(model,PlannerConfig(horizon=args.horizon,beam_width=16,proposals_per_beam=args.proposals_per_beam,support_weight=3.0,reward_weight=args.physical_reward_weight,value_weight=args.value_weight))
        if args.motion_checkpoint:
            from .anchored import MotionModel,AnchoredPlanner
            motion_ck=torch.load(args.motion_checkpoint,map_location=env.device,weights_only=False)
            if motion_ck['world_sha256']!=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
                raise ValueError('motion model was fitted to a different frozen world encoder')
            motion_model=MotionModel(motion_ck['latent_dim'],motion_ck['proprio_dim']).to(env.device)
            motion_model.load_state_dict(motion_ck['motion_model']);motion_model.eval()
            margin=None
            if args.landing_calibration:
                calibration=json.loads(args.landing_calibration.read_text())
                if calibration['motion_sha256']!=hashlib.sha256(args.motion_checkpoint.read_bytes()).hexdigest():
                    raise ValueError('landing calibration belongs to a different motion model')
                margin=calibration['xy_p90_m']
            risk_model=None
            if args.risk_checkpoint:
                from cgowm import OptionRiskCritic,RiskConfig
                risk_ck=torch.load(args.risk_checkpoint,map_location=env.device,weights_only=False)
                if risk_ck['world_sha256']!=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() or risk_ck['lower_sha256']!=lower_hash:
                    raise ValueError('risk model belongs to a different world/lower model')
                if risk_ck['risk_config']['latent_dim']!=model.latent_dim+proprio_dim or risk_ck['risk_config']['action_dim']!=4:
                    raise ValueError('risk input contract mismatch')
                risk_model=OptionRiskCritic(RiskConfig(**risk_ck['risk_config'])).to(env.device)
                risk_model.load_state_dict(risk_ck['risk']);risk_model.eval()
            if args.planning_objective=='reward_tail':
                from dataclasses import replace
                planner.config=replace(planner.config,terminal_value_weight=args.value_weight)
            planner=AnchoredPlanner(model,motion_model,planner.config,landing_margin=margin,goal_cost_radius=args.goal_cost_radius,risk_model=risk_model,
                proposal_mode='recovery' if args.action_profile=='recovery' else 'prior',objective=args.planning_objective)
    active=torch.zeros(n,dtype=torch.bool,device=env.device)
    prev=torch.zeros(n,4,device=env.device)
    episode=torch.zeros(n,dtype=torch.long,device=env.device)
    option=torch.zeros_like(episode); durations=torch.zeros_like(episode)
    state_image=torch.zeros(n,1,64,64,device=env.device)
    state_proprio=torch.zeros(n,proprio_dim,device=env.device)
    start_pose=torch.zeros(n,4,device=env.device)
    state_panorama=torch.zeros(n,1,128,128,device=env.device) if args.record_motion else None
    chosen=torch.zeros_like(episode)
    previous_distance=torch.zeros(n,device=env.device)
    target=torch.zeros(n,3,device=env.device); commanded_foot=torch.zeros_like(episode)
    collided=torch.zeros_like(active)
    labels={name:torch.zeros(n,len(grid),device=env.device) for name in
            ["candidate_support","candidate_progress","candidate_valid"]}
    if args.geodesic_guidance:
        labels['candidate_alignment']=torch.zeros(n,len(grid),device=env.device)
    batches=[]; event_errors=[]; total_falls=total_success=total_timeout=0
    decision_count=exploration_count=0
    zero_valid_count=micro_count=backward_count=0
    env_success=torch.zeros(n,dtype=torch.long,device=env.device)
    env_falls=torch.zeros_like(env_success)
    goal_hold=torch.zeros_like(env_success)
    episode_ticks=torch.zeros_like(env_success)
    episode_max_x=torch.zeros(n,device=env.device)
    episode_start_x=torch.zeros(n,device=env.device)
    episode_records=[]
    terminal_audit=[]
    max_x=torch.zeros(n,device=env.device)
    trace={k:[] for k in ["root","feet","targets","done","region"]}
    if args.record_video:
        if env.viewer is None: raise ValueError("omit --headless for video")
        (args.output/"frames").mkdir(exist_ok=True)

    def finish(ids,done,fall,success,root=None,feet=None):
        nonlocal total_falls,total_success,total_timeout
        if not len(ids): return
        root=env.root_states[:,0] if root is None else root
        feet=env.foot_positions if feet is None else feet
        dist=torch.norm(env.goals[ids,:2]-root[ids,:2],dim=-1)
        if args.geodesic_guidance:dist=env.navigation_labels.sample(ids,root[ids,:2])[:,0]
        progress=previous_distance[ids]-dist
        error=feet[ids,commanded_foot[ids]]-target[ids]
        support=(abs(feet[ids,commanded_foot[ids],2]-env.sample_height(
            feet[ids,commanded_foot[ids],:2]))<.035).float()
        # Terminal arrival must dominate in the value function so the robot aims
        # at holding in the goal; a per-option time cost makes stalling non-free.
        reward=(args.progress_reward*progress
                -5.0*fall.float()-2.0*collided[ids].float()-2.0*(1-support)
                +args.success_reward*success.float()-args.time_penalty)
        event_errors.append(error.cpu().numpy())
        if args.collect:
            image,proprio=sense(env,prev,proprio_dim,ids)
            local_x=root[ids,0]-env.route_origins[ids,0]
            region=((local_x-1)/2.4).long().clamp(0,6).cpu().numpy()
            terrain=np.array([env.atlas["routes"][int(j)]["segments"][int(k)]["kind"]
                              for j,k in zip(ids.cpu().numpy(),region)],dtype="U24")
            batch=dict(depth=(state_image[ids]*255).round().byte(),proprio=state_proprio[ids],
                action=prev[ids],reward=reward,next_depth=(image*255).round().byte(),
                next_proprio=proprio,done=done,progress=progress,support=support,
                touchdown_error=error.norm(dim=-1),fall=fall,collision=collided[ids],success=success,
                duration=durations[ids],env_id=ids,episode_id=episode[ids],option_index=option[ids],
                candidate_index=chosen[ids],**{k:v[ids] for k,v in labels.items()})
            if args.record_motion:
                delta=root[ids,:2]-start_pose[ids,:2];yaw0=start_pose[ids,2]
                motion=torch.stack((torch.cos(yaw0)*delta[:,0]+torch.sin(yaw0)*delta[:,1],
                    -torch.sin(yaw0)*delta[:,0]+torch.cos(yaw0)*delta[:,1],
                    wrap_to_pi(quaternion_yaw(root[ids,3:7])-yaw0),
                    feet[ids,1-env.sampler.swing_foot[ids],2]-start_pose[ids,3]),-1)
                batch.update(motion=motion,panorama=(state_panorama[ids]*255).round().byte())
            batch={k:v.cpu().numpy().copy() for k,v in batch.items()}
            batch["terrain_kind"]=terrain
            batch["difficulty"]=np.array([str(env.atlas['routes'][int(j)]['spec']['difficulty'])
                for j in ids.cpu().numpy()],dtype='U12')
            batches.append(batch)
        option[ids]+=1; active[ids]=False

    with torch.no_grad():
        for tick in range(args.steps):
            # Complete the old option using the old physical state, then advance
            # the clock and issue a new goal BEFORE the next lower action.
            will_switch=torch.floor(env.sampler.phase*2)!=torch.floor(
                torch.remainder(env.sampler.phase+env.dt*env.sampler.frequency,1)*2)
            ids=(will_switch&active&~env.goal_reset_pending).nonzero().flatten()
            zero=torch.zeros(len(ids),dtype=torch.bool,device=env.device)
            env.sampler.advance(env.dt,env.foot_positions,env.rigid_body_states[:,env.feet_indices,3:7])
            # The successor must be the next DECISION state (new swing foot),
            # exactly equal to the next row's input, not the old gait phase.
            finish(ids,zero,zero,zero)
            ids=(~active&~env.goal_reset_pending).nonzero().flatten()
            if len(ids):
                image,proprio=sense(env,prev,proprio_dim,ids)
                map_cache=panorama(env,ids) if args.motion_checkpoint or args.record_motion else None
                geo=geometry(env,ids,grid) if args.collect or args.behavior!="model" or args.root_geometry_guard else None
                if geo is not None:zero_valid_count+=int((~geo['candidate_valid'].any(-1)).sum())
                if args.behavior=="model":
                    mask=None
                    if args.root_geometry_guard:
                        mask=geo["candidate_valid"].clone()
                        empty=~mask.any(-1)
                        if empty.any():
                            if args.fallback=="minimal":
                                # Smallest in-place step; never commit to an
                                # invalid candidate and never teleport/reset.
                                mask[empty]=False
                                mask[empty,min_step]=True
                            else:
                                # Step onto the most-supported nearby point.
                                mask[empty]=geo["candidate_support"][empty]>=geo["candidate_support"][empty].max(-1,keepdim=True).values
                    if args.motion_checkpoint:
                        selection,_=planner.plan(image,proprio,map_cache,mask)
                    else:
                        selection,_=planner.plan(model.encode(image,proprio),mask)
                else:
                    score=10*geo["candidate_progress"]-3*(1-geo["candidate_support"])
                    if 'candidate_alignment' in geo:score+=2*geo['candidate_alignment']
                    score=score.masked_fill(~geo["candidate_valid"],-20)
                    if args.behavior=="diverse":
                        selection=torch.multinomial(torch.softmax(score/.4,-1),1).squeeze(-1)
                        if args.action_profile=='recovery':
                            from .interface import explore_supported
                            selection,_=explore_supported(selection,geo['candidate_valid'],args.random_action_prob)
                        else:
                            random=torch.rand(len(ids),device=env.device)<args.random_action_prob
                            selection[random]=torch.randint(len(grid),(int(random.sum()),),device=env.device)
                    else: selection=score.argmax(-1)
                decision_count+=len(ids)
                if args.model_exploration_prob:
                    from .interface import explore_supported
                    selection,explored=explore_supported(selection,geo['candidate_valid'],args.model_exploration_prob)
                    exploration_count+=int(explored.sum())
                state_image[ids]=image; state_proprio[ids]=proprio
                if args.record_motion:
                    state_panorama[ids]=map_cache
                    start_pose[ids,:2]=env.base_position[ids,:2]
                    start_pose[ids,2]=quaternion_yaw(env.base_quat[ids])
                    start_pose[ids,3]=env.foot_positions[ids,1-env.sampler.swing_foot[ids],2]
                if geo is not None:
                    for k,v in geo.items(): labels[k][ids]=v.float()
                chosen[ids]=selection; executed=grid[selection]
                if args.local_refinement:
                    from .refine import refine_root
                    executed=refine_root(env,ids,executed,image,proprio,map_cache,planner)
                prev[ids]=executed
                dx=.21+.09*executed[:,0]
                micro_count+=int((dx.abs()<=.061).sum());backward_count+=int((dx<-.03).sum())
                apply(env,ids,prev[ids]); commanded_foot[ids]=env.sampler.swing_foot[ids]
                target[ids]=env.sampler.target_pos[ids,commanded_foot[ids]]
                previous_distance[ids]=torch.norm(env.goals[ids,:2]-env.base_position[ids,:2],dim=-1)
                if args.geodesic_guidance:
                    previous_distance[ids]=env.navigation_labels.sample(ids,env.base_position[ids,:2])[:,0]
                active[ids]=True; collided[ids]=False; durations[ids]=0
            obs,goal,_=env.get_observations(); action,_=lower.infer(obs,goal)
            _,_,done,extras,_,_=env.step(action); durations[active]+=1
            contact=torch.norm(env.contact_forces[:,env.nonfoot_indices],dim=-1).max(-1).values>5
            collided|=contact
            root=extras["pre_reset_root"]; feet=extras["pre_reset_feet"]
            at_goal=(torch.norm(root[:,:2]-env.goals[:,:2],dim=-1)<.3)&~done.bool()
            stable=(env.projected_gravity[:,2]<-.8)&(
                root[:,2]-env.goals[:,2]>.42)&(root[:,2]-env.goals[:,2]<.95)&~contact
            goal_hold=torch.where(at_goal&stable,goal_hold+1,torch.zeros_like(goal_hold))
            success=goal_hold>=10  # 0.2 seconds upright in the terminal region.
            fall=extras["absorbing"].bool()
            env_falls+=fall.long()
            if fall.any() and len(terminal_audit)<12:
                for j in fall.nonzero().flatten().cpu().tolist():
                    terminal_audit.append(dict(tick=tick,env=j,root=root[j,:3].cpu().tolist(),
                        targets=env.sampler.target_pos[j].cpu().tolist(),
                        reasons={k:bool(v[j]) for k,v in extras["termination_reasons"].items()}))
            total_falls+=int(fall.sum()); total_success+=int(success.sum())
            total_timeout+=int(extras["time_outs"].sum()); env_success+=success.long()
            ids=((done.bool()|success)&active).nonzero().flatten()
            finish(ids,torch.ones(len(ids),device=env.device,dtype=torch.bool),fall[ids],success[ids],root,feet)
            ended=(done.bool()|success).nonzero().flatten()
            x=root[:,0]-env.route_origins[:,0]
            first=episode_ticks==0;episode_start_x[first]=x[first]
            episode_ticks+=1;episode_max_x=torch.maximum(episode_max_x,x)
            if len(ended):
                episode_records.append(dict(env_id=ended.cpu().numpy(),
                    episode_id=episode[ended].cpu().numpy(),ticks=episode_ticks[ended].cpu().numpy(),
                    start_x=episode_start_x[ended].cpu().numpy(),max_x=episode_max_x[ended].cpu().numpy(),
                    success=success[ended].cpu().numpy(),fall=fall[ended].cpu().numpy(),
                    timeout=extras['time_outs'][ended].cpu().numpy(),
                    terminal_region=((x[ended]-1)/2.4).long().clamp(0,6).cpu().numpy()))
            episode_ticks[ended]=0;episode_max_x[ended]=0;goal_hold[ended]=0
            active[ended]=False; episode[ended]+=1; option[ended]=0
            if success.any(): env._reset_idx(success.nonzero().flatten())
            max_x=torch.maximum(max_x,x)
            vals=dict(root=root[0],feet=feet[0],targets=env.sampler.target_pos[0],done=(done.bool()|success)[0],
                      region=((x[0]-1)/2.4).long().clamp(0,6))
            for k,v in vals.items():trace[k].append(v.cpu().numpy().copy())
            if args.record_video and tick%2==0:
                pos=env.base_position[0].cpu().numpy()
                env.gym.viewer_camera_look_at(env.viewer,None,
                    gymapi.Vec3(float(pos[0]+1.8),float(pos[1]-2),float(pos[2]+1.5)),
                    gymapi.Vec3(float(pos[0]+.6),float(pos[1]),float(pos[2]-.3)))
                env.gym.write_viewer_image_to_file(env.viewer,str(args.output/"frames"/f"{tick//2:06d}.png"))
            if (tick+1)%500==0:
                print(json.dumps(dict(tick=tick+1,falls=total_falls,success=total_success,
                                      mean_farthest_x=float(max_x.mean()))),flush=True)
    np.savez_compressed(args.output/"trajectory.npz",**{k:np.stack(v) for k,v in trace.items()})
    route0=env.atlas['routes'][0]
    np.savez_compressed(args.output/'terrain_env0.npz',**{
        k:route0[k] for k in ['height','x','y','route_y','start','goal']})
    (args.output/'terrain_env0.json').write_text(json.dumps({
        'spec':route0['spec'],'segments':route0['segments']},indent=2))
    if episode_records:
        np.savez_compressed(args.output/'completed_episodes.npz',**{
            k:np.concatenate([v[k] for v in episode_records]) for k in episode_records[0]})
    errors=np.concatenate(event_errors) if event_errors else np.empty((0,3))
    np.savez_compressed(args.output/"touchdown_errors.npz",xyz=errors)
    if batches:
        arrays={k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
        np.savez_compressed(args.output/"transitions.npz",**arrays,candidates=grid.cpu().numpy())
        # New observation/action dimensions need a new model initialization.
        config=ModelConfig(proprio_dim=proprio_dim,action_dim=4)
        initial=CandidateGroundedWorldModel(grid,config)
        torch.save(dict(model=initial.state_dict(),model_config=asdict(config),
                        candidates=grid.cpu(),lower_sha256=lower_hash),args.output/"initial.pt")
    metrics=dict(lower_checkpoint=str(args.lower_checkpoint),lower_sha256=lower_hash,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,behavior=args.behavior,
        motion_checkpoint=str(args.motion_checkpoint) if args.motion_checkpoint else None,
        risk_checkpoint=str(args.risk_checkpoint) if args.risk_checkpoint else None,
        landing_calibration=str(args.landing_calibration) if args.landing_calibration else None,
        planning_horizon=args.horizon if args.behavior=="model" else None,
        action_profile=args.action_profile,terrain_profile=args.terrain_profile,
        terrain_sha256=hashlib.sha256(env.atlas['height'].tobytes()).hexdigest(),
        action_contract='stance_dx21_scale9_dy18_scale6_dz6_yaw6_v1',
        planning_objective=args.planning_objective,
        zero_valid_decisions=zero_valid_count,micro_step_decisions=micro_count,backward_decisions=backward_count,
        proposals_per_beam=args.proposals_per_beam,
        model_exploration_prob=args.model_exploration_prob,
        decision_count=decision_count,exploration_draw_count=exploration_count,
        physical_reward_weight=args.physical_reward_weight,
        goal_cost_radius=args.goal_cost_radius,
        static_geometry_contract='asset_box_20x8cm_v2' if args.asset_footprint else 'proxy_16x7cm_v1',
        action_proposal_mode='bounded_xy_surface_height_v1' if args.local_refinement else 'fixed_grid_v1',
        observation_query_mode='decision_subset_cached_v1',
        root_geometry_guard=args.root_geometry_guard,physics_contract="indexed_reset_no_global_step_v3",
        proprio_dim=proprio_dim,
        progress_kind='clearance_geodesic_v1' if args.geodesic_guidance else 'euclidean',
        success_contract='within_30cm_upright_200ms_v2',
        random_action_prob=args.random_action_prob if args.behavior=='diverse' else 0,
        seed=args.seed,route_offset=args.route_offset,num_envs=n,lower_steps=n*args.steps,difficulty=args.difficulty,
        route_difficulties=[v['spec']['difficulty'] for v in env.atlas['routes']],
        reset_region_prob=args.reset_region_prob,transition_contract="decision_to_decision_v2",
        successes=total_success,falls=total_falls,timeouts=total_timeout,
        right_censored_episodes=int((episode_ticks>0).sum()),envs_with_success=int((env_success>0).sum()),
        successes_per_env=env_success.cpu().tolist(),max_x_per_env_m=max_x.cpu().tolist(),
        mean_farthest_x_m=float(max_x.mean()),
        falls_per_env=env_falls.cpu().tolist(),first_terminals=terminal_audit,
        transitions=len(errors),
        touchdown_xy_median_m=float(np.median(np.linalg.norm(errors[:,:2],axis=1))) if len(errors) else None,
        touchdown_z_median_m=float(np.median(abs(errors[:,2]))) if len(errors) else None,
        wall_seconds=time.perf_counter()-started)
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
    if args.record_video:
        subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","25","-i",
            str(args.output/"frames/%06d.png"),"-c:v","mpeg4","-q:v","3",str(args.output/"rollout.mp4")],check=True)

if __name__=="__main__": main()
