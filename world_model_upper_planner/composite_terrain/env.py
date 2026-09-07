"""Static composite-map physics. The trained lower actor stays frozen here."""
from pathlib import Path
import numpy as np
from isaacgym import gymapi, gymtorch, terrain_utils
import torch
from adapters.frozen_lower_env.env import FootholdEnv, make_sim_params
from adapters.frozen_lower_env.config import AttrDict
from adapters.frozen_lower_env.factory import disable_domain_randomization
from adapters.frozen_lower_env.sampler import FootholdSampler
from .maps import CompositeSpec, build_atlas


class ClockSampler(FootholdSampler):
    def _sample_next(self, *args):
        pass  # Static-map goals are issued by the upper planner only.

    def step(self, *args):
        return self.last_switch_ids

    def advance(self, dt, positions, quaternions):
        return super().step(dt, positions, quaternions)


class CompositeEnv(FootholdEnv):
    def __init__(self, cfg, params, device, atlas, headless):
        self.atlas = atlas
        self.surface = torch.as_tensor(atlas["height"],device=device)
        self.resolution = atlas["resolution"]
        self.region_spawns=torch.tensor([[
            [s["start_x"]+.10,s["start_y"]+v["spec"]["width"]/2,s["start_z"]]
            for s in v["segments"]] for v in atlas["routes"]],device=device)
        super().__init__(cfg,params,device,headless)
        self.sampler = ClockSampler(self.num_envs,cfg.foothold,self.device)
        self.sampler.reset(torch.arange(self.num_envs,device=self.device),
                           self.foot_positions,self.rigid_body_states[:,self.feet_indices,3:7])
        self._compute_observations()

    def _create_ground(self):
        # Verticalize abrupt support edges. Ordinary heightfield interpolation
        # would turn pits and stair risers into artificial inclined supports.
        samples=np.round(self.atlas["height"].T/.002).astype(np.int16)
        vertices,triangles=terrain_utils.convert_heightfield_to_trimesh(
            samples,self.resolution,.002,slope_threshold=.7)
        params=gymapi.TriangleMeshParams(); params.nb_vertices=len(vertices)
        params.nb_triangles=len(triangles); params.static_friction=1.0; params.dynamic_friction=1.0
        self.gym.add_triangle_mesh(self.sim,vertices.flatten(),triangles.flatten(),params)

    def sample_height(self, xy):
        index=torch.round(xy/self.resolution).long()
        inside=(index[...,0]>=0)&(index[...,0]<self.surface.shape[1])&(
            index[...,1]>=0)&(index[...,1]<self.surface.shape[0])
        z=self.surface[index[...,1].clamp(0,self.surface.shape[0]-1),
                       index[...,0].clamp(0,self.surface.shape[1]-1)]
        return torch.where(inside,z,torch.full_like(z,-.8))

    def _reset_idx(self, ids):
        # All reset sites are physical wide region-entry platforms; the route
        # and goal stay unchanged. Only collection uses these extra starts.
        prob=float(getattr(self.cfg.env,"reset_region_prob",0.0))
        if not len(ids) or prob<=0:
            return self._reset_without_global_step(ids)
        original=self.base_init_state
        state=original[None].repeat(len(ids),1)
        use=torch.rand(len(ids),device=self.device)<prob
        k=torch.randint(self.region_spawns.shape[1],(len(ids),),device=self.device)
        pose=self.region_spawns[ids,k]
        state[use,:2]=pose[use,:2]
        state[use,2]=original[2]+pose[use,2]
        self.base_init_state=state
        try:
            self._reset_without_global_step(ids)
        finally:
            self.base_init_state=original

    def _reset_without_global_step(self, ids):
        """Reset indexed robots; never advance healthy robots at zero torque.

        The inherited reset inserts two unclocked global physics steps. In
        asynchronous maps each failed robot then perturbs every healthy one.
        Let the next ordinary controlled step refresh derived body states.
        """
        if not len(ids): return
        self.dof_pos[ids]=self.reset_dof_pos[ids]; self.dof_vel[ids]=0
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.robot_actor_indices[ids]),ids.numel())
        self.root_states[ids,0]=self.base_init_state
        self.root_states[ids,0,:3]+=self.env_origins[ids]
        self.root_states[ids,0,7:13]=0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self.robot_actor_indices[ids]),ids.numel())
        self._randomize_pd_gains(ids)
        if self.cfg.domain_rand.randomize_gravity:
            lo,hi=self.cfg.domain_rand.gravity_magnitude_range
            self.gravity_magnitude[ids]=lo+(hi-lo)*torch.rand(len(ids),device=self.device)
        for value in [self.actions,self.policy_actions,self.last_actions,
                      self.episode_length_buf,self.reset_buf,self.fail_buf]: value[ids]=0
        self.goal_reset_pending[ids]=True
        for name in self.episode_sums:
            self.extras.setdefault("episode",{})[f"rew_{name}"]=float(
                self.episode_sums[name][ids].mean())/self.cfg.env.episode_length_s
            self.episode_sums[name][ids]=0

    def _check_termination(self):
        rows=torch.arange(self.num_envs,device=self.device)
        stance=1-self.sampler.swing_foot
        reference=self.sample_height(self.sampler.target_pos[rows,stance,:2])
        height=self.base_position[:,2]-reference
        low=(height<.32)|~torch.isfinite(height)
        tilt=self.projected_gravity[:,2]>-.5
        contact=torch.norm(self.contact_forces[:,self.nonfoot_indices],dim=-1).max(-1).values>40
        failed=low|tilt|contact
        self.fail_buf=torch.where(failed,self.fail_buf+1,torch.zeros_like(self.fail_buf))
        self.time_out_buf=self.episode_length_buf>=self.max_episode_length
        self.reset_buf=((self.fail_buf>self.fail_to_terminal)|self.time_out_buf).long()
        self.extras["termination_reasons"]=dict(height=low,height_above_lower_reference_limit=height>.85,
                                                tilt=tilt,nonfoot_contact=contact)
        self.extras["pre_reset_root"]=self.root_states[:,0].clone()
        self.extras["pre_reset_feet"]=self.foot_positions.clone()


def create_composite_env(args):
    root=Path(__file__).resolve().parents[1]
    ckpt=torch.load(args.lower_checkpoint,map_location="cpu",weights_only=False)
    cfg=AttrDict.from_nested(ckpt["config"])
    cfg.asset.file=str(root/"assets/SF_TRON1A/urdf/robot.urdf")
    cfg.env.num_envs=args.num_envs; cfg.env.episode_length_s=60.0
    cfg.env.termination_mode="upper_joint"; cfg.env.fail_to_terminal_time_s=.1
    cfg.env.reset_region_prob=float(getattr(args,"reset_region_prob",0.0))
    levels=[float(v) for v in getattr(args,'difficulty_levels','').split(',') if v]
    specs=[CompositeSpec(seed=args.seed*1000+getattr(args,'route_offset',0)+i,
        difficulty=levels[i%len(levels)] if levels else args.difficulty) for i in range(args.num_envs)]
    atlas=build_atlas(specs); ny,nx=atlas["routes"][0]["height"].shape
    cfg.env.env_spacing_xy=[nx*specs[0].resolution+.8,ny*specs[0].resolution+.8]
    cfg.init.spawn_xy=[.3,specs[0].width/2]
    # The reset knee angles extend the foot site ~7 cm below the base-0.663m
    # convention. Solid pillars resolve that penetration; a one-sided mesh
    # cannot. Start above the static surface so feet approach it from above.
    cfg.init.pos[2]=.74
    cfg.camera=AttrDict(enabled=False)
    disable_domain_randomization(cfg)
    env=CompositeEnv(cfg,make_sim_params(cfg,args),args.sim_device,atlas,args.headless)
    # Atlas origins are route-local centerline origins, not lower env origins.
    env.route_origins=torch.as_tensor(atlas["origins"],device=env.device)
    expected=env.env_origins.clone(); expected[:,1]+=specs[0].width/2
    if not torch.allclose(expected,env.route_origins,atol=1e-4):
        raise RuntimeError("physics env origins do not match map atlas")
    env.goals=env.route_origins+torch.as_tensor(np.stack([v["goal"] for v in atlas["routes"]]),device=env.device)
    return env
