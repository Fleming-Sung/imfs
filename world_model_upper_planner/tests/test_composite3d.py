import unittest
import numpy as np
import torch
from composite_terrain.maps import CompositeSpec, generate_route, build_atlas, FAMILIES
from lower_controller_3d.config import get_flat_config
from lower_controller_3d.sampler import FootholdSampler


class Composite3DTest(unittest.TestCase):
    def test_subset_observations_equal_full_batch_without_full_map_query(self):
        from types import SimpleNamespace
        from composite_terrain.interface import sense,panorama
        n=4;torch.manual_seed(123)
        quat=torch.zeros(n,4);quat[:,3]=1
        rigid=torch.zeros(n,2,13);rigid[:,:,6]=1
        queried=[]
        def height(xy):
            queried.append(len(xy))
            return .02*torch.sin(xy[...,0])+.01*torch.cos(xy[...,1])
        sampler=SimpleNamespace(swing_foot=torch.tensor([0,1,0,1]),phase=torch.rand(n),
            observation=lambda feet,quaternions:torch.arange(n*16).float().reshape(n,16))
        env=SimpleNamespace(num_envs=n,device='cpu',base_quat=quat,
            base_position=torch.rand(n,3),foot_positions=torch.rand(n,2,3),
            feet_indices=torch.arange(2),rigid_body_states=rigid,sampler=sampler,
            goals=torch.rand(n,3),base_lin_vel=torch.rand(n,3),base_ang_vel=torch.rand(n,3),
            projected_gravity=torch.tensor([[0.,0,-1]]).expand(n,-1),
            dof_pos=torch.rand(n,8),dof_vel=torch.rand(n,8),last_actions=torch.rand(n,2,8),
            sample_height=height)
        previous=torch.rand(n,4);ids=torch.tensor([3,1])
        full_image,full_prop=sense(env,previous,75);full_cache=panorama(env)
        image,prop=sense(env,previous,75,ids);cache=panorama(env,ids)
        self.assertTrue(torch.equal(image,full_image[ids]))
        self.assertTrue(torch.equal(prop,full_prop[ids]))
        self.assertTrue(torch.equal(cache,full_cache[ids]))
        self.assertEqual(queried,[4,4,2,2])

    def test_risk_labels_include_short_terminals_but_not_censored_safety(self):
        from composite_terrain.train_risk import risk_labels
        data=dict(env_id=np.array([0,0,1,1,1,2,3]),episode_id=np.zeros(7,int),
            option_index=np.array([0,1,0,1,2,0,0]),fall=np.array([0,1,0,0,0,0,0]),
            collision=np.zeros(7),done=np.array([0,1,0,0,0,1,1]),
            success=np.array([0,0,0,0,0,1,0]))
        labels,known=risk_labels(data)
        np.testing.assert_array_equal(known,[True,True,True,False,False,True,False])
        np.testing.assert_array_equal(labels[known,0],[1,1,0,0])

    def test_physical_goal_progress_penalizes_body_overshoot(self):
        from composite_terrain.anchored import physical_goal_progress
        prop=torch.zeros(2,75);prop[:,33]=torch.tensor([1.,.1])/6
        progress,distance=physical_goal_progress(prop,torch.tensor([[.2,0,0,0],[.4,0,0,0]]))
        self.assertTrue(torch.allclose(progress,torch.tensor([.2,-.2]),atol=1e-6))
        self.assertTrue(torch.allclose(distance,torch.tensor([1.,.1]),atol=1e-6))

    def test_full_candidate_coverage_can_escape_prior_shortlist(self):
        from types import SimpleNamespace
        from composite_terrain.anchored import AnchoredPlanner
        from cgowm.planner import PlannerConfig
        actions=torch.zeros(180,4)
        def outcomes(latent):
            n=len(latent);zeros=torch.zeros(n,180)
            progress=zeros.clone();progress[:,37]=1
            return dict(policy_logits=-torch.arange(180).float()[None].expand(n,-1),
                progress=progress,support=torch.ones_like(zeros),
                fall_logit=zeros-20,collision_logit=zeros-20,
                continuation_logit=zeros+20,q=torch.zeros(2,n,180))
        world=SimpleNamespace(candidates=actions,encode=lambda image,prop:prop,
                              predict_candidates=outcomes)
        motion=SimpleNamespace(predict=lambda z,p,a:(torch.zeros(len(p),4),p,torch.zeros(len(p))))
        image=torch.zeros(1,1,64,64);prop=torch.zeros(1,75);cache=torch.zeros(1,1,128,128)
        short=AnchoredPlanner(world,motion,PlannerConfig(horizon=1,proposals_per_beam=8,terminal_value_weight=0))
        full=AnchoredPlanner(world,motion,PlannerConfig(horizon=1,proposals_per_beam=180,terminal_value_weight=0))
        self.assertNotEqual(int(short.plan(image,prop,cache)[0]),37)
        self.assertEqual(int(full.plan(image,prop,cache)[0]),37)
        mask=torch.ones(1,180,dtype=torch.bool);mask[:,37]=False
        self.assertNotEqual(int(full.plan(image,prop,cache,mask)[0]),37)

    def test_exploration_uses_supported_actions_and_preserves_empty_fallback(self):
        from composite_terrain.interface import explore_supported
        chosen=torch.tensor([0,1,2]);valid=torch.zeros(3,180,dtype=torch.bool)
        valid[0,37]=True;valid[1,4]=True
        result,draw=explore_supported(chosen,valid,1.)
        self.assertTrue(torch.equal(result,torch.tensor([37,4,2])))
        self.assertTrue(torch.equal(draw,torch.tensor([True,True,False])))
        unchanged,draw=explore_supported(chosen,valid,0.)
        self.assertTrue(torch.equal(unchanged,chosen));self.assertFalse(draw.any())

    def test_refinement_uses_observed_height_and_bounded_continuous_actions(self):
        from types import SimpleNamespace
        from cgowm import CandidateGroundedWorldModel, ModelConfig
        from cgowm.planner import PlannerConfig
        from composite_terrain.interface import candidates
        from composite_terrain.anchored import MotionModel, AnchoredPlanner
        from composite_terrain.refine import refine_root
        rigid=torch.zeros(2,2,13);rigid[:,:,6]=1
        env=SimpleNamespace(device='cpu',foot_positions=torch.zeros(2,2,3),
            feet_indices=torch.arange(2),rigid_body_states=rigid,
            sampler=SimpleNamespace(swing_foot=torch.tensor([0,1])),
            goals=torch.tensor([[6.,0,0],[6.,0,0]]),
            sample_height=lambda xy:torch.full(xy.shape[:-1],.06))
        model=CandidateGroundedWorldModel(candidates('cpu'),ModelConfig(
            proprio_dim=75,action_dim=4,geometry_dim=16,dynamics_dim=16,hidden_dim=32))
        planner=AnchoredPlanner(model,MotionModel(32,75),PlannerConfig(horizon=1))
        initial=torch.zeros(2,4);prop=torch.zeros(2,75)
        image=torch.full((2,1,64,64),.5);cache=torch.full((2,1,128,128),.5)
        refined=refine_root(env,torch.arange(2),initial,image,prop,cache,planner)
        self.assertTrue(torch.allclose(refined[:,2],torch.full((2,),.75)))
        self.assertTrue((refined[:,0].abs()*.09<=.020001).all())
        self.assertTrue((refined[:,1].abs()*.06<=.020001).all())
        env.sample_height=lambda xy:torch.full(xy.shape[:-1],-.8)
        fallback=refine_root(env,torch.arange(2),initial,image,prop,cache,planner)
        self.assertTrue(torch.equal(fallback,initial))

    def test_landing_projection_respects_next_stance_and_yaw(self):
        from composite_terrain.anchored import predicted_landing
        state=torch.zeros(1,75);state[:,2]=-1
        state[:,22:28]=torch.tensor([.1,.05,-.65,-.1,-.05,-.65])
        state[:,42]=1  # next swing right, landed stance left
        landing=predicted_landing(torch.tensor([[.2,0,torch.pi/2,.03]]),state)
        self.assertTrue(torch.allclose(landing,torch.tensor([[.15,.1,.03]]),atol=1e-6))

    def test_static_cache_warp_and_pose_composition(self):
        from composite_terrain.anchored import crop_map,compose_pose
        cache=torch.linspace(0,1,128)[None,None,:,None].expand(2,1,128,128).clone()
        pose=torch.zeros(2,4)
        patch,known=crop_map(cache,pose)
        expected=torch.linspace(10/127,73/127,64)
        self.assertTrue(torch.allclose(patch[0,0,:,20],expected,atol=1e-6))
        self.assertTrue(known.all())
        pose[1,2]=torch.pi/2
        result=compose_pose(pose,torch.tensor([[.2,0,0,.03],[.2,0,0,.03]]))
        self.assertAlmostEqual(float(result[1,1]),.2,places=5)
        self.assertAlmostEqual(float(result[1,3]),.03,places=5)

    def test_motion_model_bookkeeping_and_gradients(self):
        from composite_terrain.anchored import MotionModel
        model=MotionModel(32,75)
        action=torch.rand(4,4);prop=torch.rand(4,75)
        motion,state=model(torch.rand(4,32),prop,action)
        self.assertEqual(motion.shape,(3,4,4))
        self.assertTrue(torch.allclose(state[0,:,30:33],action[:,[0,1,3]]))
        self.assertTrue(torch.allclose(state[0,:,41:43],prop[:,41:43].flip(-1)))
        (motion.square().mean()+state.square().mean()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_navigation_labels_are_finite_static_geometry(self):
        from composite_terrain.navigation import route_field
        route=generate_route(CompositeSpec(seed=9502000,difficulty=.25))
        field=route_field(route)
        self.assertTrue(np.isfinite(field).all())
        gx=round(float(route['goal'][0])/.05)
        gy=round((float(route['goal'][1])+2.4)/.05)
        self.assertAlmostEqual(float(field[0,gy,gx]),0.)
        self.assertGreater(float(field[0,48,6]),15.)

    def test_terminal_reset_state_does_not_train_dynamics(self):
        from cgowm import CandidateGroundedWorldModel,ModelConfig
        from composite_terrain.trainer import WorldModelTrainer
        from composite_terrain.interface import candidates
        grid=candidates('cpu')
        self.assertEqual(grid.shape,(324,4))
        model=CandidateGroundedWorldModel(grid,ModelConfig(proprio_dim=75,
            action_dim=4,geometry_dim=16,dynamics_dim=16,hidden_dim=32))
        trainer=WorldModelTrainer(model)
        batch=dict(depth=torch.rand(2,2,1,64,64),proprio=torch.rand(2,2,75),
            action=grid[:2,None],next_action=grid[:2,None],bootstrap_valid=torch.zeros(2,1))
        for name in ['reward','progress','support','touchdown_error','fall','collision']:
            batch[name]=torch.zeros(2,1)
        batch['done']=torch.ones(2,1)
        loss,metrics=trainer.loss(batch)
        self.assertEqual(float(metrics['consistency_h1']),0.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_deterministic_routes_contain_every_family(self):
        a=generate_route(CompositeSpec(seed=8)); b=generate_route(CompositeSpec(seed=8))
        np.testing.assert_array_equal(a["height"],b["height"])
        self.assertEqual(set(v["kind"] for v in a["segments"]),set(FAMILIES))
        self.assertGreater(a["goal"][0],15)

    def test_atlas_spawn_and_goal_have_real_support(self):
        atlas=build_atlas([CompositeSpec(seed=i) for i in range(12)])
        for origin,route in zip(atlas["origins"],atlas["routes"]):
            for point in [route["start"],route["goal"]]:
                xyz=origin+point; ix,iy=np.round(xyz[:2]/atlas["resolution"]).astype(int)
                self.assertLess(abs(atlas["height"][iy,ix]-xyz[2]),.003)

    def test_region_boundaries_are_supported_and_height_continuous(self):
        for seed in range(30):
            route=generate_route(CompositeSpec(seed=seed,difficulty=.6))
            for s in route["segments"]:
                ix=round(s["end_x"]/.05); iy=round((s["end_y"]+2.4)/.05)
                h=route["height"][iy,ix-1:ix+2]
                self.assertTrue(np.all(abs(h-s["end_z"])<.004))

    def test_goal_height_is_sampled_relative_to_stance(self):
        cfg=get_flat_config().foothold; cfg.z_distance=[.035,.035]
        sampler=FootholdSampler(16,cfg,"cpu")
        ids=torch.arange(16); stance=torch.zeros(16,3); stance[:,2]=.12
        target,_=sampler._sample_candidate(ids,torch.zeros(16,dtype=torch.long),stance,torch.zeros(16))
        self.assertTrue(torch.allclose(target[:,2],torch.full((16,),.155)))

if __name__=="__main__": unittest.main()
