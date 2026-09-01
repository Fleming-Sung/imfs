# CG-OWM research hypothesis and experiment contract

## Hypothesis

A world model specialized to the *closed-loop foothold option* can outperform a
model-free candidate policy in sample efficiency and out-of-distribution terrain
adaptation, provided that representation and planning are grounded by candidate
geometry and constrained to the frozen lower controller's reachable action set.

This is falsified if it cannot beat the strong teacher-student/PPO baseline's
learning curve or OOD performance under matched observations, action candidates,
lower controller, terrain splits and environment interactions.

## Method

Each upper action is a semi-Markov option: choose a foothold candidate, let the
frozen lower execute until the next genuine gait boundary, then observe the
robot-level outcome. The model contains:

1. separate geometry and robot-dynamics latents from depth and deployable
   proprioception;
2. option-conditioned latent transitions aligned at every imagined step to an
   EMA encoding of the real next observation;
3. parallel per-candidate predictions of progress, support, touchdown error,
   collision, fall, continuation, reward and twin Q;
4. privileged support, obstacle-inflated traversability, geodesic progress and
   touchdown supervision during training only;
5. discrete uncertainty-aware beam planning over the empirically reachable
   candidate set, with a learned proposal policy and pessimistic terminal Q.

There is deliberately no learned goal-probability head. Goal completion is
known task logic. There is no continuous CEM and no unconstrained action cube.

## Lessons encoded from the failed predecessor

- Action aliasing: use the same finite Cartesian candidate set as all baselines.
- Latent exploitation: real-next EMA consistency, short horizons, ensemble
  disagreement, and a mandatory horizon-1 diagnostic precede longer planning.
- Rare-event miscalibration: report positive rate, Brier score and calibration;
  safety heads are penalties, never a high-weight synthetic success bonus.
- Online feedback instability: alternate frozen collection phases and offline
  update phases. Do not update the model after every single transition.
- Misleading loss trends: promote a checkpoint only by fixed-seed closed-loop
  episode outcomes and trajectory/video inspection.

## Minimal experiment matrix

All methods use the same frozen lower, depth+proprio Actor input, 294 candidates,
training terrain seeds and held-out seeds.

| Axis | CG-OWM | Required baselines/ablations |
|---|---|---|
| Data efficiency | success vs real option transitions and wall time | PPO from scratch, BC+PPO, curriculum PPO |
| Final behavior | success/fall/timeout, path efficiency, touchdown error | same metrics and 30 s videos |
| Generalization | nominal, narrow, large-gap, combined-hard, unseen layouts | existing strong PPO checkpoint |
| Model value | H=1, H=3 beam | policy-only prior; no geometry grounding; no uncertainty |
| Calibration | Brier/ECE and risk-coverage | single model vs ensemble |

Primary claim is not allowed until at least three training seeds and disjoint
terrain seeds support it. Reward, survival, or one successful video is not proof.

## Efficient training on the current machine

- Physics collection and SGD are separated; no per-transition 512-sample UTD.
- Start with sample UTD 8 and increase only if held-out multi-step consistency
  and candidate ranking improve.
- Render depth only at synchronized upper decision boundaries. Store uint8 depth
  and compact labels in CPU replay; decode minibatches on GPU.
- Load the compressed replay once into CPU RAM; repeated NPZ decompression per
  minibatch is substantially slower than GPU training on the current machine.
- Train H=1 first. Enable H=3 planning only after H=1 closed-loop beats the
  policy prior and normal depth beats shuffled/zero-depth on future prediction.
- Use 2048 camera-free environments for lower/action gates and up to the tested
  512 camera environments for collection; choose the largest
  camera batch that improves *wall-clock transitions/s*, not nominal env count.

## Literature boundary

TD-MPC learns task-oriented latent dynamics, reward and terminal value for
short-horizon local planning. TD-MPC2 adds robust normalization/scaling and a
policy prior. CG-OWM adopts that research idea but changes the problem unit to a
closed-loop foothold option and adds candidate-grounded geometry plus constrained
discrete planning. It is not a reimplementation claim.
