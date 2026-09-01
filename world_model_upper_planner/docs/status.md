# Development status

Date: 2026-09-01

## Completed and verified

- Created the independent `world_model_upper_planner/` project. The existing
  `upper_foothold_planner/` teacher-student/PPO project has no diff.
- Copied the minimum Isaac Gym environment, target interface, assets and frozen
  `lower_model_7000.pt` into the new adapter boundary. Source and checkpoint
  hashes matched at creation.
- Implemented the CG-OWM model: split geometry/dynamics SimNorm latents,
  option transition, per-candidate outcomes, twin Q and proposal policy.
- Implemented uncertainty-aware discrete beam planning with strict static
  reachable masks. There is no continuous CEM and no learned goal head.
- Implemented sequence learning with EMA next-observation consistency, measured
  option outcomes, TD value, policy prior, and optional all-candidate privileged
  geometry labels.
- Seven unit tests pass, including 294 unique candidates, strict reachable
  planning, sparse-mask safety, obstacle geometry, and all-invalid-row safety.

## First real-data Gate

Command:

```bash
conda run -n isaacgym --no-capture-output \
  python scripts/collect_random_options.py \
  --num_envs 16 --lower_ticks 300 --seed 11 \
  --output experiments/gate1_random16x300
```

Result: 180 real frozen-lower option transitions, 4 falls, no successes in the
short random-exploration rollout, and mean option progress 0.0681 m. The data
contains deployable depth/proprio before and after each option plus measured
support, touchdown and terminal labels. No teacher or existing PPO policy was
used.

The geometry-enabled collector also labels all 294 candidate support fractions,
geodesic progress values and valid masks at every decision state, without using
those labels to choose the behavior action. Small-data overfit (180 samples,
300 updates) reduced total loss from 4.481 to 1.587 (ratio 0.354). Final h1
consistency MSE was 0.00275; all-candidate support/progress Huber losses were
0.0195/0.00070. This proves the new data/model/trainer path is executable and
learnable; it does not prove closed-loop planning.

## First complete challenging-terrain result

The first formal dataset contains 7,785 frozen-lower option transitions from
128 random-composite environments (30 simulated seconds each): 155 course
successes, 417 falls, and 0.150 m mean progress per option. Collection took
122.7 seconds at 1,565 lower-environment-steps/s. Training uses environment-ID
disjoint train/validation partitions.

The H1 model could walk, but did not beat its proposal prior on the fixed seed
924 evaluation. This is recorded as a negative result rather than hidden:

| planner | successes / episodes | success | progress / option | candidates |
|---|---:|---:|---:|---:|
| H1 learned score | 194 / 287 | 67.6% | 0.199 m | 18 |
| H1 proposal prior | 197 / 286 | 68.9% | 0.202 m | 53 |

Three-step sequence training formed 6,427 linked samples. The held-out H3
candidate regret improved from about 0.085 m initially to 0.065 m at the chosen
update 700, while H1 stayed near 0.025 m. Normal depth strongly beat shuffled
depth at the final model (for example H1 progress 0.203 versus 0.064 m and H3
progress 0.059 versus -0.031 m), so the model uses terrain geometry.

Fixed-seed 924, random-composite, 64 environments x 30 simulated seconds,
using the exact same H3 checkpoint:

| decision rule | successes / episodes | success | falls | progress / option | candidates |
|---|---:|---:|---:|---:|---:|
| H3 vectorized beam | **222 / 288** | **77.1%** | **66** | **0.212 m** | 77 |
| H1 learned score | 178 / 299 | 59.5% | 121 | 0.202 m | 16 |
| proposal prior | 182 / 286 | 63.6% | 104 | 0.200 m | 58 |

Thus multi-step latent planning contributes +17.5 and +13.4 percentage points
over the same-checkpoint H1 score and prior. This is the first evidence for a
world-model planning benefit; it is not yet a multi-seed or out-of-distribution
paper result.

## Trajectory and video audit

- `experiments/videos/h3_v1_beam_seed42_20s/rollout.mp4`: 3/3 course
  successes, no falls/collisions, continuous narrow-block crossing.
- `experiments/videos/h3_v1_beam_seed123_20s/rollout.mp4`: a reproducible fall
  at lower tick 220 while crossing a larger gap, followed by reset and 2 course
  successes. The model predicted very low mean fall probability (0.00175), so
  recoverability/fall calibration is the dominant diagnosed weakness.
- `experiments/videos/h3_v1_beam_seed333_20s/rollout.mp4`: 3/3 successes and no
  falls in viewer mode. Its headless seed scan had one fall, so it must not be
  presented as a deterministic failure example.

Every evaluation folder retains `metrics.json`, `trajectory_env0.npz`, and
`terminals.npz`; video runs additionally retain frames and MP4.

## Automation and next research gate

`scripts/run_experiment_pipeline.py` provides a resumable full pipeline:
collection, H1 training, H3 training, three same-checkpoint closed-loop
ablations, three videos, per-stage logs, manifests, and automatic result-index
generation. Existing artifacts are reused rather than overwritten.

## 2026-09-01 multi-terrain expansion

V2 adds obstacle-inflated training-only traversability labels, terrain-balanced
sampling/validation, replay across terrain families, and RAM-resident datasets.
Inference remains depth plus deployable proprioception; exact layouts are never
used by the planner. The original lower and teacher/PPO projects remain unchanged.

Large collection used 512 parallel environments x 2,500 lower ticks on the
RTX A6000. It produced 52,532 transitions (4,397 falls, 174 successes) in
668.7 seconds at 1,914 lower-env-steps/s. With the original replay, V2 trains on
60,317 transitions from 640 environment layouts. Loading the NPZ once into RAM
reduced 1,200-update H3 training to 107 seconds; repeated per-batch decompression
had been the main throughput bottleneck.

Same seed 1201, 32 environments x 20 seconds, V1 -> V2:

| terrain | V1 success | V2 success | V1 -> V2 falls | main observation |
|---|---:|---:|---:|---|
| edge cases | 80.7% | 81.6% | 17 -> 16 | retained |
| random composite | 64.7% | 71.3% | 36 -> 27 | improved |
| stepping stones | 60.6% | 70.8% | 39 -> 26 | improved |
| household | 25.2% | 58.5% | 80 -> 27 | large obstacle-domain gain |
| turns | 8.2% | 13.6% | 112 -> 95 | still a failure domain |
| mixed | 52.9% | 61.3% | 49 -> 36 | improved |

Three held-out terrain seeds (1201/1202/1203), pooled V2 outcomes:

| terrain | pooled success | per-seed success | falls | collisions |
|---|---:|---|---:|---:|
| edge cases | 73.4% (201/274) | 81.6 / 80.9 / 59.2% | 73 | 81 |
| random composite | 70.8% (199/281) | 71.3 / 60.8 / 81.1% | 82 | 74 |
| stepping stones | 66.9% (182/272) | 70.8 / 63.7 / 66.3% | 90 | 113 |
| household | 58.3% (134/230) | 58.5 / 60.5 / 56.2% | 96 | 796 |
| turns | 16.7% (53/317) | 13.6 / 25.2 / 11.5% | 264 | 921 |
| mixed | 53.8% (150/279) | 61.3 / 47.7 / 52.0% | 129 | 293 |

The deliberately extreme 4.5 m random-composite course (width 0.40--0.95 m,
gaps up to 0.20 m, obstacles) remains mostly outside the frozen lower/candidate
capability: V1 5.8%, V2 7.0%. Its collection states had only 3.7% valid
candidates and zero course successes, so it is retained as a capability boundary,
not used as evidence that more SGD should solve an infeasible action set.

Route-tangent/yaw supervision was tested in two forms. Full V3 fine-tuning
raised household to 77.8% but caused severe negative transfer (stepping stones
70.8% -> 48.6%, turns 13.6% -> 7.6%). Freezing the V2 world model and adapting
only the proposal policy still reduced most non-household terrains. Both are
rejected as the unified checkpoint; `runs/h3_v2_multiterrain_seed91/model_best.pt`
remains the selected model. Horizon 1/2/3/5 on turns gave about
14.9/14.3/15.4/10.8%, so longer rollout is not the fix.

New video evidence:

- `experiments/videos/terrain_map_v2_household_seed1202_15s/rollout.mp4`:
  successful household navigation run.
- `experiments/videos/terrain_map_v2_household_seed1201_15s/rollout.mp4`:
  obstacle-domain failure near a furniture block.
- `experiments/videos/terrain_map_v2_turns_seed1201_15s/rollout.mp4`:
  reaches the corner but does not rotate sufficiently and falls from the outer edge.

Next research gate: treat sharp turns as a lower-option/action-interface boundary.
Before more world-model training, measure closed-loop yaw realization across the
existing candidate set and determine whether repeated +/-12 degree foothold yaw
actually rotates the base reliably. If not, add an isolated lower adapter or a
temporally extended turn option; do not edit the frozen lower controller.

## 2026-09-01 V4 scale run started

The V2 replay is now treated as feasibility data rather than a sufficient final
training set. A resumable 12-shard, 512-environment scale pipeline has started,
targeting roughly 0.7--0.8 million new option transitions across seven feasible
terrain families and nominal/hard parameter ranges. Large replay merging is now
disk-memmap based; H1/H3 batch sizes were raised to 1024/512 after GPU probes.
Full design, exclusions, commands and live artifact paths are in
`docs/scale_training_v4.md`.
