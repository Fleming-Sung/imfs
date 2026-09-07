# V6 obstacle-complete world-model result

Date: 2026-09-02

V6 fixes a state-definition error found by inspecting V5 videos: household and
research obstacles are separate static actors and were absent from the Isaac
heightfield.  The privileged terrain adapter now rasterizes their true
footprints and heights into the same 64x64 ego-centric terrain observation.
This remains a true-terrain upper-planning experiment; it does not use a
teacher action, future simulator state, or task outcome at inference.  The
frozen lower policy and the existing teacher-student/PPO project are unchanged.

## Scale and runtime

- 12 terrain/difficulty shards and 2,048 environments per shard.
- 24,576 procedural layouts, 749,541 upper-option transitions, and 18.432M
  lower-environment interactions.
- Collection: 2,282.8 s; memmap merge: 20.8 s.
- H1 training: 6,000 updates / 715.9 s; H3 training: 5,000 updates / 909.7 s.
- Three held-out seeds per scenario, 32 environments x 1,000 lower ticks:
  36 independent closed-loop evaluations.

The obstacle visibility smoke audit is categorical rather than a small loss
change: V5 household observations had obstacle-height pixels in 0% of frames;
V6 has them in 100% of frames.  Ten unit tests pass, including a static-obstacle
rasterization regression test.

## Closed-loop result

| Scenario | V4 | V5 | V6 | V5 to V6 |
|---|---:|---:|---:|---:|
| research nominal | 65.5% | 64.4% | **71.6%** | +7.2 pp |
| research hard | 17.2% | 25.4% | 25.3% | -0.1 pp |
| turns nominal | 16.2% | 36.0% | **40.0%** | +4.0 pp |
| turns hard | 4.0% | 11.4% | **25.4%** | +14.0 pp |
| household nominal | 61.8% | 49.9% | **77.6%** | +27.7 pp |
| household hard | 45.5% | 32.5% | **61.3%** | +28.8 pp |
| bridge nominal | 100.0% | 100.0% | **100.0%** | 0.0 pp |
| bridge hard | 93.6% | **97.9%** | 94.7% | -3.2 pp |
| edge hard | 23.5% | 22.8% | **25.5%** | +2.7 pp |
| stones hard | 18.2% | 19.3% | **20.4%** | +1.1 pp |
| irregular nominal | 93.0% | **98.6%** | 96.5% | -2.1 pp |
| irregular hard | 53.8% | 73.0% | **77.7%** | +4.7 pp |

The large household and hard-turn improvements validate both the corrected
terrain state and curvature option.  Bridge/irregular remain strong.  Edge and
stepping-stone results remain poor despite full-scale coverage, so they are the
current bottleneck and not evidence for further undirected SGD.

## Representative videos

These are single-seed trajectory audits from the same V6 checkpoint and state
definition as the table.  They are diagnostic examples, not substitutes for
the 36-run aggregate statistics.

- [research hard](../experiments/scale_v6_obstacle/scale/videos/video_01_research_hard_seed5101/rollout.mp4)
- [turns hard](../experiments/scale_v6_obstacle/scale/videos/video_03_turns_hard_seed5101/rollout.mp4)
- [household hard](../experiments/scale_v6_obstacle/scale/videos/video_05_household_hard_seed5101/rollout.mp4)
- [bridge hard](../experiments/scale_v6_obstacle/scale/videos/video_07_bridge_hard_seed5101/rollout.mp4)
- [edge hard](../experiments/scale_v6_obstacle/scale/videos/video_08_edge_hard_seed5101/rollout.mp4)
- [irregular hard](../experiments/scale_v6_obstacle/scale/videos/video_11_irregular_hard_seed5101/rollout.mp4)

Every video directory also retains `metrics.json`, `trajectory_env0.npz`, and
`terminals.npz` for numerical and terminal-event auditing.

## Decision and next experiment

The H3 held-out selection statistics do not consistently beat the proposal
prior, especially after multiple latent transitions.  Increasing beam width or
planning horizon would therefore amplify model exploitation rather than solve
the difficult terrains.  The next controlled experiments are:

1. train and evaluate the finite-horizon option risk critic on V6 replay;
2. compare H1 and H3 closed loop with the identical checkpoint and seeds;
3. inspect edge/stones terminal trajectories and videos;
4. if risk/H1 do not materially close the gap, test a terrain-map-derived
   analytic feasibility shield while keeping dynamics and outcome prediction
   learned.  Any simulator-layout oracle is diagnostic only and will not be
   reported as the final method.

## Risk-H3 follow-up

The finite-horizon option risk critic was trained on 561,597 linked V6
sequences. Its held-out fall/collision rates are 11.43%/36.85%, predicted means
are 10.33%/34.80%, and Brier scores are 0.0796/0.1780. Candidate-wise predicted
risk ranges average 0.139/0.241, so the critic is calibrated enough to rank
actions rather than collapsing to a constant prior.

| Scenario | raw H3 | risk H3 | Delta |
|---|---:|---:|---:|
| research nominal | 71.6% | **80.7%** | +9.1 pp |
| research hard | 25.3% | **41.4%** | +16.1 pp |
| turns nominal | 40.0% | **47.1%** | +7.1 pp |
| turns hard | 25.4% | **29.8%** | +4.4 pp |
| household nominal | 77.6% | **78.7%** | +1.1 pp |
| household hard | 61.3% | **67.4%** | +6.1 pp |
| bridge nominal | **100.0%** | **100.0%** | 0.0 pp |
| bridge hard | **94.7%** | 94.2% | -0.5 pp |
| edge hard | 25.5% | **37.2%** | +11.7 pp |
| stones hard | 20.4% | **21.8%** | +1.4 pp |
| irregular nominal | 96.5% | **97.3%** | +0.8 pp |
| irregular hard | 77.7% | **83.4%** | +5.7 pp |

Risk-H3 is the selected V6 planner so far: it gives material gains on mixed,
hard, and edge terrain with only a 0.5 pp bridge-hard regression.  Stepping
stones remain unchanged and are now isolated as a geometry/action-coverage
problem.  A same-checkpoint H1 closed-loop ablation is running before any new
architecture is introduced.

## Same-checkpoint horizon ablation

| Scenario | raw H1 | raw H3 | risk H3 |
|---|---:|---:|---:|
| research nominal | 74.1% | 71.6% | **80.7%** |
| research hard | 28.6% | 25.3% | **41.4%** |
| turns nominal | **48.5%** | 40.0% | 47.1% |
| turns hard | 24.0% | 25.4% | **29.8%** |
| household nominal | 75.5% | 77.6% | **78.7%** |
| household hard | 65.4% | 61.3% | **67.4%** |
| bridge nominal | **100.0%** | **100.0%** | **100.0%** |
| bridge hard | **97.3%** | 94.7% | 94.2% |
| edge hard | 13.7% | 25.5% | **37.2%** |
| stones hard | 14.6% | 20.4% | **21.8%** |
| irregular nominal | 96.7% | 96.5% | **97.3%** |
| irregular hard | 71.6% | 77.7% | **83.4%** |

H1 is useful on nominal turns and bridge hard, but catastrophically worse on
edge/stones. Multi-step foresight is therefore necessary on discontinuous
support, while unregularized latent rollout is not sufficient. Risk-H3 remains
the unified checkpoint/planner. The next experiment is a current-observation
analytic feasibility shield feeding only the first beam-search action; using a
current mask at imagined future steps would be a coordinate error.

## Trajectory-driven V7 diagnosis

The selected risk-H3 stones video is
[here](../experiments/scale_v6_obstacle/scale/videos_risk/video_09_stones_hard_seed5101/rollout.mp4).
It contains two falls in 15 seconds. The retained trajectory shows large
target-to-foot errors on the failing approach (several 0.10--0.35 m events),
where ordinary options are typically 0.01--0.04 m. This motivated explicit
closed-loop landing modeling rather than more generic world-model updates.

Two targeted V7 shards contain 126,589 transitions from 4,096 new edge/stones
layouts and all 294 actions. Collection took about 113 seconds. A new target-
frame 2-D Gaussian landing ensemble was trained on 110,081 valid non-fall
touchdowns; falls are excluded because Isaac has already reset their physical
pose. Environment-disjoint validation gives 3.69 cm RMSE, 1.79 cm MAE, less
than 3 mm bias, and 96.0% two-sigma coverage.

The following one-seed stones diagnostics were deliberately stopped before a
full sweep because each failed its admission gate:

| planner diagnostic | success | off-support | conclusion |
|---|---:|---:|---|
| selected risk-H3 | 21.2% | 79 | baseline |
| static current-map shield | 12.3% | 51 | safer but too conservative |
| touchdown scalar penalty | 18.5% | 84 | magnitude alone is insufficient |
| landing chance score | 17.7% | 67 | safer, lower completion |
| learned inverse command | 15.8% | 77 | breaks planner/action semantics |
| exact geometry + risk (oracle) | 29.6% | 45 | upper geometry has headroom |
| exact greedy progress (oracle) | 0.8% | 106 | ignores lower dynamics |

Oracle rows are capability diagnostics only and are not method results. The
combined evidence shows that the world-model dynamics/risk term is necessary,
but the immutable lower controller is now a measured limit on narrow discrete
supports. Any lower adaptation must live in
`adapters/lower_tracking_adaptation/`; transition replay must then be
recollected under the new checkpoint hash.
