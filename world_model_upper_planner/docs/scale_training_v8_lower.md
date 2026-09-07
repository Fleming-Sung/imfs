# V8 lower-consistent world-model expansion

Date: 2026-09-02

## Why this branch is justified

V6 risk-H3 is still the selected upper planner, but trajectory inspection found
0.10--0.35 m target-to-foot errors on failing stepping-stone approaches. Static
geometry shields, touchdown penalties, a landing chance score, and command
inversion did not improve task completion. This indicates an action-realization
boundary rather than a lack of generic world-model updates.

The original lower controller and the teacher-student/PPO baseline remain
unchanged. The optional variant is self-contained under
`adapters/lower_tracking_adaptation/`. It was initialized from the copied
`lower_model_7000.pt` and fine-tuned for only 40 PPO iterations (8.192 million
lower-environment steps) with a fixed 3e-5 learning rate and moderate foothold
tracking weights. Its SHA-256 is
`bbd87522ab27aa25238f6d14af96fea5cd481ca76a89187e4fc08cb94f42dd7b`.

## Admission audit

The following fixed-seed audit evaluates 46k option transitions over all 294
upper candidates. Error is target-to-touchdown XY distance.

| commanded yaw | original error | V2 error | original fall | V2 fall |
|---:|---:|---:|---:|---:|
| -6 deg | 4.48 cm | 4.45 cm | 0.54% | 0.92% |
| 0 deg | 3.12 cm | **2.32 cm** | 0.01% | 0.07% |
| +6 deg | 4.79 cm | **3.60 cm** | 0.50% | **0.38%** |

The left-turn stability regression means V2 is not a general replacement for
the original lower controller. It passed only as a difficult-terrain research
branch because its real stepping-stone capability improved consistently:

| seed | original lower | V2 lower | V2 touchdown error |
|---:|---:|---:|---:|
| 5101 | 21.2% | **34.5%** | 3.31 cm |
| 5102 | 19.7% | **35.2%** | 3.47 cm |
| 5103 | 14.9% | **22.4%** | 3.71 cm |
| pooled seed mean | 18.6% | **30.7%** | 3.49 cm |

This is a capability probe using the V6 upper model, so the +12.1 percentage
point result is not a final method comparison: the upper transition model was
trained under the original lower dynamics.

## Representative videos

- [Original-lower risk-H3 stepping-stones failure](../experiments/scale_v6_obstacle/scale/videos_risk/video_09_stones_hard_seed5101/rollout.mp4)
- [V2-lower stepping-stones diagnostic](../experiments/lower_adaptation/video_model7040_v2_stones_seed5102/rollout.mp4)
- [V6 household-hard success](../experiments/scale_v6_obstacle/scale/videos/video_05_household_hard_seed5101/rollout.mp4)
- [V6 irregular-hard trajectory](../experiments/scale_v6_obstacle/scale/videos/video_11_irregular_hard_seed5101/rollout.mp4)

The V2 diagnostic still contains a fall; it is retained because representative
videos must expose failure modes rather than show only successful episodes.
Each directory also contains metrics and trajectory arrays.

## Replay-semantics safeguard and next run

Collection and evaluation now accept an explicit `--lower_checkpoint`. The
experiment manifest and every shard summary record its SHA-256, and reusing an
experiment namespace with a different lower hash is rejected. A 12-terrain
smoke pipeline completed collection, merge, H1/H3 training, and evaluation
under the V2 hash; 12 unit tests pass.

The formal V8 experiment was started to recollect all 12 scenario/difficulty shards with
2,048 parallel environments and 750 lower ticks per shard. No V6 transition is
mixed into this replay. H1/H3 and finite-horizon risk models will then be
retrained under the exact V2 dynamics, followed by the same 36 held-out runs
and representative hard-scenario videos. V2 is accepted only if it improves
edge/stones without sacrificing the already strong bridge, household, and
irregular families.

Status audit on 2026-09-06: the job stopped after 2/12 shards. The two complete
shards contain 124,984 transitions and 3.072M lower-environment interactions;
the remaining collection, merge, training and formal evaluation have not run.
See [progress_2026-09-06.md](progress_2026-09-06.md) for the exact inventory.
