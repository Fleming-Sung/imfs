# V5 privileged-terrain + curvature-option scale result

V5 isolates the upper-planning hypothesis from camera perception. The model
receives a simulator-true ego-centric 64x64 metric height map; the frozen lower
policy is unchanged. An isolated curvature target adapter maps the weak foot-yaw
coordinate into non-crossing inner/outer foothold asymmetry (gain 1.5).

## Data and compute

- 12 terrain/difficulty shards, 2,048 parallel environments per shard.
- 750 lower ticks per environment: 18.432M lower-env interactions total.
- 24,576 independent procedural layouts and 749,757 option transitions.
- Camera probe: 2,844 lower-env steps/s at 1,024 envs.
- True-height observation: 26,197 steps/s at 1,024 envs and 36,875 steps/s
  at 2,048 envs (about 9.2x over the matched camera probe).
- Collection stage wall time: 2,160.6 s total, including 12 simulator builds.
- H1: 6,000 updates / 608.3 s. H3: 5,000 updates / 767.1 s.

## Closed-loop three-seed results

| Scenario | V4 raw | V5 | Delta |
|---|---:|---:|---:|
| research nominal | 65.5% | 64.4% | -1.1 pp |
| research hard | 17.2% | 25.4% | +8.2 pp |
| turns nominal | 16.2% | 36.0% | +19.8 pp |
| turns hard | 4.0% | 11.4% | +7.4 pp |
| household nominal | 61.8% | 49.9% | -11.9 pp |
| household hard | 45.5% | 32.5% | -13.0 pp |
| bridge nominal | 100.0% | 100.0% | 0.0 pp |
| bridge hard | 93.6% | 97.9% | +4.3 pp |
| edge hard | 23.5% | 22.8% | -0.7 pp |
| stones hard | 18.2% | 19.3% | +1.1 pp |
| irregular nominal | 93.0% | 98.6% | +5.6 pp |
| irregular hard | 53.8% | 73.0% | +19.2 pp |

The architecture is effective but not uniformly superior. Curvature options
materially improve turning and irregular terrain, while household obstacle
avoidance regresses. The next iteration must inspect the new videos/trajectories
and combine geometry-conditioned obstacle risk with the finite-horizon risk
critic; more SGD on V5 alone is not justified by the plateaued validation trend.

## Post-video diagnosis and V6 correction

The V5 privileged observer read the Isaac heightfield, but household/research
obstacles are separate static-box actors. Consequently, the first 500 V5
household observations contained zero pixels above the flat-ground value: the
world model could not see the red boxes shown in video. This is a state-definition
bug, not an optimizer limitation. V6 rasterizes the true obstacle footprints and
heights into the ego terrain map. A smoke dataset confirmed that 100% of frames
now contain obstacle-height pixels (7.4% of pixels on average, maximum normalized
height 1.0), and the regression is covered by a unit test.

## Videos

Six 375-frame, 15-second hard-scenario videos are under
`experiments/scale_v5_turn/scale/videos/`: research, turns, household, bridge,
edge, and irregular. The fixed video seed produced successes on household,
bridge, and irregular; falls on research and turns; edge neither succeeded nor
fell within the recording window. These videos are diagnostic examples, not the
three-seed aggregate result above.
