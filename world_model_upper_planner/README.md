# Candidate-Grounded Option World Model

Independent research implementation of a TD-MPC-like upper foothold planner.
The existing `upper_foothold_planner/` teacher-student/PPO project is an
immutable baseline and is not imported or modified here.

The upper action is one closed-loop foothold option executed by a frozen lower
controller. The learned model predicts option outcomes, latent transition and
terminal value over a fixed, empirically reachable candidate set. Planning uses
uncertainty-aware discrete beam search rather than continuous CEM.

Current status:

- independent PyTorch model, loss and planner core;
- synthetic smoke test and unit tests;
- research hypothesis and comparison protocol in `docs/method.md`;
- self-contained simulator/frozen-lower snapshot in `adapters/`, with the
  original teacher-student project unchanged;
- a 60,317-transition, 640-layout replay spanning edge cases, random composite,
  stepping stones, turns, and household obstacles;
- a complete H3 beam planner that reaches 77.1% course success on the first
  fixed challenging benchmark, versus 59.5% for same-checkpoint H1 scoring and
  63.6% for its proposal prior;
- preserved metrics, trajectories and success/failure videos plus a resumable
  end-to-end experiment pipeline.

See `docs/status.md` for verified results and the next blocking Gate.

Quick check:

```bash
cd world_model_upper_planner
conda run -n isaacgym --no-capture-output python -m unittest discover -s tests -v
conda run -n isaacgym --no-capture-output python scripts/smoke_model.py
```

Full automatic run (execute inside the `isaacgym` environment):

```bash
python scripts/run_experiment_pipeline.py --name cgowm_replication --profile full
```
