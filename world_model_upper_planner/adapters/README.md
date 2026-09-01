# Adapter boundary

This directory is the only allowed place for simulator and frozen-lower
integration. Do not edit `../upper_foothold_planner/`, `../foothold/`, or their
checkpoints.

The adapter must expose option-boundary transitions with:

- deployable observation before and after the option;
- normalized candidate action and option duration;
- measured reward/progress, success/fall/collision/off-support;
- touchdown error and support fraction as training-only labels;
- episode id and terrain split id.

Exact terrain, contact force and terminal reason are training/evaluation labels
only. They must never enter the deployed Actor/planner observation.

