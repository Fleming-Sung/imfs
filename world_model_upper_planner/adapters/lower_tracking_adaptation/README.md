# Isolated lower tracking adaptation

This directory is the only allowed home for a lower-controller variant. The
immutable baseline remains `checkpoints/lower_model_7000.pt`, and no source or
checkpoint under the existing teacher-student/PPO project is modified.

Admission requires a fixed-seed tracking/capability audit. If a variant is
selected, all world-model replay and checkpoints must be recollected/retrained
under that exact lower-controller hash; mixing lower transition semantics is
invalid.

