# Frozen baseline and lower-controller boundary

The new project was created without editing `../upper_foothold_planner/`.
`git diff -- upper_foothold_planner` was empty on 2026-08-31.

The simulator/lower integration was copied into
`adapters/frozen_lower_env/`. At creation time, all eleven copied Python files
had byte-identical SHA-256 hashes to their source files. The frozen checkpoint
was also copied byte-for-byte:

```text
lower_model_7000.pt
f88a9f8ebdf64e1c976daf9182e73f7b801b057f50b45d4fe9505230ced786e2
```

Changes needed only for the new upper-planner transition contract belong in
this adapter copy. The original teacher-student/PPO baseline and the original
lower-controller source remain immutable comparison artifacts.

