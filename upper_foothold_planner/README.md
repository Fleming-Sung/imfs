# 上层落足规划器：冻结下层的 TD-MPC 风格隐空间世界模型

本目录实现一个**两层层次化控制框架**：下层是冻结的 Mind Your Steps 落足跟踪策略（50 Hz，
不可重训），上层在一个紧凑隐空间里学习落足选择的动力学，并用 CEM 在隐空间滚动选择下一个
摆动脚目标。

```
┌─────────────────────────────────────────────────────────────────────────┐
│  上层 (upper_planner, 事件驱动, 每次脚切换/跌倒/成功触发一次决策)            │
│                                                                         │
│  输入: depth 1×64×64 + proprio 36 维                                     │
│        │                                                                │
│        ▼                                                                │
│  编码器 → 隐状态 z ∈ R^128 (SimNorm)                                      │
│        │                                                                │
│        ├─ dynamics: z' = SimNorm(z + mlp(z, a))                          │
│        ├─ 任务 head: progress / goal / collision / fall / off_support /  │
│        │             collision_force / stability / support / touchdown   │
│        ├─ 选项 head: option_duration / continuation / twin-Q             │
│        └─ 监督 head: task_state → 21 维宏观运动状态                        │
│                                                                         │
│   CEM 在隐空间滚动 horizon 步 → 选出第 1 个归一化落足动作 a ∈ [-1,1]^3      │
└─────────────────────────────────────────────────────────────────────────┘
                              │ a (polar: 距离/方向/偏航)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  下层 (冻结, 50 Hz, checkpoints/lower_model_7000.pt)                      │
│  a → 实际落足目标 → 论文跟踪奖励闭环执行 → 真实宏观运动状态 (诊断用, 不进策略) │
└─────────────────────────────────────────────────────────────────────────┘
```

## 核心设计决定（V4）

1. **隐空间只由宏观运动状态监督，不做完整观测重建。** 训练时用选项结束时的 21 维宏观状态
   （基座高度、线/角速度、姿态、脚在基座系位置、步态相位、相对目标）监督 dynamics 的下一隐
   状态，而不是让隐状态去重建下一帧深度图。深度重建会把隐状态拖向视觉重构，使模型学不到
   任务信息。
2. **时间尺度 = 步态切换步数，不是 50 Hz 时间步。** 一次上层决策对应一次左右脚切换
   （一个"选项"，约 25 个下层 tick）。`option_duration`/`continuation`/时间奖励都按此尺度。
3. **第一阶段只在一个复杂场景（`random_composite`）上跑通，不做泛化。** 成功标准：绝大多数
   环境能走到目标。跑通后再谈课程/域随机化。

## 目录

- `upper_planner/` — 全部框架代码。
  - `contracts.py` — 维度契约（proprio 36 / action 3 / macro_state 21）、动作解码。
  - `upper_state.py` — 上层观测 `build_proprio`、宏观状态 `macro_state`、宏观奖励
    `macro_reward`（`UpperTaskDiagnostics`）。
  - `world_model.py` — 编码器 + 隐空间动力学 + 全部 head（`OptionTaskWorldModel`）。
  - `task_world_model_trainer.py` — 事件/margin/task_state/正则 loss。
  - `option_world_model_trainer.py` — 在 task 基础上加 twin-Q/duration/continuation/排序。
  - `cem.py` — 隐空间 CEM 规划。
  - `rollout.py` — 事件驱动的两层 rollout（下层 tick + 上层 transition）。
  - `replay.py` — replay buffer（含 `macro_state`、counterfactual 组）。
  - `factory.py` — 场景/课程构造；`env.py` — 环境封装；`train.py` — 训练入口。
- `config/default.json` — 奖励系数、模型超参。
- `scripts/upper_rollout_smoke.py` — 评估脚本。
- `tests/` — 契约与工具单测。

## 运行

训练（单场景 `random_composite`，第一阶段）：

```bash
conda run -n isaacgym --no-capture-output python train.py \
  --headless --physx --sim_device cuda:0 --pipeline gpu \
  --num_envs 128 --seed 921 --max_updates 100000 \
  --terrain_curriculum research --research_kind random_composite \
  --course_length_m 3.5 --action_profile polar_course \
  --model_variant option --balanced_replay --decomposed_reward \
  --collection_policy mixed --mixed_planner_fraction 0.5 \
  --planning_disable_goal --planning_base_safety_scale 0.5 \
  --output runs/<run_name>
```

评估：

```bash
conda run -n isaacgym --no-capture-output python scripts/upper_rollout_smoke.py \
  --headless --physx --sim_device cuda:0 --pipeline gpu \
  --terrain_curriculum research --research_kind random_composite \
  --course_length_m 3.5 --action_profile polar_course \
  --planner_checkpoint runs/<run_name>/checkpoints/model_XXXXXX.pt \
  --planning_horizon 3 --decomposed_reward \
  --num_envs 10 --lower_ticks 1500 --seed <eval_seed> \
  --output experiments/<eval_name>
```

单测：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n isaacgym --no-capture-output \
  python -m unittest discover -s tests -v
```

## 详细文档

- [docs/design.md](docs/design.md) — 每一步的输入/输出、每个 head、每个 loss、每个奖励、
  每次迭代。
- [docs/provenance.md](docs/provenance.md) — 各组件来源。
- [docs/research_log.md](docs/research_log.md) — 研究日志（设计决定与结论）。
- [docs/phase1_status.md](docs/phase1_status.md) — 第一阶段总结：框架现状、发现的现象、
  可能的问题、下一步方向。
- [docs/phase1_architecture_reassessment.md](docs/phase1_architecture_reassessment.md) — Phase 1
  架构重审与 V6 路线。
- [docs/privileged_planner_gate.md](docs/privileged_planner_gate.md) — 不使用相机的真实地形上层
  规划器、闭环 Gate 命令与验收指标。
