# 设计文档：每一步的输入/输出、每个 head、每个 loss、每个奖励、每次迭代

本文档只描述当前 V4 架构（`--model_variant option`）。旧的 compact/spatial/task 变体仅作
历史保留，不在第一阶段使用。

## 1. 观测与动作契约

### 1.1 深度图

- 输入：单相机 64×64，`preprocess_isaac_depth` 把 Isaac Gym 的负视轴深度 `-z` 映射为
  **接近度 [0,1]**（near=0.40 m，far=2.0 m），`nan`/越界按 `far` 处理。
- 形状：`(B, 1, 64, 64)`。

### 1.2 本体观测 `build_proprio` → 36 维

| 区间 | 维度 | 内容 | 缩放 |
|---|---|---|---|
| 0:3 | 3 | 投影重力 | 已归一 |
| 3:5 | 2 | 基座线速度 xy | ×0.5 |
| 5:6 | 1 | 基座角速度 z | ×0.25 |
| 6:14 | 8 | 关节位置 | rad |
| 14:22 | 8 | 关节速度 | ×0.1 |
| 22:28 | 6 | 脚在基座系位置 | m |
| 28:30 | 2 | 步态相位 cos/sin | — |
| 30:33 | 3 | 上一上层动作 | 已归一 |
| 33:36 | 3 | 相对目标 x/6, y/3, 偏航误差/π | — |

### 1.3 动作（`action_profile=polar_course`）

- 上层输出归一化 `a ∈ [-1,1]^3`，解码为支撑脚偏航坐标系下的 `(距离, 方向, 落足偏航)`：
  - 距离 ∈ [0.07, 0.22] m
  - 方向 ∈ ±35°
  - 落足偏航 ∈ ±15°
  - 双脚最小横向间距 0.10 m（防交叉，`PolarFootholdActionBounds.decode`）
- 解码器把摆动脚（左=0，右=1）映射为横向符号，保证左右对称。

### 1.4 宏观运动状态 `macro_state` → 21 维（隐空间监督信号）

| 区间 | 维度 | 内容 | 缩放 |
|---|---|---|---|
| 0:1 | 1 | 基座高度 | /0.8 |
| 1:4 | 3 | 基座线速度 | ×0.5 |
| 4:7 | 3 | 基座角速度 | ×0.25 |
| 7:10 | 3 | 投影重力 | — |
| 10:16 | 6 | 脚在基座系位置 | m |
| 16:18 | 2 | 步态相位 cos/sin | — |
| 18:21 | 3 | 相对目标 x/6, y/3, 偏航误差/π | — |

**关键**：dynamics 的下一步隐状态由 `task_state` head 预测这 21 维并监督，**不重建下一帧
深度图**。这是 V4 相对 V1–V3 的核心改动。

## 2. 模型 `OptionTaskWorldModel`

### 2.1 编码器

- 深度 CNN：Conv(1→16→32→64→64) + Flatten + Linear → 160。
- 本体 MLP：36 → 96。
- 融合：`Linear(256 → 128) → LayerNorm → SimNorm(8)`。
- 隐状态 `z ∈ R^128`，每 8 维一组做 softmax（SimNorm），防止隐状态饱和。

### 2.2 动力学

```
next(z, a) = SimNorm( z + mlp([z, a], 128→256→128) )
```

残差 + SimNorm。`a` 为归一化 3 维动作。

### 2.3 head 清单

| head | 输入 | 输出 | 激活 | 训练目标 | 归属 loss |
|---|---|---|---|---|---|
| `progress` | [z,a] | 1 | — | progress/10 | margin |
| `heading_progress` | [z,a] | 1 | — | heading/π | margin |
| `collision` | [z,a] | 1 | logit | 碰撞 0/1 | event |
| `fall` | [z,a] | 1 | logit | 跌倒 0/1 | event |
| `goal` | [z,a] | 1 | logit | 首次到目标 0/1 | event |
| `off_support` | [z,a] | 1 | logit | 落足悬空 0/1 | event |
| `collision_force` | [z,a] | 1 | softplus | 碰撞力/100 截断[0,5] | margin |
| `stability_margin` | [z,a] | 1 | — | 高度/倾角裕度 | margin |
| `support_fraction` | [z,a] | 1 | sigmoid | 足底 3×3 支撑占比 | margin |
| `touchdown_error` | [z,a] | 1 | softplus | 触地误差/0.10 截断[0,5] | margin |
| `option_duration` | [z,a] | 1 | softplus | 选项 tick 数/25 截断[0.04,4] | duration |
| `continuation` | [z,a] | 1 | logit | 1−done | continuation |
| `q_functions` (×2) | [z,a] | 1 | — | 截断 return | q |
| `policy` | z | 3 | tanh | (阶段一不使用) | policy |
| `value` | z | 1 | — | (阶段一 value_coef=0) | value |
| `task_state` | z' | 21 | — | 21 维 macro_state | task_state |

其中 `task_state` 的输入是 `next(z,a)`（预测的下一隐状态），其余 head 的输入是当前
`(z,a)`。

## 3. 损失（`OptionWorldModelTrainer.losses_sequence`）

序列长度为 `sequence_horizon=5`，第 k 步权重 `temporal_decay^k`（0.8^k）。在模型自己的
隐状态 rollout 上逐步计算。

### 3.1 基础损失（`TaskWorldModelTrainer`）

- **事件 loss**（coef `event=1.0`）：collision / fall / goal / off_support 四个 logit 对
  0/1 目标的二元交叉熵。option 变体 `balanced_events=False`，正类权重上限 50。
- **margin loss**（coef `margin=1.0`）：progress / heading_progress / collision_force /
  stability_margin / support_fraction / touchdown_error 六个回归项的 smooth L1。
- **task_state loss**（coef `task_state=1.0`）：`task_state(next(z,a))` 对真实 `macro_state`
  的 smooth L1，**仅对 alive（未 done 且 valid）样本取平均**——done 的下一步隐状态不强制
  匹配重置后的新姿态。
- **value loss**（coef `value=0.0`，阶段一关闭）。
- **正则 loss**（coef `regularization=0.01`）：隐状态各维方差下限 0.20 + 协方差非对角项。

总损失：

```
L = event·ΣBCE + margin·ΣsmoothL1 + task_state·alive_mean_smoothL1
  + value·... + regularization·...
  + q·q_loss + duration·duration_loss + continuation·continuation_loss
  + ranking·ranking_loss + policy·policy_loss
```

### 3.2 选项损失

- **Q loss**（coef `q=1.0`）：twin-Q 对截断 return（`return_target`）的 smooth L1。
  本项目 CEM 即策略改进算子，没有独立 actor，因此 Q 回归到真实有限视界 return，而不是
  max 随机动作的外推 bootstrap。
- **duration loss**（coef `duration=0.25`）：`option_duration` 对 `option_duration_ticks/25`
  （截断 [0.04, 4]）的 smooth L1。
- **continuation loss**（coef `continuation=0.5`）：`continuation_logit` 对 `1−done` 的 BCE。
- **ranking loss**（coef `ranking=0.25`）：同状态反事实动作组内，按 outcome 差异排序 score
  的 pairwise softplus。
- **policy loss**（coef `policy=0.1`）：只在 planner 动作步上让 `policy(z)` 回归实际动作。
  阶段一**不使用策略先验**（`--use_policy_prior` 不传），该项自然无梯度。

## 4. 奖励（`UpperTaskDiagnostics.macro_reward`，每个选项结束时算一次）

| 项 | 系数 | 定义 |
|---|---|---|
| progress | +10.0（CLI `--reward_progress`） | 选项前后到目标距离的减小量（m） |
| time | −0.02 | × `option_ticks/25`（按选项时长惩罚） |
| goal | +10.0 | 距离 < 0.30 m 且首次到达 |
| collision | −2.0（CLI `--reward_collision`） | 选项期间任一非足部件接触力 > 5 N |
| fall | −5.0 | 物理终止（跌倒） |
| off_support | −3.0 × (1 − support_fraction) | 落足 9 点支撑占比的分级惩罚 |

要点：
- `progress` 是**差分**（`previous_distance − distance`），每选项重置时 `previous_distance`
  重新锚定到当前位置，因此“重置到中间位置”不会白送 progress。
- `off_support` 用 3×3 九点 `support_fraction`（脚掌大部分有支撑就不算踏空），事件 head
  的二元目标为 `support_fraction < 0.5`；奖励为分级 `−3·(1−fraction)`。
- `collision` 阈值从 1 N 抬到 5 N（`collision_force_threshold_n`），消除正常行走的小腿/膝盖
  蹭地误判；物理终止阈值仍是 40 N。
- 量级原则：一次“激进但成功”的轨迹累计惩罚应小于目标奖励，避免净奖励恒为负。

reward 除以 `reward_scale=10.0` 后作为训练目标（progress 目标即真实米数）。
诊断额外记录 `collision_force` / `stability_margin` / `support_fraction` /
`touchdown_error` / `macro_state` 供各 head 监督。

## 5. 训练循环（每次迭代）

1. **下层 tick（50 Hz）**：`rollout.lower_tick` 推一步冻结下层策略，`UpperTaskDiagnostics`
   累计选项期间的碰撞力、最低高度、最大倾角。
2. **事件触发**：脚切换 / 跌倒 / 成功时，产生一条宏观 transition
   `(s_t=(depth,proprio), a_t=上一上层动作, r, s_{t+1}=(depth,proprio), done, macro_state,
   各诊断量)`；`s_t` 是决策时刻观测，`s_{t+1}` 是事件时刻观测，`done = 物理终止 | 成功`。
3. **写入 replay**：`ReplayBuffer` 存 depth/proprio/action/reward/done + macro_state +
   各监督目标 + counterfactual 组号。
4. **采样与训练**：`sample_sequence(batch, horizon=5)` 抽连续序列，`train_step_sequence`
   在隐空间 rollout 上算全部损失，反向传播 + EMA 更新 target 网络。
5. **数据收集策略**（`collection_policy=mixed`）：warmup 前全随机；之后 50% CEM 规划 +
   分层模板（20% 完全随机）。CEM 规划步会被 `from_planner` 标记。
6. **反向课程重置**（`--reset_curriculum_prob 0.5`）：以概率 0.5 在赛道中段随机支撑位置
   重生（距目标 ≥ 0.6 m、距起点 ≥ 0.5 m、3×3 全支撑），让模型经历中段与接近目标的状态，
   使目标奖励可被采样。评估脚本不带此 flag（`getattr` 默认 0.0），评估仍从起点完整跑。

## 6. 规划（CEM，隐空间）

`cem.plan` 在 `z` 上滚动 `planning_horizon=3` 步，目标函数：

```
return = Σ_t γ_eff^t · [ predict_task_reward(z_t, a_t)
                        − collision_risk·σ(collision) − fall_risk·σ(fall)
                        − 各连续风险项 − action_l2·||a_t||² ]
         + terminal_value_coef · γ_eff^H · 终端值
```

- `predict_task_reward` 由第 4 节各 head 组装（progress + time + goal·σ(goal) +
  collision·σ(collision) + fall·σ(fall) + off_support·σ(off_support)）。
- 选项变体的有效折扣 `γ_eff = discount^option_duration · continuation`。
- **阶段一规划配置**：`--planning_disable_goal`（goal 系数置 0，防止未校准 goal head 主导
  CEM）、`--planning_base_safety_scale 0.5`（collision/fall/off_support 系数 ×0.5）、
  `--terminal_value_coef 0`。
- CEM：每环境 `candidates` 条候选，`elites` 条精英重估均值/方差，`iterations` 轮。
