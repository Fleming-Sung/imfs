# 最优上层落足策略：方法与训练设置总结

> 日期：2026-08-29。本文总结最终视频 `experiments/v6_video_seed924/rollout.mp4`
> 所使用的策略：一个**只看深度 + 36 维本体观测、可部署**的候选落足 Actor，在未见地形上
> 达到 **95.5% / 95.1%** 成功率，显著超过特权教师（70.8%）。
>
> 视频：`experiments/v6_video_seed924/rollout.mp4`（30 s，seed 924，无凸起障碍
> random-composite 赛道）。

## 1. 总体架构

两层层次化控制，下层完全冻结：

```text
深度图(1×64×64) + 36-D 本体观测
        │
        ▼
上层候选 Actor（本策略）── 从 294 个候选落足中选一个
        │  归一化 (forward, lateral_abs, yaw)
        ▼
目标接口解码为支撑脚系物理落足目标
        │
        ▼
冻结下层 Mind Your Steps（model_7000.pt，TRON1-SF，50 Hz）执行落足
```

上层每步的决策间隔 = 一次步态切换（约 20–25 个 50 Hz tick），即上层是
"每步选一个落脚点"的离散决策。

## 2. 训练路线（Gate A → B → E）

| 阶段 | 内容 | 结果 |
|---|---|---|
| Gate A | 特权几何教师：用仿真精确 heightfield 算 geodesic 距离场 + 障碍几何，枚举候选评分 | 70.8% 成功率（教师上界） |
| Gate B | 教师行为克隆：收集教师决策数据集，训练深度+本体的候选 Actor | 71.6% / 70.0%（追平教师） |
| **Gate E** | **非对称 PPO 微调：Actor 只看深度+本体，Critic 看特权地形，geodesic 势能奖励** | **95.5% / 95.1%（超过教师）** |

## 3. 动作空间（cartesian_course）

- 归一化 `[-1,1]^3` 解码为支撑脚 yaw 系 `(forward, signed_lateral, yaw)`：
  - forward ∈ [0.08, 0.30] m
  - lateral_abs ∈ [0.10, 0.26] m（左右脚符号由摆动脚决定）
  - yaw ∈ [−12°, 12°]
- 候选网格：forward 12 层 × lateral 9 层 × yaw 3 层（yaw 归一化 −0.5/0/0.5 = ±6°），
  再按径向距离 [0.12, 0.35] m 过滤，得 **294 个候选**。
- 横向量直接缩放，无 polar 动作的最小间距裁切，候选之间无物理别名。

## 4. 观测（可部署，无特权信息）

- **深度**：1 × 64 × 64，归一化近距 [0,1]。
- **本体 36-D**：投影重力(3) + 基座线速度xy(2) + 基座角速度z(1) + 关节位置(8) +
  关节速度(8) + 双脚相对基座位姿(6) + 步态相位(2) + **上一上层动作(3)** +
  **相对路线目标 + 偏航误差(3)**。
  （`upper_planner/upper_state.py::build_proprio`）

## 5. 候选 Actor 网络（`upper_planner/candidate_actor.py`）

```
depth (1,64,64) → Conv2d(1,16,5,s2)→ELU→Conv2d(16,32,3,s2)→ELU
                →Conv2d(32,64,3,s2)→ELU→Flatten→Linear(4096→128)→ELU   (128)
proprio (36)   → MLP(36→128→64)                                        (64)
concat(192) → Linear(192→256)→ELU→Linear(256→256)→ELU
  ├─ candidate_logits  : Linear(256→294)   分类（策略）
  ├─ candidate_feasible: Linear(256→294)   可行性 BCE（辅助）
  └─ candidate_progress: Linear(256→294)   geodesic 进度 Huber（辅助）
```

无 GRU（单帧决策）；辅助头只在 BC 阶段监督，PPO 阶段仅用 `candidate_logits`。

## 6. Gate B：行为克隆（BC）

数据收集（`scripts/collect_teacher_dataset.py`）：

- 1024 环境 × 30 s（seed 61，地形 61000–62023），特权教师在线决策，
  相机渲染开启，记录每次上层决策的 `(depth, proprio, teacher候选索引,
  每候选support/progress/valid/score, 该决策所在episode是否成功, env_id)`。
- 产出 57641 条决策，其中 23205 条来自成功 episode（40.3%）。
- 训练只保留成功 episode 的决策（`--success_only`），按 `env_id` 划分 80/20 train/val
  （val 是**未见地形实例**）。

监督目标（`scripts/train_candidate_bc.py`）：

```text
L = CE/软目标(student_logits, teacher打分分布)
  + BCE(candidate_feasible, candidate_valid)
  + Huber(candidate_progress, geodesic_progress)
```

- **软目标蒸馏**：`softmax(teacher_score / 0.25)`，对近并列候选更有信息量
  （教师完整打分分布，而非单一 argmax）。
- 超参：60 epochs，batch 512，lr 3e-4（Adam），权重 CE 1.0 / BCE 0.5 / Huber 0.5。
- 离线指标（非行为指标）：top1 18%、所选候选有效率 80%、可行率 91%、进度 MAE 0.14 m。
- 闭环评估：seed 924 = 71.6%（222/310），seed 42 = 70.0%（217/310）。

## 7. Gate E：非对称 PPO 微调（最终策略）

`scripts/train_candidate_ppo.py` + `upper_planner/ppo_critic.py`。

**Actor**：候选 Actor，BC 热启动，离散策略（Categorical over 294 候选），
只看 depth + proprio。

**Critic**（特权，绝不部署）：MLP(42→256→256→1)，输入 =
proprio(36) + geodesic距离(1) + 支撑率(1) + 支撑脚世界xy(2) + 基座世界xy(2)。

**奖励**（geodesic 势能 shaping，绕障不被欧氏距离误惩罚）：

```text
r = r_progress + 10·success − 5·collision − 5·fall − 3·(1−support_fraction)
r_progress = d_geodesic(s_t) − γ·d_geodesic(s_{t+1})     （fall 置 0，success 置 d_t）
```

其中 `d_geodesic` 由特权教师预计算的 geodesic 距离场采样（
`privileged_planner.geodesic_distance`）。

**PPO 超参**：

| 项 | 值 |
|---|---|
| 环境数 / rollout | 512 环境 × 600 ticks（≈12 s，约 12k 条 option 转移） |
| total_updates | 30 |
| update_epochs / batch_size | 10 / 1024 |
| lr（Adam，actor+critic 共用） | 1e-4 |
| γ / λ | 0.99 / 0.95 |
| clip ε | 0.2 |
| entropy 系数 | 0.005 |
| value 系数 / max_grad_norm | 0.5 / 1.0 |
| 种子 | 61（训练地形，与评估 seed 924/42 不重叠） |

**训练过程**：探索采样下累计成功率 21.6% → 47.8%（30 updates），loss 6227 → 2975。
**贪婪评估**（`evaluate_candidate_bc.py`，未见地形）：

| 模型 | seed 924 | seed 42 |
|---|---|---|
| 特权教师 | 70.8% | — |
| BC 学生 | 71.6% | 70.0% |
| **PPO 微调** | **95.5%** (235/246) | **95.1%** (232/244) |

## 8. 复现命令

```bash
cd upper_foothold_planner

# 1) 收集教师数据集（1024 环境 × 30 s，seed 61）
python scripts/collect_teacher_dataset.py --headless --sim_device cuda:0 \
  --num_envs 1024 --lower_ticks 1500 --seed 61 --course_length_m 3.5 \
  --output experiments/v6_teacher_dataset_1024

# 2) BC 软目标蒸馏
python scripts/train_candidate_bc.py \
  --dataset experiments/v6_teacher_dataset_1024/dataset.npz \
  --output experiments/v6_bc_1024_success \
  --epochs 60 --batch_size 512 --lr 3e-4 --success_only --soft_target \
  --score_temperature 0.25 --val_env_fraction 0.2 --device cuda:0

# 3) 非对称 PPO 微调（512 环境 × 30 updates）
python scripts/train_candidate_ppo.py \
  --actor_checkpoint experiments/v6_bc_1024_success/model_best.pt \
  --num_envs 512 --rollout_ticks 600 --total_updates 30 --checkpoint_every 10 \
  --update_epochs 10 --batch_size 1024 --lr 1e-4 --entropy_coef 0.005 \
  --seed 61 --course_length_m 3.5 --output experiments/v6_ppo_real

# 4) 贪婪评估（未见地形）
python scripts/evaluate_candidate_bc.py \
  --checkpoint experiments/v6_ppo_real/ppo_final.pt \
  --num_envs 64 --lower_ticks 1500 --seed 924 --course_length_m 3.5 \
  --output experiments/v6_ppo_eval_seed924

# 5) 录视频（1 环境 + viewer + 跟随相机 → rollout.mp4）
python scripts/record_candidate_video.py \
  --checkpoint experiments/v6_ppo_real/ppo_final.pt \
  --seed 924 --lower_ticks 1500 --course_length_m 3.5 \
  --output experiments/v6_video_seed924
```

所有命令必须在 `isaacgym` conda 环境运行，且 `import isaacgym` 必须先于 `import torch`。

## 9. 关键实现细节与踩坑

1. **动作别名根因**：旧 `polar_course` 的横向最小间距裁切使 98.1% 样本退化为 0.10 m 别名，
   改为 `cartesian_course` 直接缩放横向量。
2. **geodesic 而非欧氏 progress**：侧移绕障的正确动作在欧氏距离下会被误罚，教师评分和
   PPO 奖励都改用 geodesic 距离场。
3. **候选集合离散化**：连续落足改成 294 个无别名候选，直接做分类/策略，可视化直观。
4. **软目标 BC**：教师 argmax 在近并列候选间有噪声，软目标分布比 one-hot 更有信息量
   （可行率 67%→79%，进度 MAE 0.24→0.18 m）。
5. **非对称 Critic**：Critic 看 geodesic 距离 + 支撑率 + 绝对位姿，Actor 只看部署观测，
   是 PPO 能超过教师的关键。
6. **GE 奖励终态处理**：fall 时 `r_progress=0`；success 时 `r_progress=d_t`（env 已 reset，
   不能再采样 `d_{t+1}`）。
7. **rollout 簿记**：上层决策与转移是事件驱动的，用 per-env deque 记录"决策时"的动作/
   log_prob/特权特征/支撑脚位姿，转移时弹出匹配，避免同 tick 内被新决策覆盖。
8. **深度渲染瓶颈**：带相机 512 环境仅 ~3 tps（无相机 42 tps），是数据收集/PPO 训练的主要
   耗时来源；后续可同步决策分组或降分辨率提速。

## 10. 产物清单

| 文件 | 说明 |
|---|---|
| `experiments/v6_video_seed924/rollout.mp4` | 最终策略演示视频（30 s） |
| `experiments/v6_ppo_real/ppo_final.pt` | PPO 最终 actor+critic checkpoint |
| `experiments/v6_bc_1024_success/model_best.pt` | BC 教师蒸馏 checkpoint |
| `experiments/v6_teacher_dataset_1024/dataset.npz` | 教师决策数据集 |
| `upper_planner/candidate_actor.py` | 候选 Actor 网络 |
| `upper_planner/ppo_critic.py` | 特权 Critic + 特权特征 |
| `scripts/train_candidate_ppo.py` | 非对称 PPO 训练 |
| `scripts/train_candidate_bc.py` | BC 训练 |
| `scripts/evaluate_candidate_bc.py` | 闭环评估 |
| `scripts/record_candidate_video.py` | 录像脚本 |

## 11. 分布外泛化实测（2026-08-29，未重训）

把 `ppo_final.pt` 直接拿到比训练更难的 terrains 上评估（seed 924，64 env）：

| 地形 | 参数 | 成功率 |
|---|---|---|
| 原始（训练分布） | 宽度 0.50–1.30 m，间隙 0–0.14 m，3.5 m | 95.5% |
| Hard A 窄支撑 | 宽度 0.35–0.60 m，间隙 0.14 m，3.5 m | 72.2% |
| Hard B 大间隙 | 宽度 0.50–1.30 m，间隙 0.22 m，3.5 m | 81.0% |
| Hard C 组合极限 | 宽度 0.35–0.60 m，间隙 0.22 m，5.0 m | 18.7% |

结论：单独加难（窄支撑 / 大间隙）时策略**优雅退化**（72% / 81%），证明它在感知地形
几何而非死记场景；三者叠加到极限时崩溃（18.7%），主要触及冻结下层的物理执行上限
（SF 脚掌小，0.35 m 窄支撑 + 0.22 m 间隙 + 更长射程的组合接近不可执行）。

视频：`experiments/v6_video_hardA_narrow/rollout.mp4`、`experiments/v6_video_hardB_gap/rollout.mp4`。
