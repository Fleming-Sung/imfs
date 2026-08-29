# V6 Gate A：真实地形上层规划器

## 目的

在不使用深度相机的情况下，直接读取仿真的真实支撑图和障碍几何，为冻结下层输出落足目标。
它回答一个先于学习的问题：**当前下层、动作接口和训练赛道组合是否存在可执行的上层解？**

这不是质点路径回放。每个目标仍通过 `UpperFootholdTargetInterface` 下发，由
`lower_model_7000.pt` 在 50 Hz 真实闭环执行；摔倒、碰撞、落足误差和成功都来自物理仿真。

## 输入与输出

规划器的特权输入：

- 每个环境的 `support_mask`；
- 障碍物真实 xy 矩形；
- 环境原点、任务目标；
- 当前支撑脚位置和 yaw、下一摆动脚；
- 上一上层动作，仅用于很小的动作变化正则。

输出仍是三维上层动作，但使用新的 `cartesian_course` 契约：

```text
normalized [-1,1]^3
    -> stance-yaw-frame (forward, signed lateral, foothold yaw)
    -> frozen lower foothold target
```

其中 forward 候选覆盖 0.08--0.30 m，横向绝对值覆盖 0.10--0.26 m，yaw 为 ±12°；组合后
再删除径向距离超过 0.35 m 的点。因此可以横向够到偏移支撑块，又不越出冻结下层训练的
0.35 m 径向范围。横向量直接缩放，
不再经过 polar action 的最小间距裁切，因此候选之间没有物理别名。

## 路径与候选

1. 用与现有诊断一致的足底九点 stencil 计算每个网格的支撑率；
2. 支撑率至少 5/9 且不位于障碍轮廓内的网格可作为落足中心；
3. 在可落足网格上反向计算到目标的 geodesic distance；
4. 图边允许跨越不超过 0.26 m 的无支撑间隙，但禁止穿过障碍，表示真实的离散落足序列；
5. 每次决策枚举物理 `(forward,lateral,yaw)` 网格，并按 0.35 m 径向可达圆过滤；
6. 按 geodesic progress、支撑率、路线朝向和很小的动作变化代价评分；
7. 候选的第一步线段若穿过障碍则直接拒绝。

当前 64 个 seed 924 起的 random-composite 几何审计结果为 64/64 起点到目标连通，预计算约
3 秒（CPU）。这只证明候选图有路线，不代表冻结下层一定能执行；后者必须由 Gate A 闭环评估。

## 闭环评估

当前 V5 训练仍占用 GPU 时不要启动以下命令。GPU 空闲后先跑固定起点、多环境、无相机评估：

```bash
conda run -n isaacgym --no-capture-output \
  python scripts/evaluate_privileged_planner.py \
  --headless --physx --sim_device cuda:0 --pipeline gpu \
  --num_envs 64 --lower_ticks 1500 --seed 924 \
  --terrain_curriculum research --research_kind random_composite \
  --course_length_m 3.5 \
  --random_width_min_m 0.50 --random_width_max_m 1.30 \
  --random_gap_max_m 0.14 --random_obstacle_probability 0.55 \
  --output experiments/v6_privileged_gate_seed924
```

评估入口自动启用 `--privileged_terrain_planner --action_profile cartesian_course`，并完全关闭深度
相机创建和捕获。

行为验收至少报告：

- 固定起点的 success/fall/timeout episode rate；
- 每类终止原因；
- touchdown error、off-support 和 collision；
- 每环境进度、到目标距离和路径效率；
- 候选有效数、fallback 比例、选择目标的支撑率和 geodesic progress。

随后用一个环境打开 viewer 并保存视频，检查典型成功、摔倒和徘徊轨迹：

```bash
conda run -n isaacgym --no-capture-output \
  python scripts/evaluate_privileged_planner.py \
  --physx --sim_device cuda:0 --pipeline gpu \
  --num_envs 1 --lower_ticks 1500 --seed 924 \
  --terrain_curriculum research --research_kind random_composite \
  --course_length_m 3.5 --record_video \
  --output experiments/v6_privileged_gate_seed924_video
```

## Gate 判断

- 若成功率高：开始用真实几何监督任务相关 terrain latent 和动作条件结果模型；最终规划器仍输出
  落足目标。
- 若路线选择正确但身体碰撞/摔倒：缩小或重新标定候选可达域，引入真实闭环动态可行性表。
- 若经常无候选：检查足底支撑阈值、动作范围和局部路线图，不训练视觉网络掩盖问题。
- 若几何路径错误：修 geodesic/障碍语义，不通过调 reward 或世界模型 loss 绕过。

## 实测结果（截至 2026-08-26）

分离实验（random-composite 课程）逐步定位到：凸起障碍是下层能力缺口，而非上层规划问题。
去掉凸起障碍、只保留随机宽度支撑块 + 间隙（`--random_obstacle_probability 0.0`）后：

- 平地对照：64 env × 6 s **0 摔倒、0 碰撞、0 踏空**，每步 progress 约 0.16 m，证明动作接口
  和下层动态正常。
- 复杂地形（支撑块+间隙）短测：把候选加密到 2 cm 间距、扩大径向到 0.35 m 后，6 s 摔倒从
  40 逐步降到 11。
- 30 s 正式 Gate（64 env）：**275 回合 187 成功 / 88 摔倒 = 68% 成功率**；每决策平均
  geodesic progress +19.6 cm。env0 视频确认能连续跨越多段带间隙/横向错位的支撑块并在
  约 10 s 内到达目标。
- 失败集中在横向偏移大的路线（个别 seed 0 成功/6~8 摔倒）。
- **恢复小幅逐步转向（每步 ±6°）**：30 s 成功 196 / 摔倒 81 = **70.8%**，fallback 12.0%→11.4%。
  小步转向能部分改善大横向偏移路线，但提升有限（+2.8 点）。

结论：教师达到 ~71% 成功率，已足以作为 Gate B 的数据源；继续微调教师边际收益递减。
下一步转入 Gate B/C：收集教师决策数据集，训练只依赖深度+可部署本体的候选 Actor（行为克隆
+ 候选特权标签辅助），再 DAgger 闭环修正分布偏移。凸起障碍暂不混入，先证明无凸起障碍赛道
上整套蒸馏管线可跑通。

## Gate B 结果（2026-08-26）

候选 Actor（深度 CNN 1×64×64 + 36-D proprio MLP → 294 候选分类 logits + 可行 BCE +
geodesic progress Huber；软目标蒸馏 `score_temperature=0.25`）在教师数据集上行为克隆：

- 数据集：1024 环境 × 30 s（seed 61，地形 61000–62023），57641 决策，其中 23205 来自
  成功 episode（40.3%）；按 env_id 划分 80/20 train/val。
- 离线指标：top1 18%、所选候选有效率 80%、可行率 91%、进度 MAE 0.14 m（top1 不是行为指标，
  软目标天然不打中教师 argmax）。
- 闭环（学生只看 depth+proprio，评估地形与训练完全不重叠）：
  - seed 924（64 env，教师基准集）：**310 回合 222 成功 / 88 摔倒 = 71.6%**；
  - seed 42（64 env，另组未见）：**310 回合 217 成功 / 93 摔倒 = 70.0%**。
- 结论：纯行为克隆学生（可部署观测）在未见地形上**追平特权教师**（70.8%）。Gate B 通过。

下一步：Gate C DAgger 修正闭环分布偏移（教师为 BC 上界，增益有限）；达到 70–80% 后转
Gate E 非对称 PPO 微调以超过教师；凸起障碍仍受下层能力缺口限制，暂不混入。

## Gate E 结果（2026-08-27）

非对称 PPO：Actor=BC 热启动候选分类（离散 294，只看 depth+proprio），Critic=特权 MLP
（proprio + geodesic + support + 绝对 stance/base 位姿），奖励=geodesic potential shaping
`r = d_t − γ·d_{t+1}` + success(10) − collision(5) − fall(5) − off_support(3)。
512 环境 × 600 ticks/rollout × 30 updates，lr 1e-4，entropy 0.005，γ=0.99。

- 训练（seed 61，探索采样）累计 success：21.6% → 47.8%（30 updates），loss 6227 → 2975。
- 贪婪评估（未见地形，与 BC 完全同口径）：

| 模型 | seed 924 | seed 42 |
|---|---|---|
| 特权教师 | 70.8% | — |
| BC 学生 | 71.6% | 70.0% |
| **PPO 微调** | **95.5%** (235/246) | **95.1%** (232/244) |

- 结论：非对称 PPO 把成功率从 ~71% 推到 ~95%，**显著超过特权教师**，证明 RL 微调能
  超越教师上界。Gate E 通过，达到 Phase-1 目标（可部署 Actor 在无凸起障碍赛道上
  ~95% 成功率）。

下一步：①凸起障碍（下层能力缺口，需先补下层）；②更窄支撑/更大间隙的难度课程；
③部署侧验证（Actor 仅依赖深度+36-D 本体）。
