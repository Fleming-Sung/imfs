# 第一阶段训练总结：框架现状、发现的现象与可能的问题

> 撰写日期：2026-08-26。本文只记录**趋势性的结论与结构**，不纠结单次窗口的数值抖动。

---

## 1. 训练结构（当前框架）

### 1.1 两层层次化控制

- **下层**：冻结的 Mind Your Steps 落足跟踪策略（`checkpoints/lower_model_7000.pt`），
  50 Hz，输入深度+本体+目标，输出关节动作，闭环执行上层给出的落足目标。
- **上层**（`upper_planner/`）：事件驱动的隐空间世界模型 + CEM 规划。一次上层决策 =
  一次左右脚切换（一个"选项"，约 24 个下层 tick ≈ 0.5 s）。上层不预测关节动作，只在
  冻结下层动作支持集内选下一个摆动脚目标。

### 1.2 观测与动作

- 观测：深度 `1×64×64`（归一化接近度）+ 本体 `proprio` 36 维（投影重力、线/角速度、
  关节位置/速度、脚在基座系位置、步态相位、上一动作、相对目标）。
- 动作 `polar_course`：归一化 `[-1,1]^3` → (距离 0.07–0.22 m, 方向 ±35°, 落足偏航 ±15°)，
  双脚最小横向间距 0.10 m。

### 1.3 世界模型（`OptionTaskWorldModel`）

- 编码器：深度 CNN + 本体 MLP → 融合 → **128 维隐状态，SimNorm(8) 归一**。
- 动力学：`z' = SimNorm(z + mlp(z, a))`。
- head 分两类：
  - **任务/安全**（输入 [z,a]）：progress、goal、collision、fall、off_support（事件 logit）、
    collision_force、stability_margin、support_fraction、touchdown_error（连续 margin）。
  - **选项**：option_duration、continuation、twin-Q、policy（策略先验，阶段一未用）。
  - **监督 head**（输入 z'）：`task_state` → 21 维宏观运动状态。
- 隐状态**不做完整观测重建**，只由 21 维宏观状态监督（V4 核心改动）。

### 1.4 损失

```
L = event·ΣBCE + margin·ΣsmoothL1 + task_state·alive_smoothL1
  + q·q_loss + duration·duration + continuation·continuation
  + ranking·ranking + policy·policy + regularization
```

- 事件/margin/task_state 在模型自己的隐空间 5 步 rollout 上计算，权重 `0.8^t`。
- **ranking（反事实排序）恒为 0**：`counterfactual_group` 从未被采集填充（见 §3.1）。

### 1.5 奖励（每个选项结束算一次）

```
R = 10·progress − 0.02·ticks/25 + 10·goal − 2·collision − 5·fall − 3·(1−support_fraction)
```

- progress = 选项前后到目标距离的**差分**（重置时重新锚定，不白送）。
- goal = 距目标 <0.30 m 且首次（每 episode 一次）。
- collision = 非足刚体接触力 >5 N（V5 从 1 N 抬高）。
- off_support = 3×3 九点支撑占比的分级惩罚（V5 从单点二元 −10 改来）。

### 1.6 规划与收集

- CEM 在隐空间滚动 `planning_horizon=3` 步，最大化模型预测的任务奖励。
- 阶段一规划配置：`planning_disable_goal`（goal 系数置 0）、`planning_base_safety_scale 0.5`
  （安全项 ×0.5）、`terminal_value_coef 0`。
- 收集 `mixed`：warmup 后 50% CEM 规划 + 分层模板（20% 完全随机）。

### 1.7 反向课程重置（V5）

- `--reset_curriculum_prob 0.5`：50% 概率在赛道中段随机支撑位重生（距目标 ≥0.6 m、
  距起点 ≥0.5 m、3×3 全支撑），让模型经历中段/接近目标状态，使 goal 可被采样。
- 评估脚本不带此 flag，评估始终从起点完整跑。

---

## 2. 发现的现象（结论性）

### 2.1 宏观状态监督解决了"隐状态学不到任务"（V4）

- 去掉了完整观测一致性/重建后，`loss_task_state` 收敛到 ~0.013、`prediction_task_state_mae`
  ~0.08（宏观状态预测良好），progress head MAE ~0.05 m。
- **结论**：隐空间动力学已经能被 21 维宏观运动状态有效监督，不再是"学不到任何任务信息"。

### 2.2 奖励分解：progress 稳定为正，reward_mean 被摔倒率摆动主导

- `reward_progress` 稳定在 ~+0.8~0.9（机器人每选项确实在前进）。
- `reward_fall` 摆动幅度（约 ±1.0）超过 progress 本身（0.8），所以 `reward_mean` 没有
  上升趋势，不代表 progress 没学，而是被摔倒率摆动淹没。
- 摔倒率 `done_fraction` 在 **2%~23%** 之间以约 3 分钟周期振荡——这是规划器-数据反馈环
  （模型在"激进/保守"之间来回摆），不是稳定收敛。

### 2.3 goal 与 fall 的"正相关"是指标口径问题，不是 bug

- goal 和 fall 都是"每 episode 至多一次"的终止事件。日志里的 `reward_goal`、`reward_fall`
  是**每条 transition 的事件率**，两者都乘了 `E/N = 1/平均 episode 长度`。
- 平均 episode 变短时（快速到目标或快速摔倒），两者**一起升高**；episode 变长（徘徊）时
  一起降低。真正该看的是每 episode 的结局分布（成功率 G/E、失败率 F/E），当前日志缺这个。

### 2.4 中间位置重置混淆了"进步"指标

- `distance_mean_m` 下降、`reward_goal` 出现，**很大程度是中间位置重置的功劳**——50% 的
  episode 直接把机器人传送到离目标 1~2.6 m 的位置，距离自然变小、目标自然更容易够到。
- 不被重置污染的信号是 `reward_progress`（差分，每选项 ~0.09 m），它只说明"每步在前进"，
  不代表能完成任务。

### 2.5 评估结论（决定性）：尚未学会完成任务

从起点完整赛道评估（10 env × 30 s，训练地形）：**成功 3 次 vs 摔倒 40 次**（约 7% 成功率），
每次 episode 平均只前进 **0.92 m**（赛道 3.5 m）。即模型没有学到"走完全程"这件事。

### 2.6 吞吐与 num_envs 上限

- 本机最优并行环境数 **2048**（128→13.8 / 512→42 / 1024→63 / 2048→123 tps；3072/4096
  原生段错误）。已写入记忆。
- replay 采样曾有两处 O(size) 瓶颈，已修：`padded_sequence_indices` 向量化 +
  `_balanced_sequence_probabilities` 改 subset-then-balance + 版本号缓存 + replay 容量
  固定 10 万。（注意：当前在跑的进程加载的是旧代码，tps 仍低，重启后恢复 ~120。）

---

## 3. 可能的问题（假设）

### 3.1 核心问题：模型学的是"状态风险"，不是"动作风险"

- fall/collision/off_support 这些安全 head 只在 `(状态, 实际执行动作)` 上训练，学会的是
  "这个**状态**危险"，而不是"这个**落脚动作**危险"。
- CEM 规划器需要后者：在同一点对比几十个候选落脚点，靠"每个候选分别导致多大摔倒概率"来
  选安全的。但模型的 fall 预测对候选动作区分度近零，于是规划器只能"最大化前进"，走到
  间隙/障碍就摔。
- **证据**：`loss_counterfactual_ranking = 0.0`——为动作区分度设计的反事实排序机制
  （`counterfactual_group` + ranking loss）在采集管线里从未被填充，恒为空。

### 3.2 摔倒率反馈环

- 摔倒率 2%~23% 的振荡是模型-数据反馈环：模型保守→摔倒少→fall 数据稀缺（balanced replay
  又 20× 过采样）→模型对 fall 过拟合/误校准→规划器激进→摔倒多→循环。
- 该振荡目前没有随训练收敛的趋势。

### 3.3 "fall 预测 99% 准确"是混淆，不是真实能力

- `prediction_fall_balanced_accuracy≈0.99`、`brier≈0.003` 只能说明模型**过拟合到训练分布**
  （记住了某些状态序列的终点），不代表模型学会了因果地预测"某个动作会不会摔"。真正的
  检验是固定状态下、动作变化时 fall 预测的方差——需要反事实数据才能做。

---

## 4. 下一步方向

1. **补上反事实采集**（最优先）：让 `counterfactual_group` + ranking loss 真正有数据可学，
   训练模型的动作区分度。实现方式（二选一或结合）：
   - 同状态多动作探针：决策时保存物理+采样器状态，用不同动作各滚一个选项、记录真实结果，
     再恢复状态走主轨迹；
   - 配对环境：成对环境共享地形+域随机化，一个走规划动作、一个走随机动作。
2. **加 episode 级结局统计**：`goal_episode_rate` / `fall_episode_rate` / `timeout_episode_rate`，
   替代被 episode 长度稀释的 per-transition 指标，作为真正的收敛判据。
3. 重启训练（应用已就绪的 replay 提速 + 上述改动），继续观察摔倒率是否收敛、成功率是否上升。

---

## 附：当前运行状态

- **已于 2026-08-26 应要求暂停**：后台训练进程（`v5_rc_seed924_2048env_100k_cap100k`，
  原 pid 1791909）与监测脚本已停止，GPU 已释放。暂停前的最终指标：28124 updates、
  `distance_mean_m` 2.14、`done_fraction` 13%、`reward_goal` 0.08、`loss_task_state` 0.013。
- 已就绪但未应用的改动：replay 版本号缓存（重启后 tps 可恢复 ~120）。
- 恢复训练时，从 §4 的"下一步方向"继续。
