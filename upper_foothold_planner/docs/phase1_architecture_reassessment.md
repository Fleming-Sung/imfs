# Phase 1 训练架构重审：从在线隐空间 MPC 转向教师引导的候选落足策略

> 日期：2026-08-26。目标不变：尽快训练出一个从起点完整通过训练赛道、行为上有明显提升、
> 且 Actor 输入可部署的上层策略。本文只讨论决定成败的结构，不继续围绕单次 log 抖动调参。

## 1. 结论

当前 V5 的在线 `世界模型 + CEM + mixed collection` 不适合作为 Phase 1 的主训练结构。
建议保留它作为后续 model-based 对照，但停止把“继续降低预测 loss”当作当前主线。Phase 1 改为：

1. 用冻结下层的真实闭环，先标定可控落足动作集；
2. 用仿真特权地形构造几何教师，首先证明层次接口本身能完成全程；
3. 把连续落足改成小型、无别名的候选集合，直接训练候选评分/策略；
4. 用 DAgger 收集学生真正访问到的状态，蒸馏为只使用深度和可部署本体感知的 Actor；
5. 达到较高成功率后，才用非对称 PPO 做短程微调。

这条路线把“感知、动作可行性、长程绕障、动态稳定性、模型误差、在线规划”从一个闭环中拆开，
每一层都能单独用真实行为验收。它比补一个 counterfactual ranking loss 更直接。

## 2. 为什么当前结构停滞

### 2.1 动作名义上三维，实际方向维几乎失效

`polar_course` 先用距离和方向计算横向量，再将左右脚横向量裁到至少 0.10 m。在当前
距离 0.07--0.22 m、方向 ±35° 的范围内，对均匀随机动作做解析审计，约 **98.1%** 的样本
横向量都被裁成恰好 0.10 m，横向量标准差只有约 **1.4 mm**。真实评估的平均横向落足
`0.100019 m` 与此一致。

因此 CEM 搜索的第二维几乎是别名动作。模型被要求学习 `(z,a)->outcome`，但大量不同 `a`
对应相同物理目标；同时机器人又缺少绕障所需的横向控制权。此时增加动作排序 loss 不能解决
接口本身退化的问题。

此前同一 `model_7000` 的随机化可达域审计还记录到：只有 29/250 个单腿网格、10/125 个双腿
配对动作通过 5 cm P95 标准。该结果说明下层可达域不仅狭窄，而且明显非均匀；具体候选仍需在
当前 Isaac wrapper 中重新确认，不能直接把归一化动作立方体当作均匀可控空间。

### 2.2 预测 loss 继续下降，任务行为没有继续提升

V5 续训中，`prediction_task_state_mae` 大致从 0.10 降到 0.078，说明模型越来越会拟合 replay
中的宏观状态；但后段训练的 `distance_mean_m`、`done_fraction` 和 progress 没有对应改善。
从起点的独立评估仍是 3 次成功、40 次摔倒，平均每 episode 只前进 0.92 m。

保存的实际运动窗口也显示停滞：同一续训段较早窗口的净位移/路径长度约 0.53，后段降到
约 0.32；机器人在移动，但更多是在绕行和徘徊。这说明“宏观状态预测更准”不是当前策略提升
的充分条件。

### 2.3 当前 latent 不是可靠的多步规划状态

当前 `z` 由单帧深度和 36 维本体观测编码，随后开环滚动 3 步。`task_state` 只监督基座运动、
脚、相位和相对目标，不监督下一时刻的局部地形或与真实下一观测编码的一致性。因此：

- `z'` 是否仍保留规划所需的地形几何没有明确约束；
- 单帧观测缺少若干决定跌倒风险的动态量，且没有历史 belief；
- 多步 CEM 同时依赖视觉编码误差、latent dynamics 误差和事件 head 校准误差。

V4 去掉像素重建是合理的，但“不要重建像素”不等于“无需监督任务相关地形”。真正需要的
辅助目标是局部可行落足图、候选支撑率和候选结果，而不是完整深度图，也不只是 21 维运动状态。

### 2.4 奖励和短视界规划鼓励直冲目标

当前 progress 使用欧氏目标距离差，planning horizon 只有 3 个落足，terminal value 和 goal
又被关闭。需要暂时侧移或绕开障碍时，正确动作可能产生零甚至负的短期欧氏 progress；当前
规划目标会系统性排斥这种动作。即使 fall/off-support 预测完全正确，策略也更容易在障碍前
保守停滞或反复试探，而不是沿可通行路径绕行。

### 2.5 在线模型和采集策略形成不稳定反馈环

50% 动作由当前模型的 CEM 产生，模型又立刻在这些状态上训练。安全事件的条件分布随规划器
变化，balanced replay 再放大稀有事件，形成激进/保守振荡。`fall balanced accuracy` 很高只
证明 replay 内分类容易，不能证明同一状态下模型能区分候选动作。

反事实数据能改善动作区分度，但若动作仍有别名、规划仍使用错误距离势函数、latent 仍缺少
地形 belief，单独补 ranking 不会成为最快路径。

### 2.6 梯度更新比例远高于数据量所需要的水平

当前代码中 `updates_per_transition=1` 表示每新增一条 transition 做一次 `batch_size=512` 的
梯度更新，等价于约 512 个训练样本抽取/新 transition。V5 在约 5.6 万条 transition 上已经做了
约 3.5 万次 sequence batch 更新，和安全 head 迅速拟合 replay、但闭环行为不提升的现象一致。

应把 UTD 明确定义为“训练样本数/新 transition 数”：

```text
gradient_updates += new_transitions * sample_utd / batch_size
```

Phase 1 初始建议 `sample_utd=4--16`，用 held-out action-ranking 和完整 episode 成功率决定是否
增加，而不是默认 512。

## 3. 新的上层问题定义

### 3.1 决策状态

把真实半马尔可夫状态分成三部分：

- 机器人动态：IMU 可提供的姿态/角速度、可部署的基座速度估计、关节位置速度、双脚相对位姿、
  步态相位和当前摆动脚；
- 局部地形：基座坐标系下、覆盖未来 2--3 步范围的可通行/高度表示；
- 任务：相对目标或局部路线方向，以及上一上层动作。

仿真中的绝对位姿、精确 heightfield、接触力、真实接触、终止原因和域随机参数只能给教师、
Critic 和辅助标签，不能进入最终 Actor。

### 3.2 观测与 belief latent

Actor 观测建议为：

```text
o_t = depth_t + deployable_proprio_t + previous_action + relative_route_goal
z_t = GRU(terrain_encoder(depth_t), proprio_encoder(o_t), z_{t-1})
```

这里 `z_t` 是历史 belief，不再承担开环重建整个世界的职责。建议拆成可解释的两块：

- `z_terrain`：局部可通行几何；
- `z_dynamics`：机器人当前运动/稳定性和下层执行滞后。

辅助监督只保留直接有用的量：候选目标支撑率、候选一步 fall/collision 概率、下一落足后的
位移/姿态裕度、触地误差。不重建 64×64 像素，也不在 Phase 1 训练 3--5 步 latent 世界模型。

### 3.3 动作

第一阶段使用无别名的候选落足集合，而不是连续 polar CEM：

- 在支撑脚 yaw 坐标系直接定义 `(forward, signed_lateral, yaw)`；
- 左右脚的符号由摆动脚决定，`lateral_abs` 独立参数化，不再事后 clamp；
- 先用冻结下层的平地闭环测出零摔倒、touchdown p95 合格的范围；
- 在该范围内构造约 24--48 个候选，yaw 可先由局部路线方向确定，只学习二维落点；
- 后续如需连续精度，再输出候选周围的小 residual。

离散候选不是最终能力上限，而是 Phase 1 的优化手段：它让每个状态天然拥有多个可比较动作，
可以直接做分类、ranking、Q 或 advantage 学习，也让可视化非常直观。

## 4. 特权教师与训练架构

### 4.1 Gate A：先证明层次接口可解

用仿真精确 heightfield 为每条赛道预计算从任意支撑区域到目标的 geodesic distance field。
教师在每次落足时枚举全部候选，按以下顺序过滤和评分：

1. 目标足底 3×3 或更密 stencil 的支撑率；
2. 足端摆动路径和身体包络是否撞障碍；
3. 候选是否在下层已标定的动态可达集；
4. geodesic progress，而不是欧氏直线 progress；
5. 步长变化、yaw 变化和稳定裕度。

先直接运行教师。如果教师从起点也不能高成功率完成 3.5 m 赛道，问题在下层能力、动作集、
碰撞模型或课程地形，继续训练视觉策略没有意义。

### 4.2 Gate B：教师监督预训练

利用 2048 个并行环境收集 `(history observation, candidate scores, teacher action)`。每个状态
都能从 heightfield 一次性标注全部候选，不需要为每个候选都做完整物理分叉。网络训练：

```text
L = CE(student_logits, teacher_action)
  + BCE(candidate_feasible, privileged_feasible_labels)
  + Huber(candidate_progress, geodesic_progress)
  + auxiliary_next_motion
```

这一步应离线训练，数据收集与梯度更新解耦；固定 train/validation terrain-state split，避免
用 replay 内准确率冒充动作泛化能力。

### 4.3 Gate C：DAgger，而不是 50% 在线 CEM

学生开始闭环运行后，在它真实访问的状态上继续查询教师。收集以学生为主、教师逐步退火，
但执行动作可加安全接管。这样解决 covariate shift，同时不会让一个尚未校准的世界模型改变
数据分布。

只有当纯学生从起点成功率稳定达到约 70%--80%，再用非对称 Actor-Critic/PPO 微调动态性能。
Critic 可看精确局部 heightfield、真实基座完整状态和接触诊断；Actor 始终只看部署观测。

## 5. 奖励

RL 微调使用 geodesic potential shaping：

```text
r_progress = gamma * Phi(s_next) - Phi(s),  Phi(s) = -d_geodesic(s, goal)
r = w_p*r_progress + success_bonus
    - fall_penalty - collision_penalty - unsupported_penalty
    - small_action_change_penalty
```

这样绕障所需的侧移不会被欧氏距离错误惩罚。奖励只负责排序行为，不再同时承担训练世界模型
所有 head 的职责。touchdown error、支撑率、稳定裕度继续单独记录，并可作为小权重辅助项；
不能让 survival 或总 reward 掩盖任务失败。

## 6. 训练速度

本机 RTX A6000 48 GB + Ryzen 7950X，当前 2048 env 是已验证的稳定上限。优先优化如下：

1. 删除在线训练阶段的 CEM 内环和 5 步 latent rollout；候选 actor 一次前向即可决策；
2. 修正 UTD 定义，数据采集和离线 SGD 分段执行；
3. 深度只在统一的上层决策边界渲染。当前任一异步 env 触发事件时会为全部 env 捕获深度，
   应采用同步 decision groups，避免接近每个 50 Hz tick 都渲染 2048 个相机；
4. Phase 1 可先用 32×32 或紧凑 egocentric BEV 做教师蒸馏，确认成功后再比较 64×64；
5. 物理收集使用 2048 env，视觉评估使用少量 env 并保存视频；不要用 headless 指标代替行为。

## 7. 收敛与停滞判据

唯一主指标是固定起点、固定评估集上的 episode 结局：

- success / fall / timeout episode rate；
- 到目标前的中位落足数和连续运行时间；
- geodesic path efficiency；
- touchdown error、off-support、碰撞和摔倒原因；
- 不同地形实例分别统计，而不是只报总平均。

每隔固定数量的真实 transition 做一次独立 deterministic eval。连续三次评估无明显提升时，
直接查看典型成功、摔倒和徘徊轨迹/动画，并按“感知错误、教师/路线错误、动作不可达、下层执行
失败”分类；不根据 TensorBoard 的微小 loss 变化继续训练。

## 8. 实施顺序

1. 保留当前运行进程不动，另建 V6 实验路径；
2. 修复动作参数化并做冻结下层 reachable-set gate；
3. 实现 heightfield geodesic 教师，先跑教师完整赛道 eval；
4. 实现候选 actor + deployable history belief，监督预训练；
5. DAgger 闭环收集与训练；
6. 达标后再决定是否需要 PPO residual 或恢复 model-based MPC 作为论文扩展。

如果 Gate A 成功，这条路线应很快给出肉眼可见的上层行为；如果 Gate A 失败，也能在训练大模型
之前得到明确、可定位的失败证据。
