# 来源说明（provenance）

## 下层策略

- 冻结的 **Mind Your Steps** 落足跟踪策略，SF 机器人（TRON1A）。
- checkpoint：`checkpoints/lower_model_7000.pt`；加载时需先 `import isaacgym` 再
  `import torch`（checkpoint 反序列化会拉入 isaacgym）。
- 50 Hz，输入深度 + 本体 + 目标，输出关节动作，实现论文跟踪奖励的闭环落足执行。
- 上层**不**使用下层落足误差作为奖励，只通过真实闭环执行后的稳定性/碰撞/可行区域/任务
  进度学习。

## 上层框架

- 整体思路参照 **TD-MPC / Dreamer**：学习隐空间动力学 + 隐空间 MPC（CEM）规划。
- 本实现手写、可读，不调用现成 TD-MPC 包。
- **SimNorm**（每 8 维 softmax 归一化）来自 TD-MPC 的隐状态防饱和技巧。

## 相对 TD-MPC/Dreamer 的定制

1. **半马尔可夫选项**：一次上层决策 = 一次左右脚切换（约 25 个 50 Hz tick），因此有
   `option_duration` 与 `continuation` head，折扣按选项时长衰减。
2. **宏观状态监督代替观测重建**：TD-MPC/Dreamer 常让隐状态预测下一步观测/隐状态一致；
   这里改为让 `task_state` head 预测 21 维宏观运动状态。原因见 research_log。
3. **无独立 actor**：CEM 就是策略改进算子；twin-Q 回归真实截断 return（不 max 随机动作
   外推）。
4. **动作空间**：polar（距离/方向/偏航），约束在冻结下层采样器的支持集内（距离
   0.07–0.22 m、方向 ±35°、偏航 ±15°、双脚最小横向间距 0.10 m）。

## 场景

- 第一阶段用 `terrain_curriculum=research --research_kind random_composite`：每个并行环境
  一条独立固定随机路线（组合不同宽度支撑面、小间隙、低栅栏、可绕行障碍）。
- 场景由 `factory.py` 的 `build_tiled_heightfield` 生成，静态几何走
  `cfg.terrain.static_boxes`（此时关闭 gravity 随机化）。

## 旧模型变体（历史，非当前）

- `compact` / `spatial`：带深度解码器的重建监督模型，已弃用。
- `task`：去解码器但曾用完整观测一致性，已被宏观状态监督取代。
