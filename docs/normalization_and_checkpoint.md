# 运行归一化、checkpoint 与 train/eval 一致性

> 状态（2026-08-21）：本文描述当前 `foothold/ppo.py::Normalizer` 的实际行为，并记录
> `Aug21_18-27-00_step005-020_yaw10pct` 的 train/eval 失配。结论是：该 run 的落足点
> 规划器在 train 与 eval 中一致；主要问题是 checkpoint 未保存训练期 running statistics。

## 1. Normalizer 是什么

Normalizer 不是神经网络层，也没有通过反向传播学习的参数。它是训练循环外的一组运行统计量，
持续估计输入各维的均值和方差，再把原始数值转换到相对稳定的尺度：

$$
\hat{x}_j = \operatorname{clip}\left(
\frac{x_j-\mu_j}{\sqrt{\sigma_j^2+10^{-8}}},-c,c\right).
$$

当前 `c=10`。每一个输入维度都有独立的 $\mu_j$ 和 $\sigma_j^2$；不同物理量不会混在一起
求一个总均值。例如关节角、角速度、目标位置和 gait phase 各自使用自己的统计量。

`RunningMeanStd.update()` 对一个 batch 计算均值和 population variance，再用 batch 合并公式更新
历史统计。`count` 表示至今累计了多少个样本。这个算法与保存全部历史样本后重新计算均值/方差
等价，但只需要常量内存。

归一化的作用主要有两个：

1. 让量纲和数值范围不同的输入落到相近尺度，避免大数值维度天然支配 MLP 第一层。
2. 随训练数据分布逐渐变化时，给优化器提供较稳定的输入坐标系。

它不会改变物理仿真状态、落足点位置、奖励计算或 PD 控制；它只改变送入 Actor/Critic 的数值，
以及 PPO 内部使用的奖励尺度。

## 2. 当前实现包含四组统计量

`Normalizer` 内部并不是只有一对 mean/variance：

| 统计量 | 形状（SF） | 用途 | eval rollout 是否需要 |
|---|---:|---|---|
| `actor_obs` | 30 | Actor 的本体感知观测 | 需要 |
| `goal` | 16 | 左右落足目标和 gait phase | 需要 |
| `critic_obs` | 33 | Critic 的特权观测 | 仅计算 value 时需要 |
| `return_rms` | 标量 | PPO 的 reward normalization | 纯 rollout 不需要，恢复训练需要 |

Actor 的实际输入是：

$$
\pi_\theta\left(
[\operatorname{norm}_{obs}(o_t),\operatorname{norm}_{goal}(g_t)]
\right).
$$

Critic 类似，只是把 `critic_obs` 与同一份 normalized goal 拼接。把 obs 和 goal 分开维护逐维
RMS，再拼接输入，在二者按相同 batch 和时序更新的前提下，与先拼接再做逐维 RMS 数值等价。

奖励走另一条路径。代码维护折扣回报：

$$
R_t^{run}=\gamma R_{t-1}^{run}(1-d_t)+r_t,
\qquad
\hat r_t=\frac{r_t}{\sqrt{\operatorname{Var}(R^{run})+10^{-8}}}.
$$

PPO storage 和梯度更新使用 $\hat r_t$，但环境的原始 reward、TensorBoard 中的 episode return
和 eval 轨迹仍可保留原始单位。reward normalization 不直接影响部署时 Actor 前向计算。

## 3. 训练时它如何工作

当前训练流程为：

1. 环境产生原始 `obs / goal / critic_obs`。
2. `normalizer.observations(..., update=True)` 先用当前 batch 更新 RMS，再返回 normalized tensors。
3. Actor/Critic 只消费 normalized tensors。
4. 环境产生原始 reward 后，`normalizer.rewards(..., update=True)` 更新 return RMS，并把 normalized
   reward 写入 PPO storage。
5. 下一策略步继续更新统计，因此网络与这组不断演化的输入坐标系共同训练。

因此训练完成后的策略不是一个可以脱离 RMS 单独解释的函数。完整的推理函数实际是：

$$
a_t=f_\theta\left(\frac{x_t-\mu_{train}}{\sigma_{train}}\right),
$$

其中网络权重 $\theta$ 和训练结束时的 $\mu_{train},\sigma_{train}$ 都属于模型状态。

## 4. 为什么 eval 不能新建一个空 Normalizer

新建 `RunningMeanStd` 时 `mean=0, var=1`。如果随后使用 `update=False`，则 eval 实际计算的是：

$$
\hat{x}_{eval}\approx x,
$$

而不是训练时的：

$$
\hat{x}_{train}=\frac{x-\mu_{train}}{\sigma_{train}}.
$$

两者代表不同的输入坐标系。即使原始环境状态和落足目标完全相同，Actor 第一层收到的向量也不同，
输出动作自然可能完全不同。

落足 goal 对这个问题尤其敏感。reset 时两个目标四元数的 `w` 分量通常接近 1；训练 batch 会将
这种近常量维度中心化到 0 附近，eval 的空 Normalizer 却直接把 1 送入网络。目标位置、yaw 和
`cos/sin(2*pi*phase)` 也都有各自的训练期均值和尺度。因此视觉上可能表现为“机器人过度追点”，
但 planner 给出的物理目标并没有改变，改变的是策略对目标向量的解释。

不能在 eval 中用单个环境临时重新估计 RMS 来替代训练统计：

- 统计会随 eval 轨迹漂移，使同一个状态在不同时刻对应不同输入；
- 单环境早期样本方差很差，近常量维度容易被放大并触发 clip；
- eval 状态分布已经由错误动作产生，形成循环偏差；
- 即使采很多样本，也无法保证恢复训练结束时网络适配的精确坐标系。

## 5. 本次 train/eval 失配的代码证据

修改前，训练在 `foothold/train.py` 中创建 Normalizer，并在初始观测和每个 rollout step 使用
`update=True`，但 checkpoint 只保存：

```python
{"iteration": it, "actor_critic": ac.state_dict()}
```

`model_1300.pt` 的顶层也确实只有 `iteration` 和 `actor_critic`，没有任何 mean、variance 或 count。

修改前的 eval 随后创建一套全新的 Normalizer，并始终调用 `update=False`，所以它没有恢复训练
输入分布，而是近似使用原始输入。

与此相对，受控对照表明 iteration 1300 的训练配置和 eval 配置都使用 `[0.05,0.20] m` 步长；
相同 seed 与足端状态下，sampler 的 phase switch、swing foot、target position 和 target yaw
逐元素一致。现有失败轨迹还显示，第一次非平凡目标约在第 25 步生成，而 base 在此之前已经从
约 `0.659 m` 下沉至约 `0.467 m`。这些证据不支持“eval 使用了不同 planner”。

## 6. 已实施的 checkpoint 契约

从 checkpoint format version 2 开始，训练保存：

```text
iteration
actor_critic state
actor_obs RMS: mean, var, count
goal RMS: mean, var, count
critic_obs RMS: mean, var, count
return RMS: mean, var, count
训练时实际配置或其不可变快照
checkpoint 格式版本（当前为 2）
```

其中 `normalizer` 自身带 version、`gamma` 和 `obs_clip`，四组 RMS 分别带 mean、variance 和 count。
`returns` 与当前 rollout 的环境数量绑定，属于瞬时状态，不跨运行保存。

eval 现在先检查 format version、Normalizer 和训练配置，再创建环境；随后恢复训练 RMS 并始终使用
`update=False`。环境使用 checkpoint 内的训练配置，避免当前 `config.py` 改动污染旧模型评估。
跨机器时，如果保存的 URDF 绝对路径不存在，只允许重定位到本地同型号机器人资源。

若要求可恢复训练，还应继续保存 optimizer、自适应 learning rate、随机数状态，以及需要延续的
训练进度状态；当前 version 2 只保证 eval 所需状态完整，还不是完整的训练断点恢复格式。

推荐区分两种评估模式：

- **确定性基准评估**：恢复训练 RMS，Actor 使用均值动作，关闭 observation noise、kick 和域随机化。
- **鲁棒性评估**：仍恢复同一训练 RMS，但显式开启训练分布内的噪声和域随机化，多 seed 报告统计。

二者都不应改变落足 sampler 的算法和任务参数。

## 7. 旧 checkpoint 的限制

`Aug21_18-27-00_step005-020_yaw10pct` 已保存的 checkpoint 没有 RMS，因此无法仅根据 `.pt`
文件精确恢复当时的策略输入坐标系。网络权重不能反推出唯一的训练均值和方差。

eval 会对这类旧 checkpoint 明确报错，不再退回空 Normalizer。`scripts/check_actions.py` 同样会标记
并跳过缺少 Normalizer 的 checkpoint。

如果原训练进程仍在运行，准确统计只存在于该进程内存中的 `normalizer` 对象；应优先设法从该
进程导出统计。否则只能重新训练，或明确标记为近似恢复并做额外校准实验，不能把临时收集的 RMS
当成原始训练统计。

## 8. 排查清单

出现“train 很稳、eval 秒倒”时，按以下顺序检查：

1. checkpoint 是否同时包含网络权重和全部 RMS。
2. eval 是否成功加载，而不是静默使用默认 `mean=0,var=1`。
3. eval 是否固定 RMS（`update=False`）。
4. actor observation 和 goal 的维度、排列、坐标系及缩放是否与训练一致。
5. eval 是否加载该 run 保存的配置，而不是当前源码默认配置。
6. 再比较确定性/随机动作、seed、噪声、域随机化和 planner 参数。
