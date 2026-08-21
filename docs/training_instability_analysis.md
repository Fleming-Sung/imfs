# 训练不稳定根因分析与调参方案报告

> 对象：`refactor/` 下的 SF_TRON1A 平地落足跟踪训练
> 结论时间：2026-08-16
> 结论：不稳定性**不是**超参数能单独解决的；根因是**奖励信号“时间稠密但状态无信息”**，并被一处**与论文的 fidelity 偏差**放大。观测/动作/算法链路经核查无硬 bug。

---

## 1. 摘要

1. 阶段 1~3 的 PPO 超参数调整**已经生效**：`policy_mean_std` 由上涨转为单调下降（0.135→0.098），`learning_rate` 由 400 倍震荡收敛到 ~3 倍区间（4e-5~3e-4），`kl` 降到 0.005~0.01。
2. 但 `episode_length` 仍在 `1 ↔ 354` 步之间剧烈震荡（后 150 轮 std=74）。**这证明根因不在 PPO 动力学超参数**。
3. 数据铁证：`corr(episode_return, episode_length) = 0.999`，而 `corr(step_reward, episode_length) = 0.24`，且 `step_reward ≈ 0.315 ± 0.076`（近乎常数）。
4. 结论：奖励是**每步恒定、与状态无关**的。agent 的唯一学习信号是“从这个状态还能活多久”，而存活时长是高方差、与摔倒随机性纠缠的，导致 advantage 噪声大、策略反复横跳（“好几百步→秒倒”循环）。
5. 叠加一处 fidelity bug：`sharpness.xy = 100`（论文为 **20**），把摆动脚跟踪奖励压成**近乎稀疏**（仅触地前 ~0.1m 有分），任务信号进一步被削弱。

---

## 2. 现象与数据证据

### 2.1 三个阶段的指标变化

| 指标 | 原始（44k 轮） | 阶段 1+2 后 | 阶段 3 后 |
|---|---|---|---|
| `policy_mean_std` | 0.135→**0.17**（涨） | 0.134→0.096（降） | 0.135→0.098（降） |
| `learning_rate` 区间 | 1e-6 ↔ 4e-3 | 1e-6 ↔ 4.4e-3 | **4e-5 ↔ 3e-4**（稳） |
| `kl` | 0.02~0.03 | 0.005~0.077 | **0.005~0.011** |
| `episode_length` | 1.4↔283 | 1↔399 | **1↔354（仍震）** |

阶段 3 后 lr/std/kl 全部健康，**但 episode_length 依然震荡**——这是把矛头指向“奖励结构”而非“优化器”的决定性证据。

### 2.2 关键相关性

对 `Aug16_22-30-58`（294 轮）计算：

```
corr(episode_return, episode_length) = 0.999
corr(step_reward,      episode_length) = 0.240
step_reward: mean = 0.315, std = 0.076
```

**解读**：`episode_return ≈ step_reward × episode_length`，而 `step_reward` 几乎不随“活得好不好”变化（std 只有均值的 24%）。即：**单步奖励是常数，回合回报几乎完全由存活时长决定**。

### 2.3 每步奖励构成（加权，末轮）

| 项 | 每步加权 | 占比 | 性质 |
|---|---|---|---|
| stance_xy | +0.100 | 20.2% | 免费分（支撑脚目标锁定在自身） |
| stance_yaw | +0.100 | 20.2% | 免费分 |
| nominal_joint_pos | +0.076 | 15.4% | 保持名义姿态即可 |
| gait_height | +0.072 | 14.6% | 步高（站立即接近满分） |
| swing_xy | +0.053 | 10.7% | **真正任务** |
| swing_yaw | +0.022 | 4.3% | **真正任务** |
| feet_swing | +0.012 | 2.4% | **真正任务** |
| survival | +0.005 | 1.0% | 存活 |

**站着不动白拿约 70%，真正落足跟踪只占 17%，存活项仅 1%。**

---

## 3. 根因分析

### 3.1 核心：奖励“时间稠密但状态无信息”

设每步奖励近似为常数 $r_t \approx c$（本任务 $c\approx0.315$）。那么：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} \approx c \cdot \frac{1-\gamma^{T-t}}{1-\gamma}
$$

其中 $T$ 是 episode 终止时刻（摔倒）。**价值函数唯一可学的信息是 $T$（还能活多久）**：

$$
V^\pi(s) \approx c \cdot \mathbb{E}\left[ \frac{1-\gamma^{T-t}}{1-\gamma} \;\middle|\; s_t=s \right]
$$

于是：
- “好状态”（站得稳）和“坏状态”（快摔了）的**即时奖励完全相同**（都是 $c$），只有通过“未来存活时长”这个**延迟、随机**的量来区分。
- 存活时长本身是高方差的：摔倒由 kick 扰动、域随机化（摩擦/质量/重力）、动作噪声共同决定，存在强随机性。
- 结果：GAE advantage 的方差被“存活预测误差”主导，梯度方向噪声大 → 策略在“能走的参数区”和“秒倒的参数区”之间反复横跳。

这解释了用户的观察：**不是“站着不动很难做到”，而是“站着不动很难学会”**——因为奖励在摔倒发生之前几乎不给任何“你正在变糟”的信号，唯一的信号是摔倒本身，而它稀疏且随机。

> 注：`survival` 项（0.25 × dt = 0.005/步）只占 1%，不是主因；主因是 stance/nominal/gait 这些**每步饱和的常数项**。

### 3.2 叠加：与论文的 fidelity 偏差（重点，疑似重构 bug）

对照原版 `mind_steps/footstep_configs.py`，`refactor/config.py` 存在如下偏差：

| 参数 | 论文 | refactor | 影响 |
|---|---|---|---|
| `sharpness.xy` | **20.0** | **100.0** | 摆动脚 xy 奖励从“稠密”变成“近稀疏” |
| `sharpness.gait_height` | 25.0 | 100.0 | 步高奖励过度尖锐 |
| `feet_swing` | **10.0** | 6.0 | 抬脚奖励被削 40% |
| `action_rate` | -1.0 | -3.0 | 动作平滑惩罚 3 倍 |
| `feet_slip` | -2.0 | -4.0 | 2 倍 |
| `feet_roll` | -1.0 | -4.0 | 4 倍 |

`sharpness.xy` 的数学含义：

$$
r_{swing\_xy} = \exp\!\left(-\xi \cdot \|\Delta p_{xy}\|^2\right)
$$

- $\xi=20$：脚离目标 0.3m 时 $e^{-20\times0.09}=0.165$，0.5m 时 $e^{-5}=0.0067$——**摆动相全程都有梯度**。
- $\xi=100$：脚离目标 0.1m 时 $e^{-1}=0.37$，0.2m 时 $e^{-4}=0.018$，0.3m 时 $e^{-9}\approx10^{-4}$——**摆动相大部分时间奖励≈0，只有触地前 ~0.1m 才有分**。

$\xi=100$ 把整个“迈步到目标”的过程变成一个**只在触地瞬间给分的稀疏奖励**，梯度几乎无法引导 agent 学会“朝目标迈步”。这与 3.1 的“时间稠密但无信息”叠加，形成“既无稠密任务梯度、又无状态区分度”的双重困境。

### 3.3 PPO 超参数动力学（已修复部分）

已完成的阶段 1~3 及其数学作用：

#### 阶段 1：熵系数与动作方差上限

策略为对角高斯 $\pi_\theta(a|s)=\mathcal{N}(\mu_\theta(s), \sigma^2)$，其中 $\sigma=\mathrm{clamp}(e^{\log\sigma}, \sigma_{min}, \sigma_{max})$。熵：

$$
H[\pi] = \frac{1}{2}\sum_i \left(1+\ln(2\pi\sigma_i^2)\right), \qquad \frac{\partial H}{\partial \sigma_i} = \frac{1}{\sigma_i} > 0
$$

熵对 $\sigma$ 的梯度**恒为正**。PPO 损失中的熵项 $-c_e H[\pi]$ 会**永远把 $\sigma$ 往上推**。当 $c_e$（`entropy_coef`）过大时，$\sigma$ 上涨（观测到的 0.135→0.17），探索噪声永远关不掉。
- `entropy_coef`: 0.01 → 0.003：减弱熵梯度，让 $\sigma$ 能被奖励信号压下去。
- `max_std`: 1.0 → 0.5：给 $\sigma$ 封顶，防早期乱跳。

#### 阶段 2：更新次数与 mini-batch

每轮 rollout 后，PPO 对同一批数据做 `num_learning_epochs × num_mini_batches` 次梯度更新。20 次**全量**更新（20×1）会在单批数据上严重过拟合，导致策略单轮大幅漂移；改为 5×4 后（总步数仍 20，但每步只看 1/4 数据、且跨 epoch 重排），单步方差降低、过拟合减弱。

#### 阶段 3：自适应学习率控制器

KL 自适应规则（`ppo.py:171-178`）：

$$
\mathrm{lr} \leftarrow
\begin{cases}
\max(\mathrm{lr}/s,\ \mathrm{lr}_{min}) & \mathrm{KL} > \delta \cdot m \\
\min(\mathrm{lr}\cdot s,\ \mathrm{lr}_{max}) & \mathrm{KL} < \delta / m \\
\end{cases}
$$

其中 $\delta$=`desired_kl`、$s$=`kl_scale`、$m$=`kl_margin`。原来 $\delta{=}0.02,s{=}1.5,m{=}1.5$：KL 在 $[0.013, 0.03]$ 这个窄带外就触发 ×/÷ 1.5，而实测 KL 恰好落在这个边界附近 → **乒乓**。改为 $\delta{=}0.01,s{=}1.1,m{=}1.2$，死区变宽、步长变温和，lr 才稳定下来。

### 3.4 观测/动作/算法核查结论（无硬 bug）

| 核查项 | 结论 |
|---|---|
| 前一步动作时序 | 正确：`policy_actions` 在 `step()` 开头更新，obs 里读到的是上一步动作 |
| `done`/`absorbing` 掩码 | 正确（`done_buf` 已在 `_reset_idx` 清零前保存；GAE 中 fail=absorbing、timeout 非 absorbing） |
| 观测维度 | 30 = 6+3n（重力 3 + q 8 + 角速度 3 + qdot 8 + 上一步动作 8），与论文一致 |
| 观测缩放 | dof_vel×0.1，与论文一致 |
| 观测噪声 | dof_pos 0.03 / dof_vel 0.30(×0.1) / ang_vel 0.20 / gravity 0.015，量级合理 |
| goal 编码 | 16 维（左右足 3 位置 + 4 四元数 + 2 相位），与论文一致 |
| 动作 | 8 DoF，$a\in[-1,1]$，$q_{target}=\mathrm{clamp}(q_{nom}+a)$，与论文附录 F 一致 |

**因此“观测/动作/算法配置有 bug”这个方向可以排除。** 剩余的不稳定来自奖励结构本身。

---

## 4. 与同类项目对比

| 维度 | legged_gym（Rudin 2022，能跑） | 原版速度基线（PF_TRON1A，能跑） | 原版 foothold（论文） | 本 refactor |
|---|---|---|---|---|
| 任务信号 | 速度跟踪（稠密、方向明确） | 速度跟踪（`tracking_lin_vel`=1.0） | 落足跟踪 | 落足跟踪 |
| 任务奖励稠密度 | 每帧有明确目标速度 | 每帧明确 | `sharpness.xy=20`（稠密） | `sharpness.xy=100`（近稀疏） |
| 免费/风格项占比 | 小（正则项 ≤ 主项的 1/10） | 小 | 存在但被 $\xi=20$ 的稠密跟踪项平衡 | 70% 白拿 |
| `action_scale` | 0.25（位置偏移） | 0.25 | 1.0 | 1.0（忠实） |
| `entropy_coef` | 0.01 | 0.02 | 0.02 | 0.003（阶段 1） |
| 控制频率 | 100Hz（decimation=4） | 100Hz | 50Hz（decimation=8） | 50Hz（忠实） |

**核心差异**：能跑的项目都有**稠密、方向明确的任务奖励**（速度跟踪），而本 refactor 的落足跟踪奖励被 $\xi=100$ 稀疏化、又被 70% 的“站立白拿分”稀释。这是唯一能解释“原论文同套 PPO 超参（20 epoch、entropy 0.02）能收敛、而本 refactor 不收敛”的差异。

---

## 5. 分阶段解决方案

### 第 0 步（立即，纯 config）：先修 fidelity 偏差，恢复到论文数值

```python
# rewards.sharpness
"xy": 20.0,              # 100 -> 20（最关键：把摆动奖励变稠密）
"gait_height": 25.0,     # 100 -> 25
# rewards.scales
"feet_swing": 10.0,      # 6 -> 10
"action_rate": -1.0,     # -3 -> -1
"feet_slip": -2.0,       # -4 -> -2
"feet_roll": -1.0,       # -4 -> -1
```

保留阶段 1~3 已调好的 PPO 超参（entropy 0.003、5×4、lr 温和化）。跑 20k 轮，观察 `swing_xy` 在 `reward_w` 中占比是否上升、`episode_length` 方差是否下降。

### 第 1 步（若仍不收敛）：奖励再平衡，去免费分

```python
# rewards.scales
"stance_xy": 1.0,  "stance_yaw": 1.0,   # 5 -> 1（免费分降权）
"nominal_joint_pos": 1.0,                # 4 -> 1
"swing_xy": 8.0,   "swing_yaw": 8.0,     # 5 -> 8（抬高任务项）
```

目标：让 `swing_*` 三项占总奖励 >50%，`stance_*` 降到 <20%。

### 第 2 步（若仍不收敛，需改 rewards.py）：消除稀疏 + 降低存活预测方差

1. **摆动相稠密化**：把“整段摆动用一个终点目标”改成“按相位沿摆线插值目标位置”，使 $r_{swing\_xy}$ 在摆动相**每帧**都有非零梯度（这是从源头解决 3.1 的核心手段）。
2. **降低存活方差**：减少 `kick_probability`、收窄 `gravity_magnitude_range`/`friction_range`，或对 reward 做 `clip_reward`，让“存活时长预测”这个学习目标本身变稳定。
3. **缩短 horizon 或提高 $\gamma$**：$\gamma=0.995$ 下 20s episode 的折扣已足够，但若存活预测太难，可缩短 `episode_length_s` 以降低方差。

### 第 3 步（备选，需改 env）：若怀疑状态不可观测

- 给 actor 观测补 `base_lin_vel`（当前只有 critic 有），消除“看不到自身平移速度”的盲区。
- 给 goal 增加“距下一次迈步的剩余时间”，帮助预测相位切换。

---

## 6. 附录：各超参数的数学含义与作用

| 超参数 | 公式/定义 | 作用 | 过大 | 过小 |
|---|---|---|---|---|
| `entropy_coef` $c_e$ | 损失 $-c_e H[\pi]$ | 探索下限，防止 $\sigma$ 过早塌缩 | $\sigma$ 永远上涨，探索关不掉 | $\sigma\to\sigma_{min}$ 过早确定化 |
| `min_std`/`max_std` | $\sigma\in[\sigma_{min},\sigma_{max}]$ | 动作噪声硬边界 | 动作抖动大 | 探索不足 |
| `init_noise_std` | 初始 $\sigma$ | 开局探索量 | 开局乱 | 开局太保守 |
| `clip_param` $\epsilon$ | $\min(rA,\ \mathrm{clip}(r,1\pm\epsilon)A)$ | 策略更新信任域 | 更新过大 | 更新过慢 |
| `num_learning_epochs` | 每批数据重放次数 | 数据利用率 | 过拟合单批 | 数据浪费 |
| `num_mini_batches` | 批内切分份数 | 单步方差 | 每步方差小但慢 | 每步方差大 |
| `gamma` $\gamma$ | $G_t=\sum\gamma^k r_{t+k}$ | 奖励折扣、credit 跨度 | 依赖远期 | 只看近期 |
| `lam` $\lambda$ | GAE 的偏差-方差权衡 | $A_t=\sum(\gamma\lambda)^k\delta_{t+k}$ | 高方差 | 高偏差 |
| `desired_kl` $\delta$ | KL 目标 | 自适应 lr 锚点 | 更新太保守 | 更新太激进 |
| `kl_scale` $s$ | lr 乘除系数 | 自适应 lr 灵敏度 | 乒乓 | 响应迟钝 |
| `kl_margin` $m$ | 死区宽度 | 允许 KL 波动范围 | 死区过大失去控制 | 死区过小乒乓 |
| `max_grad_norm` | $\|g\|\leftarrow \min(1,\frac{c}{\|g\|})g$ | 梯度裁剪 | 抑制过大 | 不裁剪 |
| `value_loss_coef` | 损失 $c_v \|V-R\|^2$ | critic 拟合权重 | 忽略策略 | critic 拟合不足 |
| `sharpness.xy` $\xi$ | $r=\exp(-\xi\|\Delta p\|^2)$ | 跟踪奖励的“锐度” | 奖励稀疏 | 奖励平坦 |

**核心结论**：先修 `sharpness.xy`（100→20）与相关 fidelity 偏差（第 0 步），这是当前最确定、最便宜、风险最低的修复；奖励再平衡（第 1 步）与摆动相稠密化（第 2 步）是真正消除“时间稠密但无信息”的结构性手段。
