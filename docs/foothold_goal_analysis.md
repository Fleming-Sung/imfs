# 落足点目标（Foothold Target）设计机制分析

> 本文对照原论文参考实现（`loco-mujoco`，路径见下文）逐项审查当前 `foothold/` 复现代码的落足目标机制，
> 回答"为什么训练上千轮后机器人仍只学会站桩、不探索正确落足"这一核心问题，并给出修改建议。

> **状态（2026-08-21）**：第 2/3/4 节的修改已实施——步长范围改为 `[0.05, 0.20]`、补齐"距摆动脚 ≥ 0.05 m"
> 拒绝采样、足端朝向惩罚降到 10%（`swing_yaw/stance_yaw` 4.0→0.4，`feet_roll` -4.0→-0.4）。

## 0. 代码位置速查

| 模块 | 当前代码 | 原论文参考 |
|---|---|---|
| 落足目标采样 | `foothold/sampler.py`（`FootholdSampler._sample_next`） | `loco-mujoco/loco_mujoco/core/observations/goals_foot_placement.py`（`GoalDoubleFootPlacement.sample_goal`） |
| 目标→观测 | `foothold/sampler.py`（`FootholdSampler.observation`） | 同上（`get_obs_and_update_state`，`dim=16`） |
| 跟踪/风格奖励 | `foothold/rewards.py`（`Rewards`） | `loco-mujoco/loco_mujoco/core/reward/foot_placement.py`（`CrispBoosterLocomotionFootPlacementReward`） |
| 参数 | `foothold/config.py`（`foothold`、`rewards` 两节） | `loco-mujoco/experiments/humanoid_foot_placement/train/conf_t1.yaml` |
| 目标喂入策略 | `foothold/ppo.py`（`PPO.act`）、`foothold/train.py` | `loco_mujoco` 中 goal 直接 append 进观测 |

---

## 1. 检查项 1：落足目标是否被 agent 感知？（喂入方式对比）

### 1.1 原论文的喂入方式

原论文 `GoalDoubleFootPlacement.dim = 16`，目标是**观测的一部分**：在 `_init_from_mj` 里用
`self.obs_ind = np.arange(current_obs_size, current_obs_size + 16)` 把 16 维目标拼在本体感知观测之后，
策略输入 = `[本体感知 obs, 16 维 goal]`。

16 维结构（`get_obs_and_update_state`，默认 `root_frame=False`，即**支撑脚坐标系**）：

$$
\mathrm{goal} = \underbrace{[\Delta p_{\text{left}}]}_{3}
                 \oplus \underbrace{[\Delta q_{\text{left}}]}_{4\,(w,x,y,z)}
                 \oplus \underbrace{[\Delta p_{\text{right}}]}_{3}
                 \oplus \underbrace{[\Delta q_{\text{right}}]}_{4}
                 \oplus \underbrace{[\cos 2\pi\phi,\ \sin 2\pi\phi]}_{2}
$$

其中 $\Delta p_{\text{foot}} = R(\text{stance})^{-1}\big(p^{\text{target}}_{\text{foot}} - p_{\text{stance}}\big)$
是目标脚位置相对**支撑脚**、旋转到支撑脚坐标系后的偏移；$\Delta q$ 同理是相对朝向四元数（做 $w\ge 0$ 半球修正保证连续）。

### 1.2 当前代码的喂入方式

`foothold/sampler.py::observation()` 返回完全相同的 16 维（`torch.cat` 顺序也一致）：

```python
return torch.cat((rel_pos[:, 0], rel_quat[:, 0], rel_pos[:, 1], rel_quat[:, 1], phase), dim=-1)
# = [左目标位置(3), 左目标四元数(4), 右目标位置(3), 右目标四元数(4), cos/sin(2πφ)(2)]
```

喂入策略在 `foothold/ppo.py::PPO.act`：

```python
actor_input = torch.cat((obs, goal), dim=-1)   # 30 + 16 = 46 维
critic_input = torch.cat((critic_obs, goal), dim=-1)
```

`foothold/train.py` 建网络时输入维度已含 goal：
`ActorCritic(env.num_obs + num_goal, env.num_critic_obs + num_goal, ...)`。

### 1.3 结论

**agent 确实能感知落足目标，且格式、坐标系、维度（16）都与原论文一致**，这一环没有功能缺陷。
唯一的实现差异是当前把 `obs / goal / critic_obs` 三者分别做 `RunningMeanStd` 归一化（`ppo.py::Normalizer`），
而原论文是统一拼接后再归一化——这不会导致"感知不到目标"。

> 因此"只学会站桩"的根因不在喂入，而在下面检查项 2（目标不可达）+ 检查项 4（朝向项过重）导致
> 跟踪奖励长期≈0，agent 退化为"最大化生存/静态奖励"。

---

## 2. 检查项 2：采样范围是否合理（步长适配）★核心问题

### 2.1 数值对比

| 参数 | 原论文 T1（23 DoF 人形） | 当前 SF（8 DoF） |
|---|---|---|
| 步长范围 | `xy_distance_range: [0.2, 0.5]` | `step_distance: [0.05, 0.20]`（已按 SF 腿长适配） |
| 步角范围 | `angle_range_deg: [-30, 30]` | `step_angle_deg: [-30.0, 30.0]` |
| 目标 yaw | `yaw_range_deg: [-30, 30]` | `target_yaw_deg: [-30.0, 30.0]` |
| 摆动高度 | `gait_height: 0.05` | `swing_height: 0.05` |
| 基准高度 | `goal_height: 0.65` | `goal_height: 0.65` |
| 名义脚距 | `still_feet_distance: 0.2` | `hold_feet_distance: 0.20` |

### 2.2 腿长对比（关键）

从 SF 的 URDF 关节 origin 实测：

- 大腿（hip→knee）：$\sqrt{0.150^2+0.25981^2}\approx 0.30$ m
- 小腿（knee→ankle）：$\sqrt{0.150^2+0.25981^2}\approx 0.30$ m
- **整条腿（hip→ankle）伸直 ≈ 0.60 m**

而原论文 T1 是完整 23 DoF 人形、更高更长。步长上限 $d_{\max}$ 受腿长 $L$ 约束（双足步长约为 $L$ 的
一半到 $2/3$）。对 SF：$d_{\max}\approx 0.3$ m 左右；而当前范围上界 $0.5$ m **接近腿长，不可达**。

### 2.3 为什么不可达会"杀死"跟踪奖励

跟踪奖励（`foothold/rewards.py::_reward_swing_xy`）：

$$
r_{\mathrm{swing\_xy}} = w_{\mathrm{swing\_xy}}\cdot e^{-\xi_{\mathrm{xy}}\,\|\Delta p_{xy}\|^2}\cdot dt,
\qquad w=5.0,\ \xi=100
$$

若目标 0.5 m 而机器人最多迈 0.3 m，误差恒 ≥ 0.2 m：

$$
e^{-100\times 0.2^2}=e^{-4}\approx 0.018 \;\Rightarrow\; r\approx 5.0\times 0.018\times 0.02\approx 0.0018
$$

而"生存 + 静态姿态"奖励（见 4.3）恒为 $\approx(0.25+4.0)\times dt=0.085$，**比跟踪奖励高一个数量级**。
于是 agent 学到的最优策略就是"不迈腿、站桩拿静态奖励"。

### 2.4 已实施的修改

已把步长范围缩到与腿长匹配（腿长 ≈0.60 m，上界取约腿长的 1/3）：

```python
"step_distance": [0.05, 0.20],          # 上界从 0.50 → 0.20
"step_distance_curriculum": {"start": [0.05, 0.10], "end": [0.05, 0.20], "ramp_iterations": 1000},
```

---

## 3. 检查项 3：采样生成机制是否合理（格式/内容/裁剪）

### 3.1 目标位置生成

当前 `foothold/sampler.py::_sample_next`：

```python
d     = uniform(step_distance)                    # d ∈ [0.2, 0.5]
alpha = uniform(step_angle_deg, degrees=True)     # α ∈ [-30°, 30°]
angle = movement_yaw + alpha                      # 全局方向 + 抖动
target = stance_pos + [d·cos(angle), d·sin(angle), 0]
```

原论文 `_no_tracking_candidate` 完全一致：

$$
p^{\text{target}} = p_{\text{stance}} + d\begin{bmatrix}\cos(\theta_{\text{mov}}+\alpha)\\ \sin(\theta_{\text{mov}}+\alpha)\\0\end{bmatrix}
$$

其中 $\theta_{\text{mov}}$ 是 episode 开始时从 `direction_range_deg:[-180,180]` 采的全局运动方向（当前
`movement_direction_deg:[-180,180]` 一致）。

### 3.2 横向防交叉裁剪（两者一致）

把目标变换到支撑脚 yaw 坐标系，做**单侧半平面**裁剪（当前 `minimum_lateral_separation=0.10`，原论文
`feet_distance=0.1`）：

$$
y_{\text{local}} \leftarrow \begin{cases}
\max(y_{\text{local}},\; 0.10), & \text{左脚摆动（右支撑）}\\
\min(y_{\text{local}}, -0.10), & \text{右脚摆动（左支撑）}
\end{cases}
$$

即保证摆动脚目标始终在支撑脚**外侧** ≥ 0.10 m，避免腿部交叉。

### 3.3 原论文的拒绝采样（已补上）

原论文 `_check_valid` 对候选目标做**拒绝采样**（JAX 批 2048 个候选取第一个合法者）：

```python
valid_stance = ‖target − stance_foot‖ > pillar_min_center_dist   # 平地 ≈ 0.05 m
valid_swing  = ‖target − swing_foot‖  > pillar_min_center_dist   # ← 已补上
```

它额外要求目标离**摆动脚当前位置**也 ≥ 0.05 m，防止采出"原地踏步/退化为当前脚位置"的退化目标。
当前已补上：`foothold/config.py` 新增 `min_swing_distance: 0.05`、`max_rejection_attempts: 10`；
`foothold/sampler.py` 拆出 `_sample_candidate()`，并在 `_sample_next()` 里对"距摆动脚当前距离 < 0.05 m"
的候选做逐环境重采样（hold 态按论文走 `_hold_still_proc`，跳过 `_check_valid`、不参与拒绝采样）。

### 3.4 目标朝向

两者一致：`yaw = feet_direction + uniform(target_yaw)`，再相对支撑脚 yaw 裁剪到 $\pm 90°$：

```python
yaw = stance_yaw + clamp(wrap_to_pi(yaw - stance_yaw), -π/2, π/2)
```

### 3.5 hold（静止）机制

两者一致：以 `hold_probability`（当前 0.10，原论文 `still_proportion=0.05`）在重采样步态参数时进入
hold 态，目标改为名义脚距（`hold_feet_distance=0.2`）、朝向跟随支撑脚、相位归零。

> 小结：格式、内容、横向裁剪、yaw 裁剪、hold 机制、"距摆动脚"拒绝采样均与原论文一致。

---

## 4. 检查项 4：足端朝向惩罚是否应减小 ★核心问题

### 4.1 自由度差异

| 关节 | 原 T1（23 DoF） | 当前 SF（8 DoF） |
|---|---|---|
| hip yaw / roll | ✅ | ❌ |
| ankle roll / yaw | ✅ | ❌ |
| 腿 | hip pitch + knee pitch + ankle pitch | abad + hip pitch + knee pitch + ankle pitch |

SF **没有踝关节 roll/yaw、没有髋关节 roll/yaw**，因此足端的 roll 和 yaw 无法独立调节，只能靠
整条腿的构型（abad 外展 + hip/knee/ankle pitch）"大致"逼近。

### 4.2 当前权重 vs 原论文（完全照抄，未适配）

| 项 | 当前 `config.py` | 原论文 `conf_t1.yaml` |
|---|---|---|
| swing yaw 跟踪 | `swing_yaw: 0.4`（原 4.0 的 10%） | `swing_orn_w: 4.0` |
| stance yaw 跟踪 | `stance_yaw: 0.4`（原 4.0 的 10%） | `stance_orn_w: 4.0` |
| foot roll 惩罚 | `feet_roll: -0.4`（原 -4.0 的 10%） | `feet_roll_coeff: -4.0` |
| foot yaw diff/mean | 无 | `feet_yaw_diff_coeff: 0.0`（原论文已关） |

原论文的 4.0/-4.0 是为 23 DoF 人形调的；SF 没有对应自由度，这些项变成了"永远难以满分"的噪声项，
并且 `feet_roll=-4.0` 会**惩罚任何侧倾的腿构型**，直接抑制迈腿探索。

### 4.3 静态奖励的来源（与检查项 4 叠加）

`foothold/rewards.py::_reward_nominal_joint_pos`：SF 没有躯干关节，行走时该项恒为 $e^0=1$，乘以
scale 4.0 后**恒贡献 $4.0\,dt$** 的静态奖励；再加生存 $0.25\,dt$。

> 注：原论文 T1 的 `tracking_nominal_joint_pos_names` 是头部/手臂/腰部 7 个"躯干"关节（行走时跟踪它们、
> 不跟踪腿），因此原论文这项**也是近似常数** $4.0\,dt$（躯干本来就停在名义位）。所以"静态奖励"不是
> SF 独有、也不是根因——**根因仍是检查项 2（目标不可达使跟踪奖励≈0）**：只有当跟踪奖励上限
> （≈$32\,dt$）被激活时，静态的 $4.25\,dt$ 才显得微不足道。

### 4.4 已实施的修改

已把三项足端朝向惩罚统一降到原来的 10%（4.0 → 0.4，-4.0 → -0.4）：

```python
"swing_yaw": 0.4,   # 4.0 → 0.4（SF 无踝/髋 yaw）
"stance_yaw": 0.4,   # 4.0 → 0.4
"feet_roll": -0.4,   # -4.0 → -0.4（SF 无踝 roll，只能整体构型逼近）
```

`nominal_joint_pos` 的行走态常数奖励暂未改动（与原论文语义一致，见 4.3），留待后续实验决定。

---

## 5. 检查项 5：摆动/支撑腿奖励逻辑是否正确

### 5.1 跟踪项（`foothold/rewards.py`）

摆动脚三通道：

$$
r_{\mathrm{swing}} = w_{xy}e^{-\xi_{xy}\|\Delta p_{xy}\|^2}
                   + w_z e^{-\xi_z \Delta p_z^2}
                   + w_{\psi}e^{-\xi_{\psi}\Delta\psi^2}
$$

- `swing_xy`：$\Delta p_{xy}=p_{\text{swing}}-p^{\text{target}}_{\text{swing}}$（xy 平面，**世界系**）。
- `swing_z`：$\Delta p_z = p_{\text{swing},z}-\big(p^{\text{target}}_{\text{swing},z} + \mathbf{1}[\text{half}\le0.5]\,h_{\text{swing}}\big)$，
  即前 1/4 相位要求脚抬到目标高度 + 摆动高度。
- `swing_yaw`：$\Delta\psi=\mathrm{wrap}(yaw_{\text{swing}}-yaw^{\text{target}})$。

支撑脚项在**触地瞬间锁存**（`_update_stance_latch` 把上一步摆动脚跟踪质量锁进 `stance_reward`），
整段支撑相复用，与原论文 `stance_*_reward = last_swing_*_reward`（`goal_resampled` 时锁存）一致：

$$
r_{\mathrm{stance}} = \mathrm{latch}\big(r_{\mathrm{swing}}\big)\big|_{\text{touchdown}}
$$

### 5.2 迈腿激励项

`feet_swing`（相位窗口内摆动脚离地）：

$$
r_{\mathrm{feet\_swing}} = \mathbf{1}\big[|\phi-\phi_c|<w_{\text{win}}\big]
  \cdot \mathbf{1}[\text{摆动脚无接触}] \cdot \mathbf{1}[\neg\text{hold}] \cdot dt
$$

其中 $\phi_c=0.25$（左脚）/ $0.75$（右脚），$w_{\text{win}}=0.10$；与原论文
$|\mathrm{gp}-0.25|<0.5\times0.2$ 一致。

`gait_height`（相位中点把脚抬到目标高度 + $h_{\text{swing}}$）：

$$
r_{\mathrm{gait\_height}} = w\,e^{-\xi\, \mathrm{deficit}^2\, \mathrm{gait}(\phi)\,\mathbf{1}[\neg\text{hold}]}\,dt
$$

其中 $\mathrm{gait}(\phi)$ 在相位中点取峰值的三角窗，与原论文 `desired_gait_height_ada_sharp` 一致。

### 5.3 结论

**摆动/支撑腿的奖励公式与锁存逻辑与原论文一致，逻辑正确。** 它确实能在"目标可达"时激励机器人在该
迈腿时迈腿；当前"不迈腿"是检查项 2（目标不可达）+ 检查项 4（朝向惩罚过重）导致的激励失效，而非
本项公式错误。

---

## 6. 总结与修改优先级

| 优先级 | 项 | 问题 | 建议 |
|---|---|---|---|
| ✅ 已改 | 检查项 2 | 步长范围照抄 T1，SF 腿短不可达 | `step_distance` 上界 0.50 → 0.20，范围 [0.05, 0.20] |
| ✅ 已改 | 检查项 4 | 朝向/roll 权重未按自由度适配 | `swing_yaw/stance_yaw` 4.0→0.4，`feet_roll` -4.0→-0.4 |
| ✅ 已改 | 检查项 3 | 缺"距摆动脚"拒绝采样 | 已补 `min_swing_distance=0.05` + 逐环境重采样 |
| ⏳ 待定 | 检查项 4 | 行走态 `nominal_joint_pos` 恒为常数奖励 | 考虑去掉或降权，避免掩盖跟踪奖励 |
| — | 检查项 1 | 目标喂入 | ✅ 已一致，无需改 |
| — | 检查项 5 | 摆动/支撑奖励逻辑 | ✅ 已一致，无需改 |

**核心结论**：机器人"只学站桩"的直接原因是 **步长上界（0.5 m）对腿长仅 0.6 m 的 SF 不可达**，
使跟踪奖励恒≈0；叠加 **朝向/roll 惩罚按 23 DoF 人形照搬过重**，进一步抑制迈腿探索；最终 agent
退化为最大化"生存 + 恒定姿态奖励"。上述修改（步长 [0.05, 0.20]、朝向惩罚 10%、拒绝采样）已于
2026-08-21 实施，并据此以 8192 并行环境 headless 重新启动训练。
