# 奖励函数分析报告

> 生成时间：2026-08-20
> 数据来源：`foothold/dump_rewards.py`，128 环境 × 200 步、**初始随机策略（未训练）**。
> 加权规则（`rewards.py:compute()`）：`weighted = scale × raw`，其中 `raw` 已包含 `dt`；`总奖励 = Σ weighted`。
> `dt = env.decimation × env.dt = 4 × 0.005 = 0.02`。

## 0. 重要说明（测量口径）

下面的"加权后均值"是**启动初期**（未训练策略、机器人快速摔倒）的量级：

- 跟踪项（`exp(-100·err²)`）在重置瞬间目标=当前足位，因此接近其上界 `scale × dt`；
- 惩罚项因机器人很快摔倒、误差尚未积累而偏小。

因此本报告用于**相对排序与公式核对**；稳态跟踪下的实际值会不同（跟踪项下降、`base_height`/`orientation`/`action_rate` 等惩罚上升）。

---

## 1. 按加权绝对值排序（实测）

| 排名 | 奖励项 | 实测加权均值 | 符号 | 权重 scale | 代码位置 |
|---|---|---|---|---|---|
| 1 | `stance_xy` | +0.0919 | 正 | 5.0 | `rewards.py:105` |
| 2 | `nominal_joint_pos` | +0.0800 | 正 | 4.0 | `rewards.py:139` |
| 3 | `gait_height` | +0.0785 | 正 | 4.0 | `rewards.py:126` |
| 4 | `stance_yaw` | +0.0746 | 正 | 4.0 | `rewards.py:112` |
| 5 | `swing_z` | +0.0644 | 正 | 4.0 | `rewards.py:86` |
| 6 | `swing_xy` | +0.0576 | 正 | 5.0 | `rewards.py:82` |
| 7 | `swing_yaw` | +0.0501 | 正 | 4.0 | `rewards.py:94` |
| 8 | `orientation` | −0.0267 | 负 | −5.0 | `rewards.py:168` |
| 9 | `feet_slip` | −0.0186 | 负 | −3.0 | `rewards.py:158` |
| 10 | `action_rate` | −0.0172 | 负 | −3.0 | `rewards.py:151` |
| 11 | `ang_vel_xy` | −0.0126 | 负 | −0.2 | `rewards.py:177` |
| 12 | `feet_swing` | +0.0082 | 正 | 6.0 | `rewards.py:115` |
| 13 | `survival` | +0.0050 | 正 | 0.25 | `rewards.py:79` |
| 14 | `torques` | −0.0037 | 负 | −2e-4 | `rewards.py:171` |
| 15 | `feet_roll` | −0.0029 | 负 | −4.0 | `rewards.py:162` |
| 16 | `root_acc` | −0.0016 | 负 | −1e-4 | `rewards.py:186` |
| 17 | `base_height` | −0.0007 | 负 | −2.0 | `rewards.py:148` |
| 18 | `energy` | −0.0003 | 负 | −2e-3 | `rewards.py:174` |
| 19 | `dof_vel` | −0.0001 | 负 | −9e-4 | `rewards.py:180` |
| 20 | `dof_acc` | −0.0000 | 负 | −1e-7 | `rewards.py:183` |
| 21 | `dof_pos_limits` | −0.0000 | 负 | −1.0 | `rewards.py:189` |

**已禁用（scale=0，不参与总奖励）**：`stance_z`（`rewards.py:109`）、`knee_height`（`rewards.py:136`）。

---

## 2. 各项精确公式

记号：`P_swing/P_stance` 摆动/支撑足世界位置；`t_swing/t_stance` 目标位置；`ψ` 足 yaw；`t_yaw` 目标 yaw；`φ` 步态相位；`q/q̇` 关节角/角速度；`q_default` 名义关节角；`a_t` 当前策略动作；`τ` 关节力矩；`g_proj` 投影重力（本体系）；`ω` 基座角速度（本体系）；`v_foot` 足端世界速度；`roll` 足 roll 角。所有 `raw` 已乘 `dt`，最终再乘 `scale`。

### 2.1 跟踪项（正奖励）

**`stance_xy`**（`rewards.py:105`，scale 5.0，sharpness 100）
$$R_{stance,xy}=5.0\cdot S_{stance}[0]\cdot dt$$
其中 $S_{stance}[0]$ 是**触地切换瞬间锁存**的上一摆动足 xy 跟踪质量（`_store_swing_reward`，`rewards.py:46`）：
$$S_{stance}[0]=\exp\left(-100\cdot\lVert P_{swing,xy}-t_{swing,xy}\rVert^2\right)\big|_{\text{落地前一瞬}}$$

**`nominal_joint_pos`**（`rewards.py:139`，scale 4.0）
$$R_{nom}=\begin{cases}4.0\cdot\exp(-4.0\cdot\lVert q-q_{default}\rVert^2)\cdot dt, & \text{hold\_still}\\[2pt]4.0\cdot 1.0\cdot dt, & \text{行走（SF 无躯干关节，空集→exp(0)=1）}\end{cases}$$

**`gait_height`**（`rewards.py:126`，scale 4.0，sharpness 100）
$$\text{half}=\frac{\varphi\bmod 0.5}{0.5},\quad w=\begin{cases}2\,\text{half}, & \text{half}<0.5\\2(1-\text{half}), & \text{否则}\end{cases}$$
$$\text{deficit}=\max(t_{swing,z}+0.05-P_{swing,z},\ 0)$$
$$R_{gait}=4.0\cdot\exp(-100\cdot\text{deficit}^2\cdot w\cdot[\neg \text{hold\_still}])\cdot dt$$

**`stance_yaw`**（`rewards.py:112`，scale 4.0）
$$R_{stance,yaw}=4.0\cdot S_{stance}[2]\cdot dt$$
$S_{stance}[2]=\exp(-100\cdot\Delta\psi_{swing}^2)$ 锁存于落地瞬间（同上）。

**`swing_z`**（`rewards.py:86`，scale 4.0，sharpness 100）
$$\text{desired}_z=t_{swing,z}+0.05\cdot[\text{half}\le 0.5]$$
$$R_{swing,z}=4.0\cdot\exp\left(-100\,(P_{swing,z}-\text{desired}_z)^2\right)\cdot dt$$

**`swing_xy`**（`rewards.py:82`，scale 5.0，sharpness 100）
$$R_{swing,xy}=5.0\cdot\exp\left(-100\,\lVert P_{swing,xy}-t_{swing,xy}\rVert^2\right)\cdot dt$$

**`swing_yaw`**（`rewards.py:94`，scale 4.0，sharpness 100）
$$R_{swing,yaw}=4.0\cdot\exp\left(-100\,\text{wrap}(\psi_{swing}-t_{yaw,swing})^2\right)\cdot dt$$

**`feet_swing`**（`rewards.py:115`，scale 6.0）
$$\text{center}=\begin{cases}0.25,& \text{左摆动}\\0.75,& \text{右摆动}\end{cases},\quad \text{in\_window}=|\varphi-\text{center}|<0.10$$
$$R_{feetswing}=6.0\cdot\left[(\text{in\_window}\land\neg c_{swing}\land\neg\text{hold\_still})\lor(\text{hold\_still}\land c_L\land c_R)\right]\cdot dt$$
$c_f$ 为足 f 的接触标志（法向接触力 z 分量 > 1.0）。

**`survival`**（`rewards.py:79`，scale 0.25）
$$R_{surv}=0.25\cdot dt$$

### 2.2 惩罚项（负奖励）

**`orientation`**（`rewards.py:168`，scale −5.0）
$$R_{ori}=-5.0\cdot\lVert g_{proj,xy}\rVert^2\cdot dt$$

**`feet_slip`**（`rewards.py:158`，scale −3.0）
$$R_{slip}=-3.0\cdot\sum_{f\in\{L,R\}}\lVert v_{f}\rVert^2\cdot c_f\cdot dt$$

**`action_rate`**（`rewards.py:151`，scale −3.0）
$$R_{ar}=-3.0\cdot(1+[\text{hold\_still}])\cdot\lVert a_t-a_{t-1}\rVert^2\cdot dt$$

**`ang_vel_xy`**（`rewards.py:177`，scale −0.2）
$$R_{\omega xy}=-0.2\cdot\lVert \omega_{xy}\rVert^2\cdot dt$$

**`torques`**（`rewards.py:171`，scale −2e-4）
$$R_{\tau}=-2\times10^{-4}\cdot\lVert\tau\rVert^2\cdot dt$$

**`feet_roll`**（`rewards.py:162`，scale −4.0）
$$\text{roll}=\operatorname{atan2}\left(2(wx+yz),\,1-2(x^2+y^2)\right),\quad (x,y,z,w)=\text{足四元数}$$
$$R_{roll}=-4.0\cdot\sum_{f}\text{roll}_f^2\cdot dt$$

**`root_acc`**（`rewards.py:186`，scale −1e-4）
$$R_{ra}=-1\times10^{-4}\cdot\Big\lVert\frac{v_{root,t}-v_{root,t-1}}{dt}\Big\rVert^2\cdot dt$$
$v_{root}$ 为 6 维世界系基座速度（线+角）。

**`base_height`**（`rewards.py:148`，scale −2.0）
$$R_{bh}=-2.0\cdot\left(P_{base,z}-0.65\right)^2\cdot dt$$

**`energy`**（`rewards.py:174`，scale −2e-3）
$$R_{en}=-2\times10^{-3}\cdot\sum_j\max(\tau_j\,\dot q_j,\,0)\cdot dt$$

**`dof_vel`**（`rewards.py:180`，scale −9e-4）
$$R_{dv}=-9\times10^{-4}\cdot\lVert\dot q\rVert^2\cdot dt$$

**`dof_acc`**（`rewards.py:183`，scale −1e-7）
$$R_{da}=-1\times10^{-7}\cdot\lVert\dot q_{acc}\rVert^2\cdot dt,\quad \dot q_{acc}=\frac{\dot q_t-\dot q_{t-1}}{dt}$$

**`dof_pos_limits`**（`rewards.py:189`，scale −1.0）
$$\text{margin}=0.5\,(1-0.98)\,\text{range},\quad \text{outside}=\#\{q<q_{min}+\text{margin}\ \lor\ q>q_{max}-\text{margin}\}$$
$$R_{lim}=-1.0\cdot\text{outside}\cdot dt$$

---

## 3. 总奖励

$$R_{total}=\sum_k \text{scale}_k\cdot \text{raw}_k$$

由 `rewards.py:compute()`（第 29–44 行）逐项累加得到；`clip_single_reward`、`clip_reward` 当前均为 `None`（不裁剪）。
