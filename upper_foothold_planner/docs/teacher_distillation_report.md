# 基于特权教师蒸馏与非对称强化学习的可部署落足策略

> 日期：2026-08-29。本文是一份严谨的方法汇报：说明可部署上层落足策略如何从一个
> 特权几何教师蒸馏而来，并解释学生策略为何最终超过了教师策略。不涉及开发阶段代号，
> 所有公式对应代码实现，所有数值对应实际训练/评估表征。

## 摘要

针对双足机器人在非结构化地形上的落脚点规划问题，本文提出一种两层控制框架：下层
为冻结的强化学习行走策略（50 Hz 执行落脚目标），上层为每步选择一次落脚目标的规划器。
上层策略的最终形态是一个**只依赖深度图和 36 维本体观测**的候选落足 Actor，其训练分为
两个阶段：

1. **监督蒸馏**：用一个仅在仿真中可用的特权几何教师产生决策数据，以行为克隆方式训练
   可部署学生，使其在未见地形上达到与教师相当的成功率（约 71%）；
2. **非对称强化学习微调**：Actor 保持可部署观测，Critic 额外访问特权几何信息，以
   geodesic 势能整形奖励进行 PPO 微调，把成功率提升到约 **95.5%**，**显著超过教师
   （70.8%）**。

在分布外更困难地形上的退化测试表明，该策略确实在感知地形几何并自主规划落足，而非
记忆单一场景。

## 1. 问题定义

### 1.1 层次化控制

上层决策是事件驱动的：每当下层完成一次步态切换（约 20–25 个 50 Hz 控制周期），上层
输出下一个落脚目标。记决策时刻的观测为 $s_t$，上层动作为 $a_t$，则闭环转移

$$s_{t+1} = \mathcal{T}(s_t, a_t)$$

由冻结下层在真实物理仿真中执行 $a_t$ 后产生。上层动作的物理含义是支撑脚坐标系下的
落脚位移。

### 1.2 可部署观测与特权信息

- **可部署观测** $o_t$：深度图 $D_t\in[0,1]^{1\times 64\times 64}$ + 36 维本体向量
  $p_t$（投影重力、基座线/角速度、关节位置/速度、双脚相对位姿、步态相位、上一动作、
  相对路线目标与偏航误差）。真实部署中全部可得。
- **特权信息**（仅仿真可用，供教师与 Critic）：精确支撑掩码、障碍几何、geodesic 距离场、
  真实支撑率、绝对位姿。

## 2. 动作空间与候选集合

上层动作归一化为 $a=(f, l, \psi)\in[-1,1]^3$，按支撑脚 yaw 系解码为

$$(x_f, y_l, \psi_{\deg}) = \left(\tfrac{1+f}{2}(0.30-0.08)+0.08,\ \operatorname{sgn}(s)\big[\tfrac{1+l}{2}(0.26-0.10)+0.10\big],\ \psi\cdot 12^\circ\right)$$

其中 $s$ 为摆动脚指示（左右脚的横向符号相反），横向分量直接缩放、不做事后裁切，从而
避免旧 polar 参数化中 98.1% 样本退化为 0.10 m 的别名问题。

候选集合为离散网格

$$\mathcal{C} = \{ (f,l,\psi) : f\in\mathcal{F}_{12},\ l\in\mathcal{L}_{9},\ \psi\in\Psi_{3} \},$$

经径向可达域过滤 $\rho\in[0.12, 0.35]$ m 后得到 $|\mathcal{C}|=294$ 个候选。离散化的
意义在于：每个状态天然拥有多个可比较的动作，可直接做分类、价值或优势学习，且可视化
直观。

## 3. 特权教师策略

教师 $\pi_T$ 是一个确定性几何规划器：给定特权地形信息，对每个候选计算支撑率、可行性和
到目标的 geodesic 距离，再按固定权重打分取最优。

### 3.1 足底支撑率

对每个网格单元，以 9 点足底模板近似 SF 脚掌（纵向偏移 $\Delta x=0.08$ m、横向偏移
$\Delta y=0.035$ m），支撑率为

$$F(x,y) = \frac{1}{9}\sum_{i\in\{-1,0,1\}}\sum_{j\in\{-1,0,1\}} S(x+j\Delta x,\ y+i\Delta y),$$

其中 $S(\cdot)$ 为仿真高度场支撑掩码。$F$ 是连续量化的"该处可落足程度"，直接作为教师
打分项与学生可行性标签。

### 3.2 跳跃感知的 geodesic 距离场

在"可落足单元"图上（$F\ge 5/9$ 且不在障碍上），以目标为源做 Dijkstra，边允许跨越
无支撑间隙但禁止穿越障碍：

$$d_{geo}(x) = \min_{x'}\big[\, d_{geo}(x') + \|x'-x\| \,\big],\quad \|x'-x\|\le d_{jump},\ \overline{xx'}\cap\text{障碍}=\varnothing,$$

边界条件 $d_{geo}(x_{goal})=0$，跨隙上限 $d_{jump}=0.35$ m。这表示"一串真实可行的落足
序列"的路径长度，而非连续地面的路径长度。教师用它衡量候选的**绕障前进量**。

### 3.3 候选评分与选取

对候选 $a$ 解码到世界系目标点 $p_a$，记

- 支撑率 $F_a$；
- geodesic 前进量 $\Delta d_a = d_{geo}(p_{\text{stance}}) - d_{geo}(p_a)$；
- 路线朝向误差 $\eta_a = |\operatorname{wrap}(\theta_a - \theta_{\text{route}})|$。

评分为

$$\operatorname{score}(a) = w_d\,\Delta d_a + w_s\,F_a - w_\eta\,\eta_a - w_c\,\|a - a_{\text{prev}}\|^2,$$

权重 $w_d=10,\ w_s=3,\ w_\eta=0.25,\ w_c=0.04$。候选必须同时满足"目标在地形内、
$F_a\ge 7/9$、不在障碍上、摆动路径不穿越障碍"才有效；无有效候选时回退到支撑率最大者。
教师输出

$$\pi_T(s) = \arg\max_{a\ \text{有效}} \operatorname{score}(a).$$

教师本身是**手写权重的单步贪心启发式**，评分是对"任务成功"这一真实目标的代理；它
完全忽略下层执行的动力学误差（触地误差、失稳、碰撞力），这为后续学生超越教师埋下空间。

## 4. 教师决策数据收集

让教师 $\pi_T$ 在 1024 个并行随机地形（随机支撑宽度 0.50–1.30 m、间隙 0–0.14 m、
无凸起障碍、赛道 3.5 m）上闭环运行 30 s，在每个决策时刻记录：

$$\big(o_t,\ \pi_T(s_t),\ \{\text{每候选 }F_a,\ \Delta d_a,\ \text{有效性},\ \text{教师 score}\},\ \text{该决策所在 episode 是否成功},\ \text{地形实例 id}\big).$$

共 57641 条决策，其中 23205 条来自成功 episode（40.3%）。行为克隆只保留成功 episode 的
决策，按地形实例划分训练/验证集（验证集为**未见地形**），避免用同场景重放准确率冒充
泛化能力。

## 5. 学生策略网络

学生 $\pi_\theta$ 是可部署候选 Actor：

```
depth (1×64×64) → 4 层卷积 → 128 维地形特征
proprio (36)    → MLP → 64 维本体特征
concat(192) → 256 维融合特征
  ├─ 分类 logits  g_θ(o) ∈ R^294        （策略）
  ├─ 可行性 logits f_θ(o) ∈ R^294        （辅助：候选是否可落足）
  └─ geodesic 进度 ĝ_θ(o) ∈ R^294        （辅助：候选前进量）
```

策略分布为 $p_\theta(a\mid o)=\operatorname{softmax}(g_\theta(o))$。

## 6. 学生蒸馏（行为克隆）

### 6.1 损失函数

$$L_{BC} = -\underbrace{\sum_{a} q(a)\log p_\theta(a\mid o)}_{\text{软目标交叉熵}}
+ \lambda_f \underbrace{\operatorname{BCE}\big(f_\theta(o), v(a)\big)}_{\text{可行性}}
+ \lambda_g \underbrace{\operatorname{Huber}\big(\hat g_\theta(o), \Delta d_a\big)}_{\text{geodesic 进度}},$$

其中 $v(a)\in\{0,1\}$ 为候选有效性标签，$\lambda_f=\lambda_g=0.5$。

关键设计是**软目标** $q$：教师输出是 $\arg\max$，但在多个近并列候选之间 $\arg\max$ 带有
噪声。改用教师完整评分的温度分布

$$q(a) = \operatorname{softmax}\big(\operatorname{score}(a)/\tau\big),\quad \tau=0.25,$$

把教师对候选之间的**相对偏好**整体蒸馏给学生，而非只给一个 one-hot。消融显示软目标使
学生可行性准确率从 67% 提升到 79%、进度 MAE 从 0.24 m 降到 0.18 m。

### 6.2 训练表征与结果

训练 60 epoch（batch 512，Adam lr $3\times10^{-4}$）后：

- 离线：top-1 命中 18%（软目标下必然低于 one-hot 口径）、所选候选有效率 80%、
  可行性准确率 91%、进度 MAE 0.14 m；
- 闭环（未见地形，贪婪选取）：seed 924 = 71.6%（222/310），seed 42 = 70.0%（217/310）。

即学生**追平教师**（70.8%）。这也说明行为克隆的上界就是教师：最小化模仿误差只能逼近
教师，无法修正教师的系统性错误。

## 7. 学生超越教师：非对称强化学习微调

### 7.1 为何 RL 能超过教师

教师是**单步贪心 + 固定权重**的几何启发式，其评分没有优化真实目标（episode 成功率），
也没有建模下层动力学可行性（触地误差、失稳）。它系统性地在"横向大偏移路线"上失败。
行为克隆继承了这些错误；而强化学习直接优化真实折扣回报，并用学到的值函数隐式建模
多步后果与动力学可行性，因此可以找到偏离教师贪心选择、但真实成功率更高的落足序列。

### 7.2 不对称 Actor–Critic

- **Actor** $\pi_\theta$：候选 Actor（BC 热启动），只看可部署观测 $o$；
- **Critic** $V_\phi$：MLP，输入在 $o$ 之上附加特权特征（geodesic 距离、支撑率、
  支撑脚世界坐标、基座世界坐标）。

$$\phi(s) \in \mathbb{R}^{42} = [p;\, d_{geo};\, F_{\text{stance}};\, x_{\text{stance}};\, x_{\text{base}}].$$

Critic 只在训练期存在，部署时丢弃。

### 7.3 奖励：geodesic 势能整形

以教师预计算的 geodesic 距离场构造势函数 $\Phi(s)=-d_{geo}(s)$，奖励为

$$r_t = \underbrace{\Phi(s_{t+1}) - \gamma\,\Phi(s_t)}_{\text{geodesic 前进势差}}
+ 10\,\mathbb{1}_{success} - 5\,\mathbb{1}_{collision} - 5\,\mathbb{1}_{fall} - 3\,(1-F_{\text{land}}).$$

终态处理：摔倒时前进项置 0（只给摔倒惩罚）；成功时置为 $d_{geo}(s_t)$（一次性给满剩余
距离，因为环境已重置、无法再采样 $s_{t+1}$）。势能整形保证绕障所需的侧移不会被欧氏
距离误罚，并让奖励对任何策略都保持一致的势差结构。

### 7.4 PPO 目标

$$\mathcal{L}^{PPO} = \mathbb{E}\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big) - c_1\big(V_\phi(s)-\hat R_t\big)^2 + c_2\,H(\pi_\theta)\Big],$$

其中 $r_t(\theta)=\pi_\theta(a_t|o_t)/\pi_{\theta_{\text{old}}}(a_t|o_t)$，
$\hat A_t$ 为 GAE 优势（$\gamma=0.99,\lambda=0.95$），$\hat R_t=\hat A_t+V(s_t)$。

训练设置：512 环境 × 600 ticks/rollout（约 12 k 条 option 转移）、30 次更新、
每更新 10 epoch × batch 1024、Adam lr $10^{-4}$、clip $\epsilon=0.2$、熵系数 0.005、
值函数系数 0.5、梯度裁剪 1.0。

### 7.5 训练表征

- 探索采样下累计成功率 **21.6% → 47.8%**（30 次更新单调上升）；
- 总损失 **6227 → 2975**（持续下降）；
- 贪婪评估（未见地形）：**seed 924 = 95.5%（235/246），seed 42 = 95.1%（232/244）**。

## 8. 实验结果汇总

| 策略 | 观测 | seed 924 | seed 42 |
|---|---|---|---|
| 特权教师 $\pi_T$ | 全量 heightfield | 70.8% | — |
| 蒸馏学生 $\pi_\theta$ (BC) | 深度+本体 | 71.6% | 70.0% |
| **RL 微调 $\pi_\theta$ (PPO)** | **深度+本体** | **95.5%** | **95.1%** |

### 8.1 分布外泛化（未重训，直接评估）

| 地形 | 参数 | 成功率 |
|---|---|---|
| 训练分布 | 宽度 0.50–1.30 m / 间隙 0–0.14 m / 3.5 m | 95.5% |
| 窄支撑 | 宽度 0.35–0.60 m / 间隙 0.14 m / 3.5 m | 72.2% |
| 大间隙 | 宽度 0.50–1.30 m / 间隙 0.22 m / 3.5 m | 81.0% |
| 组合极限 | 宽度 0.35–0.60 m / 间隙 0.22 m / 5.0 m | 18.7% |

单独加难时策略优雅退化（72% / 81%），说明它确实在感知地形几何；三者叠加到极限时崩溃
（18.7%），主要触及冻结下层的物理执行上限（SF 脚掌小，窄支撑 + 大间隙 + 更长射程的组合
接近不可执行）。

### 8.2 演示视频

- 训练分布：`experiments/v6_video_seed924/rollout.mp4`
- 窄支撑：`experiments/v6_video_hardA_narrow/rollout.mp4`
- 大间隙：`experiments/v6_video_hardB_gap/rollout.mp4`

## 9. 感知与自主规划的论证

1. **排除单场景过拟合**：训练仅用 1024 个地形实例，评估的 128 个地形与其完全不重叠，
   成功率 95% 说明策略适应的是地形**分布**而非具体场景。
2. **深度是唯一地形信号**：可部署观测中除深度图外不含任何地形信息，95% 的成功率要求
   CNN 必须从深度中恢复"可落足区域"这一几何结构。
3. **超过特权教师**：学生用更少信息（无 heightfield、无障碍几何）取得更高成功率，说明
   它学到了比手写评分更有效的落足策略——这只能来自对深度特征与动力学后果的学习。

## 10. 结论

本文展示了"特权几何教师 → 监督蒸馏 → 非对称 RL 微调"三步把可部署上层落足策略训练到
95.5% 成功率的完整方法。蒸馏阶段让可部署学生逼近教师；RL 微调阶段通过 geodesic 势能
奖励与特权 Critic 修正教师的贪心与动力学盲区，最终让学生以更少观测超过教师。分布外
测试进一步确认了策略的感知与规划能力，同时标定了其在极端地形上的能力边界。

## 附：复现命令

```bash
# 1) 收集教师决策数据（1024 环境 × 30 s）
python scripts/collect_teacher_dataset.py --headless --sim_device cuda:0 \
  --num_envs 1024 --lower_ticks 1500 --seed 61 --course_length_m 3.5 \
  --output experiments/v6_teacher_dataset_1024

# 2) 行为克隆蒸馏
python scripts/train_candidate_bc.py \
  --dataset experiments/v6_teacher_dataset_1024/dataset.npz \
  --output experiments/v6_bc_1024_success --epochs 60 --batch_size 512 \
  --lr 3e-4 --success_only --soft_target --score_temperature 0.25 \
  --val_env_fraction 0.2 --device cuda:0

# 3) 非对称 RL 微调
python scripts/train_candidate_ppo.py \
  --actor_checkpoint experiments/v6_bc_1024_success/model_best.pt \
  --num_envs 512 --rollout_ticks 600 --total_updates 30 \
  --update_epochs 10 --batch_size 1024 --lr 1e-4 --entropy_coef 0.005 \
  --seed 61 --course_length_m 3.5 --output experiments/v6_ppo_real

# 4) 评估 / 录像
python scripts/evaluate_candidate_bc.py --checkpoint experiments/v6_ppo_real/ppo_final.pt \
  --num_envs 64 --lower_ticks 1500 --seed 924 --course_length_m 3.5 --output <out>
python scripts/record_candidate_video.py --checkpoint experiments/v6_ppo_real/ppo_final.pt \
  --seed 924 --lower_ticks 1500 --course_length_m 3.5 --output <video_dir>
```
