# 复现规格与来源审计

## 三类来源

1. 论文 PDF：最终方法和附录报告值，优先级最高。
2. 论文随附 `loco-mujoco` 源码：补足实现细节，但不运行、不修改，也不作为运行时依赖。
3. `tron1_RL`：PF/SF 机器人资产、Isaac Gym 建模、PD 控制和已验证训练链路的事实来源；不修改、不作为运行时依赖。

## 观测、动作与坐标系

Actor 输入按论文源码顺序组织：projected gravity、关节位置、base angular velocity、关节速度、上一动作、目标和 gait phase。Actor 不接收 base linear velocity、接触标志或接触力。Critic 顺序为 projected gravity、关节位置、base linear/angular velocity、关节速度、上一动作、目标；全部使用无噪声版本。`FreeJointVel` 是 MuJoCo 原始世界系 `qvel`，因此 Actor 的 base angular velocity 和 Critic 的六维 base velocity 均使用 Isaac root-state 世界系速度；只有奖励源码显式变换后的速度惩罚使用 body 系。

论文代码的 `GoalDoubleFootPlacement.dim == 16`，实际顺序为：

```text
left_target_position(3), left_target_yaw_quaternion(4),
right_target_position(3), right_target_yaw_quaternion(4),
cos(2*pi*phase), sin(2*pi*phase)
```

目标位置和朝向在当前 stance foot 的完整姿态坐标系中表示；采样器的平面防交叉裁切单独在 stance yaw frame 中完成。当前 stance foot 的目标保持为相位切换瞬间记录的位置/朝向；swing target 在整段 swing phase 内保持常量。静止模式下 phase 两维均为零。论文源码给 Actor 的目标 quaternion 使用 scalar-first `[w,x,y,z]`，Isaac Gym 物理状态使用 scalar-last `[x,y,z,w]`；本项目在目标观测边界显式转换，使 Actor 顺序与论文源码一致。

动作是每个关节相对 nominal pose 的 PD position-target offset。与论文源码一致，策略输出先固定裁切到 `[-1,1]`、乘 `action_scale=1`，nominal+offset 再按每关节 URDF 位置范围裁切，最终 PD 力矩按每关节执行器范围裁切。不能使用 TRON locomotion 中根据实时 `q/qdot` 反推 position-offset 范围的状态依赖预裁切；它会改变论文的动作语义。

| 机器人 | DoF/action | Actor proprio+last action | goal+phase | Actor 总维度 | Critic 总维度 |
|---|---:|---:|---:|---:|---:|
| PF | 6 | 24 | 16 | 40 | 43 |
| SF | 8 | 30 | 16 | 46 | 49 |

## 目标采样

- 每个 episode 固定 movement direction 和 feet direction。
- 每次左右 swing/stance 切换时采样下一落足目标。
- 平面距离 `d`、方向扰动 `alpha`、目标 yaw 扰动 `beta` 和 z offset 独立均匀采样。
- 通过 stance yaw frame 的 lateral clipping 防止交叉腿。
- gait phase 为 `[0,0.5)` 左脚摆动、`[0.5,1)` 右脚摆动，以固定 cadence 前进。
- 以 `p_hold` 进入站立模式，目标为名义脚间距且 phase 输入为 `(0,0)`。
- 第一阶段只训练 flat `z=0`；3D terrain/pillar 支撑属于第二阶段，不能混入平地成功判据。

## 奖励

论文主奖励：swing foot 的 XY、Z、yaw 分离指数跟踪奖励；在 touchdown 后将最终 swing 精度锁存为整个 stance phase 的奖励；指定相位窗口内的离地奖励；摆动膝盖高度奖励。

附录 Table V/VI 报告通用 3D task 使用 `w_xy=w_z=w_yaw=5`、`xi_xy=xi_yaw=100`、`xi_z=200`，但平地实验明确设置 `w_z=0`、`w_knee=0`；本项目 flat 配置因此使用 `w_xy=w_yaw=5`，同时保留 z/knee term 的实现但令权重为零。随附开发 YAML 中 yaw 权重 4 不覆盖论文最终表。`w_feet=6`、窗口半宽 `0.1`；base height `-10`、action rate `-3`、contact foot slip `-4`。源码额外 gait-height 项单独保持其 YAML sharpness 100。所有 reward term 必须分别保留 raw、weighted、per-step 和 episode-sum 指标。

接触/接触力只用于 reward、Critic 可选特权量和 eval 诊断，绝不进入 Actor。

## Episode 终止

论文源码对 Booster 使用 `HeightBasedTerminalStateHandler`，T1 的目标高度约 `0.65 m`、健康范围为 `[0.30,0.90] m`。该范围是形态参数而不是跨机器人的绝对常数。早期 SF 对照使用的 `[0.60,0.95] m` 下限相对目标高度只留 `0.15 m`，明显比论文的约 `0.35 m` 更严格，并在视频中诱导出贴着阈值的双支撑静止行为；因此它只保留为已记录对照，不能宣称已完成最终形态映射。正式 SF 使用相同的目标相对裕量：SF 映射目标 `0.75 m` 加上 T1 的 `-0.35/+0.25 m` 得到 `[0.40,1.00] m`。URDF 中 base 碰撞盒最低点约在 root 下方 `0.167 m`，该下限允许深度动态屈膝而不要求静止站立。本项目不沿用 `tron1_RL` 的 projected-gravity 终止，否则会改变论文探索分布；eval 仍独立报告全 link 接触和姿态，不能仅凭高度判摔。

## PPO

论文附录最终值：8192 env、50 horizon、20 epochs、1 minibatch、`gamma=0.995`、`lambda=0.95`、clip `0.2`、entropy `0.01`、value coefficient `0.5`、max grad norm `1.0`、ELU MLP `[512,256,128]`、可学习 action std 初值 `0.135`。自适应学习率：初值 `1e-5`，范围 `[1e-6,1e-2]`，KL target `0.02`，margin/scale 均为 `1.5`。观测与奖励均使用 running normalization，history length 为 1。

网络初始化严格复现随附 Flax 实现：每个隐藏层使用 gain `sqrt(2)` 的正交初始化，输出层 gain `0.01`，全部 bias 为零。观测 RMS 的初始 count 为 `1e-6`、batch variance 加 `1e-6`；reward-return RMS 初始 count 为 `1e-4`。当前实现通过 `running_obs_clip=10` 对 normalized observation 使用 `[-10,10]` 裁切，这与此前“标准化后不额外裁切”的复现目标不一致，属于待单独核验的 fidelity 差异；它不是本次 train/eval 失配的原因，因为该失配发生在训练 RMS 根本没有被 eval 恢复之前。Critic clipped MSE 先乘显式 `0.5`，再乘配置的 `vf_coef=0.5`。Advantage 标准差使用 population variance。以上初始化与损失细节不能退回 PyTorch Linear 默认初始化或双倍 Critic 权重。

论文仓库中的当前 `conf_t1.yaml` 是调试/开发态（2 env、1000 total steps、10 epochs、32 minibatches、`3e-4`），与附录不一致。本项目正式配置采用附录值，并在小规模 smoke test 时只通过 CLI 覆盖规模，不能静默修改正式默认值。

运行归一化的 mean、variance 和 count 与网络权重共同定义策略函数，属于 checkpoint 的必要状态。
format version 2 checkpoint 已保存 actor observation、goal、critic observation 和 discounted-return
四组 RMS、Normalizer 超参数及训练配置；eval 恢复训练 RMS 后冻结更新，并使用 checkpoint 配置创建
环境。新建 `mean=0,var=1` 的 Normalizer 不能替代训练统计，也不能用单环境 eval 轨迹在线重估。
`Aug21_18-27-00_step005-020_yaw10pct` 的旧 checkpoint 只保存了 Actor-Critic 权重，因而其 eval
输入坐标系与训练不一致，当前 eval 会明确拒绝加载；完整机制、证据及兼容约束见
[`normalization_and_checkpoint.md`](normalization_and_checkpoint.md)。

自适应学习率的实现顺序也与论文代码一致：先以当前学习率执行梯度更新，再用该 minibatch 的
`mean((ratio - 1 + eps) - log(ratio + eps))` 近似 KL 调节下一次更新的学习率。不能在当前梯度
之前调节，也不能用解析高斯 KL 静默替换论文使用的 ratio 估计量。

### TRON1-SF 初始化审计

Booster 论文配置以屈膝姿态初始化，而 `tron1_RL` 的 SF 默认关节角均为零。项目保留了 locomotion actor 的转换工具用于研究初始化映射，但完整物理审计发现现有 SF checkpoint 并不是全向速度跟踪器：`model_3500` 主要站立，`model_10000` 只在负 x 命令下形成明显步态。因此它不能作为论文复现成功的依据，正式复现不使用该初始化。

由于原 locomotion actor 接收未做 running normalization 的输入，迁移后先在并行环境中收集论文观测的 RMS 统计，并解析地折叠第一层：`W_norm=W_raw*std`、`b_norm=b_raw+W_raw*mean`。这使校准时刻的原始输入策略与标准化输入策略代数等价，同时后续仍按论文继续更新 running statistics。该适配和校准参数必须在 checkpoint 与日志中标记，不能表述为论文原始的 from-scratch setting。

论文 goal sampler 在 reset 后的第一个半周期把 swing target 放在脚的当前姿态，stance latch 以满值初始化；第一次 phase switch 才采样首个非平凡目标。这个 warm-up 同时保留 swing、stance、gait-height 和 feet-swing 奖励，不能因迁移初始化而屏蔽。

迁移 seed 的 Critic 是随机初始化的。可通过 `actor_freeze_iterations` 进行显式 Critic warm-up：此阶段冻结 actor、action std 与观测 RMS，只更新 Critic 和 reward-return RMS；结束后恢复完整论文 PPO 与 running observation normalization。这一阶段只用于迁移初始化，不改变 from-scratch 默认值（默认 0）。

Isaac Gym 的 indexed root/DOF reset 会立即更新对应 tensor，但 rigid-body tensor 要到下一次 physics refresh 才可靠。reset 返回观测会清零当前/历史动作，立即重建 base body-frame velocity 与 projected gravity，并在一帧的 rigid-body 等待期输出双脚 identity pose 的中性目标；下一帧再以权威足端状态初始化 episode goal。机体系速度使用独立缓存，不能作为 root-state 共享 tensor 的 view。

并行环境异步 reset 时，只能从当步 `last_switch_ids` 中移除被 reset 的 env，不能清空整个列表；否则只要 8192 个环境中任意一个 reset，其他所有环境同一步的 touchdown latch 都会丢失，训练日志会长期显示虚假的 reset 初始满额 stance reward。

防止双腿交叉的横向裁切是单侧半平面投影：左摆动脚使用 `y=max(y,+d_min)`，右摆动脚使用
`y=min(y,-d_min)`。错误侧目标应投影到最近的 `+/-d_min` 边界，不能写成
`side * max(abs(y), d_min)`；后者会把错误侧大偏移镜像到另一侧，显著扩大目标。

论文采样器还原了显式的 `just_fwd_bwd` 模式：开启时运动方向只从 `{-pi, 0}` 采样，其他目标、
奖励和物理均不变。该设置属于论文 Appendix G 的楼梯训练，不是 Appendix F 的最终平地策略；在
TRON1 上只能作为有明确标签的形态诊断对照，不能代替全方向平地成功。它不施加 base/足端位置
约束，也不强制产生接触。

确定性平地可用于排错，但正式训练和最终鲁棒性评估必须恢复论文域随机化。不得通过缩短目标步长、外力辅助或强制修改 base/足端状态绕过原始任务。

论文路径只随机化 link 质量与 base COM，不额外启用 TRON 速度策略继承的独立惯量倍率。MuJoCo
全局重力扰动在 Isaac Gym 中按每个刚体自身质量施加等效外力，不能把整个机器人的合力集中施加
在 base；旧 checkpoint 则保留原来的兼容分支，避免改变既有对照评估。

随附 T1 开发配置除二值 `r_feet=6` 外还实现了 `tracking_swing_z=4` 与 `gait_height=4`：前者在摆动区间前半段跟踪 `target_z+5 cm`、后半段跟踪落足高度，后者以相位三角权重单侧惩罚足高低于 5 cm。开发 YAML 两项均写 sharpness 100，但论文 Table VI 的正式 z tracking sharpness 是 200；两者因此分开配置。Appendix F 明确规定最终平地策略令 z 跟踪权重 `omega_2=0`，flat 配置禁用 `tracking_swing_z`，后续 3-D fine-tuning 恢复论文值；gait-height 仍按开发源码的 100。

action-rate 必须使用当前裁切前的 policy sample 与上一 policy step 的原始 sample，即 `||a_t-a_{t-1}||^2`；PD 路径随后才裁到 `[-1,1]`。TRON 历史 buffer 的列 0 是 `a_{t-1}`，列 1 是 `a_{t-2}`。接触足 slip 按论文源码使用足端 XYZ 三维线速度，而不是只用 XY。关节限位项按源码统计落在各关节范围内侧 98% 之外的关节数量，不使用“越过 URDF 硬限位后的距离”。

正式 PPO 默认仍采用论文配置的 adaptive learning rate。迁移 curriculum 可显式传入 `--fixed_learning_rate --learning_rate 1e-5`，避免尚未对齐的新 Critic 令 adaptive KL schedule 在最初几轮把学习率放大一个数量级；该覆盖会完整写入 `config.json`。

若 actor 初始化文件包含为行为等价性校准并折叠过的 RMS，迁移阶段使用 `--freeze_normalizer` 保持该输入坐标系；否则短步长 curriculum 的 command RMS 会在解冻瞬间改变 actor 的有效函数。该状态会记录为 `diagnostics/normalizer_frozen`。

TRON 的 PD/action 尺度使 locomotion seed 的 `action_rate=-3` 初期代价较大。冷启动消融可显式覆盖 `action_rate_scale`、`feet_swing_scale`、`swing_z_scale`、`gait_height_scale`；最终阶段必须恢复源码正式值 `-3/6/4/4`，并在正式值下重新评估。


## 成功判据

不能以存活或阶段切换数单独宣布成功。至少同时报告：

- 连续运行时间、episode 长度、fall/reset rate；
- base 姿态、base 高度、速度和漂移；
- 每一步 touchdown 的 2D/3D/yaw 误差与阈值成功率；
- swing 是否在指定窗口离地、stance foot slip、错误脚触地；
- 动作、关节位置/速度/力矩和接触力轨迹；
- 可视化视频中自然且持续的左右交替行走。

达标目标是至少一个 PF 或 SF agent 在平地稳定连续行走且能跟踪变化落足点；随后再扩展另一机器人和 3D z tracking。
