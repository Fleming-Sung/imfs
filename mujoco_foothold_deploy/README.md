# SF_TRON1A foothold policy：MuJoCo 部署

本目录独立部署 `model_7000.pt`，运行时不依赖也不修改上层训练项目。机器人动力学模型来自
LimX 官方 `robot-description/pointfoot/SF_TRON1A/xml/robot.xml`。原文件副本保存在
`assets/SF_TRON1A/xml/robot_official.xml`；`robot_deploy.xml` 只增加左右足底 site、目标球、
地形插入点和离屏渲染尺寸，保留官方 body、joint、足底 visual/collision geom 和 actuator。

## 三个场景

- `training_demo`：不指定路线或任务，在无限平地上复现 `model_7000` 训练时的目标 sampler。
  每次半周期切换都以当前实际支撑脚为基准重新采样下一目标，用于直接观察策略在 MuJoCo 中的
  原始效果。这是优先推荐的模型演示入口。

- `flat`：桩顶高度的长平台，左右脚目标保持固定步长 0.15 m、固定横向间距 0.20 m。
  它用于独立检查 MuJoCo 模型、Actor 输入、PD 控制和目标坐标变换能否让策略在明确目标下行走。
- `plum_piles`：先在与桩顶等高的实体平台上走 8 个交替落足目标，再走向本轮固定的随机梅花桩。
  平台阶段仍由策略全程控制，是进入周期步态的 runway，不是静止站立阶段。

reset 时只做一次整体高度平移，使两个官方足底 site 恰好接触支撑面，避免穿透造成 MuJoCo
特有的初始冲量。第一个物理步开始后不锁定 base、不约束腿、不强制落足，也不把机器人传送到
目标位置。

## 运行

先进入本目录。直接观察训练模型效果：

```bash
conda run -n isaacgym --no-capture-output python run.py \
  --scenario training_demo --realtime
```

限制为短步距、窄步角，并固定宏观方向为世界系 +x：

```bash
conda run -n isaacgym --no-capture-output python run.py \
  --scenario training_demo --realtime \
  --step-distance 0.07 0.12 \
  --step-angle-deg -5 5 \
  --movement-direction-deg 0 0 \
  --lateral-separation 0.20
```

这些范围也可直接修改 `config/training_demo.json`。命令行参数优先于配置文件。

实时窗口验证梅花桩：

```bash
conda run -n isaacgym --no-capture-output python run.py \
  --scenario plum_piles --realtime
```

实时窗口验证平地：

```bash
conda run -n isaacgym --no-capture-output python run.py \
  --scenario flat --realtime
```

无窗口运行并保存完整时长视频（示例为 12 秒）：

```bash
MUJOCO_GL=egl conda run -n isaacgym --no-capture-output \
  python run.py --scenario plum_piles --headless --record-video --duration 12 --seed 7
```

`--seed` 在固定路线场景中决定布局，在 `training_demo` 中决定 sampler 随机流；同一 seed 和
相同动力学轨迹可重复得到相同结果。梅花桩一轮开始后不再移动，而 `training_demo` 目标会按训练
逻辑在切步时在线生成。输出目录名为 `outputs/<time>_<scenario>_seed<seed>/`，也可用
`--output-dir` 指定。

## 输出

- `scene.xml`：完整可复现场景；固定路线场景的平台和桩位会写入其中；
- `foothold_layout.json`：全部目标的世界坐标、左右脚分配和 `platform/pile/flat` 类型；
- `sampled_target_events.json`：`training_demo` 每次实际采样的目标位置、四元数、脚和 hold 状态；
- `training_demo_settings.json`：本轮最终生效的步距、步角、宏观方向和实际抽到的方向；
- `planning_events.json`：区域规划实验中每次切步的固定支撑中心、实时支撑脚候选、最终安全区
  选点、中心偏移和候选是否原本就在安全区内；
- `trajectory.npz`：每个 20 ms 控制周期的 base、全部 9 个 robot body、关节位置/速度/加速度、
  控制力矩、足底位置/速度/接触力、实时目标、原始及归一化观测、原始及裁切动作；
- `trajectory_schema.json`：body/joint/foot 顺序、四元数和速度分量约定；
- `metrics.json`：落足事件、分支撑面误差，以及分开的 base 过低和倾角过大时间。阈值只是诊断
  信号，不能单独证明摔倒，必须结合视频和完整轨迹判断；
- `rollout.mp4`：指定 `--record-video` 后保存，与整轮测试等长，默认 50 FPS。

`support_height` 当前对平台和所有桩统一生效。规划器每个控制周期直接读取 MuJoCo 支撑面 site
的实时世界坐标，所以后续扩展逐桩动态高度时不需要改变 Actor 接口。

## 目标如何进入策略

1. 相位按训练配置的 1 Hz 前进，左右脚每半周期交替；
2. 相位切换时，固定路线场景推进到下一支撑点，`training_demo` 从实时支撑脚重新采样；
3. 每个控制周期读取当前两个目标的世界坐标；固定地形目标来自 MuJoCo site；
4. 用当前支撑脚足底位置和 `ankle_*_Link` 姿态转换到支撑脚坐标系；
5. 按训练顺序构成 16 维 goal：左右脚位置各 3、四元数各 4、相位 2；
6. 使用 checkpoint 内的 goal RMS 归一化，与 30 维本体观测拼成 Actor 输入。

Actor 输出按训练配置裁切至 `[-1, 1]`，再经过名义关节角偏置和 PD 控制。MuJoCo 使用官方
1 ms 步长，每个 20 ms 策略动作执行 20 个物理子步。

### `training_demo` 与训练 sampler 的对应关系

`training_demo` 读取 checkpoint 中的 foothold 配置，并复现：当前实际支撑脚位置/yaw、
`0.07–0.35 m` 步距、`±30°` 步角、`±30°` 目标 yaw、摆动脚
`0.05 m` 拒绝采样、1 Hz phase、每 10 个 gait 更新一次的 10% hold。随机数实现使用 NumPy
而不是训练端 PyTorch，因此相同 seed 不要求产生相同数列，但状态转移和采样分布一致。

训练原配置的 `minimum_lateral_separation` 是 `0.10 m`。对前进方向的短步目标，这会直接把
左右脚中心距离压到约 10 cm，因此 `training_demo` 默认覆盖为更接近机器人自然脚距的
`0.20 m`。这只改变上层目标规划，不施加任何物理辅助。使用 `--lateral-separation 0.10`
可以恢复训练原值；`training_demo_settings.json` 会同时记录本轮值和训练原值。

该场景匹配的是策略接口和目标生成条件，不声称 MuJoCo 与 Isaac Gym 的接触动力学完全相同。
当前使用官方 MuJoCo 质量/惯量、名义 PD、9.81 m/s² 重力和 0.6 摩擦（位于训练随机化的
0.5–1.5 范围内），关闭训练噪声、kick 和参数随机化，以显示确定性的 nominal policy 效果。
reset 的一次足底高度对齐是避免两个仿真器处理初始穿透方式不同的必要适配，之后不再干预状态。

目标球附带 RGB 姿态轴：红/绿/蓝分别表示目标局部 x/y/z；左脚目标球为绿色，右脚为红色。
轨迹同时保存 Actor 实际消费的 `command_target_position`、`command_target_quaternion_wxyz`
和 `command_phase`，避免把物理步之后更新的目标误认为当前动作输入。

训练 sampler 在 episode 开始时抽取一个固定 `movement_yaw`，后续每一步的原始采样方向为
`movement_yaw + step_angle`。它是宏观行进方向，不是包含 waypoint 和横向误差反馈的轨迹规划器。
画面中的青色直线从初始双脚中心沿 `movement_yaw` 延伸 8 m，用来显示这条名义中心线；机器人
偏离后，sampler 仍从实时支撑脚继续沿同一宏观方向附近采样，不会主动回到青线。

## 2026-08-24 能力边界与场景结论（model_7000，seed 7）

`outputs/20260824_162422_training_demo_seed007/` 是用户最新一轮 50 秒手动验证：99 次落足全部
小于 0.10 m，均值 0.0296 m、P95 0.0511 m、最大 0.0653 m；未触发 base 过低或 60° 倾角
信号。base 从约 `(0, 0, 0.749)` 移动到 `(4.902, 1.093, 0.692)`。但抽到的宏观方向约为
45.0°，实际 base 位移方向只有约 12.6°，所以它证明了实时相对落足目标跟踪，不证明世界系
路线跟踪。

固定 0° 宏观方向、左右目标间距 0.20 m 的 20 秒受控测试中，前向采样距离 0.12/0.15/0.18 m
分别为 39/39、39/39、39/39 次落足小于 0.10 m；0.20 m 为 38/39，最大误差 0.129 m。
因此离散支撑实验采用 0.15–0.18 m，而不把 0.20 m 当作可靠设计点。即便宏观方向为 0°，
这些测试仍出现约 1.0–1.5 m 横向漂移和明显 stance-foot yaw 漂移，固定稀疏路线必须另做
闭环规划，不能直接使用训练 sampler。

`outputs/showcase_transition_clean_seed7_12s/` 是区域规划器的诊断结果。平台 8 次落足全部小于
0.10 m，两个无重叠碰撞几何的过渡目标也都小于 0.10 m；第一个真正孤立桩目标误差为
0.247 m，进入孤立桩区约 0.68 秒后整机翻倒，视频可见身体越过平台边缘后失去支撑。因此当前
不能把 `model_7000` 宣称为梅花桩部署成功，也不应靠扩大桩到彼此相接来制造成功画面。

`config/experimental_regional_piles.json` 保留这一失败实验，便于继续研究。它的世界桩位固定，
规划器只在固定桩面的安全内圈选点，不移动机器人或地形；黄色点表示支撑中心，红/绿姿态球表示
Actor 实际收到的实时目标。当前更可信的展示仍是 `training_demo` 的长时平地落足跟踪。下一步
若要做显著场景，应先解决固定世界路线与实时 stance-frame 采样的漂移，再选择宽 stepping-stone
场景；不建议现在直接包装成窄梅花桩或带高度变化的综合障碍成功演示。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n isaacgym --no-capture-output \
  python -m unittest discover -s tests -v
```
