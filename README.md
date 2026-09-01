# Mind Your Steps — TRON1-SF 落足点跟踪与上层落脚规划（IsaacGym）

复现论文 *Mind Your Steps: A General Learning Framework* 的随机落足点跟踪
（foothold tracking），并在此基础上开展**上层落脚规划**研究：在冻结下层行走策略之上，
学习一个每步选择落脚目标的上层规划器。机器人：**TRON1-SF 双足（8 自由度）**，仿真基于
IsaacGym（自编译 `gym_38.so`）。

## 仓库组成

| 目录 | 说明 | 状态 |
|---|---|---|
| `foothold/` | 下层策略训练包：环境、落足采样器、奖励、PPO、训练/评估入口 | 完成，冻结 |
| `upper_foothold_planner/` | 特权教师蒸馏 + 非对称 PPO 的上层规划器（可部署，95.5% 成功率） | 完成，作为 immutable 基线 |
| `world_model_upper_planner/` | **候选约束选项世界模型（CG-OWM）**：TD-MPC 风格上层规划器，离散 beam 搜索 | **活跃研究** |
| `mujoco_foothold_deploy/` | MuJoCo 部署资产/配置/脚本 | 部署辅助 |
| `resources/` | 机器人 URDF + STL mesh（`SF_TRON1A` / `PF_TRON1A`） | 资产 |
| `scripts/`、`docs/`、`reports/` | 诊断脚本、设计与分析文档、汇报 | 文档/工具 |

## 当前研究进展（`world_model_upper_planner/`，2026-09-01）

CG-OWM 是"世界模型 + 规划"路线的重新实现，针对上一版不收敛的根因做了结构改动：

- **候选约束**：动作空间为 294 个经验可达的离散落足候选，不再用连续 CEM（消除动作别名）；
- **不确定性感知的离散 beam 搜索**替代连续 CEM 规划；
- 严格保持边界：冻结下层 + 部署观测（深度 + 36 维本体）+ 训练期特权几何标签。

已验证（固定 seed 924，64 环境 × 30 s，random-composite，同一 H3 checkpoint）：

| 决策规则 | 成功率 | 每步 progress |
|---|---:|---:|
| H3 向量化 beam 规划 | **77.1%**（222/288） | 0.212 m |
| H1 学习打分 | 59.5% | 0.202 m |
| proposal 先验 | 63.6% | 0.200 m |

多步隐空间规划相对同 checkpoint 的 H1 打分带来 **+17.5 个百分点**，是首个世界模型规划
收益的闭环证据。

V2 多地形扩展（held-out seeds，beam 规划）：edge cases 81.6%、random composite 71.3%、
stepping stones 70.8%、household 58.5%、mixed 61.3%；**turns 13.6% 仍是失败域**（急转弯的
下层 yaw 实现能力是当前瓶颈）。

进行中：**V4 大规模训练**（12 分片 × 512 环境，目标约 70–80 万条 option 转移，覆盖
7 类地形 × nominal/hard 难度；已完成 research_nominal/hard、turns_nominal 三片）。

详见 `world_model_upper_planner/docs/`（`status.md`、`latest_results.md`、
`scale_training_v4.md`、`method.md`）。

## 环境

- Python 3.8，conda 环境 `isaacgym`
- IsaacGym（自编译 `gym_38.so`）+ PyTorch 2.4（CUDA 11.8）
- 运行前需先 `import isaacgym`（再 `import torch`，代码已按此顺序处理）

## 下层策略训练（foothold/）

```bash
cd mind-steps
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -m foothold.train --headless --num_envs 8192
```

关键超参位于 `foothold/config.py` 的 `foothold` 与 `rewards` 两节；
训练日志默认写入 `logs/<时间戳>_<run_name>/`，TensorBoard 事件在其 `tb/` 子目录。

## 评估（foothold/）

```bash
python -m foothold.eval --checkpoint logs/<run>/model_100.pt --steps 500
```

评估要求 version 2 checkpoint，其中包含训练期 Normalizer/RMS 和训练配置。旧 checkpoint 只有
网络权重，无法精确恢复训练输入坐标系，eval 会明确拒绝加载而不是静默使用空 Normalizer。

## 上层规划器快速开始

- **teacher-student / PPO**：见 `upper_foothold_planner/docs/teacher_distillation_report.md`
  （方法）与 `upper_foothold_planner/README.md`（命令）。
- **world model（CG-OWM）**：见 `world_model_upper_planner/README.md`，含单测、smoke、
  完整流水线命令。

## 文档

- `docs/normalization_and_checkpoint.md` — 运行归一化原理、checkpoint 契约与 train/eval 失配分析
- `docs/foothold_goal_analysis.md` — 落足点目标机制分析（5 项检查，含公式与修改记录）
- `docs/reproduction_spec.md` — 复现规格
- `docs/provenance.md` — 上游来源与资产校验（SHA-256）
- `upper_foothold_planner/docs/` — 特权教师蒸馏与非对称 PPO 的完整方法/结果
- `world_model_upper_planner/docs/` — CG-OWM 方法、状态、结果索引与 V4 规模训练设计
- 其余 `docs/*.md` — 奖励、训练稳定性等专题报告

## 备注

`logs/`、`experiments/`、`runs/`、`checkpoints/`、`outputs/`、`.vscode/`、`pretrained/`、
`eval_data/`、数据集（`*.npz`）、视频（`*.mp4`）、日志（`*.log`）等均已在 `.gitignore`
中排除，不会被提交。
