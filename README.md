# Mind Your Steps — TRON1-SF 落足点跟踪复现（IsaacGym）

复现论文 *Mind Your Steps: A General Learning Framework* 中的随机落足点跟踪
（foothold tracking）任务，目标为 **TRON1-SF 双足机器人（8 自由度）**，仿真基于 IsaacGym。

## 目录结构

- `foothold/` — 主代码包：环境、落足采样器、奖励、PPO、训练/评估入口
- `scripts/` — 诊断/审计/评估脚本（`diag_*.py` 为调试期的一次性脚本）
- `docs/` — 设计与问题分析文档（见下文）
- `resources/` — 机器人 URDF + STL mesh（`SF_TRON1A` / `PF_TRON1A`）
- `Mind Your Steps: A General Learning Framework.pdf` — 参考论文与附录

## 环境

- Python 3.8，conda 环境 `isaacgym`
- IsaacGym（自编译 `gym_38.so`）+ PyTorch 2.4（CUDA 11.8）
- 运行前需先 `import isaacgym`（再 `import torch`，代码已按此顺序处理）

## 训练

```bash
cd mind-steps
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -m foothold.train --headless --num_envs 8192
```

关键超参位于 `foothold/config.py` 的 `foothold` 与 `rewards` 两节；
训练日志默认写入 `logs/<时间戳>_<run_name>/`，TensorBoard 事件在其 `tb/` 子目录。

## 评估

```bash
python -m foothold.eval --checkpoint logs/<run>/model_100.pt --steps 500
```

## 文档

- `docs/foothold_goal_analysis.md` — 落足点目标机制分析（5 项检查，含公式与修改记录）
- `docs/reproduction_spec.md` — 复现规格
- `docs/provenance.md` — 上游来源与资产校验（SHA-256）
- 其余 `docs/*.md` — 奖励、训练稳定性等专题报告

## 备注

`logs/`、`.vscode/`、`pretrained/`、`eval_data/`、根目录调试产物等均已加入 `.gitignore`，不会被提交。
