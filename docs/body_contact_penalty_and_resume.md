# 膝盖/大腿触地惩罚 与 断点续训

> 状态（2026-08-22）：本文记录两项改动——(1) 机器人的膝盖、大腿等关节接触地面时增加
> 奖励惩罚；(2) 训练支持从已有 checkpoint 断点续训。涉及
> `foothold/config.py`、`foothold/env.py`、`foothold/rewards.py`、`foothold/train.py`。

## 1. 膝盖/大腿触地惩罚

### 1.1 动机

原有奖励中只有 `base_height`、`orientation`、`feet_slip` 等与摔倒/足端打滑相关的惩罚，
但机器人在摔倒过程中或低姿态行走时，膝盖、大腿（hip/abad 链接）等身体部件会先接触地面，
此时躯干高度可能仍处于 healthy 区间内，不会立即触发终止，策略也得不到足够的负面信号。
因此新增一项显式惩罚，对非足端身体部件触地给予负奖励。

### 1.2 实现

- `foothold/config.py`
  - `rewards.scales` 新增 `"body_contacts": -2.0`（负权重，作为惩罚）。
  - `rewards` 新增 `body_contacts_threshold: 1.0`：净接触力 z 分量超过 1 N 即判定为接触，
    与现有足端接触判定阈值（`feet_swing` 中 `> 1.0`）保持一致。
  - `rewards` 新增 `contact_penalty_bodies: ["abad", "hip", "knee"]`：按链接名子串匹配，
    覆盖左右 `abad_L/R_Link`、`hip_L/R_Link`、`knee_L/R_Link` 共 6 个身体部件。
- `foothold/env.py`
  - 在建环境时用 `find_actor_rigid_body_handle` 匹配上述链接名，生成
    `self.body_contact_indices`（LongTensor）。
- `foothold/rewards.py`
  - 新增 `_reward_body_contacts()`：统计接触地面的身体部件个数，返回
    `count * dt`；`compute()` 中按 `body_contacts` 权重缩放为负奖励。

### 1.3 惩罚量级

单个触地部件约为 $-2.0/\text{s}$（`scale × count × dt`），多个部件同时触地时线性叠加。
该量级与现有 `action_rate`、`feet_slip`（均为 $-3.0$）同一量级。

### 1.4 兼容性

旧 checkpoint 的 `config` 不含 `contact_penalty_bodies` / `body_contacts_threshold`，
代码中通过 `getattr(..., default)` 兜底：旧配置下 `body_contact_indices` 为空，
`_reward_body_contacts()` 返回 0，eval 行为不受影响。

## 2. 断点续训

### 2.1 动机

原训练脚本每次从零开始，无法利用已有模型继续训练。本次改动使训练可从中途 checkpoint
恢复策略权重、归一化统计量、优化器状态与学习率。

### 2.2 实现

- `foothold/train.py`
  - `checkpoint_state()` 新增可选参数 `ppo`：当传入时额外保存
    `optimizer`（优化器 state_dict）与 `ppo_learning_rate`（当前学习率）。
  - 新增 `load_resume()`：加载 checkpoint，恢复 actor_critic、normalizer、optimizer
    与学习率，返回续训起始 iteration。
  - 新增命令行参数 `--resume <checkpoint路径>`。
  - 训练循环从 `start_iter` 开始（步长 curriculum 与 ETA 计算同步修正），
    续训来源记录到新 run 的 `config.json`（`resume_from` 字段）。

### 2.3 用法

```bash
python -m foothold.train --headless --resume logs/<run>/model_1000.pt
```

说明：

- `--max_iterations` 是**总**迭代数（绝对目标）。例如从 1000 步续训、想再训 30000 步，
  需传 `--max_iterations 31000`。
- 续训写入一个新的日志目录，不覆盖原 run 的日志。
- 旧 version-2 checkpoint（无 `optimizer` 字段）也可 resume：策略权重与 normalizer 正常恢复，
  优化器状态从头开始。

## 3. 改动文件清单

| 文件 | 改动 |
|---|---|
| `foothold/config.py` | 新增 `body_contacts` 权重、触地阈值与惩罚 body 列表 |
| `foothold/env.py` | 新增 `body_contact_indices` 索引 |
| `foothold/rewards.py` | 新增 `_reward_body_contacts()` |
| `foothold/train.py` | 新增 `--resume`、`load_resume`，checkpoint 保存 optimizer/学习率 |
