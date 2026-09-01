# V4 大规模多场景训练

日期：2026-09-01。目标是把 V2 的 60,317 条可行性验证 replay 扩展为覆盖多场景、
多参数和多难度的正式训练集，并继续保持冻结下层、部署观测和训练期特权标签的边界。

## 数据矩阵

正式流水线使用 12 个独立分片，每片 512 个程序生成环境、3,000 个 50 Hz 下层 tick：

- `research` 混合场景：nominal / hard；
- `turns`：nominal / hard；
- `household`：nominal / hard；
- `narrow_bridge`：nominal / hard；
- `edge_cases`：hard；
- `stepping_stones`：hard；
- `irregular_support`：nominal / hard。

每个难度同时改变路线长度、支撑宽度、间隙、障碍密度或桥宽。30% episode 使用
中途重置以覆盖长路线后半段；行为分布由安全多样采样、35% 可行候选均匀采样和 15%
非安全探索组成。所有状态仍保存 294 个训练期几何标签，动态结果只记录真实执行动作。

没有把全宽栏杆放进主数据：当前冻结下层没有落足高度动作，该场景中所有候选均不可通行；
将其大量加入只会制造能力集外数据。旧通用 corridor mesh 短测也出现异常高频快速摔倒，
在接口原因查清前不进入正式 replay。

## 规模与吞吐

- 目标新增数据：约 70--80 万条上层 option transition；加 V2 replay 后约 80 万量级。
- 原始物理交互预算：12 × 512 × 3,000 = 18.432M lower-env steps。
- 512 环境新增典型场景探针：300 tick / 5,864 transitions / 61.8 s，吞吐
  2,485 lower-env-step/s，跌倒 338 次、成功 7 次。
- 大数据合并使用只读 NPY memmap，不再把多个压缩 NPZ 同时复制进 32 GB 主存。
- H1 使用 batch 1024、H3 使用 batch 512；已通过完整 forward/backward 探针。
- 训练按 `terrain_kind/difficulty` 平衡，而不是按原始行数采样。

## 自动流水线

入口：

```bash
conda run -n isaacgym --no-capture-output \
  python scripts/run_scale_training.py --profile scale --phase all
```

流程按 `collection -> memmap merge -> H1 -> H3 -> 12场景×3种子闭环评估` 自动执行。
每个阶段以 artifact 为恢复点，命令、耗时和状态写入
`experiments/scale_v4/scale/manifest.json`，stdout 和逐阶段日志保存在同一实验树。

正式采集已启动。模型是否晋级只看多场景闭环 success/fall/timeout、路径和视频，不按
训练 loss 的微小变化判断。
