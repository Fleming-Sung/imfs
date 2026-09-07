# composite3d_v4_nav 自动实验结果

所有正式模型测试均从路线起点出发，禁用全图训练引导；保留当前支撑约束。
H1与H3使用同一个检查点，区别只是规划时域。各测试尾部未结束episode作为截尾记录。

| 测试 | 完成数 | 跌倒数 | 截尾数 | 最远x均值(m) |
|---|---:|---:|---:|---:|
| video_seed9603 | 0 | 3 | 1 | 11.80 |
| video_seed9602 | 0 | 2 | 1 | 17.31 |
| h3_64_seed9601 | 4 | 203 | 63 | 12.37 |
| h1_64_seed9601 | 23 | 128 | 64 | 14.81 |

视频不是成功证明，需结合完整测试表：

- seed9602：[视频](../experiments/composite3d/composite3d_v4_nav/video_seed9602/rollout.mp4)、[轨迹](../experiments/composite3d/composite3d_v4_nav/video_seed9602/trajectory.png)
- seed9603：[视频](../experiments/composite3d/composite3d_v4_nav/video_seed9603/rollout.mp4)、[轨迹](../experiments/composite3d/composite3d_v4_nav/video_seed9603/trajectory.png)

原始数据、日志和逐阶段状态：`experiments/composite3d/composite3d_v4_nav/`。
