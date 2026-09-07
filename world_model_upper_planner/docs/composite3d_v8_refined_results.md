# composite3d_v8_refined 自动实验结果

所有正式模型测试均从路线起点出发，禁用全图训练引导；保留当前支撑约束。
本轮仅运行有界局部细化H1，不构成H1/H3对照。各测试尾部未结束episode作为截尾记录。

| 测试 | 完成数 | 跌倒数 | 截尾数 | 最远x均值(m) |
|---|---:|---:|---:|---:|
| video_seed9802 | 0 | 2 | 1 | 11.15 |
| h1_64_seed9901 | 26 | 98 | 64 | 15.71 |
| h1_64_seed9801 | 29 | 126 | 64 | 15.23 |
| video_seed9803 | 0 | 4 | 1 | 7.81 |

视频不是成功证明，需结合完整测试表：

- seed9802：[视频](../experiments/composite3d/composite3d_v8_refined/video_seed9802/rollout.mp4)、[轨迹](../experiments/composite3d/composite3d_v8_refined/video_seed9802/trajectory.png)
- seed9803：[视频](../experiments/composite3d/composite3d_v8_refined/video_seed9803/rollout.mp4)、[轨迹](../experiments/composite3d/composite3d_v8_refined/video_seed9803/trajectory.png)

原始数据、日志和逐阶段状态：`experiments/composite3d/composite3d_v8_refined/`。
