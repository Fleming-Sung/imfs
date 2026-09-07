# composite3d_v6_multilevel 自动实验结果

所有正式模型测试均从路线起点出发，禁用全图训练引导；保留当前支撑约束。
H1与H3使用相同检查点/信息，区别只是规划时域。各测试尾部未结束episode作为截尾记录。

| 测试 | 完成数 | 跌倒数 | 截尾数 | 最远x均值(m) |
|---|---:|---:|---:|---:|
| video_seed9802 | 1 | 1 | 1 | 18.24 |
| h3_64_seed9801 | 20 | 142 | 64 | 13.77 |
| h1_64_seed9801 | 18 | 150 | 64 | 13.74 |
| video_seed9803 | 0 | 1 | 1 | 14.71 |

视频不是成功证明，需结合完整测试表：

- seed9802：[视频](../experiments/composite3d/composite3d_v6_multilevel/video_seed9802/rollout.mp4)、[轨迹](../experiments/composite3d/composite3d_v6_multilevel/video_seed9802/trajectory.png)
- seed9803：[视频](../experiments/composite3d/composite3d_v6_multilevel/video_seed9803/rollout.mp4)、[轨迹](../experiments/composite3d/composite3d_v6_multilevel/video_seed9803/trajectory.png)

原始数据、日志和逐阶段状态：`experiments/composite3d/composite3d_v6_multilevel/`。
