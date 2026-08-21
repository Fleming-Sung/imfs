# 本地参考来源

本项目在 2026-08-14 从以下相邻目录做了一次性复制/对照。它们不参与运行时 import。

- `../tron1_RL/tron1_rl/*.py`：基础 Isaac Gym、PF/SF 配置、手写 PPO 与 runner。
- `../tron1_RL/resources/PF_TRON1A`、`SF_TRON1A`：机器人 URDF 和 mesh。
- `../loco-mujoco/experiments/humanoid_foot_placement` 与对应 `loco_mujoco/core`：只读算法细节参考，未复制执行代码。
- `Mind Your Steps: A General Learning Framework.pdf`：论文与附录。

关键资产 SHA-256：

```text
PF robot.urdf  86a4e3e2b0bac3352f120bd4f07c1d1d6ae8111b7c50871af8b589bbc54b5695
SF robot.urdf  febc6baeda21f77fd9b648bfe6c75e147a7d36c2d41834c8cc3a196a060c2a27
SF model_10000.pt  1746158d9cf3fd0712e1fcfb2a587f75a999ec6cb92d8b1aee76c6d9f0410b71
SF model_3500.pt   1ba7d65587a0e3219a4c281aaef2cd517dff0aa8dd435aa5a82c6e7a07b608bd
```

`pretrained/SF_TRON1A_locomotion_model_10000.pt` 是上述 SF checkpoint 的
逐字节本地副本，仅用于审计原始速度策略的命令响应和初始化映射；运行时不访问
上层 `tron1_RL`。复制后已再次计算 SHA-256，结果与来源一致。
`pretrained/SF_TRON1A_locomotion_model_3500.pt` 同样是逐字节副本，用来判断
最终 checkpoint 的站立偏置是否由训练后期产生。

后续实现只改当前目录。若需重新参考上游，先做校验和/差异审计，禁止直接修改参考目录。
