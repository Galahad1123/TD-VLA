# 原版 TurboVLA 同预算 Clean 基线

2026-11-15：按用户要求，使用与 TD-VLA 相同的数据与更新预算训练原版 TurboVLA。配置为 `configs/train/turbovla-clean-v1.json`，实验 ID 为 `turbovla-clean-baseline-v1`。

## 比较口径

- 训练入口保持 `scripts/train_multistream.py`；`mode=baseline` 直接实例化固定上游的 `TurboVLA`，使用原版并行 ACT Action 头，不创建 Track/Stage 查询或预测头。
- 只有 Action L1 参与优化，权重 1。Track/Stage 权重显式设为 0；共享数据加载器仍读取标签，但标签不进入模型输入或 Action 损失。
- 使用相同六任务 Clean 数据、270/30 episode 划分、训练集归一化、三视角预处理及固定任务指令，任务均匀采样，seed=42。
- 使用相同 BERT、DINOv3 和 GroundingDINO 基础初始化，不加载 RoboTwin 成品策略或已训练 TD-VLA 权重。仅冻结 BERT。
- 双卡每卡 batch 64，全局 128，训练 45,000 步，共 576 万个样本窗口；每 5,000 步保存。相同更新预算不意味着完全相同运行时间或 FLOPs。
- AdamW、学习率、warmup/cosine、权重衰减、EMA、梯度裁剪及精度策略均沿用 TD-VLA 配方。
- 每 1,000 步对同一 240 帧固定验证子集计算在线模型 Action L1。训练后闭环比较应使用相同 Clean 场景种子、指令、执行窗口与每任务 20 次规模；本轮仅启动训练，不自动附加评测。

## 执行与验收

1. 核对配方仅 `purpose`、`mode`、`loss_weights` 不同。
2. 在真实预训练权重上运行双卡每卡 64 的 8 步短测，验证无辅助头、BERT 冻结、损失和梯度有限、检查点可加载。
3. 短测通过后从相同基础初始化重新开始正式 45,000 步训练，不从短测权重续训。
4. 保存命令、Git 提交、PID、配置和短测摘要；清理短测大检查点与冗余日志。正式训练运行期间保留正在使用的产物。

```bash
source scripts/env.sh
envs/policy/.venv/bin/torchrun --standalone --nproc-per-node=2 \
  scripts/train_multistream.py --config configs/train/turbovla-clean-v1.json \
  --data data/clean-six-50-selected-v1 \
  --split configs/data/clean-six-50-split-v1.json \
  --pretrained-root /root/autodl-tmp/td-vla/pretrained/base-v1 \
  --output outputs/turbovla-clean-baseline-v1
```

正式启动与验收记录保存在 `logs/turbovla-clean-baseline-v1/`。训练是否完成须同时检查状态与进程，不以本文作为完成证明。
