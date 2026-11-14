# P2 小规模数据与短程训练复现

本阶段是数据和训练链路诊断。正式 Clean-only 对照仍须获取基础编码器预训练权重、冻结 P3 数据/标注协议。

## 环境与资产

沿用 [P1 环境](uv-environments.md)。更新后执行 `bash scripts/setup_env.sh policy` 以安装锁定的 HDF5 读取依赖。
本机 `uv sync --locked` 包含全部轻量测试依赖，核心 td_vla 包本身仍不依赖 GPU。
`source scripts/env.sh` 同时设置 NCCL_IB_DISABLE=1、NCCL_PROTO=LL。

P2 发现默认/Simple 协议在 4 MiB 广播时触发 CUDA 非法访存，P1 的微型 DDP 测试不足以覆盖这个问题。
关闭 P2P、关闭 cuMem host 均未解决；LL 通过 4 KiB、4 MiB、64 MiB、256 MiB 广播/归约以及完整模型训练。
这是该容器和当前依赖组合的兼容设置，不是所有 Blackwell GPU 的性能结论。

```bash
source scripts/env.sh
"$STARVLA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/check_collectives.py
for obj in 060_kitchenpot 020_hammer 110_basket 081_playingcards 057_toycar; do
  .venv/bin/python scripts/fetch_minimal_assets.py --archive objects.zip --object "$obj"
done
```

资产脚本一次只启动一个，避免同时写清单。小型连续 ZIP 区间会一次预取，保持 ZIP CRC 和来源 revision 校验。
本次玩具车资产远端较慢，使用本机读取相同归档区间、逐成员 CRC 验证后中转，服务器再与官方目录核验并写回执。

## 数据生成与 v4 标注

每个任务、种子启动独立进程；已存在的 HDF5（含 partial）不会被覆盖。失败专家回合保留，但不进入训练。
使用正式文档中的成功种子清单复现；扩大数据时从事先选定的新种子区间顺序尝试，不能依据模型得分挑种子。

```bash
"$ROBOTWIN_PYTHON" scripts/collect_pilot.py --task move_stapler_pad --seed 0 --output data/p2-pilot
```

其余任务同样调用 collect_pilot.py。原始数据的 `manifest.jsonl` 保存成功/失败、帧数和文件 SHA256。
旧版标签、依据和叠图已清理。新版生成命令（输出目录必须不存在）：

```bash
.venv/bin/python scripts/annotate_pilot.py --data data/p2-pilot --output data/p2-labels-v4 \
  --track-seconds 3 --track-points 10 --confirm-seconds 0.12 --min-depth 0.0001
```

每条成功回合保存 `labels.npz`、`boundaries.json`、`report.json` 和一张 `review.jpg`。
版本根目录 `manifest.json` 记录标签版本、源代码校验值与逐回合结果。失败专家回合仅记跳过；事件缺失的成功回合保留 Track，Stage 标为待检查。
`track` 为 `[N,K,2,2]`；`track_valid` 为 `[N,K,2]`，由 `temporal_valid` 与 `projection_valid` 合成，`in_frame` 不参与有效性判定。
同时保存目标时间、实际采样时间、未来索引与偏移。`stage` 与 `stage_valid` 为 `[N]`。
全部无效的 Track 回合统计值可为 null，读取统计报告时须正确处理；训练只读取完整 manifest 对应的数据版本。
新的双臂出画监督、稀疏 Track 和 Oracle 事件分段约定见 [标注定义](../data/p2-annotation-rules.md)。

## 短程训练

使用上游 TurboVLA 的完整宽度 ViT-L + BERT + 原版 action decoder。BERT 冻结，其余预期路径训练。
诊断权重来自 P1 的随机 fixtures，BERT 词表也是简化测试词表；因此不能评价语言理解或预训练策略能力。
需要重新建立 fixtures 时运行 P1 的 check_model.py（固定 seed=0），其文件保存在 cache/model-fixtures。
本轮没有加载官方混合数据训练的 RoboTwin 成品权重。

```bash
source scripts/env.sh
"$STARVLA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_pilot.py \
  --data data/p2-pilot --fixtures "$TD_VLA_STORAGE_ROOT/cache/model-fixtures" \
  --output outputs/p2-example-continuous --steps 32 --save-every 16 --batch-size 2 \
  --fixed-batch --initialization random-fixture-diagnostic

"$STARVLA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_pilot.py \
  --data data/p2-pilot --fixtures "$TD_VLA_STORAGE_ROOT/cache/model-fixtures" \
  --output outputs/p2-example-resumed --steps 32 --save-every 16 --batch-size 2 \
  --fixed-batch --initialization random-fixture-diagnostic \
  --resume outputs/p2-example-continuous/step-0016.pt

"$STARVLA_PYTHON" scripts/check_resume.py \
  outputs/p2-example-continuous/step-0032.pt outputs/p2-example-resumed/step-0032.pt \
  --output logs/p2/example-resume.json
```

每次使用新的 output 目录。`torch.load(weights_only=False)` 仅用于自己生成、可信的完整训练检查点，不能用于不可信下载。
去掉 `--fixed-batch` 会按步变更样本，任务均匀抽样；本次另做 8 步该模式的运行验收。
参数为每卡 batch=2（全局 4）、AdamW 5e-5、weight_decay=.01、两步 warmup 后常数 LR、裁剪=1、EMA=.99；
这些仅是诊断参数，不是已选择的双卡正式 recipe。

默认三个先导任务各一个训练回合、一个验证回合；不采样失败回合、终止帧作为起点或未来 RGB。
归一化为训练回合命令的均值/标准差，标准差下限 .01；完整原始文件指纹与划分保存在 config.json。
同一任务/种子重复、原始文件完全重复会拒绝分割；恢复时配置和数据指纹必须一致。
Masked L1 以两卡有效元素总数归一化，避免不同末尾长度造成 rank 权重偏差。

检查点含原版模型、优化器、调度器、EMA、各 rank Python/NumPy/CPU/CUDA RNG、步数、配置和数据指纹。
provenance.json 保存代码提交、源码/锁文件 SHA、Torch/CUDA/NCCL 版本及通信设置。

诊断结束后按工程约定清理重复恢复检查点、临时日志与测试输出。当前远端只保留最终连续运行的 step-0032.pt 及必要运行元数据；文中 step-0016.pt 需重新运行生成后才能进行恢复复现。
