# TD-VLA 第一版实现与诊断验收

日期：2026-11-14。架构依据：[第一版模型架构](../model-architecture-v1.md)。

## 实现范围

- `src/td_vla/models/multistream.py`：复用锁定 TurboVLA 的编码、融合、状态投影与 ACT，保留原动作 query/头，新增 20 个 Track queries、1 个 Stage query、无界二维轨迹头与共享 5 类阶段头。
- `full` 全双向；`block` 仅流内交互；两者参数完全一致，只改 mask。`baseline` 直接调用原 TurboVLA。固定同一初始化种子时，共有模块保持相同初始值。
- `src/td_vla/training/losses.py`：Action L1、Track Smooth L1（beta=1）、Stage CE。先排除无效元素再算损失，双卡按全局有效数归一化。
- 权重起点 `1.0 / 0.1 / 0.1`，配置在 `configs/train/td-vla-v1.json`。分别记录各项原始损失、加权总损失和输出头/上游模块的梯度范数；不是最终 recipe。
- Stage 先取 5 类 argmax，再按任务阶段数截断；保留原始类别和越界标记。任务编号不进入未来标签或 Oracle 输入路径。
- 数据使用当前三视角 RGB、任务文本与机器人 command state。原始 qpos 为 `[左6关节, 左夹爪, 右6关节, 右夹爪]`，显式重排为上游 `[左6关节, 右6关节, 左夹爪, 右夹爪]`；提供逆转换函数。
- 新训练入口不复用旧 pilot 的 z-score。关节使用训练集 min/max，常量关节归一为 0；夹爪阈值为 0.49。验证值不反向影响训练统计，轨迹不裁剪。
- 固定划分：[270 训练 / 30 验证清单](../../configs/data/clean-six-50-split-v1.json)，六任务各 45/5，seed=42。该清单与入选 manifest、原始和标签 SHA256 绑定。每个进程首次读取一个 episode 时复核数据哈希；文件在运行期间应保持只读。
- 图像按需读取并缩放，不将所有 RGB 加载入内存；只缓存小型标签。输出仍处于原图 `u/W,v/H` 坐标系，没有 crop/随机几何增强。
- `scripts/train_multistream.py` 支持三组模式、任务均匀采样、BERT 冻结、DDP、完整恢复与 EMA。恢复检查模型/配置/数据清单/源码签名及 world size，逐 rank 恢复 Python、NumPy、Torch、CUDA RNG。

## 已完成验证

- 初始已有测试 44 项通过。
- 本地数据及已有测试通过；模型测试在本地无 PyTorch 的轻量环境中显式跳过，在远端 policy uv 环境执行。
- 远端新增模型、数据与真实编码器测试通过：形状、block 与上游动作等价、跨流依赖、Stage 截断、加权损失、无效 NaN 屏蔽、全无效零损失、冻结 BERT 和其他模块梯度、state_dict 往返、完整恢复。
- 独立审查发现并修复：划分重叠/运行数据被修改的漏检、数据集分辨率未读取模型配置。补充了数据修改、分区重叠、14 维动作重排往返测试。
- 额外两进程 Gloo 测试：一个 rank 全无有效 Track，另一个 rank 有效，归约后的梯度与合并 batch 相同。

### 双卡固定小样本诊断

完整模型参数 410,607,668，可训练参数 324,556,084。使用已有随机 BERT/DINO fixture、完整深度编码器（BERT fixture 使用精简词表）与六任务入选数据；每卡 batch=2，固定各 rank 的采样，运行 32 步。

| 损失 | 第 1 步 | 第 32 步 |
| --- | ---: | ---: |
| Action L1 | 0.576196 | 0.141175 |
| Track Smooth L1 | 0.522852 | 0.067490 |
| Stage CE | 2.401110 | 0.010971 |
| 加权总损失 | 0.868593 | 0.149021 |

rank 0 峰值显存约 9.19 GB（8.56 GiB），最后一步计算及数据读取约 0.45 秒，不含检查点写盘。这个测量仅对应本次设置。

从第 16 步恢复至第 32 步，与连续运行的模型、优化器、调度器、EMA、逐 rank RNG 及签名逐项精确相同。修复数据校验后另做最终三组动态采样与恢复验证，见下方最终记录。

这些结果证明多头能接收梯度、固定小样本损失下降与恢复一致，不代表六任务收敛，也不代表闭环成功率。单 batch 验证中的阶段准确率不作为正式研究结果。

## 复现入口

在远端仓库使用已有 uv policy 环境：

```bash
source scripts/env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2

envs/policy/.venv/bin/python scripts/prepare_training_split.py \
  --data data/clean-six-50-selected-v1 \
  --output data/clean-six-50-selected-v1/split-v1.json
```

划分文件已存在时无需再生成；命令拒绝覆盖。仓库保存同内容副本，路径均相对于数据视图根目录解析。

```bash
envs/policy/.venv/bin/torchrun --standalone --nproc-per-node=2 \
  scripts/train_multistream.py \
  --config configs/train/td-vla-v1.json \
  --mode full \
  --data data/clean-six-50-selected-v1 \
  --split configs/data/clean-six-50-split-v1.json \
  --fixtures /root/autodl-tmp/td-vla/cache/model-fixtures \
  --output outputs/td-vla-v1-example --steps 32 --save-every 16 --fixed-batch
```

模式改为 `baseline` / `block` 即运行其他组，输出目录必须不同；去掉 `--fixed-batch` 使用每步变化的任务均匀采样。恢复使用新的输出目录与 `--resume <checkpoint>`，`--steps` 是恢复后的目标总步数。改变损失权重属于新实验，不能伪装为相同配置的精确恢复。

## 当前边界与下一步

- 诊断入口明确只接受 `random-fixture-diagnostic`，尚未接入/验收正式基础视觉语言预训练及 GroundingDINO 融合初始化。不能把这些诊断权重作为正式 baseline。
- 当前验证只对少量 batch；完整验证集按任务统计、阶段边界误差和运动/静止 Track 分组指标仍待接入。
- 返回模型输出与阶段解码已可用，但 RoboTwin 策略服务及 Clean 闭环评测适配尚未验收。
- 当前没有 Temporal 历史、动作自回归、动态损失权重或按需省略 query。
- 正式权重、训练预算和闭环评测属于后续 recipe 阶段。本轮不启动正式长训练。

## 最终验证与产物

本地完整测试：50 passed，3 skipped（本机轻量环境没有 PyTorch）；远端 policy 环境新增测试：15 passed；同步到实际仓库后，完整回归 **59 passed**。完整回归中的旧资产测试需要根项目 dev 组的 `remotezip==0.12.3`，本次通过清华源安装到临时目录并以 PYTHONPATH 注入，测试后清理，不修改 policy 锁文件。Ruff 和 git diff --check 通过。上游 timm/transformers 各有一条弃用警告，不影响本轮执行。

审查修复后的 baseline、block、full 分别完成 4 步双卡动态采样训练；full 再从第 2 步恢复至第 4 步，完整状态也逐项精确相同。必要配置、损失摘要和恢复结果保留于双机 `logs/td-vla-v1/`。诊断产生的大型检查点、控制台日志和隔离验证目录在完成比较后清理；原始数据、正式划分与已有环境保持保留。

本轮清理六个诊断输出目录，共 55,383,032,700 字节（约 55.4 GB），以及临时验证目录和控制台日志。分支 `feature/td-vla-v1` 已通过主仓库同步双机；源码提交 `90ea83c`，后续文档提交仅补充最终验收记录。
