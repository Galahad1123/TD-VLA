# 第一版基础预训练初始化与正式训练配方

日期：2026-11-14。模式 `full`，只使用六任务 Clean 入选训练数据。

## 权重来源

固定清单：[configs/pretrained.json](../../configs/pretrained.json)。所有文件在加载前验证来源、revision、文件列表和 SHA256；不加载 RoboTwin 成品策略。

- BERT：`google-bert/bert-base-uncased`，revision `86b5e0934494bd15c9632b12f734a8a67f723594`，hf-mirror 优先。
- DINOv3 ViT-L/16 LVD-1689M：官方 HF revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`。HF 账号已登录但尚未获准访问；改用 [ModelScope 公开分发](https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m)，revision `99b6205b0321c4d9726dd072d8c6a80aa6be72df`。该仓库元数据是用户上传，不将其称为 Meta 官方上传。
- DINOv3 权重 SHA256：`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`，与 HF 官方公开 LFS 元数据相同；实际完整下载后再次验证。保留随权重发布的 `LICENSE.md`，使用仍遵守 [DINOv3 许可证](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。
- GroundingDINO：官方发布者 `ShilongLiu/GroundingDINO` revision `a94c9b567a2a374598f05c584e96798a170c56fb` 的 `groundingdino_swint_ogc.pth`。镜像吞吐较低时，使用 [ModelScope 公开副本](https://modelscope.cn/models/AI-ModelScope/GroundingDINO)，revision `ad6aab4bfda74f778aac9890a0c6af2d067a9936`；两方公布 SHA256 均为 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`。

实际权重位于 `/root/autodl-tmp/td-vla/pretrained/base-v1/{bert,vision,fusion}/`，每个组件附 `manifest.json`；记录 canonical HF 来源与实际公开分发来源，不把替代下载冒充 HF 下载。

加载顺序：先从基础 BERT/DINOv3 构造上游模型，再按 TurboVLA 官方映射从 GroundingDINO 加载 **BERT、文本投影、text_layers、fusion_layers**。因此最终冻结的是 GroundingDINO 对应 BERT 权重，tokenizer 沿用 bert-base-uncased。动作、状态投影、视觉投影及新辅助头按固定种子初始化；不加载 GroundingDINO 检测头或 RoboTwin 动作策略。要求四组目标张量完整、形状一致，缺失时直接报错。

## 固定配方

配置：[configs/train/td-vla-clean-v1.json](../../configs/train/td-vla-clean-v1.json)。

| 项目 | 设置 |
| --- | --- |
| GPU / batch | 双卡；每卡64，全局128，累积1 |
| 更新预算 / 保存 | 45,000步；每5,000步保存完整状态，共9个 |
| 优化器 | AdamW，betas=(0.9,0.95)，weight_decay=0.01 |
| 学习率 | 峰值5e-5；1000步线性warmup，随后cosine至5e-6 |
| 冻结 / 裁剪 | BERT冻结；其余预期模块训练；全局梯度范数裁剪1 |
| EMA | 0.999 |
| 损失权重 | Action=1、Track=0.1、Stage=0.1 |
| 数据 | 270训练/30验证，固定episode划分；任务均匀抽样 |
| 验证 | 每1000步，验证集每回合均匀选8帧，共240帧；使用当前在线权重 |
| 日志 | 每50步及首末步，分项损失、学习率、梯度、显存；status.json记录进度 |

验证按任务报告Action L1、原图归一化Track坐标MAE、Stage截断准确率、原始准确率和越界率。固定子集只用于过程诊断，不等于完整验证集或Clean闭环评测。`status.json` 的 `running` 可能在异常退出后变旧，监控须同时核对PID和更新时间。

`--steps` 可限制短测停止位置，但cosine总长度始终使用配置中的45,000步。恢复要求模型、优化器、调度器、EMA、逐rank RNG、配置与数据/代码签名一致。当前选定超参数是第一轮配方，不宣称已调优。

## 运行入口

```bash
source scripts/env.sh
# 公开镜像版本与canonical官方文件哈希相同；无需安装ModelScope SDK。
envs/policy/.venv/bin/python scripts/prepare_pretrained.py \
  --output pretrained/base-v1 --components bert vision fusion --public-distribution

envs/policy/.venv/bin/torchrun --standalone --nproc-per-node=2 \
  scripts/train_multistream.py --config configs/train/td-vla-clean-v1.json \
  --data data/clean-six-50-selected-v1 \
  --split configs/data/clean-six-50-split-v1.json \
  --pretrained-root /root/autodl-tmp/td-vla/pretrained/base-v1 \
  --output outputs/td-vla-clean-full-v1
```

下载与权重检查不发送、记录token；Hugging Face认证来自用户在服务器配置的凭据存储。公开分发下载不使用该凭据。

## 验收进度

- 65项远端完整回归通过；本地50项通过、4项policy环境测试跳过。
- 新增周期验证边界的双卡诊断：第2步恢复至第4步，模型/优化器/调度器/EMA/逐rank随机状态精确一致。
- 独立审查未发现阻断性训练逻辑问题。
- 真实基础权重严格加载381个张量：BERT 199、文本投影2、text_layers 72、fusion_layers 108。
- 每卡64完成16步真实权重短测与第8/16步固定子集验证；梯度与损失有限，峰值分配显存55,548,263,424字节（约51.7 GiB）。真实权重恢复比较及正式启动完成后另附记录。
