# TD-VLA 第一版模型架构：基于 TurboVLA 的并行多流解码

状态：第一版模型与诊断训练入口已实现，验收见 [实现与验证记录](setup/td-vla-v1-validation.md)。最新实验范围、冻结与初始化约定及阶段进度见 [项目路线图](project-roadmap.md)；部分实现参数待数据验收后确定。

本文整理本轮讨论形成的最新决定。与 [研究定位](research-design.md) 中早期 Temporal–Depth 设想冲突时，第一版以本文为准；工程组织仍遵循 [工程约定](engineering-conventions.md)。TD-VLA 保留为项目名称，不表示第一版已经实现完整的 Temporal–Depth 自回归架构。

## 1. 核心目标与范围

在保留 TurboVLA 并行动作块预测能力的前提下，增加双臂夹爪目标轨迹与当前任务阶段输出，验证隐藏特征之间的跨流交互是否有效。

第一版明确采用：

- 当前多视角图像、任务指令与机器人状态作为输入。
- 原版 TurboVLA 的视觉编码、语言编码、融合与状态投影路径。
- 一个共享 Transformer Decoder，处理 Action、Track、Stage 三类 query。
- query 之间采用全双向 self-attention。
- 连续动作块、连续二维轨迹与离散阶段类别在同一次 decoder 前向中产生。
- 冻结 BERT，其余模块从训练开始联合优化，不采用先冻结感知与融合、仅训练新增模块的阶段性方案。

第一版不增加：Temporal 历史模块、记忆摘要、跨窗口 KV cache、动作离散 tokenizer、逐动作步自回归或输出值的串行反馈。按需跳过辅助流计算留到第二版。

原版控制能力支持以最小改动开展研究，但不证明单时刻输入足以区分所有阶段。若阶段存在观测歧义，应分析失败案例，历史信息作为后续研究，而非第一版默认新增模块。

## 2. 与 Moshi 的关系

Moshi 的 Temporal Transformer 汇聚时间上下文，Depth Transformer 利用该上下文和前序生成 token，沿 codebook 顺序生成当前步输出。其核心实现将 Temporal 输出投影后与前序 token embedding 相加；Depth 推理缓存只在当前生成步内部有效。

本项目借鉴的是将同一窗口的多个输出流放到专门模块中交互的思路，第一版做了明确改动：

| 方面 | Moshi 的对应机制 | 第一版设计 |
| --- | --- | --- |
| 时间历史 | Temporal Transformer | 不新增，沿用当前观测条件 |
| 输出形式 | 离散 token | 连续动作、连续轨迹、阶段分类 |
| 流间信息 | 前序 token 与隐藏状态 | 同一 decoder 内的隐藏特征 |
| 窗口内推理 | Depth 顺序采样 | 全双向 query 并行计算 |
| 缓存 | Temporal 与 Depth 不同生命周期 | 不新增跨窗口模型缓存 |

因此，第一版准确定位为“受 Moshi Depth 机制启发的并行多流解码”。全双向 attention 本身不是待主张的新算法；研究价值需要通过机器人多流输出、控制表现及公平对照验证。不得将隐藏特征交互写成条件于最终动作采样结果的自回归联合分布。

## 3. 总体结构

```text
当前多视角图像 → 视觉编码 ─┐
                          ├→ 原版视觉语言融合 → F_t ─┐
任务指令 → 文本编码 ───────┘                         │
当前机器人状态 → 原版状态投影 → R_t ────────────────┤
                                                    ↓
                                        memory = concat(F_t, R_t)
                                                    ↓ cross-attention
Action queries ─┐                         共享 Transformer Decoder
Track queries ──┼──────────────────────→  全双向 self-attention
Stage query ────┘                                   ↓
                                        按 query 类型拆分特征
                                             ↓      ↓      ↓
                                        动作头  轨迹头  阶段头
                                             ↓      ↓      ↓
                                        动作块  双臂轨迹 阶段 ID
```

每次策略更新读取最新观测并重新预测。控制端仍可缓存尚未执行完的 action chunk；这与模型保存跨窗口隐藏记忆不同。预测窗口长度和执行后重新请求策略的间隔分别定义。

## 4. Query、Decoder 与输出头

Query 是输入 decoder 的可学习向量，不是一套独立 decoder。原 ACT 为未来动作位置提供 action queries；共享 decoder 将这些位置转换成与场景相关的隐藏特征，输出头再映射到动作数值。

记当前条件 memory 为：

\[
C_t=[F_t;R_t]\in\mathbb{R}^{B\times N\times D}.
\]

将三类 query 拼接：

\[
Q=[Q_A;Q_P;Q_S],\qquad
H=D_{\theta}(Q,C_t;M_{\mathrm{full}}).
\]

按类型拆分得到 \(H_A,H_P,H_S\)，分别经过输出头：

\[
\hat A_t=f_A(H_A),\qquad
\hat P_t=f_P(H_P),\qquad
\ell_t^S=f_S(H_S).
\]

保留原 Action queries 和动作头的结构；新增 Track queries、一个 Stage query 及相应输出头。查询的流类型、未来位置和左右手身份必须可区分，具体 embedding 实现待定。

当前实现的 query 布局如下；v4 配置取 H_P=10，使用独立 Track query 表，其索引为 time × 2 + arm：

| 类型 | Query 数量 | 输出形状（含 batch） |
| --- | --- | --- |
| Action | \(H_A\) | \([B,H_A,D_A]\) |
| Track | \(2H_P\)，每手每时刻一个 | \([B,H_P,2,2]\) |
| Stage | 1 | \([B,5]\) 类别 logits |

RoboTwin 原版动作维度为 14；预测长度沿用基线配置作为起点，最终实验数值另行确定。Action 取未来 50 个记录步；Track 独立采用未来 0.3、0.6、…、3.0 秒，共 10 个采样点。数据通常间隔约 60 ms，但以实际时间戳为准；不要求二者末点严格重合。此处以最新数据标注约定为准。

## 5. 三类输出的语义

### Action

连续预测整个双臂 action chunk。Action queries 内部保持双向交互，不能施加逐时间位置下三角 mask 将其改成动作自回归。沿用原版动作定义、左右维度顺序及归一化，并通过适配层转换为仿真接口。

### Track

主视角中左右夹爪两指中点的未来目标轨迹，左右身份固定，不动的手也保留输出。监督来自专家轨迹回放，表达期望运动，不保证部署时实际运动完全一致。

每个点输出二维坐标。具体坐标归一化、机器人中点定义、投影与图像变换、有效性 mask 需在数据协议中确定。不能将缺失标签当作坐标零值。

### Stage

预测当前任务内部的阶段 ID，每个已知任务单独划分阶段，具体含义由任务指令提供上下文。不生成阶段描述文本。阶段持续时允许重复预测同一类别。

第一版采用一个跨任务共享的 **5 类 Stage 分类头**，不采用任务专属分类头（2026-11-14 确认）。共享的是输出参数与阶段序号，不要求同一编号在不同任务中表示同一种操作。六任务的阶段定义沿用[数据标注约定](data/p2-annotation-rules.md)。

- 标签使用从 0 开始的任务内编号，某任务有 `C_task` 个阶段时，有效标签为 `0 … C_task-1`，最后一类表示该任务完成。
- 训练对 5 个 logits 使用标准交叉熵，目标为真实任务内阶段 ID；不按任务屏蔽 logits，也不合并类别概率。仅对有效 Stage 标签计算损失，`stage_valid=False` 或 `-1` 标签应排除，不能通过截断标签将其变为有效监督。
- 推理先取 `raw_stage = argmax(logits)`，再解码为 `stage = min(raw_stage, C_task-1)`。即按用户的从 1 开始计数方式，5 类输出用于 3 阶段任务时，预测第 4、5 类均截断为第 3 类。实际从 0 开始的编号中，原始输出 2、3、4 均解码为 2。
- `C_task` 来自已知任务的固定词表配置，不依赖当前或未来 Oracle 状态。当前六任务分别为：`move_stapler_pad=4`、`lift_pot=3`、`handover_block=5`、`beat_block_hammer=4`、`stack_blocks_two=5`、`place_object_basket=5`。
- 评测保留原始预测，同时报告截断后的阶段准确率和原始越界率（`raw_stage >= C_task` 的比例），均在有效标签帧上统计，并按任务报告。截断会把越界输出归入完成阶段，不能用截断后的准确率替代对原始越界行为的检查。

上述分类头、训练损失与推理解码已实现；完整闭环评测接入另行验收。Stage 描述当前状态，Action 与 Track 描述未来窗口；预测 query 不包含未来真实观测或真实标签。

## 6. 注意力机制及对照

第一版的 full attention 即全双向 query self-attention，不将 full 与 bidirectional 作为两个不同方案。所有 query 还通过 cross-attention 读取共同的当前 memory。

全双向 mask 的允许关系如下，行为读取者、列为被读取者：

| | Action | Track | Stage |
| --- | --- | --- | --- |
| Action | 允许 | 允许 | 允许 |
| Track | 允许 | 允许 | 允许 |
| Stage | 允许 | 允许 | 允许 |

未来位置 query 是可学习位置向量，不是未来真实数据，因此它们的双向交互不构成未来信息泄漏。辅助特征也会影响动作，这是预期的交互机制，同时需要评估负迁移风险。

必要对照采用分块独立 mask：仅允许同类 query 之间 self-attention。它仍共享 decoder 参数与条件 memory，不是三个完全独立的网络。

第一版实验组：

1. 原版 TurboVLA：动作控制基线。
2. 分块独立多流 decoder：增加辅助监督与共享参数，不允许跨流 query 交互。
3. 全双向多流 decoder：与第 2 组保持相同 query 数量、参数、监督及训练设置，仅改变跨流 mask。

第 2、3 组直接检验跨流特征交互的贡献。第 1、2 组差异还包含 query、输出头与监督变化，不能全部归因于一个因素。

## 7. 训练决定与归因边界

最新决定是冻结 BERT，从一开始联合训练视觉、融合与输出等其余模块。实现时显式审计冻结标志、requires_grad 和实际梯度；冻结 BERT 不等于冻结全部文本投影与融合模块。

用户提出直接依赖已有任务检查点可能难以归因。全量微调描述“哪些参数更新”，初始化描述“从哪些权重开始”，两者分别决策：

- 正式 Clean-only 子集对照采用基础编码器预训练权重与官方融合初始化，重新训练任务策略，不从 RoboTwin Clean + Randomized 成品策略开始。
- 不将联合训练解释为所有模块随机初始化；官方任务检查点仅作为接口复现参考。
- 正式架构对照使用相同的共有模块初始化来源、训练数据和训练预算；新增模块采用一致初始化规则，记录随机种子。
- 官方检查点评测可作为复现参照，但不能替代同配方训练的架构对照。

总体监督可以写为：

\[
\mathcal L=\lambda_A\mathcal L_A+
\lambda_P\mathcal L_P+
\lambda_S\mathcal L_S.
\]

当前实现：Action 使用有效坐标平均 L1；Track 使用有效坐标平均 Smooth L1（beta=1），不裁剪出画目标；Stage 使用有效帧平均的 5 类交叉熵。掩码先于损失算术生效，双卡按全局有效元素数归一化，全无效时产生可反传的零损失。

初始权重固定为 `λ_A=1.0、λ_P=0.1、λ_S=0.1`，在 `configs/train/td-vla-v1.json` 中独立配置。这是诊断与后续 recipe 搜索的起点，不是已验证的最优比例。各项损失尺度不同，0.1 并不意味着梯度贡献严格为动作的十分之一；需结合动作表现、原始/加权损失及梯度量级调整。第一版不加入动态权重或阶段类别重加权。

该方案没有前序输出值的串行反馈，不需要 Moshi 式 Depth teacher forcing，也不把真实动作、轨迹或阶段作为 decoder 输入。三类损失通过共享 decoder 及感知融合模块共同反向传播。

第一版无需为 Temporal 训练构造历史窗口；样本仍需包含当前输入与未来监督，数据划分必须按完整 episode 进行。验证指标、延迟口径和数据标注遵循研究与工程文档。

## 8. 参数规模与资源说明

已核验官方 README：LIBERO 使用约 0.2B 版本，RoboTwin 报告的是约 0.4B 版本。不能把 0.2B 无条件当作当前 RoboTwin 配置规模。最终需根据选定配置统计总参数量、可训练参数量和实际训练显存。

远端两张约 96 GB 显存的 RTX PRO 6000 Blackwell Max-Q 为全量训练提供资源基础，但具体 batch size、序列长度、精度和吞吐仍需实测，不据此预先承诺训练规模。

## 9. 第二版：自然按需生成

第一版默认计算全部 query。仅隐藏辅助输出不等于省掉辅助计算，临时删除 query 也不保证动作性能不变。

第二版候选 mask：

| | Action | Track | Stage |
| --- | --- | --- | --- |
| Action | 允许 | 禁止 | 禁止 |
| Track | 允许 | 允许 | 禁止 |
| Stage | 允许 | 禁止 | 允许 |

由此支持 Action、Action+Track、Action+Stage、全部输出。真正省计算时移除未请求的辅助 query，而不只是屏蔽输出头。位置和身份编码需稳定，不能因删除某类 query 而改变其余 query 的含义。

第二版复用 decoder 实现，但需要按新 mask 训练或微调；不能直接修改第一版权重的 mask 就声称性能保持。确定性推理、相同输入和位置定义下，动作具有不依赖辅助 query 的计算路径，实际数值差异仍需测试。

Temporal 历史、预测值反馈、离散 token TD 等均为独立后续方向，不因项目名称而自动纳入第一版。

## 10. 实施前剩余决定与检查项

正式基础预训练与融合权重已完成严格加载及每卡64短程验收，第一轮损失权重与学习率配方见[正式训练记录](setup/pretrained-training-v1.md)。未决项：完整离线及 Clean 闭环评测接口、性能验收阈值及后续超参数调优。任务、v4 标签、初始窗口、query 布局与 270/30 episode 划分已落实。训练预算已约定每卡 batch=64、双卡总 batch=128、45,000 个更新步、每 5,000 步保存；详见[路线图 P5](project-roadmap.md)。

实现后重点检查：动作 chunk 仍并行；三类输出形状与左右身份正确；mask 方向正确；所有约定模块实际收到梯度；真实未来标签未进入输入；多流两组除 mask 外配置一致；全量微调后控制与辅助指标均被报告。

## 11. 官方实现依据

- [Moshi 的 Temporal 与 Depformer 实现](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py)
- [TurboVLA 主模型与 encode_condition](https://github.com/H-EmbodVis/TurboVLA/blob/main/turbovla/models/turbovla.py)
- [TurboVLA Action queries 与 ACT decoder](https://github.com/H-EmbodVis/TurboVLA/blob/main/turbovla/models/action_head.py)
- [TurboVLA 参数规模与训练说明](https://github.com/H-EmbodVis/TurboVLA/blob/main/README.md)

以上链接指向可变的上游分支；正式实现和复现需锁定具体提交。实现与测试的具体范围以验收记录为准；诊断结果不代表正式控制性能。
