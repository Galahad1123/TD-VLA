我完全同意。而且我认为应该调整一下研究思路：先把我们真正想提出的模型定义清楚，再针对这个模型的每一个技术特征做 Prior Art Search。

上一轮调研虽然找到了很多相关工作，但把「多任务辅助预测」「多模态联合生成」「多流 Transformer」「多 Codebook」混在了一起。这些工作并不一定真正构成对我们架构的直接先例。

我认为，我们需要先回答一个问题：

我们到底是想提出一种新的 VLA 输出表示，还是一种新的 VLA 自回归建模架构？

我的建议是：把创新重点放在后者，前者作为支撑。

# 一、首先明确：我们真正想改变什么？

传统 VLA 的建模方式是：

\[ (O_t,L)\longrightarrow A_{t:t+H} \]

其中：

- \(O_t\)：当前视觉观测。
- \(L\)：语言指令。
- \(A_{t:t+H}\)：未来 Action Chunk。

也就是说，VLA 把未来机器人行为压缩为一个 Action Chunk，模型主要学习：

\[ P(A_{t:t+H}\mid O_t,L) \]

但我们认为，机器人行为实际上包含多个相互关联的信息维度：

- Action：机器人应该怎么动？
- Spatial Dynamics：物体和空间中的点将如何运动？
- Task State：当前正在执行什么子任务？
- Future State：执行动作后，世界将变成什么样？

这些信息不是独立的，它们存在复杂的时间和条件依赖。

例如，机器人执行「把桌上的杯子放进盒子」：

[DiT4DiT: 联合建模视频动态和动作以实现可泛化机器人控制 | alphaXiv](https://images.openai.com/static-rsc-4/4wHoSUtRrVzuyi3kkp-UkLNZOq25PvTyZCa4ZEiv5KblQy9PIxFk6oDimiSp8PZ33_NCGt5gO1kjKlSsn2V-ujSMFn51Qr1vvTHC1OO1pD23D6U-hdiuGSeon9ZUY0uS0ALQWDbfnQJ3D2JK9Flss3pogbdLdi90ToU36AfKVtw?purpose=inline)

[alphaxiv.org](https://www.alphaxiv.org/zh/overview/2603.10448v2?utm_source=chatgpt.com)

在这个任务中：

1. Subtask：确定当前目标是抓取杯子。
2. Point Track：预测杯子和抓取点的空间运动。
3. Action：生成机器人抓取和移动轨迹。
4. Future State：预测杯子进入盒子后的视觉状态。

传统 VLA 将这些信息隐式地压缩进 Action Prediction。

我们想做的是：

将这些不同的信息显式表示为不同的 Token Stream，让模型在预测 Action 的同时，也预测与 Action 相关的其他结构化信息。

这才是我们研究的出发点。

# 二、我建议将核心架构定义为：Temporal-Depth Factorized VLA

暂时给它一个工作名称：

TDF-VLA：Temporal-Depth Factorized Vision-Language-Action Model

中文可以叫：

时序-深度因子化多流视觉语言动作模型。

这里的 Depth 指的是同一个时间位置上不同 Token Stream 的建模维度，不是 3D 空间深度。

我认为这个名称比简单叫 Multi-stream VLA 更准确，因为真正需要强调的是 Temporal Transformer 和 Depth Transformer 的二维因子化建模。

## 1. 核心表示：将未来预测组织成二维 Token Grid

假设我们需要预测未来 \(T\) 个决策时间窗口。

每个时间窗口不再只有 Action Token，而是有 \(K\) 种不同类型的 Token。

例如：

\[ Z= \begin{bmatrix} z_{1}^{A} & z_{1}^{P} & z_{1}^{S} & z_{1}^{V}\\ z_{2}^{A} & z_{2}^{P} & z_{2}^{S} & z_{2}^{V}\\ \vdots & \vdots & \vdots & \vdots\\ z_{T}^{A} & z_{T}^{P} & z_{T}^{S} & z_{T}^{V} \end{bmatrix} \]

其中：

| 维度  | 表示                 | 作用             |
| ----- | -------------------- | ---------------- |
| \(A\) | Action Tokens        | 预测机器人动作   |
| \(P\) | Point Track Tokens   | 预测空间运动     |
| \(S\) | Subtask Tokens       | 预测任务阶段     |
| \(V\) | Future Visual Tokens | 预测未来视觉状态 |

## 我们希望构建的二维 Token Grid

横轴是 Stream / Depth，纵轴是 Temporal Horizon。

Time

Action

Point

Subtask

Future

t=1

Token

Token

Token

Token

t=2

Token

Token

Token

Token

t=3

Token

Token

Token

Token

t=4

Token

Token

Token

Token

Action

Spatial

Task

Future State

这里有一个关键设计：

每一种 Stream 都有自己的 Tokenizer 和 Codebook，而不是把所有信息强行压缩到同一个 Codebook。

例如：

| Stream       | Tokenizer            | Token 表示       |
| ------------ | -------------------- | ---------------- |
| Action       | Action VQ / RVQ      | Action Chunk     |
| Point Track  | Trajectory VQ        | 一组点的未来轨迹 |
| Subtask      | Text Tokenizer       | 当前任务阶段     |
| Future Image | Image VQ / Latent VQ | 未来视觉状态     |

各个 Codebook 可以具有不同的词表大小，但是它们的 Embedding 都映射到统一的 Transformer Hidden Dimension。

这就形成了一个异构 Token Grid（Heterogeneous Token Grid）。

我认为这本身就是一个值得研究的表示设计。

## 2. Temporal Transformer：沿时间维度建模

Temporal Transformer 负责建模不同预测时间窗口之间的依赖关系。

\[ h_t= T_{\mathrm{temp}}(Z_{<t},O,L) \]

它需要理解：

- 前一个时间窗口机器人做了什么？
- 物体之前如何运动？
- 当前任务处于什么阶段？
- 之前预测的未来状态是什么？

这部分可以直接借鉴 Moshi 的 Temporal Transformer。

它的核心职责是建模跨时间的依赖。

## 3. Depth Transformer：沿 Stream 维度建模

这是我们最重要的架构创新候选。

在当前时间窗口 \(t\)，模型需要预测：

\[ Z_t= [z_t^A,z_t^P,z_t^S,z_t^V] \]

Depth Transformer 不同于普通的多个独立输出 Head。

它需要让不同 Stream 之间存在显式的条件依赖：

\[ P(Z_t\mid Z_{<t}) = \prod_{k=1}^{K} P(z_t^k\mid Z_{<t},z_t^{<k}) \]

也就是说：

- 当前 Stream 的预测可以使用过去所有时间窗口的信息。
- 当前 Stream 的预测还可以使用当前时间窗口已经生成的其他 Stream。

这使得不同输出不只是共享一个 Backbone，而是具有明确的联合自回归关系。

## TDF-VLA 的模型结构

Observation Encoder + Language Encoder

Image / Proprioception / Language

Temporal Transformer

Across-time modeling

Depth Transformer

Across-stream modeling

Action VQ Head

Point Track VQ Head

Subtask Head

Future Visual VQ Head

Joint Multi-stream Prediction

这里还需要澄清一个技术细节：Moshi 的 Depth Transformer 本身也是自回归的。

因此，严格采用 Moshi 的设计时，不是一个普通前向就无条件同时产生所有 Stream，而是在每个 Temporal Step 内，按照 Depth 顺序生成不同 Stream 的 Token。

我们也可以设计并行 Depth Head，但那将是另一种架构，不能再直接称为 Moshi 式的 Depth Autoregression。

# 三、真正的创新点应该怎样定义？

我建议把创新分成三个层次。

核心创新 1

Heterogeneous Multi-codebook Representation

使用不同的离散 Tokenizer / Codebook，将 Action、Spatial Dynamics、Task State、Future Visual State 编码成异构 Token Stream，并组织成共享的时序表示。

核心创新 2

Temporal-Depth Factorized Autoregressive Modeling

通过 Temporal Transformer 建模跨时间依赖，通过 Depth Transformer 建模同一时间窗口内不同异构 Stream 的条件依赖。

应用创新 3

Joint Predictive Action Modeling

将 Action Prediction 与 Spatial / Task / Future State Prediction 联合起来，使模型学习动作、空间变化和任务状态之间的结构化关系。

其中，前两个才是我们应该重点主张的架构创新。

第三个更多是这个架构在 VLA 上的应用价值。

## 四、我建议不要把 CoT 作为第一版的核心 Stream

这里我想提出一个不同于你原始设想的建议。

你的原始五个 Stream 是：

1. Action。
2. 2D Point Track。
3. Subtask。
4. CoT。
5. Future Image。

我建议第一版改成四个：

\[ \boxed{ \{Action, Spatial, Task, Future\} } \]

其中不单独设置自由形式的 CoT Stream。

原因是 CoT 存在几个问题：

- 很难定义稳定、客观的 Token 监督。
- 不同模型生成的推理文本可能存在很大差异。
- 自由文本可能引入大量冗余 Token。
- 文本推理正确，不代表动作一定正确。
- 可能让模型学会复述训练集中的语言模式，而不是真正提高物理预测能力。

如果确实希望包含推理，我建议将其结构化成：

- Target Object。
- Contact State。
- Motion Intention。
- Goal Position。
- Task Completion State。

这些信息比自由形式 CoT 更容易获得监督，也更容易评估。

例如：

原始 CoT

我应该先抓住杯子的把手，然后将杯子移动到盒子上方，最后松开夹爪。

结构化 Task / Spatial Tokens

Target: Cup

Contact: Handle

Intent: Grasp

Goal: Inside Box

这样做既保留了任务推理信息，也减少了语言生成的负担。

# 五、这里有一个非常关键的问题：Stream 的生成顺序是什么？

我认为这会直接影响架构能否真正成立。

不能仅仅为了方便，将 Action、Point Track、Subtask、Future Image 按任意顺序放进 Depth Transformer。

它们之间存在物理因果关系。

例如：

\[ Subtask\rightarrow Action\rightarrow Future\ State \]

但是 Point Track 比较特殊。

它有两种不同含义：

情况 A：目标轨迹（Desired Track）

例如机器人希望物体移动到哪里。

这种情况下：

\[ Subtask\rightarrow DesiredTrack\rightarrow Action \]

情况 B：实际运动预测（Predicted Track）

例如机器人执行某个动作后，物体预计如何移动。

这种情况下：

\[ Action\rightarrow PredictedTrack\rightarrow FutureState \]

两者不能混为一谈。

因此，我建议将 Point Track 定义得更加明确：第一版先预测动作执行后物体的未来运动轨迹。

那么一个比较合理的 Depth 顺序是：

## 推荐的因果生成顺序

1. Subtask Token

当前需要完成什么目标？

2. Action Token

为了完成目标，机器人应该怎么动？

3. Point Track Token

执行动作后，物体和关键点会怎么运动？

4. Future Visual Token

最终会形成什么视觉状态？

当然，这只是一个可检验的因果分解，不代表唯一正确的顺序。

尤其是 Subtask 如果在一个 Action Chunk 中持续不变，不需要每个时间步重复生成，而应该使用持续有效的状态 Token 或事件驱动更新。

这也是我们需要研究 Multi-rate Token Modeling 的原因。

# 六、怎样证明这真的是一个架构创新，而不是普通 Multi-task Learning？

这是我认为最重要的实验设计问题。

假设我们使用相同的训练数据、相同的输出监督和相近的模型参数量。

应该比较以下五种模型：

| 模型            | 结构                               | 要验证的问题         |
| --------------- | ---------------------------------- | -------------------- |
| Baseline        | Action-only VLA                    | 基准性能             |
| Multi-head VLA  | Shared Backbone + 多个独立 Heads   | 额外监督是否足够     |
| Interleaved VLA | 单一 1D Transformer 交错生成 Token | 2D 因子化是否必要    |
| MoT-VLA         | 多 Expert + Joint Attention        | 独立专家是否更合适   |
| TDF-VLA         | Temporal + Depth Transformer       | 我们的架构是否有优势 |

其中最重要的是第二个和第五个对比。

因为如果 Multi-head VLA 和 TDF-VLA 的表现完全相同，就很难说明 Depth Transformer 带来了实质性收益。

还应该做两个关键消融：

消融一：去掉 Depth Transformer。

改成 Temporal Transformer + 独立输出 Head。

观察不同 Stream 之间的联合预测能力是否下降。

消融二：去掉独立 Codebook。

改成所有信息使用共享 Tokenizer，观察异构 Tokenization 是否真的有帮助。

最终需要证明：

1. 多流监督比单纯 Action Prediction 更有帮助。
2. 独立 Codebook 比简单共享表示更有效。
3. Depth Transformer 比独立 Head 更能建模不同 Stream 的关系。
4. Temporal-Depth Factorization 比普通 1D AR 更适合长时序、多输出 VLA。

# 七、最后，我建议将我们的创新点暂时定稿为这个版本

推荐的研究定位

# Temporal-Depth Factorized Multi-stream VLA

A Hierarchical Autoregressive Architecture for Joint Prediction of Robot Actions and Future World Dynamics

核心思想

现有 VLA 通常将机器人未来行为表示为单一 Action Chunk，难以显式刻画动作、空间动态和任务状态之间的结构化依赖。

我们提出一种 Temporal-Depth Factorized VLA，将未来机器人行为表示为由异构 Codebook 构成的多流 Token Grid。Temporal Transformer 建模跨时间的动态依赖，Depth Transformer 建模同一时间窗口内不同信息流之间的条件依赖，从而实现 Action、Spatial Dynamics、Task State 和 Future Visual State 的联合自回归预测。

预期贡献

1. 一种异构多 Codebook 的 VLA 输出表示。
2. 一种 Temporal-Depth 二维因子化自回归 Transformer。
3. 一种 Action 与未来物理状态联合预测的 VLA 学习框架。

英文可以进一步浓缩成一句：

> We propose a temporally and depth-factorized autoregressive VLA that jointly models heterogeneous action, spatial, task, and future-state token streams using modality-specific codebooks.

我的最终建议：先把创新点限定为「异构 Codebook + Temporal-Depth Factorized Autoregression」，不要一开始把 CoT、多模态、多任务、世界模型全部当成创新点。

这样后续检索就能非常精准地围绕三个问题展开：

1. 是否有人已经将不同物理意义的输出编码成独立 Codebook，并构建统一 Token Grid？
2. 是否有人已经在 VLA 中使用 Temporal × Stream 两个维度的因子化自回归 Transformer？
3. 是否有人已经通过 Depth Transformer 显式建模 Action、Spatial Dynamics、Task State 和 Future State 之间的联合条件分布？

这三个问题才真正决定我们是否有架构层面的创新。

## 5. 我们的创新点究竟还剩多少？

这是最重要的判断。

我认为需要把「新颖性」分成三个层次。

### 层次 A：不能再主张的创新

以下说法现在都不够新：

| 原来可能的创新表述                            | 为什么不够新           |
| --------------------------------------------- | ---------------------- |
| 同时预测 Action 和 Point Track                | ModAR、TrAct 已涉及    |
| 同时预测 Action 和 Future Image               | ViPRA、PhysGen 已涉及  |
| 同时预测 Subtask、CoT 和 Action               | G0.5 已涉及            |
| 使用多个独立 Codebook                         | M²Tok 已涉及           |
| 使用统一 Transformer 处理 Text、Image、Action | MM-ACT 已涉及          |
| 多流之间通过 Attention 交互                   | Flex-π、HiF-VLA 已涉及 |

这些可以是我们的设计组成部分，但不能作为核心贡献。

### 层次 B：有潜力的核心创新

我认为真正有潜力的是以下三个点。

创新点 1：Heterogeneous Temporal-Depth Token Grid

不同模态不仅具有独立 Codebook，还被组织在一个统一的二维时序结构中。

\[ Z\in\mathcal C_A\times \mathcal C_P\times \mathcal C_S\times \mathcal C_V \]

每个时间窗口都有相应的多模态 Token 表示。

它与 G0.5 的单一序列、MM-ACT 的并行 Masked Prediction、ModAR 的逐模态 Diffusion 有不同的组织方式。

创新点 2：Temporal-Stream Factorized Autoregression

将联合概率分解为两个条件建模层次：

\[ \boxed{ P(Z_{1:T}\mid O,L) = \prod_t \prod_k P(z_{t,k}\mid Z_{<t},z_{t,<k},O,L) } \]

这才是我们从 Moshi 借鉴并迁移到 VLA 的核心机制。

与普通多头模型不同，Stream Transformer 可以明确学习：

- Subtask 与 Action 之间的依赖。
- Action 与 Point Track 之间的依赖。
- Point Track 与 Future State 之间的依赖。

同时，Temporal Transformer 负责学习这些信息在不同预测时间窗口之间的变化。

创新点 3：Multi-rate Heterogeneous Stream Modeling

这是我认为可以进一步发展成真正有价值的技术贡献。

VLA 中不同信息流的时间尺度并不一致：

- Action：高频。
- Point Track：中频。
- Subtask：低频。
- Future State：按照关键时刻或预测 Horizon 产生。

如果我们能够让 Temporal Transformer 建模统一的物理时间轴，同时让 Depth Transformer 管理不同模态的 Token 块和条件依赖，那么这就不只是把 Moshi 搬到机器人领域。

它解决了一个新的问题：

如何在统一的自回归模型中建模不同时间尺度、不同语义结构的机器人状态。