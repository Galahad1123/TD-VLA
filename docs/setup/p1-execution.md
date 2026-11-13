# P1 执行记录

目标：工程骨架、uv 开发/策略/仿真三环境、双机同步及运行验收。

- 执行分支：setup/p1-uv；新仓库无既有代码分支，直接在用户指定目录初始化。
- 已通过授权权限机制初始化 Git，未删除受保护元数据。
- 远端清华源连通，GitHub ls-remote 连通；实际安装及 GPU 验收待执行。
- Python 使用 uv 管理版本，不依赖现有 Conda。
- 初始兼容基线：torch 2.7.1 + torchvision 0.22.1 + cu128；以运行证据判断兼容性。
- 不执行正式训练和批量数据生成。开发调试产物不等同于正式实验。

## 执行中的调整

- Python 固定为本机已有的 uv CPython 3.10.19；远端 GitHub 下载缓慢，使用同版本中转包，SHA256 为 `b26ea5ba4050298b237d4e15baa82601a964e95884724170e3942c465b8995da`。中转仅解决本次部署，常规安装仍用 uv python install。
- 远端 uv 由现有 Python 引导下载清华源的 uv wheel，不创建 Conda 项目环境。
- 上游源码通过 Git bundle 中转；保留未完成检出在数据盘 cache，正式 external 路径为固定提交。
- 容器已删除 CUDA apt 源配置；使用 dpkg --force-confmiss 恢复，安装 CUDA 12.8 用户态工具链，未替换驱动。
- numcodecs 0.15.1 不支持 Python 3.10，解析失败后改为 0.13.1，锁定成功。
- SAPIEN 强制依赖 opencv-python；sim 只安装该包，避免与 headless 版本覆盖同一 cv2 命名空间。policy 独立使用 headless。
- 大型 HF 文件优先镜像；串行速度低，改为 aria2 多连接下载。完整校验回执写数据盘。
- DINOv3 官方配置镜像返回 HTTP 403，已询问已有授权路径；官方权重推理尚未验收。
- 工程首次提交 `6581c96` 已推送 GitHub `setup/p1-uv`，远端已从该仓库 fetch；后续修正会追加提交。
- 原始三个 Markdown 文件使用 CRLF，保留原文，不为消除初始提交的空白提示重写用户材料。
- Git 忽略规则调整为根目录产物路径，避免误忽略 `src/td_vla/data`。
- 资产下载增加最小模式：读取官方 ZIP 的范围，仅提取 aloha-agilex、048_stapler、公共 JSON；保持资产字节不变，记录 CRC/SHA256。Clean 模式不下载随机背景纹理。完整资产下载仍保留为单独入口。

## 独立代码审查与修正

已完成一次只读独立审查，无 Critical。Important 已处理：

- 三份锁文件的一致性：根项目工具组变更导致子环境锁过期，实际复现 --check 失败后更新；新增 scripts/check_locks.sh，三项目检查通过。
- 存储脚本此前静默接受已有目录/错误链接；新增两种冲突回归测试，确认失败后修复为保留原内容并报错。
- 最小资产入口此前只按文件存在复用本地 ZIP；现在要求来源、revision、文件名及 SHA256 全部匹配。五种来源/内容场景测试通过。
- 完整资产说明补上上游配置模板生成命令。
- 根产物软链接的 Git 忽略回归测试先失败后通过；src/td_vla/data 不再被误忽略。

当前本机测试：9 passed；ruff 通过；三份 lock --check 通过。

审查列出的 Minor 暂留：上游 fetch 首次中断后的无 HEAD 目录需要人工检查后恢复；随机模型测试权重写入中断时需清理该测试缓存后重跑。二者不影响已完成安装，但不能把它们描述为完全自动恢复。

## GPU 与仿真运行证据（继续验收中）

- 两卡 CUDA BF16 SDPA 前后向通过：logs/p1/gpu.json。
- 完整宽度 TurboVLA Transformer 计算图（随机测试权重、简化测试词表）前后向通过，输出 [1,50,14]，BERT 无梯度，其余路径梯度有限：model-graph.log。该结果不代表官方策略推理或控制效果。
- 上游服务/客户端跨 policy、sim 环境通信通过：websocket-client.log。
- cuRobo 原生扩展编译成功；Open3D wheel 实际依赖 ipywidgets，已在锁文件补齐，uv pip check 再验证通过。
- move_stapler_pad、Clean、seed 0 专家成功，执行过经 WebSocket 返回的保持动作：episode/episodes.json。旧 OIDN 2.0.1 报错且图像有噪点，正在升级支持 Blackwell 的 OIDN 2.3.3，不能据此认为渲染链路完全验收。
- NCCL 默认初始化挂起；仅 NCCL_CUMEM_HOST_ENABLE=0、仅 loopback、仅关闭 P2P 均未解决。仅 NCCL_IB_DISABLE=1 时 all-reduce、DDP 更新及两卡参数一致性检查通过：ddp-no-ib.log。单机环境入口采用该设置，不影响 GPU 计算，不宣称验证过多节点 IB。
- 官方策略检查点已下载并记录 revision/校验；其 state_dict 包含 BERT 和 DINOv3 权重。独立 DINOv3 基础模型仍因镜像授权限制/官方网络失败未验收。

## 最终验收（2026-11-14）

以上“继续验收中”段落保留故障定位历史；最终状态以 [验收记录](p1-validation.md) 为准。

- 最终 env.sh 下 DDP 验收通过，ddp-final.log。
- OIDN 2.0.1 的设备初始化检查失败；固定 2.3.3 后 CUDA 去噪和光追检查通过。
- 修复后的 Clean seed 0 专家回合及保持动作通信通过，episode-final/episodes.json；没有 OIDN 错误。
- cuRobo 恢复版本标签后重建为 0.7.8，pip check 通过；后续脚本核验 SHA 后提供固定构建版本。
- 关键日志、三视角截图回传本机 logs/p1，依赖补丁、OIDN 来源校验及单机 NCCL 设置均已落档。
- 本阶段不以官方检查点策略的闭环表现作为通过条件；DINOv3 授权基础模型推理尚未验收。
