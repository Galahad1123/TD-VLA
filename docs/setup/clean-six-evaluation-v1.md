# 六任务 Clean 闭环评测（45,000 步）

本轮按用户要求，每任务 20 个有效测试场景，共 120 回合。使用 `td-vla-clean-full-v1/step-45000.pt` 的在线模型参数，不使用 EMA。加载时严格匹配模型键、训练配置签名和数据划分校验值。

- 任务：beat_block_hammer、handover_block、lift_pot、move_stapler_pad、place_object_basket、stack_blocks_two。
- 使用固定 RoboTwin 版本的 demo_clean、aloha-agilex、官方 `take_action`、成功判定和步数上限。
- 每任务从测试种子 100000 开始；先回放专家，仅专家可解场景进入策略评测。专家失败单独记录，不计入策略分母；策略失败不得换种子删除。
- 预测 50 步，执行 50 步后重新观测，无动作集成。每个 episode 重置动作缓存。
- 输入三视角 RGB、当前关节状态及训练时固定任务指令。PIL bilinear 缩放与 ImageNet 归一化完全沿用训练；关节顺序、训练集 min/max 与夹爪阈值复用数据模块。
- 模型同时输出 Action、Track、Stage，仅 Action 驱动机器人；每次请求保存 Track、Stage 和往返耗时。
- 固定指令与 20 次规模属于本项目子集评测，不能标为官方 100 次完整协议或语言泛化评测。
- 两张卡各运行一个模型服务与一个仿真进程，各串行处理三个任务；服务仅监听本机。

入口：`scripts/evaluate_robotwin.py server`（policy uv 环境）与 `evaluate`（sim uv 环境）。使用 `--help` 查看参数。实验路径 `logs/td-vla-clean-step45000-eval20-v1/`。逐任务保存配置、episode 结果、专家筛选失败、辅助预测与状态；汇总结果在运行完成后生成。异常使该任务失败并保留最小证据，不能静默记作正常完成。

验证：轻量测试覆盖预处理、动作块切换及跨回合 reset；启动前检查模型严格加载及真实仿真端到端通信。正式运行固定代码提交并记录命令及 PID。结束清理本次试运行和冗余日志，保留结果、来源、配置与未解决故障证据。
