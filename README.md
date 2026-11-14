# TD-VLA

基于 TurboVLA 并行 action chunk 的多流 VLA 研究：同步输出动作、双臂夹爪二维目标轨迹及任务内阶段。
第一版共享 decoder 使用全双向跨流注意力，不增加跨窗口历史模块。

P1 工程与双机环境已通过；P2 六任务小样本及原版动作模型诊断训练/恢复已验收，
见 [P2 验收记录](docs/setup/p2-validation.md)和[复现说明](docs/setup/p2-pilot.md)。正式预训练基线尚未训练。
研究计划见 [项目路线图](docs/project-roadmap.md)。

## 本机开发

安装 uv 0.11.28，然后：

```bash
uv sync --locked --python 3.10.19
uv run --locked pytest -q
uv run --locked ruff check src scripts tests
```

核心包不依赖 GPU 或仿真器。Python 使用 uv 管理版本；不需激活 Conda。

## 远端环境

详见 [uv 双机环境说明](docs/setup/uv-environments.md)。策略和仿真是两个独立的 uv 项目，
分别保存锁文件，不要将两个环境合并，也不要直接运行上游 Conda/pip 安装脚本。

上游版本见 `configs/upstreams.json`；资产来源见 `configs/assets.json`。
第三方代码和大型产物不随本仓库分发。
