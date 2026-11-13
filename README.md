# TD-VLA

基于 TurboVLA 并行 action chunk 的多流 VLA 研究：同步输出动作、双臂夹爪二维目标轨迹及任务内阶段。
第一版共享 decoder 使用全双向跨流注意力，不增加跨窗口历史模块。

当前处于 P1 工程与环境搭建；运行结果见 [执行记录](docs/setup/p1-execution.md)。
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
