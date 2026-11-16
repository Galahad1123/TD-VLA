# TD-VLA

基于 TurboVLA 并行 action chunk 的多流 VLA 研究：同步输出动作、双臂夹爪二维目标轨迹及任务内阶段。
第一版共享 decoder 使用全双向跨流注意力，不增加跨窗口历史模块。

P1 工程与双机环境已通过；P2 六任务小样本及原版动作模型诊断训练/恢复已验收，
见 [P2 验收记录](docs/setup/p2-validation.md)和[复现说明](docs/setup/p2-pilot.md)。正式实验结果统一维护在[实验结果目录](docs/experiments/README.md)。
旧版辅助标签已清理；[新标注约定](docs/data/p2-annotation-rules.md)已实现，六任务 v4 重标注结果见[验收报告](docs/setup/p2-annotation-v4-validation.md)。
六任务各 50 条入选专家轨迹已就绪，300 条均有正常 Track/Stage 标签；歧义回合已按授权补采替换，训练读取入口见[入选版本](docs/setup/clean-six-50-selection.md)。
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

TD-VLA 第一版共享多流模型、加权辅助损失、270/30 数据划分与双卡诊断训练入口已实现。用法和验收边界见 [模型实现验收](docs/setup/td-vla-v1-validation.md)。
