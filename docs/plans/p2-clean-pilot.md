# P2 Clean 先导数据与基线短程训练执行计划

Spec: docs/project-roadmap.md 第 5、7 节；用户授权同步推进 P2 与原版基线短程训练。

目标：可重现的小规模真实 Clean 回合、空间/阶段标注与可视化、双卡动作基线训练/恢复证据。
约束：不改任务动力学和成功函数；不以时间比例/专家调用序号定义阶段；不将未来标签输入模型；不把诊断训练当作正式对照。
执行分支 experiment/p2-clean-pilot；沿用双机主目录和已锁定环境，避免复制大型环境。

## Task 1: 原始采样与投影

Files: scripts/collect_pilot.py, scripts/fetch_minimal_assets.py, src/td_vla/annotation/geometry.py, tests/test_geometry.py。
先用已知相机坐标构造投影、背面、越界、resize 与未来窗口末端测试并观察失败，再实现。
按全局 15 个物理步采样（250 Hz -> 16.67 Hz），显式保存初始/终止帧与实际时间；主相机 RGB/深度/K/E、双腕 RGB、14 维关节状态、两指几何点、TCP、物体姿态/接触/官方成功标记。
两指几何点由碰撞网格指尖区域定义，保存 local 坐标与来源；不使用 TCP 替代。
先导每任务先取得 2 条成功回合；失败种子原样记录、不混入训练。
Expected: 三先导成功、可追溯 HDF5 与校验、投影叠图可见、几何正反变换一致。

## Task 2: 阶段与窗口

Files: src/td_vla/annotation/stages.py, scripts/annotate_pilot.py, tests/test_stages.py。
先写状态规则测试，包括掉落/无效和成功终止，然后实现任务内粗粒度规则。
先验证三个先导再覆盖另外三个任务，每任务 2 条成功小样本；不硬锁阶段数。
保存当前阶段、valid、依据，允许恢复回退；瞬时触碰不等于稳定抓持。
未来 Action/Track 采用同一 t+1..t+50 索引，轨迹投影使用当前主相机矩阵，越界/遮挡/末端各自 mask。
Expected: 轨迹可视化、阶段分布/切换图、成功终止帧保留、无时间比例特征；不确定处保留 mask 与限制记录。

## Task 3: 原版短程训练

Files: src/td_vla/data/pilot.py, scripts/train_pilot.py, scripts/check_resume.py, configs/pilot.json。
使用上游原版 TurboVLA 并行动作头，冻结 BERT，双卡 DDP，BF16，L1，AdamW，梯度裁剪及 EMA。
仅训练 episode 统计归一化，整回合分割；保存 optimizer/scheduler/EMA/RNG/step/数据指纹。
基础模型授权若仍不可用，短程诊断可显式用随机计算图权重；不能标为预训练基线训练或正式 recipe，报告限制。
Expected: 真实数据有限 loss/梯度，视觉及动作路径更新，完整 checkpoint 恢复与连续运行对照，显存/吞吐日志。

## Task 4: 验收与同步

Files: docs/setup/p2-validation.md, docs/project-roadmap.md。
运行测试与双机检查，独立代码审查一次，修复重要问题；更新文档与进度，提交并同步 GitHub 和远端。
重点审查：时间错位、未来信息泄漏、左右顺序、失败/不确定标签、恢复状态/数据划分、相机坐标与遮挡边界。
