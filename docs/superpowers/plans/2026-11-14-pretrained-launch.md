# TD-VLA 正式初始化与训练启动计划

> 使用 executing-plans 在当前 feature/td-vla-v1 分支执行。用户已授权正式准备、短程验收及45,000步启动，不另行重复确认。

目标：冻结BERT，基础DINOv3 ViT-L + GroundingDINO语言/融合初始化，每卡64、总batch128、45,000步、每5,000步完整保存。

规格：docs/model-architecture-v1.md、docs/data/p2-annotation-rules.md、docs/project-roadmap.md。

- [ ] 获取官方BERT/DINOv3/GroundingDINO权重，固定revision、SHA256和来源。镜像优先，不使用RoboTwin成品策略。
- [ ] 测试优先实现严格初始化映射、权重清单校验与学习率调度；确认缺失/不匹配张量不能静默跳过。
- [ ] 扩展正式训练配置、定期固定验证子集、状态和完整恢复；保持诊断入口可用。
- [ ] 正式预训练权重每卡64双卡短测，验证梯度、损失、耗时、显存与恢复；记录检查点空间需求。
- [ ] 全部验证通过后后台启动45,000步，报告PID、日志和预计完成时间；清理临时产物，保留正式运行及必要证据。

执行记录：
- 已有架构和预算直接执行。学习率起点5e-5，1000步线性warmup，之后cosine到5e-6；AdamW betas=(0.9,0.95)、weight_decay=0.01、EMA=0.999、clip=1。总步数始终以配置45,000确定，短测仅限制停止步数，不改变调度曲线。
- 权重访问阻塞：hf-mirror官方DINOv3 ViT-L revision ea8dc2863c51be0a264bab82070e3e8836b02d51 返回403，远端无HF token；已异步请求用户完成获批账号登录，不绕过访问限制。BERT和融合权重准备不依赖此项。

- 用户补充登录后仍返回GatedRepo，已提交官方申请。按用户要求检索其他渠道，找到ModelScope公开分发；DINOv3与GroundingDINO完整权重哈希均与官方公开LFS元数据一致。下载使用无认证公开接口，保留许可证及实际分发来源，原访问限制未被修改。
- 65项远端完整回归通过；新增周期验证边界的双卡诊断2→4步精确恢复通过。
- 真实权重已齐全，正在进行每卡64的正式初始化16步验收。
