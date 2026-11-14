# P1 验收记录

验收日期：2026-11-14（Asia/Shanghai）。P1 工程骨架与双机运行环境通过，未启动正式训练或批量数据生成。

## 部署

本机项目为 `/home/hello/lizk/vlas/TD-VLA`；实验机 `F43-6000x2` 的项目为 `/root/TD-VLA`。
代码同步分支为 `setup/p1-uv`，仓库为 `Galahad1123/TD-VLA`。运行前固定 commit，环境报告记录 SHA。
根开发环境、policy、sim 分别由 uv 0.11.28 和独立锁文件管理，Python 3.10.19，不使用 Conda 项目环境。

远端两张 RTX PRO 6000 Blackwell Max-Q，驱动 595.71.05；torch 2.7.1+cu128，
CUDA 编译工具链 12.8，cuRobo 0.7.8，SAPIEN 3.0.0b1 + OIDN 2.3.3。
上游完整提交见 configs/upstreams.json；HF revision 见 configs/assets.json；OIDN 下载校验见 configs/oidn.json。

## 结果与证据

日志相对于远端 `/root/autodl-tmp/td-vla/logs/p1`；最终验收日志、环境信息与 episode-final 图像已回传本机 `logs/p1`，不提交 Git。2026-11-14 已清理安装/调试过程日志和重复测试图像，仅保留最终验收证据。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 本机核心包、存储冲突、资产来源与 Git 忽略规则 | 9 passed | pytest；干净安装亦通过 |
| Ruff、三项目锁一致性、Shell 语法 | 通过 | check_locks.sh 等本机命令 |
| 双卡 BF16 SDPA 前后向 | 通过 | gpu.json、ddp-final.log |
| NCCL all-reduce、DDP 更新、两卡参数一致性 | 通过 | ddp-final.log |
| 上游完整宽度 Transformer 随机权重前后向 | 输出 1×50×14，BERT 冻结，梯度有限 | model-graph.log |
| SAPIEN CUDA 去噪器、光追图像 | 通过，旧版失败检查在新版转为通过 | render-before-fix.log、render-rt.log |
| policy 与 sim 跨环境 WebSocket | 通过 | websocket-final.log |
| Clean move_stapler_pad 专家回合 | seed 0 成功，实际执行通信返回的保持动作 | episode-final.log、episode-final/episodes.json |
| 原生 cuRobo 构建和依赖一致性 | 0.7.8，uv pip check 通过；专家任务实际调用规划器 | curobo-rebuild.log |

## 兼容处理及复现

完整步骤见 [uv 双机环境](uv-environments.md)。两个必要的硬件修复是单机关闭 NCCL IB 路径、
升级 SAPIEN 附带 OIDN。未更换驱动，未修改上游任务成功条件或 Clean 配置。
原生扩展及安装后补丁不由 uv.lock 独立表达，重建必须执行 setup_curobo.sh、patch_sim.py、setup_oidn.py。
第一次构建和下载可能较慢；本次使用相同 Git 提交和已校验官方下载文件中转。

根分区验收时约 20 GB 已用、11 GB 可用；数据盘约 25 GB 已用、1.1 TB 可用。
环境在根分区，大型缓存、权重、资产和日志在数据盘。阶段产物的第二份副本约定存放本机
outputs/archive/<run-id>；当前只有烟雾测试日志和图片，已存入本机 logs/p1。
已核对 [AutoDL 持久化规则](https://www.autodl.com/docs/instance_data/)：关机保留，释放删除；
实例具体到期日需看控制台，P3 前仍需安排完整数据集的异地存储容量。

## 能力边界

- 随机模型计算图验证与专家回合分别验证计算、通信和仿真，不能作为官方策略复现分数。
- 官方参考检查点已下载并校验；独立 DINOv3 模型镜像返回授权错误，官方权重推理尚未验收。
- 仅准备先导任务所需最小资产，尚未生成六任务 Clean 数据集、轨迹或阶段标注。
- 当前未接入正式训练入口或 DeepSpeed。P4 仍需完成训练、断点恢复和模型服务集成。
- 上游可打印缺少 PyTorch3D 的提示；本阶段只用 RGB，不运行该可选点云路径。
