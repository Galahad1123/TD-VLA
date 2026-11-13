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
