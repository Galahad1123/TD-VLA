# uv 双机环境

## 版本与边界

- uv 0.11.28，uv 管理 CPython 3.10.19。
- 根项目为轻量开发环境；envs/policy 与 envs/sim 分别拥有独立 uv.lock。
- 策略和仿真均使用 torch 2.7.1、torchvision 0.22.1、cu128。
- 不直接安装 TurboVLA robotwin extra：它约束 torchvision 0.21.0；本项目显式列出所需依赖。
- RoboTwin 保持 sapien 3.0.0b1、mplib 0.2.1。使用三视角 RGB，不需要 PyTorch3D 点云采样。
- P1 的 GPU/服务环境不承诺上游 DeepSpeed 训练入口可用；正式 DDP 训练接入属于 P4。
- cuRobo v0.7.8 单独编译，依赖全部在 sim 锁中；编译后执行 uv pip check。
  uv sync 会移除锁外的 cuRobo，所以每次 sim 同步后需要重新运行 setup_curobo.sh。

## 安装顺序

在 Ubuntu 22.04 实验机的项目根目录执行：

```bash
bash scripts/setup_storage.sh
source scripts/env.sh
uv python install 3.10.19
uv run --project . --python 3.10.19 scripts/fetch_upstreams.py
bash scripts/setup_env.sh dev
bash scripts/setup_env.sh policy
bash scripts/setup_system.sh
bash scripts/setup_env.sh sim
bash scripts/setup_curobo.sh
"$ROBOTWIN_PYTHON" scripts/patch_sim.py
# Blackwell: download the exact URL in configs/oidn.json to the data cache.
curl -fL --retry 3 -o "$TD_VLA_STORAGE_ROOT/cache/oidn-2.3.3.tar.gz" \
  https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz
"$ROBOTWIN_PYTHON" scripts/setup_oidn.py "$TD_VLA_STORAGE_ROOT/cache/oidn-2.3.3.tar.gz"
python3 scripts/download_assets.py robotwin
```

脚本不更新驱动；CUDA 12.8 编译工具链与系统 CUDA 13.0 并存。通用包使用清华源，
CUDA wheel 使用 PyTorch 官方专用源。Ubuntu 系统包沿用实例已有系统源。
原生编译与下载缓存都在数据盘，虚拟环境在代码目录。

下载中断会保留 `.partial`，完成后保存 SHA256 和来源凭据文件。
HF 文件按固定 revision 获取，镜像失败后回退官方。策略成品权重只作接口参考：

```bash
python3 scripts/download_assets.py reference_policy
```

镜像不能绕过上游访问授权。不得将 token 写入代码或日志。

## 验收命令

```bash
source scripts/env.sh
"$STARVLA_PYTHON" scripts/check_gpu.py
"$STARVLA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/check_gpu.py
"$ROBOTWIN_PYTHON" scripts/check_render.py --output "$TD_VLA_STORAGE_ROOT/logs/p1/render"
"$ROBOTWIN_PYTHON" scripts/check_render.py --ray-tracing --output "$TD_VLA_STORAGE_ROOT/logs/p1/render-rt"
```

实际 GPU、任务回放与通信结果另见执行记录。安装完成和基础渲染通过都不代表专家任务成功。

## 双机同步和存储

开发分支推送指定 GitHub 仓库后，远端 fetch 并检出确定 commit。正式实验运行中不更新工作区。
网络受限时可中转相同 commit 的 Git bundle，不能把文件复制当作版本一致性的证据。

远端代码 `/root/TD-VLA`，环境位于其 `.venv` 和 `envs/*/.venv`。
大文件根目录默认 `/root/autodl-tmp/td-vla`，可用 TD_VLA_STORAGE_ROOT 覆盖。
正式运行目录为 outputs/<run-id> 和 logs/<run-id>；阶段结束后在本机项目的
outputs/archive/<run-id> 保留检查点、配置、指标及环境报告的第二份副本，并校验 SHA256。
本阶段只有验收产物，已回传本机 logs/p1；完整数据集的异地存储容量须在 P3 生成前另行安排。
根据 [AutoDL 存储说明](https://www.autodl.com/docs/instance_data/)，关机保留数据、
释放实例清空数据；普通容器连续关机 15 天会释放，实例租用截止时间须以控制台为准。
本机与远端目录均不应被视为无限容量或永久可靠存储。

## 扩展运行检查

```bash
"$STARVLA_PYTHON" scripts/check_model.py --fixtures "$TD_VLA_STORAGE_ROOT/cache/model-fixtures"
"$ROBOTWIN_PYTHON" scripts/prepare_robotwin_assets.py
(cd external/RoboTwin && "$ROBOTWIN_PYTHON" script/update_embodiment_config_path.py)
"$ROBOTWIN_PYTHON" scripts/check_episode.py --output "$TD_VLA_STORAGE_ROOT/logs/p1/episode"
```

`check_model.py` 使用完整尺寸的上游计算图和随机测试权重，验证前后向及 BERT 冻结；
它不下载基础预训练模型，也不证明官方检查点兼容或控制成功率。随机测试权重不用于训练。

通信测试使用上游 WebsocketPolicyServer/WebsocketClientPolicy：
在 policy 环境启动 `scripts/check_websocket.py server`，在 sim 环境运行同脚本 `client`。
服务仅监听 127.0.0.1；返回确定性保持动作。`check_episode.py --echo-port 5694` 会将真实仿真
观测发送给该服务并执行一次保持动作，然后运行专家脚本。该流程不能标为学习策略评测。

第三方代码使用其原许可，详见 external 中上游 LICENSE、TurboVLA/third_party/licenses，
以及 cuRobo/LICENSE_ASSETS。资产清单记录来源，不授予再分发许可。

慢网络下只运行 P1 Clean 订书机任务，可用最小资产流程代替完整三压缩包：

```bash
uv sync --locked --python 3.10.19 --group asset-tools
.venv/bin/python scripts/fetch_minimal_assets.py
(cd external/RoboTwin && ../../.venv/bin/python script/update_embodiment_config_path.py)
```

该入口使用 [remotezip](https://github.com/gtsystem/python-remotezip) 范围读取官方压缩包，
保留未修改的资产文件及逐文件 CRC/SHA256；仅支持当前先导任务，不能用于宣称完整六任务资产齐备。
不要在同一资产链接上混用完整与最小安装；脚本遇到已有不同目标时会停止。

## Blackwell 兼容修复

- env.sh 默认 NCCL_IB_DISABLE=1。本容器暴露的 IB 接口使 NCCL 初始化挂起；
  关闭 IB 后双卡 all-reduce 和 DDP 更新通过。未关闭 GPU P2P，未验证多机训练。
- SAPIEN 保留 3.0.0b1，单独将 OIDN 换为 2.3.3；该版本的
  [官方发布说明](https://github.com/RenderKit/oidn/releases/tag/v2.3.3)列出 Blackwell 支持。
  按 [SAPIEN 上游问题记录](https://github.com/haosulab/SAPIEN/issues/243)替换三个库及加载路径，
  configs/oidn.json 固定下载校验值，setup_oidn.py 保留旧库和加载脚本备份。
- 每次重新安装 SAPIEN 后重新执行 patch_sim.py、setup_oidn.py；锁文件不包含 wheel 安装后的补丁。
  光追验收会检查 CUDA 去噪器初始化失败，不能仅凭图片非空判定通过。
- Git bundle 中转浅克隆时需一并保留 .git/shallow。setup_curobo.sh 先核验源码 SHA，
  再显式提供 0.7.8 给 setuptools-scm，避免仅按 SHA 获取源码时缺失 tag 造成版本号错误。

P2 补充：真实模型的大消息通信还需要 NCCL_PROTO=LL，已在 env.sh 配置。原因与复现命令见 [P2 说明](p2-pilot.md)。P1 的微型 DDP 结果不代表所有通信规模都已验证。
