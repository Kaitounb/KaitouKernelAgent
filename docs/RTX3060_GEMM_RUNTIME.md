# RTX3060 GEMM 0.1.3 运行与回退

2026-11-24：0.1.3正式包、入口及实际回退、独立正确性、完整性能资格和恢复备份均已验收。发行清单：[rtx3060-gemm-v0.1.3](../releases/rtx3060-gemm-v0.1.3/manifest.json)，production_release=true仅限下述目标和固定cuBLAS。

## 目标与输入

本包限 GPU `GPU-0140b895-e4e9-7c67-aba1-c7a953df1847`、RTX3060/SM86、driver580.95.05、Ubuntu22.04、默认170W动态时钟、准入温度<83°C。运行镜像固定 `sha256:6ce2012265c68b8553adf8d7807dcf47972a9156d1c7f0a2a64db11f9827e55a`（Torch2.14.0+cu130/CUDA13.0）。

输入为两个普通 `.npy` 文件：连续二维FP16，A[M,K]与B[K,N]，M/N/K各1..4096，有限且绝对值≤4。输出FP32[M,N]。不是通用CUDA源码执行入口；没有优化提速或单请求尾延迟保证。3070的Attention/MLA继续使用原0.1.1通道，3060未新增其生产资格。

## 在3060上执行

SSH使用本机私有连接说明，不在命令或文档中写口令。服务器登录后：

```bash
DEPLOY=/home/featurize/nka-persistent/gemm-release-0.1.3-gdjlm25g
"$DEPLOY/kernel-agent-3060" status --root "$DEPLOY/workspace/production-store"
"$DEPLOY/kernel-agent-3060" run \
  --root "$DEPLOY/workspace/production-store" \
  --a "$DEPLOY/workspace/a.npy" \
  --b "$DEPLOY/workspace/b.npy" \
  --output "$DEPLOY/workspace/my-result.npy"
```

`workspace/a.npy`与`b.npy`是已准备的公开17×23、23×19示例。自己的输入也放在`workspace`。宿主无需安装NumPy/Torch；控制器及worker都使用固定镜像。宿主launcher默认选择qualified通道；直接调用Python模块时需显式`--channel qualified`。

成功返回JSON，包含`status=PASS`、`production_release=true`、执行版本、binary SHA、fallback与run_directory。输出通过临时文件和原子替换写入。资源正忙返回BUSY/退出2；准入失败返回ERROR/非零，不写新输出。保留每次运行的execution.json、attempt记录、输入/产物SHA和有界原始stdout/stderr。错误详情见指定run目录及`launcher-records`。项目GPU锁串行化项目请求，不能阻止用户另行启动的其他GPU进程。

## 算子版本回退

```bash
"$DEPLOY/kernel-agent-3060" activate \
  --root "$DEPLOY/workspace/production-store" --version sm86-cublas-original
"$DEPLOY/kernel-agent-3060" rollback --root "$DEPLOY/workspace/production-store"
```

默认`sm86-cublas-container`失败或产物校验失败时，最多再尝试一次固定可信`sm86-cublas-original`；crash、有限CUDA后CPU超时及坏产物路径均已实测。清理未确认时停止。两种版本都使用cuBLAS，不宣称后备更快。状态目录可从对应发行catalog重建；不要混用不同channel的store。

## 应用包回退与恢复

```bash
"$DEPLOY/kernel-agent-3060-switch" preview
"$DEPLOY/kernel-agent-3060" status \
  --root "$DEPLOY/workspace/preview-recovery-store" --channel development
# 需要时使用同一preview store执行run，并显式附加 --channel development。
"$DEPLOY/kernel-agent-3060-switch" release
"$DEPLOY/kernel-agent-3060" status --root "$DEPLOY/workspace/production-store"
```

preview指向已验收0.1.3rc2；该包没有生产证书，必须显式development通道并使用独立preview store。`release`恢复0.1.3及原production store。两次切换原子替换`current`指针，限定两个已安装prefix并持GPU锁；正式包运行文件与RC2一致。

## 隔离和证据边界

可信控制器有Docker socket；worker为UID/GID65534、零cap、NoNewPrivs、只读根目录、networknone、私有PID namespace，仅挂载产物和单次输入两个只读目录；无socket/模型凭据/独立清单/参考，内存2GiB、CPU2、PID32、超时45秒及有界输出。此通道只执行已审查固定绑定。

私有独立验收16/16正确，最大误差9.536743e-6。原检查脚本错传控制容器PID，原16项PID观测全部排除；RC2与GA分别用真实存活宿主PID补充公开实测并逐文件审计。GA安装原FAIL记录保持，补充审计验证9输出11尝试、54CPU、两fault、坏产物、版本/真实包回滚及资格负例后接受；不把原FAIL改写成PASS。

性能资格为完整三AA三AB共1728段：最大单对0.657401%、最大组0.251983%、最低非退化保护0.999578594×，通过3%/1.5%/.97部署标准。指标是持续CUDA Graph批均值，不包含容器启动和端到端请求时延。

## 备份与重建

本机镜像备份`artifacts/runtime-images/rtx3060-6ce2012265c68b85.docker.tar.gz`，3150558880字节，SHA `b5ea3ed336720de0a50866a2137153471e5f2380268311d74e0027d62a52b8e0`；已在服务器对同一gzip归档执行docker image load并查询正确image ID，本机流式验证全部6层DiffID/descriptor/SHA通过。应用与RC2回退prefix备份`artifacts/runtime-images/rtx3060-0.1.3-application.tar.gz`，SHA `b2dc803c594f4541ae37b8ad26bdd28064c26318a2f9e0b491266d2f18b57d0d`，707文件逐SHA/权限校验通过，不含私有验收清单或口令。

恢复时先核对两份归档SHA，上传到已授权同目标服务器，再执行：

```bash
docker image load --input rtx3060-6ce2012265c68b85.docker.tar.gz
docker image inspect sha256:6ce2012265c68b8553adf8d7807dcf47972a9156d1c7f0a2a64db11f9827e55a --format '{{.Id}}'
# 先检查目标目录不存在；在空恢复目录解压，避免覆盖运行中的部署。
mkdir -m 700 recovery-staging
tar -xzf rtx3060-0.1.3-application.tar.gz -C recovery-staging
```

归档内recovery-manifest.json列出每个文件SHA、大小和权限；解压后逐项核对，再恢复两个gemm-release目录到`/home/featurize/nka-persistent/`原路径。工具依赖宿主`/environment/miniconda3/bin/python3`、Docker组权限和同一NVIDIA CDI配置。归档不含current软链接；在GA目录执行`ln -s "$DEPLOY/prefix" "$DEPLOY/current"`（仅在该链接不存在时），随后按上方status核验。若原实例/目录仍在，直接使用已部署入口，无需解压覆盖。应用成员校验为CPU归档验证，未声称在另一GPU上恢复运行。

当前固定wheel：`artifacts/wheels/nvidia_kernel_agent-0.1.3-py3-none-any.whl`，SHA `b3a98c60ae85ee5bf492354ce52b04599837374e9c65c152891db78305c41fed`；证书SHA `627007afe4a43206eac362923428f12f7272016635152695ba31f01291a26c81`。换GPU/驱动/镜像必须重新建立目标资格，不能继承本卡的生产结论。
