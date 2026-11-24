# SM86 GEMM 开发部署与回退

此入口用于离线验收／开发部署演练，每次调用创建独立受限容器。默认cuBLAS，保留原4-warps搜索参照；当前无性能晋升，独立验收和完整生产发布仍未通过。数据输出／CLI状态明确`development-only`和`production_release=false`。

最新开发交付已扩展为[dev5双binary离线评测](DUAL_BINARY_EVALUATION.md)，原[dev4入口](OFFLINE_OPTIMIZER.md)证据继续保留，新prefix／空cwd、预算／拒绝和版本回滚已验证；默认cuBLAS与两个allowlist版本不变。本文dev2命令／证据作为原运行包复现记录保留，最终dev4 wheel和差异验收见上述链接。

## 已声明的环境

目标为已有Docker/NVIDIA runtime的RTX3070/SM86、Linux/WSL2、Python3.12.3、NumPy2.5.3（本次实际版本）和固定镜像`sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a`。镜像内PyTorch2.14.0+cu130、CUDA13.0，Torch extension使用现有成熟绑定。镜像缺失时明确失败，不自动拉取其他版本。[实际wheel](../evidence/20261122-p3-02-install-rollback-dev2/remote/dist/nvidia_kernel_agent-0.1.0.dev2-py3-none-any.whl) SHA256为`877f7d6f31fcb2f6830397a3f6846831b009411559c7ffd7d06b738e3a2f46ea`。wheel包含两个已审查Linux二进制、合同、运行适配代码与NVIDIA版权／许可，不包含镜像和cuBLAS库。

输入为连续row-major FP16 NPY文件，A[M,K]/B[K,N]，维度1..4096、有限值且绝对值≤4；输出为FP32 [M,N] NPY。输入header在分配前校验，非法请求在创建GPU worker前拒绝。实际已测shape范围以对应报告为准，不把合同上限当作穷尽测试。

## 安装与运行

使用已授权的3070环境，安装到新目录，不修改原虚拟环境。需要目标原有NumPy及pip/setuptools；包构建／安装均禁网络：

```sh
NKA_PY=/home/dministrator/cuda-lab/.venv/bin/python
NKA_PREFIX=/tmp/nka-runtime-installed
NKA_STORE=/tmp/nka-runtime-store
# wheel路径由实际构建结果给出；prefix/store均使用未存在的新目录。
"$NKA_PY" -m pip install --no-index --no-deps --target "$NKA_PREFIX" /path/to/nvidia_kernel_agent-0.1.0.dev2-py3-none-any.whl
PYTHONPATH="$NKA_PREFIX" "$NKA_PY" -m kernel_agent.cli init --root "$NKA_STORE"
PYTHONPATH="$NKA_PREFIX" "$NKA_PY" -m kernel_agent.cli run --root "$NKA_STORE" --a /path/a.npy --b /path/b.npy --output /path/result.npy
PYTHONPATH="$NKA_PREFIX" "$NKA_PY" -m kernel_agent.cli status --root "$NKA_STORE"
```

工作区构建命令为`python -m pip wheel --no-deps --no-build-isolation --wheel-dir <dist> .`。NumPy是可信控制侧输入／输出解析依赖；pip --no-deps不会静默下载它。每次run返回实际版本／binary SHA／回退原因／attempts／证据目录。NPY输出通过临时文件和原子replace写入指定路径。

## 版本与实际回退

两个开发部署版本均只启用cuBLAS：`sm86-cublas-container`（66fcc8…，容器新编译）、`sm86-cublas-original`（93fb2f…，原可信产物）。版本manifest绑定合同／image／Torch／CUDA／SM86和binary SHA；活动版本缺失、损坏或manifest不匹配时，先校验独立保存的旧cuBLAS fallback。运行异常／超时且清理已确认后最多新起一次fallback worker；fallback也失败则返回错误。清理无法确认时停止，不启动后续GPU尝试。

```sh
PYTHONPATH="$NKA_PREFIX" "$NKA_PY" -m kernel_agent.cli activate --root "$NKA_STORE" --version sm86-cublas-original
PYTHONPATH="$NKA_PREFIX" "$NKA_PY" -m kernel_agent.cli rollback --root "$NKA_STORE"
# 仅开发恢复演练：有限CUDA工作后host异常／等待超时，再由独立旧cuBLAS新容器恢复。
PYTHONPATH="$NKA_PREFIX" "$NKA_PY" -m kernel_agent.cli run --root "$NKA_STORE" --a /path/a.npy --b /path/b.npy --output /path/result.npy --inject-failure crash
```

activate/rollback使用文件锁、fsync和原子版本指针切换；失败回退不改活动版本，也不改变搜索best。版本目录、发布指针、参考／裁决／模型凭据不挂入候选容器。用户只激活包内已验证开发版本，不能靠自改manifest注册未知二进制。

## 运行边界

候选容器只有两份宿主只读bind（固定artifact/worker与该次输入），无宿主可写mount、无外网、无模型凭据、CapEff0／NoNewPrivs1、UID65534。outputs为8MiB tmpfs，tmp128MiB，PID32、CPU2、CPU内存2GiB且swap同上限；容器文件大小上限16MiB，Docker log-driver none。stdout上限由可信shape决定，stderr64KiB；限额前缀和截断原因留档，不保存无限洪泛日志。

控制侧将管道作为有界字节流，先查长度再解析flat JSON及非pickle NPY；不读取候选指定的宿主路径。dev2在启动client前创建日志；写日志／close错误也执行清理，清理异常保留未知状态并停止后续GPU尝试。超限／超时先关闭client，再清理本次UUID容器；确认器区分“No such object”和daemon错误，超时会记录host PID及清理结果。

GPU显存不是上述CPU内存配额。当前没有无限GPU kernel取消、驱动重置、共享内核／驱动恶意代码隔离结论；不把有限CUDA后host超时恢复推广到GPU挂死。dev2不开放自由模型源码。仓库后续P3-02 builder已完成有界产物／日志验收；P3-01审查后的真实源码也已独立验证且按新A/A拒绝，不改变dev2 wheel或部署allowlist。没有新性能、在线推理延迟或正式生产发布结论。

[有界runtime回归](../evidence/20261122-p3-02-bounded-runtime-confirmation-fix/REPORT.md) · [安装／回退／回滚](../evidence/20261122-p3-02-install-rollback-dev2/REPORT.md) · [上游来源与具体缺口](../provenance/adaptations/20261122-p3-02-bounded-runtime.md)

## 后续P3-02闭环验收

同日有界builder及审查源码→独立GPU验证已完成，[报告](../evidence/20261122-p3-02-bounded-builder/REPORT.md)。P3-02按声明环境的进程／文件／网络／资源／恢复范围验收；dev2 wheel内容及部署allowlist未变，新builder从仓库源码入口复现。后续真实模型源码结构变化／新证据拒绝已按[P3-01限定范围](P3_01_SOURCE.md)验收；正式生产发布和封存独立验收仍未通过。
