# 3070 GEMM 0.1.2rc1 预览运行

2026-11-24：已完成实际安装与恢复验证，**GEMM 性能资格未通过**，本版本的 GEMM 仅提供 `development` 通道。`qualified` 缺真实证书时在启动 GPU 前拒绝。原 0.1.1 固定 Attention/MLA 生产入口保持原范围。

离线 wheel：`artifacts/wheels/nvidia_kernel_agent-0.1.2rc1-py3-none-any.whl`；SHA256 `1f196c84e93ad406ddef58f55118041a43fa04538bcd9f15fa837d6fbf341264`。该版本不可覆盖重建。

在 3070 的 WSL 可信控制用户下查询现有预览状态：

```sh
/home/dministrator/cuda-lab/nka-persistent/gemm-0.1.2rc1-zuewzhv5/kernel-agent status \
  --root /home/dministrator/cuda-lab/nka-persistent/gemm-0.1.2rc1-zuewzhv5/validation/store \
  --channel development
```

准备连续行优先 FP16 的 `A[M,K]`、`B[K,N]` NPY 后运行，返回 FP32 NPY：

```sh
/home/dministrator/cuda-lab/nka-persistent/gemm-0.1.2rc1-zuewzhv5/kernel-agent run \
  --root /home/dministrator/cuda-lab/nka-persistent/gemm-0.1.2rc1-zuewzhv5/validation/store \
  --channel development \
  --a /absolute/path/a.npy --b /absolute/path/b.npy \
  --output /absolute/path/c.npy
```

支持维度 1..4096、有限输入且绝对值≤4；完整合同随包保存。仅运行两份固定 cuBLAS 产物，最多一次可信回退。版本切换使用同入口的 `rollback`，恢复默认使用 `activate --version sm86-cublas-container`，保留相同 root/channel。新应用状态可用 `init --root /absolute/new/parent/state` 创建，state 必须尚不存在、父目录已存在。

实测：新 prefix/空 cwd 的49CPU检查；GEMM8输出10尝试、两既有算子各一次 qualified 功能冒烟、直接容器边界1输出，合计11输出13尝试。crash/有限CUDA后的主机timeout、坏产物回退、版本及rc1→0.1.1→rc1实际包回滚通过。592文件逐SHA与全部帧/NPY/NumPy FP64独立复核通过。容器UID65534、零capabilities、NoNewPrivs、只读输入/产物、外网拒绝和清理实测通过。这些是功能与恢复结论，不代替性能资格。

普通时钟三AA最坏7.699345%拒绝；100W三AA最坏15.011395%拒绝（最高55°C、热计数无增加、已恢复240W）；均超过用户3%部署界限，0AB、0证书。没有新增模型调用或重跑旧封存数据。

[预览审计](../evidence/20261124-sm86-gemm-0.1.2rc1-installed/verification.json) · [发行清单](../releases/sm86-gemm-v0.1.2rc1-preview/manifest.json) · [100W资格拒绝](../evidence/20261124-sm86-gemm-power-qualification/verification.json) · [原Attention/MLA生产入口](SM86_ATTENTION_MLA_RUNTIME.md)。
