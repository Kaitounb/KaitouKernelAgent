# 固定 SM86 算子部署（dev6）

2026-11-23 当前已完成0.1.1的3070固定Attention/MLA生产交付，操作入口、合同、可运行示例和回退以[最新运行说明](SM86_ATTENTION_MLA_RUNTIME.md)为准。本文下方为切换计划或dev6历史记录。

dev6 在新 prefix 与空 cwd 中实际安装运行，支持 Softmax、融合 Attention 和经典 MLA core Z/自然 LSE。CUDA 计算来自固定 PyTorch/FlashInfer；不提供任意模型源码加载。当前是开发部署；冻结默认MLA独立8/8调用、16/16输出及204文件CPU复核已通过，最终生产发布仍未完成（[验收范围](MLA_RC1_ACCEPTANCE.md)）。GEMM 默认仍为 cuBLAS。

运行条件与已测环境：RTX3070/SM86，Linux Docker GPU 支持，固定镜像 `sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a`，镜像内 Torch2.14.0+cu130/CUDA13.0。宿主 Python3.12 和 NumPy；模型 API/Torch SDK 无需装入宿主运行环境。SM89 正确性证据不能用于加载此 SM86 AOT 包。

仓库中的 wheel 为 `artifacts/wheels/nvidia_kernel_agent-0.1.0.dev6-py3-none-any.whl`；SHA256 `d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38`。AOT zip 为 `artifacts/operator-runtime/sm86-cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb.zip`；SHA256 `939bf2d6d71a8d47fff652f944addb16857167e9f4ffbabd83c37a7e9a5ae5fa`。解压前核对 SHA；zip 内是公开 SDK、7份 native CUDA 库、Softmax 库及原许可，不含凭据或参考数据。

在声明的 Linux 环境安装与检查（目录名可自行选择）：

```sh
python3 -m pip install --no-index --no-deps --target ./kernel-agent-dev6 artifacts/wheels/nvidia_kernel_agent-0.1.0.dev6-py3-none-any.whl
python3 -m zipfile -e artifacts/operator-runtime/sm86-cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb.zip ./operator-sdk
PYTHONPATH="$PWD/kernel-agent-dev6" python3 -m kernel_agent.operators.cli inspect --bundle "$PWD/operator-sdk"
```

准备 `inputs/x.npy`（C contiguous FP16、shape `[7,129]`、有限值绝对值不超过10000）及 request.json：

```json
{"operation":"softmax","bundle_id":"cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb","fault":null,"rows":7,"width":129,"dtype":"float16"}
```

```sh
PYTHONPATH="$PWD/kernel-agent-dev6" python3 -m kernel_agent.operators.cli run --bundle "$PWD/operator-sdk" --request "$PWD/request.json" --inputs "$PWD/inputs" --output "$PWD/outputs" --work "$PWD/run-001"
```

output/work 必须是新目录。输出按一组 NPY 完成后原子移入 output；完整 worker/cleanup/hash 记录在 work。一般调用不带故障字段；显式 `--inject-failure crash` 或 `timeout` 仅供恢复演练。失败后仅允许一次 fresh worker，清理未确认立即停止。损坏或非catalog bundle在执行前拒绝，不改当前部署或allowlist。

Attention 输入为 BSHD FP16 的 q/k/v（维64或128），可选全局 bool `[Sq,Sk]` mask，causal 为 bottom-right。MLA 为 FP16 absorbed q `[B,H,512]`、q_rope `[B,H,64]`，随机排列分页 C/rope cache，int32 indptr/indices；H16/32/64/128、B1–16、L0–8192、page1/16/64、split1/2/4。合同和准确字段见 `kernel_agent/operators/runtime_contract.py` 与 `docs/OPERATOR_CONTRACTS.md`；输入总量上限192MiB。MLA 输出是 latent Z，不能当作完整模型层吞吐。

worker 使用 UID65534、无网络、只读根目录和3个只读挂载、有界tmpfs/PID/CPU/内存/stdout/stderr。AOT 包和输入不含参考/凭据。运行范围不包含无限 GPU kernel 的取消或 GPU 驱动隔离。

原 dev5 wheel 与 prefix 不覆盖。应用回滚时切回已归档 dev5 prefix/虚拟环境，使用原 `kernel-agent` GEMM 入口；新增 operator 命令在旧版本不提供。GEMM artifact 的 activate/rollback 仍按 `docs/SM86_RUNTIME.md`，与应用包版本回滚分别记录。安装与故障实测见 `evidence/20261123-operators-dev6-installed/verification.json`；全部原始文件和 wheel 保留。
