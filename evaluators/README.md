# 成熟评测组件的任务适配

## GEMM 构建与正确性

`gemm_kernelbench.py` 用固定 KernelBench 的原始 loader、correctness 和 CUDA Event 函数运行仓库内的 cuBLAS GEMM。当前合同为正维度、二维连续行优先、同 CUDA 设备，FP16 输入、FP32 累加和输出。该入口只执行仓库内的可信实现。

在项目根目录，用既有 CUDA 版 PyTorch Python、NumPy、requests 和 CUDA 13.0 工具链运行。下例在独立目录安装固定的 KernelBench 运行依赖，生成编译缓存和结果。

```bash
run_dir=$(mktemp -d)
python -m pip install --no-deps --target "$run_dir/deps" -r evaluators/requirements-kernelbench.txt
kb_src="$PWD/provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src"
CUDA_HOME=/usr/local/cuda-13.0 \
  TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=1 \
  TORCH_EXTENSIONS_DIR="$run_dir/extensions" \
  PYTHONPATH="$kb_src:$run_dir/deps" \
  python -m evaluators.gemm_kernelbench --output "$run_dir/result.json"
```

结果包含上游正确性报告、独立 CPU FP64 检查、输入不可变检查、CUDA Event 原始样本和产物哈希。当前 Event 数据用于接入检查，正式配对 A/A、候选故障回归和隔离验收另行记录。

候选不能控制参考、容差、计时边界或 incumbent。只有独立低权限 GPU worker 通过隔离验收后，才允许自动执行模型生成的 CUDA 源码。

[固定来源与采用表](../provenance/UPSTREAM_REUSE.md) · [最小接入顺序](../docs/CUDA_ENGINEERING.md)

## 冻结合同、可信模板与故障回归

`python -m evaluators.gemm_regression --warps 4 --output <new.json>` 仍通过固定 KernelBench 的 `load_custom_model` 和原始 correctness 入口构建／检查，额外验证完整开发集、CPU FP64、输出合同、输入快照／guard、非默认 stream 的生产者与消费者，以及固定故障候选。仅加载仓库内固定可信源码。`--warps` 允许 1/2/4/8，`--sanitizer-smoke` 选择精简边界集供 memcheck/synccheck 使用。

在既有 3070 连接可用时，`python3 experiments/p1_02/run_3070.py --evidence evidence/<new-run>` 复用现有隔离依赖目录，不安装软件，运行四种配置及 Sanitizer，逐项保存编译命令、源码、日志、结果和二进制 hash。临时依赖目录丢失时先核查实际缺项，再按上方锁定 requirements 恢复。

回归包括常数、输入污染、尾部遗漏、缓存旧答案、错误 dtype／shape、NaN、错误 stream 依赖和主机异常。它证明本次检查可拒绝这些具体错误，尚不证明可以安全执行恶意源码。超时、容器恢复、实际晋升状态保护另行验收。独立验收 workload 不进入此入口。
