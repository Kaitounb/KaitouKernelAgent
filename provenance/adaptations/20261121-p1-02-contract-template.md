# P1-02 合同、workload、可信模板与故障回归（实施前记录）

2026-11-21，开始实现前记录。沿用工程 cuBLAS 的 FP16 行优先输入、FP32 累加／输出与 `atol=3e-5, rtol=1e-4`，不因候选失败放宽容差。

| 来源 | 保留 | 必要适配／缺口 |
|---|---|---|
| KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`, `eval.py::load_custom_model/run_and_check_correctness` | 原文件、ModelNew/load_inline 构建、原始 correctness | 已有接入没有冻结 workload、CPU FP64 的全开发集、输出合同和故障回归；补任务适配，参考在候选执行前生成 |
| 同 commit `unit_tests/test_eval_adversarial.py::test_result_cache_reuse/test_input_modification/test_non_default_stream` | 缓存旧答案、污染输入、stream 绕过的故障分类 | 按当前 FP16→FP32 GEMM 注入固定可信故障；stream 以显式生产者／消费者依赖检查，不用上游过快阈值代替正确性 |
| NVIDIA CUDA Samples v13.0 `3f1c50965017932fc81e6d94a3fc9e04c105b312`, `Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu::simple_wmma_gemm` | warp tile 映射、16³ FP16/FP32 fragment、K 循环、WMMA 装载／乘加／存储；原 BSD notice | 原例 B 是列优先、alpha/beta 且仅 16 倍数；改为合同的行优先 B、alpha=1/beta=0；warp 数限 1/2/4/8；不满足整 tile／地址对齐的输入直接调用已验证 cuBLAS，不新增加载器／尾块框架 |
| CUDA 13.0 `cublasGemmEx`（既有 `kernels/gemm/cublas_sm86.cu`） | 可信回退、PyTorch 当前 stream 与 handle | 共用张量合同校验；候选不可更改参考或晋升规则 |
| KDA `acdd79343c5e9d0c7cb96289f26fd3f1051ce785`, `docs/agent-flow.md` | contract→candidate→correctness→measurement→evidence | 本次仅冻结合同与保留裁决证据，无真实模型调用、无晋升结论 |

构建／验证预定命令：

```text
python -m evaluators.gemm_regression --output <new-result.json>
python -m evaluators.gemm_regression --warps 1|2|4|8 --output <new-result.json>
python experiments/p1_02/measure_aa.py --binary <previous-trusted.so> --sha256 <recorded> --session 0|1|2 --output <new.json>
compute-sanitizer --tool memcheck|synccheck <python> -m evaluators.gemm_regression --sanitizer-smoke --output <new.json>
```

只运行仓库内固定可信代码与故意故障。正式独立验收清单按 family 从未出现在开发集的形状生成，单独 evaluator 持有，发布冻结后再执行；本次不运行该清单。模型源码执行仍依赖 P3-02。正式 A/A 取决于 GPU 空闲证据；竞争负载不停止、不干预。

追加验证范围（开发集正确后、搜索前）：`experiments/p1_02/calibrate_gemm_aa.py` 复用已有 P0-03/P1-02 Event/Graph 配对顺序，补原试跑缺少的双独立 Graph、三个开发锚点、1/16 输入轮转与完整 CPU 对照；不采用同图结果推断独立图噪声。`timeout_recovery.py` 采用 Python 标准 `subprocess/start_new_session`、进程组 TERM/KILL，迁移 P0-04 的有限 GPU 工作后主机超时场景；这是可信故障夹具，不是新候选框架或完整隔离。原 cuBLAS 二进制与本次重新编译的合同版各自单独记录 hash，后者用于锚点校准。
