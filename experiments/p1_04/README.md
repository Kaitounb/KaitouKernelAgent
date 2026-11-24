# P1-04：可信 CUDA 的 NCU 硬件反馈

来源与实施前取舍见 [适配记录](../../provenance/adaptations/20261122-p1-04-ncu.md)。固定 KernelAgent 原始 `load_ncu_metrics`、`metrics_to_prompt`、`NCU_METRIC_SECTIONS` 直接导入；CUDA collector 从其 `profile_triton_kernel` 适配。原始文件 hash 校验通过后才导入，保留 Apache-2.0 文本。

## 采集与解析

```text
python3 -m experiments.p1_04.run_3070 --warps 4 --evidence <new-profile-dir>
python3 -m experiments.p1_04.run_3070 --warps 2 --evidence <another-new-dir>
```

当前入口复用已授权 21022 连接、原 `.so`、CUDA/PyTorch/NCU 路径。先检查实际缺项。远端缺少 pandas，独立 parser 目录 `/tmp/nka-p1-04-parser-deps` 安装 pandas 2.3.3 及其非 NumPy 依赖，继续使用原 NumPy，不改系统／CUDA 环境。完整实际版本在各次 execution.json。重建时按记录版本安装到独立 target，而非覆盖原环境。

`run_3070` 当前使用首次查询保存的 `/tmp/nka-p1-04-metrics.txt`。恢复机器或更换 NCU/卡时必须重新执行 `ncu --query-metrics --query-metrics-mode all`，记录实际查询与工具版本再运行；不要沿用别卡的 availability。查询体积较大，本批保留其原始 gzip 和 SHA256。

固定 driver 用 CPU FP64 参考、预热十次，然后以 CUDA profiler start/stop 只包住三次可信 GEMM；kernel regex 仅允许 gemm_wmma。collector 显式 `--replay-mode kernel --cache-control all --clock-control none`，同时保留 CSV 和 `.ncu-rep`。NCU 时长只是 replay 诊断，不输出正式 CUDA Event 成绩。已预热不表示 profiler 一定温缓存；本协议显式冲刷 cache。

原 parser 默认 LAST 只能选择一行。适配层仅按 launch ID 拆行，分别调用原 parser，再拼回全部三行并交给原 prompt formatter；不重新实现 metric 的数值清洗。单位来自原 CSV。缺失／非有限字段为 null + unavailable，实际观测的 0（如 barrier）仍是合法 0。未在 query 枚举的旧 memory_dependency metric 单独标 unavailable；launch 软件统计虽不在硬件 query 中，按实际 CSV 验证。

## 一次真实反馈与独立裁决

```text
<model-venv>/bin/python -m experiments.p1_04.propose --profile-run <baseline-profile-dir>/run --output <new-proposal-dir>
python3 -m experiments.p1_03.run_3070 --proposal <proposal-dir>/proposal.json --evidence <new-evaluation-dir>
python3 -m experiments.p1_03.replay --evidence <evaluation-dir> --output <new-replay.json>
python3 -m experiments.p1_04.compare --baseline <baseline>/run --candidate <candidate>/run --feedback <proposal>/feedback.json --output <new-comparison.json>
```

提案沿用 KernelAgent provider，DEEPSEEK_API_KEY 仅本机环境；max_tokens=768，一次调用、零自动重试。用户已允许源码及开发数据用于提案，精确独立验收清单仍不读取、不传输。模型拿到 38 个已核对 fact ID、实际单位／范围／scope 和当前源码，只能提交合法 warp 参数、假设、竞争解释与已知 metric 的方向预测；fact引用和 JSON 结构由控制层检查。

`compare` 只检验三次 replay 中位数的方向是否符合模型预测，不给因果或统计显著性结论。额外 raw launch/device 元数据显示 driver shared=1024 bytes、user static/dynamic=0；不能从 allocated 字段推断实现用了 shared tile。收集的 block/register/shared/thread 限额可计算资源上界，不等于实际 occupancy。

正式 correctness/A/A/A-B 使用完全独立的新进程，不加载 profiler；源代码、数值合同、晋升门槛不变。保留前次 8-warps 拒绝，本次不自动重采以获取更好结果。当前参照只由原冻结 gate 裁决更新。

本阶段不是 B/C 对照：P1-03 与本次 prompt/source 信息和提案不同，不能推断 NCU 带来收益；P3-03 后续对齐对照条件。也未完成自由源码隔离、发布或回滚验收。
