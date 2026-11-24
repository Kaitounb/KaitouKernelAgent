# P1-05 冻结最终复测与 V0 收尾

本轮复用 KernelBench 接入、现有 `gemm_candidate::checks/paired` 和冻结 4-warps 原 `.so`，不生成 kernel、不新增模型调用、不运行 profiler。来源／验证命令与段长校准规则在 [实施前记录](../../provenance/adaptations/20261122-p1-05-final.md)。

## 重放与复测入口

```text
python3 -m experiments.p1_05.audit --evidence evidence/20261122-p1-05-final --long-evidence evidence/20261122-p1-05-final-long --output <new-audit.json>
python3 -m unittest discover -s tests -p test_v0_summary.py
```

以上只做 CPU 复算，无模型、GPU 或密钥需求。新 GPU 复测依赖 HANDOFF 中已验证的 3070 环境和原可信 `.so`／KernelBench 依赖路径：

```text
python3 -m experiments.p1_05.run_3070 --freeze evidence/20261122-p1-04-feedback/freeze-current.json --evidence <new-short-dir>
```

每次使用新目录并保存全量失败；前三进程是 best4 对 initial4，后三进程是 cuBLAS 对 best4。现有路径不是全新机器安装证明；若临时 artifact 已丢失，先按原 source/build flags 恢复并重新验证来源／binary，不能跳过 hash 校验。全新安装／容器分发属于 P3-02。

## 本次计时校准变更

首批正确性全部通过，但原 12-replay 自身 A/A 最坏单对偏离 3.042%，因此不当成稳定最终计时；原始 1,728 个主复测段保留，工程 cuBLAS 对照的 864 段另列。未改变候选或原搜索门槛。

根据保留数据的段长确定更长调用数。100ms 草案留档但未跑 GPU；考虑首批最坏离群本已发生在约81ms段，实际追加预先固定为250ms目标、每case选择一次repeat count。best 与 initial identity相同，一组4-vs-4自身即为A/A，不重复另一组相同对照。追加仅一次三进程，各9对／32节点和全部8workload／两缓存；每行replays与calls_per_segment显式记录。单对2%／分组1.5%界限保持。

```text
python3 -m experiments.p1_05.plan_long --source <short-dir> --target-segment-ms 250 --output <short-dir>/long-plan-v2.json
python3 -m experiments.p1_05.run_long_3070 --source <short-dir> --plan <short-dir>/long-plan-v2.json --evidence <new-long-dir>
python3 -m experiments.p1_05.summarize --evidence <short-dir> --long-evidence <long-dir> --output <new-summary.json>
```

`pairing_long.py` 的计时函数来自原 `paired` 函数体，唯一 loop 改动是把写死12变成已冻结的 replays 参数，返回额外计数；无新计时后端。`long_worker` 只验证同身份4-vs-4，不能作为新的搜索晋升入口。初批失败与追加数据均保留在结果中，不把校准批重命名为成功批。

## 统计与范围

每session逐workload计算9对log ratio，再对3session等权；t(2)=4.303的区间仅为探索性描述。两缓存分列，all-workload几何均值8项等权。延迟换算使用每行实际calls_per_segment，不能对长段仍除384。

主收益分母是最初CUDA模板4warps；当前best仍同一identity，名义收益1.0、胜/平/负=0/8/0。实测A/A抖动另列。cuBLAS比值是`cuBLAS_latency/best_latency`，只表示工程库差距，不算agent收益。独立验收4项尚未运行，不把8项复测写成12项独立验收。

Event是GPU流上的时间窗口。host分配/capture在计时操作之外，但提交流水、GPU/OS调度和时钟等仍可影响窗口；不称完整eager部署延迟，也不承诺完全排除host干扰。原始所有段保留，未删最慢点。

V0仅为受约束GEMM提案／验证／profile／拒绝／回放通道；不是连续多轮服务、自由源码执行器或已发布生产运行时。P3-02将处理低权限隔离、可安装产物、实际回退及回滚。

## 实际验收状态（2026-11-22）

一次长段追加仍有两组 dev-03 单对偏离 3.216%／3.324%，`TIMING_NOT_ACCEPTED`；P1-05 保持未勾选。数据审计 PASS 只表示复算一致。三批共 3,456 段、672 组正确性全部保留，不再追加挑选通过批次。[报告](../../docs/V0_REPORT.md)
