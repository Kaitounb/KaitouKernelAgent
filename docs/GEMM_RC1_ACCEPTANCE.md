# GEMM RC1 独立正确性与恢复验收

2026-11-22。`gemm-sm86-rc1` 固定 dev5 wheel、默认 cuBLAS 和两份已审查产物后，完成一次性独立验收。冻结文件保持原样，当前验收状态由本报告及证据表达，不回写冻结前的历史状态字段。

| 实际检查 | 结果 |
|---|---|
| 4 个封存 workload × 2 seeds × default/fallback 两产物 | 16/16 独立 GPU 输出正确 |
| FP16 量化输入 → 可信 CPU FP64 → FP32 参考 | 最大绝对误差 1.1444091796875e-5，冻结 atol=3e-5、rtol=1e-4 |
| GPU 输入哈希、offset guards、producer/consumer stream、输出合同 | 16/16 通过 |
| worker UID65534、固定 image/binary、容器 cleanup | 16/16 通过 |
| 公开输入默认、crash、timeout、活动产物损坏回退、旧版本、回滚 | 6 场景／8 次尝试通过 |
| 私有原始证据 | 195 份文件完整哈希复核通过；清单和参考未挂载给 worker |

精确 shape/seed 没有展示给模型。原始私有 ZIP 和派生输入/参考位于本机 `.private/gemm-sm86-rc1-evidence/`，公开摘要只给聚合计数、误差与哈希。传输曾被自动审批拒绝；用户明确授权该 payload/3070 目的地后才执行，原拒绝及解决记录均保留。没有自动重试独立 GPU 验收。

冻结 ID：`1756bace31e202585de4164d3f4dd68d53eee5da22c8f364e06334495f615549`。私有原始 ZIP SHA256：`5e9b0751487904793686c1a44ed87a215aec054ad133d312ff037ba01ea16268`。

本次没有新增模型调用、性能采样、候选晋升或 allowlist 修改。P1-05 和旧源码 A/A 失败保持原结论；这次正确性验收不替代最终性能复测。包内状态仍是 development-only，production_release=false。完整项目继续推进 Attention/MLA、B/C、跨卡和正式实验。

复核：`python3 -m tools.verify_gemm_release_freeze`；在持有私有原始证据的可信控制机执行 `python3 -m tools.audit_gemm_rc1`。不要重跑 `run_sealed_gemm`；现有 reservation 已完成且拒绝重用。

[公开摘要](../evidence/20261122-gemm-independent-acceptance/summary.json) · [CPU 哈希审计](../evidence/20261122-gemm-independent-acceptance/verification.json) · [公开恢复原始证据](../evidence/20261122-gemm-rc1-public-recovery/archive-manifest.json) · [冻结协议](../releases/gemm-sm86-rc1/protocol.json)
