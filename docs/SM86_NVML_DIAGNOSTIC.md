# 3070 NVML 开关机制诊断

2026-11-24，3060对照完成后，按用户要求回3070执行一次预先固定的八区块诊断。**关闭窗口内NVML仍出现5.272610%的单对偏差，不能视为稳定性修复。**

| 区块 | 窗口内NVML | Event最大单对偏差 |
|---:|---|---:|
| 0 | 开 | 2.258496% |
| 1 | 关 | 2.180501% |
| 2 | 关 | 3.162684% |
| 3 | 开 | 4.663732% |
| 4 | 关 | 5.272610% |
| 5 | 开 | 17.024809% |
| 6 | 开 | 1.774013% |
| 7 | 关 | 4.997371% |

完整保留144段/72对、137.303秒Event；没有截取通过区块或按结果追加区块。该诊断只有公开M1/dev03/count16，不替代8workload双策略完整三AA/三AB；不发资格证书。

## 方法与原始证据

同3070 UUID/driver591.74/Windows boot `2026-11-23T06:52:14.5000000Z`，实际HAGS On证据来自同次启动的原DxDiag。既有0.1.1 prefix、两份原SM86 cuBLAS、固定原Docker image/低权限只读worker沿用；无需新CUDA编译或模型提案。未发现获准清理名单中的游戏/监测进程。

复用KernelBench既有共享物理ring/Graph/Event路径。两侧均同一可信cuBLAS，固定32调用/Graph、每段1348replays、每区块9对；1348取自已归档公开HAGS-On AA0/dev03/count16原校准。ON/OFF顺序在采样前固定为`on/off/off/on/off/on/on/off`，四次各自含两个capture奇偶次序。pilot仍保留，但不据本次结果改变工作量。

SM锁1200MHz，显存请求7001MHz；ON区块已有NVML样本SM均1200，显存报告6800/7000，最高52°C。OFF窗口没有GPU查询，仅区块结束后查询。不能据区块前后数值推断OFF整个窗口的实际瞬时频率。原功率240W保持，内外finally两次reset均exit0；收尾GPU0%/45°C/240W。各worker保护和清理记录均有效。

同一段记录Event、perf_counter、CLOCK_MONOTONIC_RAW、Unix时间戳，CUDAEvent/Graph命令序列未改。RAW配对比复现ON最大17.023980%、OFF最大5.271720%；全部72对Event与RAW配对比最大绝对差0.000107852（约0.010785个百分点）。perf_counter/Unix与RAW存在更大差异，完整记录在[cross-clock.json](../evidence/20261124-sm86-nvml-mechanism-utf8/cross-clock.json)。现有证据不能据此认定WSL/驱动/调度中的具体根因。

首次准备在PowerShell stderr默认UTF8解码处失败，发生于clock set/GPU worker前，0Event/10文件完整归档。修正仅为输出UTF8及Python保留原始bytes，另root冻结；无已采样本重跑。

- 成功诊断：141原始文件逐SHA、生成support/不可变wheel/帧/见证/双时钟/全部统计独立CPU重算PASS。
- 原始归档SHA：`1119b5984e5c4ee09d5f45b5e29edcb4aaf3081f184604ff698203689aad11ab`。
- 协议SHA：`141d33736a43b9cb367fea9082474f595160194efe7282a59c2c4bd788b2f6fd`。
- [诊断审计](../evidence/20261124-sm86-nvml-mechanism-utf8/verification.json)；[准备失败审计](../evidence/20261124-sm86-nvml-mechanism/verification.json)。

复算不运行GPU：

```bash
/tmp/nka-delivery-verify-venv/bin/python -m experiments.full_delivery.audit_sm86_nvml_diagnostic evidence/20261124-sm86-nvml-mechanism-utf8
```

## 下一步已经准备

[原HAGS状态恢复计划](../evidence/20261124-sm86-hags-restore-plan/plan.json)已经只读核对：当前注册表DWord2，原备份为键值缺失，原运行环境确为HAGS Off。可复用既有`windows_hags_setting.ps1 -Action Restore`，按备份恢复，再由用户在维护窗口手动重启；随后以DxDiag实态而非注册表推断确认。之前HAGS-Off完整三AA最坏2.040408%，属于值得确认的历史条件，不保证恢复就能通过。新确认将单独冻结三AA→条件三AB，全部旧拒绝保留。

**当前没有修改注册表或重启Windows。** 重启会中断应用/WSL/SSH，FRP自动恢复未确认，需要维护窗口。另一路线是在已通过完整对照的3060上独立完成新验收、安装、回滚及GEMM生产发布；不会将其证书用于3070。
