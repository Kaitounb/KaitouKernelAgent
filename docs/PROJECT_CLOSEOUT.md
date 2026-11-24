# 工程交付结项

2026-11-24：**当前已确认范围的工程交付完成**。3060 GEMM、3070 Attention/MLA的限定生产通道，以及本次3060 NCU诊断均已完成各自实际验收。后续进入使用、维护或用户另行选择的研究/学习阶段，无需继续租4090来满足本范围。

| 交付内容 | 结果与入口 |
|---|---|
| RTX3060 GEMM 0.1.3 | 完整三AA三AB/1728段，最大单对0.657401%；独立16/16；故障/坏产物回退、真实包切换、隔离与资格负例通过。[运行/恢复](RTX3060_GEMM_RUNTIME.md) |
| RTX3070 Attention/MLA 0.1.1 | 保持原固定SDK qualified资格与实卡证据；本轮未改包或重跑。[运行说明](SM86_ATTENTION_MLA_RUNTIME.md) |
| RTX3060 NCU | 3次真实cuBLAS kernel，每次19指标；报告重新导入、固定上游解析与三份输出FP64复核通过。[NCU报告](RTX3060_NCU.md) |
| 优化Agent开发与实验 | 真实DeepSeek提案、受控编译/验证/筛选/留档路径已运行；正式54组消融结论保持0候选筛选通过，没有稳定提速声明。[研究结果](FORMAL_STUDY_RESULTS.md) |
| 恢复与运维 | 3060生产镜像、应用/RC2回退包和NCU独立工具均有本机校验备份；最终qualified、GPU空闲、170W、无项目容器 |

## 结项范围

生产执行限各版本已经审查、验收的固定实现与目标GPU/驱动/镜像/合同；不开放任意模型源码生产执行。NCU是独立诊断组件，不改变原GEMM资格证书或正式性能结果。

下列内容保持原记录，没有因本次NCU通过而改为通过：3070 GEMM最终生产性能、原4090目标C完整跨卡研究、原P6全部目标/全部算子整体资格、通用源码生产执行。教学手算与学习复盘继续延后。完整原计划并非逐项全部验收，当前工程范围结项不覆盖这些项。

预算上限是700次新调用，目前累计190次（历史3次另计）；本次安装和NCU0新增模型调用/0新CUDA编译/0正式Event采样，旧sealed未读/传/重跑。未花完预算无需为结项继续消耗。此前失败、重复/拒绝及无收益结论保留。

## 证据与后续

- [3060正式发行清单](../releases/rtx3060-gemm-v0.1.3/manifest.json)
- [3070 Attention/MLA正式清单](../releases/sm86-operators-v0.1.1/manifest.json)
- [本次NCU独立审计](../evidence/20261124-3060-ncu-smoke-v2/verification.json)
- [本次结项清单](../releases/project-closeout-20261124/manifest.json)

日常运行按各自运行文档；新硬件/驱动或新算子需另建目标资格。不要重启历史一次性stage/freeze/qualification/independent/install/activate/backup/build入口，也不要重建同版本wheel。已完成工程工作可到此收尾，未来工作按用户新指令另开范围。
