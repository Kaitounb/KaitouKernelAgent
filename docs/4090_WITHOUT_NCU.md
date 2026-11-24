# 4090 不使用 NCU 的运行范围

可以继续使用现有4090。NCU用于采集硬件性能指标，普通CUDA运算、正确性检查和CUDA Event计时不依赖性能计数器权限。项目首个SM86 GEMM生产发布按PLAN.md的运行闭环、隔离、交付回退、运维门槛验收，4090 NCU不在这四项运行依赖中。

| 能做的工作 | 当前真实边界 |
|---|---|
| 可信CUDA SDK正确性、边界、stream及Sanitizer | 既有SM89结果保留；更换驱动/CUDA/Torch或GPU后重新建立环境身份，不能直接继承旧二进制资格 |
| 普通CUDA Event/Graph计时 | 独立A/A与A/B、原阈值及完整原始样本；Nsight profile时长不混入普通计时 |
| 受控SDK参数搜索 | 可以使用执行反馈；相同配置/身份的已完成结果复用，旧拒绝不为获取好结果重采 |
| 固定已审查SDK的部署验证 | 仍需实际安装、环境校验、故障恢复和回滚验收；已有开发结果不自动成为生产资格 |

完整P6-03目标卡方案C需要目标卡真实硬件反馈，目前保持未运行。不会用3070指标代替4090指标，也不会将没有NCU的执行反馈方案改称C。原冻结实验协议、模型预算和失败记录不变。未来若要完成这部分研究，再使用平台实际开放性能计数器的环境。

当前4090实例还有独立的隔离限制：无Docker，user/network namespace由宿主拒绝。此实例继续只执行可信SDK与受控配置；任意模型生成CUDA源码仍在已验证的3070低权限容器中构建和运行。获得NCU权限也不会自动解决隔离问题。

无需仅为了运行现有可信SDK更换服务器。如果以后迁移，先核实CUDA/Torch/架构和所需执行隔离；是否支持NCU取决于是否继续硬件反馈研究。旧4090备份已完成，原实例未修改或释放；新连接信息未收到，不自动迁移。

相关证据：[两卡实际支持矩阵](SERVER_COMPATIBILITY.md)、[跨卡结果](CROSSCARD_RESULTS.md)、[当前准入项](PRODUCTION_BLOCKERS.md)、[迁移备份](SERVER_MIGRATION_BACKUP.md)。性能计数器权限说明见[NVIDIA官方ERR_NVGPUCTRPERM说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。
