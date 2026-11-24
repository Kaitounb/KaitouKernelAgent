# 最终准入修复实施前记录

用户明确要求解决最终稳定性并开放4090 NCU权限，允许在已授权资源上做必要修复。保留旧失败、旧封存验收、原wheel和既有冻结协议；不得降低2%/1.5%稳定性阈值或选择性删除样本。

复用：KernelBench 423217d9fda91e0c2d67e4a43bf62f96f6d104f1 的CUDA Event/Graph评测路径，经既有 source_timing.paired_witness；既有 cuBLAS qualification host/SourceRunner/check/assess；KernelAgent e0647170da36ef9b059ac0bd3d60103aa4ed378b 的NCU collector/parser，经项目ncu_adapter/mla_profile_entry。CUDA算法、受限worker、私有数据协议与上游pins不改。先只读检查两卡权限、时钟、功耗、driver与宿主边界，再决定最小修复；无实际修复不得通过新名字重采旧身份。

官方运维依据：https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters 及 https://docs.nvidia.com/deploy/nvidia-smi/ 。计数器权限在宿主驱动控制；容器root身份本身不代表宿主权限。不通过逃逸或漏洞取得权限，不影响其他用户负载；不自动重启/卸载共享GPU驱动。

验证顺序：只读环境/失败原始段分析→记录修复前后状态→预先冻结新的环境/测量协议→一次完整三session AA及条件AB→独立CPU复算。NCU须真实原生报告/计数器与DSO绑定，不能以help/version成功代替。原始日志全部保留，0新模型调用起步。

3070实际缺口已确认：WSL的nvidia-smi锁频Unknown Error，但Windows管理员原生nvidia-smi.exe可成功设置/恢复。8秒可信Torch有限负载验证SM稳定1200MHz、CUDA P2显存实际6800MHz（请求7001MHz；空闲7000），温度35–36℃、约57W；此仅证明锁频可用，未证明旧离群因果。新资格只增Windows原生锁频及全程遥测/恢复，原250ms校准/32call Graph/9pair/双策略/8workload/三AA和条件三AB以及2%/1.5%/.97门槛完全不变。复用原 cublas_qualification_host.py 全文，生成独立 locked 版本（旧文件不改），只在既有GPU flock后加setup，run记录时间范围，finally关闭遥测并恢复。全局新环境身份显式包含1200MHz SM/7001MHz请求内存和P2实际6800MHz；不能将通过解释为未锁频环境合格。不得重跑这一新资格身份挑选通过。

锁频一次资格实际结束：320QA/864段，时钟452个高负载样本严格1200/6800MHz、最高54℃，仍单组dev04/ring16出现257.114ms对249.991ms，2.849%拒绝，无AB；这说明锁频不是充分修复，不能据此认证。原始613文件收回。对运行中worker只读观察10秒cpu.stat：cpu.max=200000/100000，nr_throttled及throttled_usec均为0，不能把失败归因于已证明CPU配额节流，也不盲目移除资源限制。

进一步诊断复用已安装Nsight Systems2025.3.2与官方graph级CUDA/NVTX trace（https://docs.nvidia.com/nsight-systems/UserGuide/index.html）。只对已审查原66 cuBLAS可信绑定和3公开失败关联shape做预定3×9×2段时间线，记录CPU提交与GPU Event耗时、Graph活动及空隙；profiler数据仅诊断，绝不作为AA/AB通过证据或重采旧资格。驱动保持原生1200/7001请求设置，finally恢复；无新CUDA算法、无模型候选、无独立清单读取。根据真实trace决定是否需要测量方法修复，禁止直接放宽阈值或筛除离群。

第一份Nsight时间线只有88750 runtime与162 NVTX，无任何GPU活动表，因此不能据此计算GPU空隙。162诊断段保留，仅CPU提交事实可用。NVIDIA官方论坛工程答复给出WSL GPU→CPU时间戳转换修复：CuptiUseRawGpuTimestamps=false（https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536）。已只读证实将HOME设为项目临时目录后nsys -z指向该目录；下一版仅修改专用配置，用户全局配置不动。先短小可信Torch Graph smoke证明原生GPU Graph活动存在，再条件执行原冻结162段；修复后的时间线时间戳精度不等同Nsight默认，不能用它生成性能资格。

修复后Nsight已真实捕获78438个GPU Graph，162个NVTX范围逐runtime correlationId关联replays数量全部一致。dev06 repeat2/pair4，Event为251.18199/256.69763ms，相差5.51564ms；CPU提交仅4.07/3.85ms，慢侧Graph最长1.727898ms而普通中位约0.585ms；最大跨Graph空隙0.282237ms。不能称已证明CPU供给不足，亦不能在无WDDM调度轨迹时断言具体外部原因。官方时间戳修复精度限制单列，原始Event与修复后的时间线分开保存。

测量精度修复：原指标本来就是Graph批量平均，不是单次尾延迟。为降低已观察5–11ms级短时设备波动对250ms窗口的影响，预先固定持续平均窗口为1000ms（四倍），pilot公式变ceil(4000/median of4-replay pilot)，上限8192，保留32calls/Graph、9pair、16行×三session、双输入策略、全部开发workload、原2%单对/1.5%组均/.97部署保护门槛，不剔除离群、不选择重跑结果。只做一次此新完整协议，旧250ms两次拒绝永久保留。worker非Sanitizer timeout从180→600秒以容纳全16行×18段，其他隔离/CPU配额不变；一进程仍三独立sessions。GPU算法与默认/回退artifact、输入、参考和数值容差不变。验收范围必须写成SM1200MHz/内存请求7001MHz（CUDA实际6800MHz）、1秒持续平均计时，不能认证未锁频或单请求尾延迟。无真实pass前生产false。

持续平均首AA1.6498%通过，第二AA2.040408%仍超2%，不得四舍五入或只选首会话。第三会话按冻结计划完成，不再追加更长窗口挑通过。进一步仅检查Windows官方D3DKMTGetProcessSchedulingPriorityClass与GPU engine owner；先只读，不在当前冻结运行中修改优先级。官方参考Microsoft Learn d3dkmthk/D3DKMTGetProcessSchedulingPriorityClass、D3DKMTSetProcessSchedulingPriorityClass与D3DKMT_SCHEDULINGPRIORITYCLASS（HIGH=4，REALTIME=5，后者不使用）。WSL GPU-PV并不能保证vmmem优先级可改或影响GPU，必须先以实际PID/owner/返回码核实；不可对未知VM/其他进程或System批量修改。

持续平均一次完整三AA结束：此批独立QA为320（两批新资格合计640），864段最坏2.040408%（session1/dev07/ring16），无AB，1585个加载遥测严格1200/6800、最高53℃，恢复通过。全部614文件收回，CPU审计进行中。必须保留拒绝，不因仅超0.040408个百分点放宽/舍入。

Windows只读调查确认实际运行Win10 build19045；GPU引擎CIM只给pid0，D3DKMT只读句柄Query返回AccessDenied，因此该探针不足以确认/修改GPU-PV进程优先级，未设置任何进程优先级。DxDiag进一步给RTX3070 WDDM2.7、Hardware Scheduling Supported:True Enabled:False，完整原文只存本机0600私有硬件目录，公开只保留该状态与SHA。NVIDIA官方明确建议WSL2开启硬件加速调度：https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/ 。下一步只为已确认支持但关闭的HAGS准备单值备份/启用/回退，不改驱动、不卸载GPU、不触碰其他调优项。注册表当前HwSchMode不存在，目标DWORD2；实际生效需要Windows重启，涉及用户应用/SSH中断，须单独确认维护窗口后才重启。启用设置不等于已生效或已通过AA，重启后必须再次以DxDiag确认实际Enabled:True再冻结新环境协议。
