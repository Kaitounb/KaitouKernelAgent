# 4090 性能计数器平台工单

以下内容已准备，可提交平台人工服务；本项目没有代发消息或工单。用户已确认目前只有容器SSH，无法提供宿主连接或平台管理入口，因此此项需要平台执行。

## 可直接发送的请求

> 我的RTX4090容器实例需要使用NVIDIA Nsight Compute采集已审查的CUDA MLA kernel性能计数器。当前驱动570.86.10，`/proc/driver/nvidia/params`显示`RmProfilingAdminOnly: 1`。容器内UID=0，但CapEff/CapBnd均为`00000000a80425fb`，缺少CAP_PERFMON及CAP_SYS_ADMIN；没有Docker/containerd宿主管理socket，现有NCU采集返回ERR_NVGPUCTRPERM。
>
> 请优先为这个实例配置最小所需性能采集权限：驱动R565以上支持CAP_PERFMON，请评估为该容器增加PERFMON capability并保留NVIDIA设备映射；若你们的部署/工具组合不支持此路径，请提供平台允许的专用profiling实例或受控采集方式。NVIDIA官方还列出容器SYS_ADMIN配置方案，是否可用请由平台管理员评估。请不要直接删除/重建当前实例或丢弃/tmp内项目数据，变更前请说明是否需要停机及数据保留方式。
>
> 目标不是开放容器逃逸或修改其他租户权限；只需本实例指定GPU上可信SDK kernel的只读性能采集。请回告采用的方案，以及capability/驱动策略生效后的状态。只提供`ncu --version`或`--query-metrics`成功不能说明计数器权限已经开放，需要实际kernel采集无ERR_NVGPUCTRPERM并生成原生.ncu-rep。

平台若选择允许所有用户采集，则涉及宿主驱动`NVreg_RestrictProfilingToAdminUsers=0`策略及共享环境影响，应由平台确定；不能在容器内写一个modprobe配置文件便声称已生效。这里不提供自动卸载驱动、重启GPU或中断其他租户的命令。

## 实际验收

先只读检查新的`CapEff/CapBnd`与`RmProfilingAdminOnly`，保存变更前后记录。只有确认权限或环境发生实际变化后，才运行一次可信SDK MLA profile；现有失败不重复采集。复用固定KernelAgent collector/parser、GPU flock、公开开发输入、真实DSO hash及原生报告，另存新证据目录。

项目本机已有 `experiments/full_delivery/run_mla_profile.py --target server` 接线和 `mla_profile_entry.py`。重新运行前需绑定已冻结SM89原生DSO（fa2 cfba14b0...、cascade b8050d9b...），不能让旧启动器默认JIT重建并将新文件冒称旧身份。精确旧字节及build.ninja已在 `evidence/20261123-sm89-frozen-native-archive` 归档。CPU parser成功不能代替真实计数器采集。

原始权限盘点：`evidence/20261123-admission-repair/4090-before/inventory.json`。平台答复确认容器内无法自行补出capability，记录在同目录上层的 `platform-response.json`。

依据：[NVIDIA官方ERR_NVGPUCTRPERM处理说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。
