# 正式实验前：候选源码的有界NCU接入

上游仍为KernelAgent e0647170... profiler/ncu_profiler.py（原文件SHA44e65a...）及已验收 ncu_adapter.py::profile_cuda/stop_profile_session/parse_profile；CUDA绑定复用原source_worker::{load,tensor_hash}、fixtures.make_inputs，不新增ABI或loader。容器捕获/清理/NKA1/flat ZIP输出复用 bounded_io.capture、protocol、source_packets。现有固定image、原GEMM运行/构建组件不改。

具体缺口：模型生成CUDA只允许在低权限GPU worker执行，不能把新DSO移到宿主root做profile。新增独立profile worker，仍UID65534/network none/read-only/capdrop/no-newprivs/PID32/CPU2，无凭据或参考；只增加可信Nsight安装目录的只读挂载，输出由有界stdout返回，无宿主RW。profile模式内存暂设4GiB、tmpfs256MiB、deadline240秒；普通运行原2GiB不改。先实测实际权限/依赖，不安装新工具，不追加SYS_ADMIN等能力。若权限不足则记录unavailable，禁止伪造计数器。

先用原93库3launch确认实际19字段；固定无匹配kernel故障须在控制侧拒绝；随后用已审查并通过320QA的398226...新库恢复，绑定GPU输入/输出hash与原trusted-host oracle。NCU参数仍API start/stop、kernel replay、cache-control all、clock-control none；profile wall单列，绝不能替代Event。只有profile阶段成功后，才用于正式C候选反馈。验证命令：`python3 -m experiments.full_delivery.bounded_profile_probe --output <new evidence>`；CPU parse/frame/hash审计与GPU/NCU实测分列。
