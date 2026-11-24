# 用户指定3060对照，随后回3070

2026-11-24：目标为RTX3060/SM86/12GiB、原生Ubuntu22.04、driver580.95.05，GPU0%/1MiB。Docker29.1.4和NVIDIA运行时/免交互sudo可用；两已有Python3.11均无Torch、宿主无CUDA工具链。先复用实际3070固定公共镜像ce00f6dc...（Torch2.14+cu130/Python3.12/CUDA13.0.2），两个目标分别绑定，不回填3070失败。原3070 ordinary7.699345%、100W15.011395%均保留。

固定KernelBench423217d9fda91e0c2d67e4a43bf62f96f6d104f1及已有source_runner/worker、两已核cuBLAS二进制、共享ring/Graph/Event和部署v2 gate继续使用。无新数学设计/模型调用；旧封存清单不读不传。仅迁移已验证不可变Docker镜像，不传.private或项目私有数据。采用Docker原生image save/load，经gzip/base64有界流、全流SHA/字节数/源exit、目标load后精确image ID三重核验；Mac不保留镜像大归档，源/目标凭据只在本机私有文件和SSH子环境。原镜像/3070入口不改。

工具 tools/transfer_3060_image.py；验证命令 python3 -m tools.transfer_3060_image。记录源/目标目录和传输日志，随后低权限只读GPU镜像冒烟、明确绑定目标3060后再冻结正式对照。可信父控制器可在同镜像运行并仅其挂载Docker socket，候选worker保持UID65534/零cap/只读/网络none及无socket；不将父进程权限算给worker。任何新适配须先记录差异再跑公开QA/计时。用户明确要求之后回3070再试，若重测将标为用户请求的确认/诊断复测并展示既往失败，不覆盖旧样本或仅择通过签发生产资格。

环境准备修订：官方Torch cu130目录确认2.14.0+cu130可直接获取，原镜像NumPy2.5.3。20GB镜像SSH仅约0.25MiB/s，已只终止本任务精确接收进程，186646528字节partial和全部日志保留，无GPU；不等待数小时传输。NVIDIA官方CUDA-base pull在CloudFront blob EOF失败，未安装/未跑GPU。3060已有Ubuntu24.04.3基础镜像，与原运行镜像OS一致；改用现有base+官方apt Python3.12、CUDA13 sanitizer、官方pip Torch2.14.0+cu130/NumPy2.5.3，记录实际base/包清单/新image ID。**新镜像不继承旧镜像资格**；需重新GPU合同/真实边界/320QA/32Sanitizer/故障恢复，冻结新目标后才计时。builder源码build_3060_runtime.py和Dockerfile原样归档；不升级宿主驱动或系统包。

评测主控制器拟复用已通过sm89_cublas_host的precheck→冻结→三AA→条件三AB流程；改为SM86/Torch14、现有两binary的REUSED_VERIFIED标记（不虚称新编译）、原Docker support、目标image精确绑定与每run Docker标签清理。保留数学/输入/共享ring/Graph/pilot/统计规则，只补可信父控制器Docker-socket运行环境；worker无socket/无凭据/无参考。parent环境只用NVML查询，避免保持额外CUDA compute context。

原海外APT构建380秒停滞，精确停止该CPU-only docker build，0GPU/无成功image，日志保留。国内Ubuntu/PyPI目录探测0.32/0.36秒可达；镜像构建改用Aliyun Ubuntu标准GPG签名源/Tsinghua PyPI传输。Torch采用官方PyPI2.14.0的cp312 Linux wheel，官方元数据SHA fecffb58f51fd643d213acd68da21cc3fc19bea05a3bc64b4ee55128f47a4963 用pip require-hashes强校验；实际torch.__version__必须2.14.0+cu130、torch.version.cuda13.0、NumPy2.5.3才通过CPU构建。官方PyPI元数据cuda-toolkit13.0.3依赖单独留档，不能将新image描述为原CUDA13.0.2 devel image逐文件相同。CUDA Sanitizer继续官方NVIDIA apt，记录全部实际包和image SHA后再GPU准入。

新target GPU兼容/边界探针使用原DockerRunner/protocol/bounded_io/output的独立副本，原包不改；仅在原pipe_worker返回之后添加实际映射cudart/cublas/cublasLt/torch_cuda/c10_cuda路径和SHA平坦metadata，既有GEMM/同步/输出不改。用于之后3070真实加载库对照，不属于计时收益或新候选。所有库hash在GPU工作完成后、仅诊断探针执行，不插入正式Event路径。

实际GPU准入缺口：新镜像CPU构建通过后，legacy `--gpus device=0`下root与UID65534均cuInit100/NVML Unknown；宿主featurize可cuInit0/设备数1，设备0666，非DAC权限或CUBLAS ABI（尚未到.so调用）。Docker采用systemd/cgroupv2，NVIDIA toolkit1.18.1已存在GPU0 CDI spec。按NVIDIA官方 [NVML Unknown说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html) 与 [CDI说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)，优先单次容器 `--runtime=runc --device nvidia.com/gpu=0 -e NVIDIA_VISIBLE_DEVICES=void`，避免legacy注入冲突；不改daemon、不重启、不加cap或放宽workerUID/网络/只读。先CuInit/NVML低权限准入，再修正实际GPU探针；全部初始失败记录保留，无正式Event。

CDI下实际cuBLAS输出/ABI/全边界通过，随后320QA、32Sanitizer零错误、两finite fault/新进程恢复全通过。源旧镜像CPU加载映射与3060真实CUDA探针的cudart/cublas/cublasLt/torch_cuda/c10_cuda五个SHA逐一完全相同，两个既有SM86 .so不变；因此不能把新镜像简单描述为数值CUDA库升级或据此预称修复3070。GPU/driver/OS仍不同，属于对照不是单变量因果证明。新3060三AA/条件三AB协议在正式Event前冻结，保持原3%/1.5%/.97与NVML0.5s/32Graph/1s校准，不新增模型调用或旧sealed执行。

下一项按用户要求回3070：复用原KernelBench/ordinary Docker support（上述上游pin不变），新增8个固定block的NVML ON/OFF机制诊断，顺序on/off/off/on/off/on/on/off，M1公开dev03/16输入，固定1348replays来源为旧HAGS On公开AA0 pilot，不因本次结果改计算量。1200MHz/7001请求且finally恢复；新增perf_counter/RAW/Unix时间戳围绕原Event+Graph序列，不插CUDA计算或profiler。ON用原0.5s WSL NVML并记录查询延迟，OFF无窗口内查询；只比较机制，不用144段代替完整三AA、不签证书。旧失败全部保留。脚本sm86_nvml_diagnostic_host.py/run_sm86_nvml_diagnostic.py；CPU support生成/编译锚点PASS，须等3060完整结束后只执行一次。

3070首个诊断host在PowerShell完成返回时遇到stderr GBK→UTF8默认解码失败；在boot/idle admission及任何clock set/GPU worker之前，0Event。外层finally复位命令和所有原始文件归档保留。仅修正PowerShell Console.OutputEncoding=UTF8，Python保留stdout/stderr原始bytes并显式decode stdout UTF8，另新root/out以utf8后缀绑定；同8block/计算/阈值不动，不重采已采GPU样本。

实测收尾：3060完整320QA/32Sanitizer/两fault恢复、三AA三AB1728段通过，960文件CPU审计PASS，最坏0.479820%/组0.185669%、最低保护0.999335845×，收尾0%/1MiB/40°C/170W/无项目容器。3070修正编码后完整8block144段，ON17.024809%、OFF5.272610%；141文件逐SHA和原生成support、帧、见证、所有双时钟统计CPU审计PASS，不能以关监测当充分修复。Event与RAW配对比最大差0.000107852；未定位WSL/驱动具体根因。两项时钟内外finally复原/240W保持，无正式3070资格。

原HAGS恢复只读计划prepare_sm86_hags_restore已核对当前2和原backup缺失状态，复用既有windows_hags_setting.ps1 Restore/Enable及DxDiag，不新设计系统控制；尚未改注册表或重启。待用户选择继续3070维护重启确认或3060独立生产交付，0新API/编译/封存执行。
