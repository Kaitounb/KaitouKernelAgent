# 可运行交付与未完成验收

2026-11-24：3060 NCU真实3 kernel/19指标、报告重导入与FP64复核完成，当前限定工程交付已结项。[工程结项与延期范围](PROJECT_CLOSEOUT.md) · [NCU](RTX3060_NCU.md)。旧目标研究状态保持，以下历史阻塞不覆盖本次已交付范围。

2026-11-24 最新范围：RTX3060 GEMM 0.1.3正式交付已完成，完整三AA三AB/独立16项/真实回退/入口和恢复备份通过。[3060正式运行](RTX3060_GEMM_RUNTIME.md)。本次无新增阻塞；3070 GEMM原拒绝及4090目标C延期仍保留，3070 Attention/MLA0.1.1保持。下方为历史或其他目标，不能据其旧production=false覆盖本发行。


2026-11-23 当前交付已完成：3070固定可信Attention/MLA的0.1.1 qualified通道已实际部署。三AA三AB1080段通过，最坏单对1.935923%（原2%门槛）、组偏差0.447360%、AB最低0.999478866×；1668文件和1632输出数组独立CPU复核PASS。Attention新合成独立8/8、239私有文件/NumPy FP64重算PASS；MLA继承同AOT/未变数值合同的原独立8调用16输出，旧sealed不读不传不重跑。rc1安装/恢复37CPU、13输出17尝试保持；rc2严格SDK准入40CPU/2输出和四拒绝PASS；正式0.1.1安装40CPU、9输出11尝试（7qualified+2显式预览回退）、五项真实资格负例、正式→rc2→正式包切换及5160文件/NPY/边界CPU审计PASS。12运行文件与rc2逐字相同，wheel f2ef0ec1f054a9331a4134e6545e6ce2a84b5aae22923f81063cb5c7418a2706，证书261b6fcb4c44ba3de7c88998b6d7e7f375350c3eb3f37b1d208be6eab0f9d3f4。原独立分配输入540段/5.598675%拒绝保留；共享物理只读input ring方法更正在采样前冻结且只执行一次，原标准不变。生产范围为当前SM86固定FlashInfer Attention和page16/split1 MLA core，不覆盖Standalone Softmax、3070 GEMM旧失败或自由源码；原4090目标C按用户决定延后未运行。无项目GPU/API在途，无新增DeepSeek（仍190/700），无新CUDA编译或系统设置变化。入口及两个可运行示例见docs/SM86_ATTENTION_MLA_RUNTIME.md；发行清单releases/sm86-operators-v0.1.1/manifest.json。以下旧检查点按当时范围阅读，不重新启动已消耗批次/独立验收或重建旧版本wheel。

2026-11-23 首个生产通道已交付：0.1.0的SM89固定可信cuBLAS qualified已在当前4090部署，入口/状态见docs/SM89_RUNTIME.md。证据：320QA+32Sanitizer；三AA三AB1728段(最大0.664015%、最低保护0.999909889×)；新服务端合成独立16/16、246私有文件本机FP64复核PASS(1.907349e-6)，未读/传/重跑旧3070/MLA封存数据；rc2预览安装27CPU/9输出11尝试及实际dev7包回滚PASS；正式0.1.0安装30CPU/6输出8尝试(5qualified+1显式预览退回)、581文件CPU审计PASS。正式wheel76224f3435764e5484e3434235e9f95a4b70cbe92476ac9f72c9df86bd3fb18f，资格证书9451476fc9ac796ff1e5c4617c9414b0b7006366a0790f184618a8747a386c77，发行清单releases/sm89-gemm-v0.1.0/manifest.json。可信控制root、worker UID/GID65534/零cap/NoNewPrivs、IPv4/IPv6拒绝/Unix IPC保留、控制目录不可读/产物输入只读/有界日志超时、故障和版本/实际包回滚均实测；无PID/mount/network namespace或RAM cgroup独立限额，不执行任意模型源码。production=true仅限SM89_REVIEWED_FIXED_CUBLAS；SM86和Attention/MLA及完整目标C保持各自状态，不冒称全项目生产通过。所有GPU/API任务已结束，0新DeepSeek(仍190/700)，无新增Event重采。新控制Python和入口在/data，旧包/证据/权限保留；当前SSH显式用.private/server-gpu-next-connection.json。以下历史检查点按当时范围阅读。

2026-11-23 4090 v2完整实测已通过并归档：SM89/Torch2.7.1+cu126/CUDA12.6.85/driver570.86.10；两可信cuBLAS绑定fresh build，320QA、32Sanitizer(0errors)、crash/timeout拒绝及两fresh恢复通过。三AA+三AB共1728原始Event段，AA最坏单对0.664015%、最坏组偏差0.570454%（也满足旧2%），AB最低保护0.999909889×>=.97。917文件逐SHA/NKA1/NPY/见证/Graph/pilot/统计CPU审计AUDIT_PASS_STABILITY_PASS；protocol8a023ee57b57948c72866304e74dabd4f50fe27a2249278473c31e4572961065，归档68eb546027baf486b5db2ff46bca89876f7fa6bcf42696d8e0bb99d89ea5a86b。报告docs/SM89_CUBLAS_STABILITY.md，evidence/20261123-sm89-cublas-v2。所有会话已结束，GPU0%/1MiB/无compute进程；0新API(仍190/700)/0NCU/0sealed/0宿主时钟变更，无best晋升或优化收益。实例无Docker，本轮是可信native固定SDK，不构成完整源码worker隔离或生产发布；production仍false。下一项SM89目标安装/回滚及相应执行边界，P6-03完整目标C、P6-04/05仍未整体验收；SM89结论不回填3070。旧连接/备份保持，新连接须显式使用.private/server-gpu-next-connection.json。不要重跑本批GPU入口，只读audit_sm89_cublas。

2026-11-23 用户修订部署标准：新增workloads/gemm_deployment_stability_v2.json，A/A单对<3%、组均值偏差<1.5%，完整三session等原采样要求不变；仅部署准入，原正式研究/搜索门槛及旧2%拒绝不改。四批历史CPU逐SHA/原始段复核完成：完整锁频1秒2.040408%（最大组偏差0.286466%）与250ms2.849277%在新v2下A/A单项通过；原4.274482%及不完整HAGS On/5.142371%不通过。明确为观察后标准修订/回顾复核，0新GPU/API、0AB/证书；不把HAGS Off成绩继承给当前HAGS On。下一步等4090连接，在新SM89环境先正确性再冻结v2/三AA→三AB；NCU可选，隔离条件另验收，目标卡通过不回填3070。报告docs/DEPLOYMENT_STABILITY_V2.md，结果evidence/20261123-deployment-stability-v2/verification.json。P6-03/04/05未整体验收，预算190/700。以下旧2%检查点按当时政策阅读。

2026-11-23 重启后最终检查点：用户手动重启3070已完成，DxDiag实际HAGS Supported/Enabled均True，bootUTC=2026-11-23T06:52:14.5000000Z；同GPU/驱动591.74/固定Docker镜像。原dev6/AOT从归档按15控制文件/1456文件SHA恢复到持久目录。HAGS新协议ec6bbbc...保留原1秒host与全部阈值：320QA通过，仅完成AA0/288段，最坏单对5.142371%（原2%门槛）；下一轮前GPU空闲条件失败停止，3AA未完成、0AB、锁频恢复成功。590文件归档+物理环境/帧/参考/Graph/clock CPU审计AUDIT_PASS_INCOMPLETE_QUALIFICATION_REJECT；首次后台启动的0QA/0Event WSL失败21文件另存，不重采任何已采身份。证据evidence/20261123-cublas-hags-qualification/verification.json；本轮无GPU/API在途，无资格证书/生产发布。4090可不用NCU执行可信SDK正确性和Event；NCU只阻塞完整P6-03目标C，不是首个SM86 GEMM生产运行门槛。现4090无已验收源码隔离，仍不执行自由生成CUDA；不为NCU单独要求换服务器，旧实例和备份未动。P6-03/04/05保持未勾，预算仍190/700，独立sealed不再执行。详docs/4090_WITHOUT_NCU.md、docs/PRODUCTION_BLOCKERS.md。下方旧检查点按历史阅读。

现有开发包 `0.1.0.dev6` 保留；新增 `0.1.0rc1` 未发布预览已新prefix安装、19项CPU/5输出/7尝试及实际回滚通过，见[RC1说明](RC1_PREVIEW.md)。两者生产发布标志仍为 false。正式研究和运行时准入分开：没有通过筛选的模型候选不会替换可信默认实现。

| 可用入口 | 实际实现 | 实测和证据 |
|---|---|---|
| `kernel-agent-operators --channel qualified` | **0.1.1生产范围：当前SM86固定Attention/MLA core（page16/split1）**，完整SDK白名单与低权限Docker | 三AA三AB/1080段、Attention新独立8/8、40CPU/9最终GPU输出11尝试及真实包回滚通过；[运行](SM86_ATTENTION_MLA_RUNTIME.md) |
| `kernel-agent-sm89 --channel qualified` | **0.1.0生产范围：SM89固定可信cuBLAS**，低权限/拒绝IP网络/哈希白名单与有界回退 | 新独立16/16、30CPU、6GPU输出8尝试、实际包回滚及581文件审计PASS；[部署入口](SM89_RUNTIME.md) |
| `kernel-agent` | 默认 cuBLAS，独立审查版本、故障回退、原子版本回滚；有界离线 CUDA 优化 | GEMM RC1 独立16/16输出；公开恢复；dev6→dev5实际应用回滚 |
| `kernel-agent-operators` | 可信 PyTorch CUDA Softmax、FlashInfer fa2 Attention/MLA；AOT，无运行时下载编译 | 新prefix/空cwd安装，5份独立CPU FP64正确输出、2次预期失败，7次GPU尝试；两卡公开/开发/147配置检查另存 |
| 正式搜索控制器 | 固定初始父版本 A/B/C、全局模型预算、防重发、结果缓存、独立外置裁决 | 见正式协议；新源码只经已测低权限容器，4090不执行自由模型源码 |

安装包位于 `artifacts/wheels/nvidia_kernel_agent-0.1.0.dev6-py3-none-any.whl`，SHA256 `d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38`。不要用现有源码重建另一个同版本 wheel。新的研究辅助模块随项目源码交付，不能称为原 dev6 wheel 已包含它们。

SM86 算子 AOT 归档位于 `artifacts/operator-runtime/sm86-cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb.zip`，SHA256 `939bf2d6d71a8d47fff652f944addb16857167e9f4ffbabd83c37a7e9a5ae5fa`。运行包、原生二进制、环境检查和第三方许可绑定在 manifest 中；不支持的卡型/软件组合在执行前拒绝。

详细命令与声明环境：

- [GEMM运行/回退](SM86_RUNTIME.md)、[离线源码优化](OFFLINE_OPTIMIZER.md)、[双二进制外置筛选](DUAL_BINARY_EVALUATION.md)。
- [Softmax/Attention/MLA AOT运行](OPERATOR_RUNTIME.md)。
- [真实两卡支持矩阵与NCU权限](SERVER_COMPATIBILITY.md)。
- [正式实验协议和信息边界](FORMAL_STUDY_PROTOCOL.md)。

独立正确性：GEMM一次性独立16/16继续有效；MLA默认page16/split1在用户具体批准后按冻结协议一次执行，8/8 GPU调用、16/16输出通过。204私有文件的SHA、帧、NPY、CPU FP64参考结果及隔离记录复核PASS；候选不挂载完整清单与参考，模型未见精确shape/seed。使用冻结dev6 wheel，RC1相关13个文件静态逐字一致；不重复独立验收。[MLA验收报告](MLA_RC1_ACCEPTANCE.md)。该流程没有性能采样，不能替代最终三session性能资格。

原预览manifest与此前审批拒绝原文保留；新状态另存 `releases/full-project-rc1-mla-acceptance-update`，引用新授权与验收证据。

4090目标卡计数器仍返回 `ERR_NVGPUCTRPERM`。普通可信SDK调参可继续，目标卡硬件反馈再优化未运行，不能写成完整跨卡消融。经典FlashMLA固定版本需要SM90及CUDA≥12.8；两张现有卡不支持，专家成绩为null。

原WMMA/MLA已失败的A/A身份完整保留并禁止挑选复测；新身份的A/A、A/B和编译失败按真实结果分别归档。未达到收益目标时报告未达到，保留cuBLAS/SDK默认。最终生产标志必须由实际准入结果更新，不能因接口可运行或实验完成而直接改为true。

生产默认cuBLAS最终三AA也未通过：320正确性通过、864段最坏4.274%，无AB/资格证书。当前两个准入阻塞及证据见[生产阻塞](PRODUCTION_BLOCKERS.md)。

2026-11-23修复补充：默认cuBLAS两个新增冻结资格各320QA/864段，锁频250ms2.849%、1秒持续平均2.040408%，仍拒绝，0AB/0资格证书。HAGS实际支持但关闭，现已备份配置DWORD2、待用户维护重启和实际生效验证；4090需平台增加计数器capability。旧包/二进制/封存验收未变，生产仍false。[修复证据](STABILITY_REPAIR.md) · [维护方案](3070_HAGS_MAINTENANCE.md) · [平台工单](4090_NCU_SUPPORT_REQUEST.md)。
