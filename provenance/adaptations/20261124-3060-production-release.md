# RTX3060 固定可信 GEMM 生产交付

用户在3060对照通过后明确要求推进正式交付。复用上游KernelBench423217d9fda91e0c2d67e4a43bf62f96f6d104f1和现有Registry/deployment/DockerRunner/pipe_worker/NKA/NPY/原子版本指针，参考已交付SM89的独立catalog/CLI、独立验收、preview→GA及真实包回滚。CUDA二进制/数值合同/原pipe_worker/原3070组件不改。

已确认具体缺口：通用DockerRunner硬编码旧3070 image/legacy --gpus；3060实际必须使用已验收的新image/nativeCDI。采用独立cdi_runner（原runner仅image和GPU参数替换）、独立CDI catalog/资格工厂和CLI，复用原Registry(data,catalog)/deployment(registry_factory,runner_factory)扩展点，避免改变已有3070/SM89已签runtime的共享文件。CDI数值worker仍使用同目录原pipe_worker.py。专用数据目录复用相同数值合同和两个SM86产物，原资格目录不变。新不可变版本0.1.3rc1→0.1.3，新增kernel-agent-3060命令；旧版本/3070入口不覆盖。

目标固定3060 UUID GPU-d464a726-39c2-9ad5-d856-f5d00c1fa870、driver580.95.05、image7ed9d551...、Torch2.14+cu130/CUDA13、默认170W/动态频率、温度<83°C。只准两个reviewed cuBLAS，不开放任意模型源码；qualified需证据证书/全运行代码SHA/合同/二进制/policy/GPU/driver/image/profile，未验收则拒绝。

复用完整3AA3AB/1728段对照与960文件审计，不重采已通过计时。新preview安装先实际边界/故障/两binary回滚/GA→已验收rc1 development→GA真实包切换（3060没有可工作的旧0.1.1镜像，不试图执行不支持的旧目标）；独立验收在preview wheel固定后由服务端生成全新合成清单，一次16输出，私有目录不展示shape/seed，不读取/传输旧sealed。客户端只保存新私有证据，CPU脚本逐SHA/NKA/NPY/NumPy FP64复算；随后签证书/构建唯一GA/实际qualified执行与回滚/完整负例/收尾，最后文档/发行manifest。

宿主Python3.11无Torch，使用已验证同image可信父容器（user1000+Docker group、socket仅父持有、netnone/RO/zeroCaps/NNP，同主机绝对路径）运行CLI/评测；GPU worker保持65534、2只读mount、无socket/参考/全清单。launcher约束用户输入/输出/状态路径在该部署目录workspaces中；不会暴露任意宿主目录。完整失败原始证据保留，0新模型/新CUDA编译/旧sealed。

验证命令将按实际stage/preview/independent/audit/GA分别追加；完成后才更新TASKS验收。

实际CPU结果：54回归通过；新12运行文件与0.1.3rc1 wheel逐字冻结、原3070 GEMM的10运行文件与0.1.2rc1逐字相同、hostlauncher三CPU检查通过。固定setuptools80.9.0/wheel0.45.1隔离构建，wheel c81d6a10d78b034d60f5d2163bdf55720eeb06a5ee0e9140076dc26dcad102e9；原版本不重建。build_cdi_release、install_cdi_release、cdi_release_validate及independent/audit/issue脚本均准备，未把静态结果当实卡。

首次安装transport在SSH认证前关闭，stderr原样保留；域名系统DNS198.18.0.172、两独立DoH一致真实183.223.25.19，直连保留旧host-key严格核验同样KEX前关闭，非已证实密码错误或CUDA失败。临时SSH helper诊断扩展已撤回，原默认接口不变。当前没有新安装/独立验收证据，签证producer对缺真实preview审计的拒绝已CPU实测，0certificate/GA/GPU/API。恢复时用resume_cdi_release_stage先只读清点，已有reserved validation不重播；旧失败/数据保持。

验证：PYTHONPATH=tests /tmp/nka-delivery-verify-venv/bin/python -m unittest test_registry test_deployment_policy test_qualified_channel test_sm89_runtime test_sm89_qualified test_operator_qualification test_gemm_deployment_policy test_gemm_target_binding test_cdi_release；python3 -m experiments.full_delivery.build_cdi_release --version 0.1.3rc1（已消耗，不重建）；install_cdi_release --phase preview的首次SSH失败不重放，只走安全清点恢复。脚本语法及launcher静态检查见evidence/20261124-3060-release-cpu。

用户重新租赁3060：新连接另存.private/server-gpu-3060-next-connection.json(0600)，旧凭据/证据保持。只读预检确认GPU UUID变为GPU-0140b895-e4e9-7c67-aba1-c7a953df1847，driver仍580.95.05、SM86/12GiB、nativeUbuntu22.04/kernel5.15、170W/空闲；旧root/image不存在，只有同配置Ubuntu24.04 base7a398144...。CDI已有GPU0规范。复用原已验证Dockerfile与国内源/官方Torch SHA pin构建新image；原0.1.3rc1固化旧UUID/image，不重建或修改它，新绑定使用0.1.3rc2，继承相同数值二进制/合同/worker。因目标GPU/image变化，新增完整320QA/32Sanitizer/故障恢复/三AA→条件三AB，一次冻结新目标、完整原始证据，不继承旧物理卡成绩，不读取旧sealed。源build_3060_next_runtime.py只增加新namespace/parent创建及base ID断言。

新目标实际image已构建：sha256:6ce2012265c68b8553adf8d7807dcf47972a9156d1c7f0a2a64db11f9827e55a，约3.16GB，CPU构建1265.7秒（下载和Docker29封装较慢），原始完整日志保存。实际GPU ABI/UID/网络/只读/资源边界通过，5核心数值库SHA与旧image相同；新目标320QA/32Sanitizer/两fault恢复通过。0.1.3rc2 SHA5b45358a5dd5b8608760faedf3442546bc51fa48db87680db95095db33190412已实卡安装54CPU/7输出9尝试/回退/边界/版本回滚及419文件独立CPU审计PASS。新完整计时protocol19912fbc...仅一次，当前执行中，原阈值/样本数/Graph/共享ring不变；计时期间不做镜像导出或重I/O。后续新私有16/GA/激活/镜像备份尚未执行。

镜像备份复用Docker官方save/load和现有SSH lossless SHA方式。新Docker29返回的image ID实际是manifest-list6ce...，config是3067...，不能复用“ID等于config hash”的旧简化假设。只新增离线归档校验适配，按官方OCI descriptor/layout核验全部blob digest/size、index→manifest→config引用链和各层未压缩DiffID；不重写镜像加载器。参考 https://docs.docker.com/engine/storage/containerd/ 、https://specs.opencontainers.org/image-spec/image-layout/ 、https://specs.opencontainers.org/image-spec/config/ 。CPU合成归档用于验证legacy/config与OCI/index两格式及损坏拒绝；实际新镜像导出/恢复须待正式GPU计时结束。

2026-11-24 PID诊断更正：冻结私有host.py使用控制容器os.getpid而非native宿主PID，16个host_pid_visible观测全部不作宿主隔离证据；数值及其余边界原始帧不改。原失败审计代码/日志私有留存。公开补充使用SSH native进程存活PID、相同RC2 wheel/runner/image与默认私有PID namespace，已完成一次实际GPU检查；CPU整档审计后，独立正确性继承该共享隔离组件证据。无私有清单重跑或新样本/计时。验证：audit_cdi_pid_supplement → audit_cdi_independent；证书必须绑定补充审计SHA及明确披露。

GA安装首次validator的8项功能/负例及9th boundary数值已完成，但末尾仍用控制容器PID断言，原result FAIL及468文件归档保留。未修改冻结wheel或重跑功能批；另对同一GA包执行真实native-host PID公开补充。CPU安装审计严格要求原失败只为host_pid_visible诊断、其余9输出/11尝试和54CPU/回滚/负例全验、并绑定GA实际PID补充；原始FAIL不覆盖，正式接受由补充后的审计单独给出。

后续安装工具修复：cdi_release_validate新增必需--host-pid，由外层native SSH installer传入os.getpid；不再取容器PID。已执行批次的远端source/清单/FAIL/result和冻结wheel不修改、不重跑。cdi_independent_host.py是本次已冻结验收源，保持原字节，仅供证据复核不作未来重新运行入口。
